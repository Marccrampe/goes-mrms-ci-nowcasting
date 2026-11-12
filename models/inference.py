#!/usr/bin/env python3
"""Portable strict-loading inference for a miniature GOES--MRMS sample."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

try:
    from .resunet import build_final_model, build_goes_only_model
except ImportError:
    from resunet import build_final_model, build_goes_only_model


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def physical_inputs(npz_path, normalization_path, include_mrms=True):
    data = np.load(npz_path)
    required = {"goes_bt_k", "mrms_reflectivity_dbz", "mrms_valid"}
    missing = required.difference(data.files)
    if missing:
        raise RuntimeError("Miniature sample is missing: " + ", ".join(sorted(missing)))
    goes = np.asarray(data["goes_bt_k"], np.float32)
    radar = np.asarray(data["mrms_reflectivity_dbz"], np.float32)
    valid = np.asarray(data["mrms_valid"], np.float32)
    if goes.ndim == 4:
        goes, radar, valid = goes[None], radar[None], valid[None]
    if goes.ndim != 5 or tuple(goes.shape[1:3]) != (4, 7):
        raise RuntimeError("goes_bt_k must have shape [N,4,7,H,W] or [4,7,H,W]")
    if radar.shape != (goes.shape[0], 4, 1, *goes.shape[-2:]):
        raise RuntimeError("MRMS reflectivity shape does not match GOES")
    if valid.shape != radar.shape:
        raise RuntimeError("MRMS validity shape does not match reflectivity")
    cfg = json.loads(Path(normalization_path).read_text())
    mean = np.asarray(cfg["mean_k"], np.float32).reshape(1, 1, 7, 1, 1)
    std = np.asarray(cfg["std_k"], np.float32).reshape(1, 1, 7, 1, 1)
    goes = (goes - mean) / std
    radar = np.clip(radar, 0, 70) / 70 * valid
    x = np.concatenate([goes, radar, valid], axis=2) if include_mrms else goes
    if not np.isfinite(x).all():
        raise RuntimeError("Non-finite model inputs")
    return torch.from_numpy(np.ascontiguousarray(x))


def load_model(checkpoint_path, device):
    # The public inference checkpoint contains tensors and a minimal, path-free configuration.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint.get('config', {})
    include_mrms = config.get('include_mrms', True)
    model = build_final_model() if include_mrms else build_goes_only_model()
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def predict(model, x):
    """Original four fixed 128-pixel tiles, averaging probabilities in overlaps."""
    h,w=x.shape[-2:]
    if h<=128 and w<=128:
        return model(x).sigmoid()
    if min(h,w)<128 or max(h,w)>256:
        raise RuntimeError('Original four-tile inference requires both dimensions between 128 and 256')
    import itertools
    probability=torch.zeros((len(x),2,h,w),device=x.device,dtype=x.dtype)
    count=torch.zeros((h,w),device=x.device,dtype=x.dtype)
    for y,col in itertools.product((0,h-128),(0,w-128)):
        probability[...,y:y+128,col:col+128]+=model(x[...,y:y+128,col:col+128]).sigmoid()
        count[y:y+128,col:col+128]+=1
    if not (count>0).all():raise RuntimeError('Uncovered inference pixel')
    return probability/count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Miniature physical-input NPZ")
    parser.add_argument("--normalization", default="preprocessing/dataset/normalization.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--expected-checkpoint-sha256")
    args = parser.parse_args()
    if args.expected_checkpoint_sha256:
        actual = sha256(args.checkpoint)
        if actual != args.expected_checkpoint_sha256:
            raise RuntimeError("Checkpoint SHA-256 mismatch")
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    x = physical_inputs(args.input, args.normalization, model.include_mrms).to(device)
    with torch.no_grad():
        probability = predict(model,x).cpu().numpy().astype(np.float32)
    if not np.isfinite(probability).all() or probability.min() < 0 or probability.max() > 1:
        raise RuntimeError("Invalid probabilities")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, probability=probability)
    print("OUTPUT", Path(args.output).resolve())
    print("SHAPE", probability.shape)


if __name__ == "__main__":
    main()

