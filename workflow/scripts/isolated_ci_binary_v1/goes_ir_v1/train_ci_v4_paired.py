"""Paired soft/binary CI training. N2 raw ResUNet, common local ignores, fixed updates."""
from ci_nowcasting.paths import resolve_path as _ci_resolve, project_root as _ci_project, source_root as _ci_source
import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler, Subset
PROJECT = Path(_ci_resolve(_ci_resolve('/project')))
DATA = PROJECT / 'data/ci_isolated_mrms_binary_v1/goes_ir_parallax_native_filled_v1'

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, out_channels)
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), nn.GroupNorm(8, out_channels))
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        x = F.silu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.silu(x + residual)

class EncoderStage(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.blocks = nn.Sequential(ResidualBlock(in_channels, out_channels, stride=stride), ResidualBlock(out_channels, out_channels))

    def forward(self, x):
        return self.blocks(x)

class DecoderStage(nn.Module):

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.block = nn.Sequential(ResidualBlock(out_channels + skip_channels, out_channels), ResidualBlock(out_channels, out_channels))

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)

def ci_loss(logits, targets, masks, positive_weight=50.0, dice_weight=0.5, negative_weight=1.0):
    pixel_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    weights = 1.0 + (positive_weight - 1.0) * targets
    weights = weights * (1.0 + (float(negative_weight) - 1.0) * (targets == 0).to(targets.dtype))
    weighted_mask = weights * masks
    bce = (pixel_loss * weighted_mask).sum() / weighted_mask.sum().clamp_min(1.0)
    probabilities = torch.sigmoid(logits) * masks
    masked_targets = targets * masks
    numerator = 2.0 * (probabilities * masked_targets).sum(dim=(2, 3))
    denominator = probabilities.sum(dim=(2, 3)) + masked_targets.sum(dim=(2, 3))
    dice = (numerator + 1.0) / (denominator + 1.0)
    positive_leads = masked_targets.sum(dim=(2, 3)) > 0
    if positive_leads.any():
        dice_loss = (1.0 - dice[positive_leads]).mean()
    else:
        dice_loss = logits.sum() * 0.0
    total = bce + dice_weight * dice_loss
    return (total, bce.detach(), dice_loss.detach())

class ConvLSTMCell(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.gates = nn.Conv2d(2 * channels, 4 * channels, 3, padding=1)
        self.channels = channels
        nn.init.zeros_(self.gates.bias)
        with torch.no_grad():
            self.gates.bias[channels:2 * channels].fill_(1.0)

    def forward(self, x, state):
        if state is None:
            state = (torch.zeros_like(x), torch.zeros_like(x))
        h, c = state
        i, f, o, g = self.gates(torch.cat([x, h], 1)).chunk(4, 1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return (h, c)

def early_fusion_features(x, include_mrms, input_mode):
    """Raw four-frame inputs plus three causal differences, when requested.

    GOES deltas use the existing channel standardization. Radar deltas use
    the existing 0..70 dBZ scaling, gated by BOTH scans' validity masks.
    Pair-validity channels accompany radar deltas; validity itself is not
    differenced. No future scan is used and no raw channel is removed.
    """
    channels = 9 if include_mrms else 7
    if x.ndim != 5 or x.shape[1] != 4 or x.shape[2] != channels:
        raise RuntimeError('Expected B,4,C,H,W with the configured channels')
    raw = x.flatten(1, 2)
    if input_mode == 'raw':
        return raw
    if input_mode != 'raw_delta':
        raise ValueError('Unknown input mode ' + input_mode)
    goes_delta = x[:, 1:, :7] - x[:, :-1, :7]
    if include_mrms:
        pair_valid = x[:, 1:, 8:9] * x[:, :-1, 8:9]
        radar_delta = (x[:, 1:, 7:8] - x[:, :-1, 7:8]) * pair_valid
        delta = torch.cat([goes_delta, radar_delta, pair_valid], dim=2)
    else:
        delta = goes_delta
    return torch.cat([raw, delta.flatten(1, 2)], dim=1)

class V3Model(nn.Module):

    def __init__(self, architecture='resunet', include_mrms=False, width=48, input_mode='raw'):
        super().__init__()
        if input_mode not in {'raw', 'raw_delta'}:
            raise ValueError('Unsupported input mode')
        if architecture != 'resunet' and input_mode != 'raw':
            raise ValueError('This ablation adds explicit deltas only to ResUNet')
        self.architecture = architecture
        self.include_mrms = include_mrms
        self.input_mode = input_mode
        channels = 9 if include_mrms else 7
        b = width
        encoder_channels = 4 * channels if architecture == 'resunet' else channels
        if input_mode == 'raw_delta':
            encoder_channels += 3 * (9 if include_mrms else 7)
        self.enc1 = EncoderStage(encoder_channels, b, 1)
        self.enc2 = EncoderStage(b, b * 2, 2)
        self.enc3 = EncoderStage(b * 2, b * 4, 2)
        self.enc4 = EncoderStage(b * 4, b * 8, 2)
        self.bottleneck = EncoderStage(b * 8, b * 12, 2)
        if architecture == 'convlstm':
            self.temporal = ConvLSTMCell(b * 12)
        self.dec4 = DecoderStage(b * 12, b * 8, b * 8)
        self.dec3 = DecoderStage(b * 8, b * 4, b * 4)
        self.dec2 = DecoderStage(b * 4, b * 2, b * 2)
        self.dec1 = DecoderStage(b * 2, b, b)
        self.head = nn.Conv2d(b, 2, 1)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.head.bias, -4.0)

    def encode(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        return (self.bottleneck(e4), (e1, e2, e3, e4))

    def forward(self, x):
        if x.ndim != 5 or x.shape[1] != 4:
            raise RuntimeError('Expected B,4,C,H,W')
        if self.architecture == 'resunet':
            z, skips = self.encode(early_fusion_features(x, self.include_mrms, self.input_mode))
        else:
            state = None
            for frame in x.unbind(1):
                z, skips = self.encode(frame)
                state = self.temporal(z, state)
            z = state[0]
        e1, e2, e3, e4 = skips
        z = self.dec4(z, e4)
        z = self.dec3(z, e3)
        z = self.dec2(z, e2)
        z = self.dec1(z, e1)
        return self.head(z)

class FractionBatchSampler(Sampler):
    """Fixed update budget with exact epoch-level issuance-class quotas.

    Natural uses the retained train cohort's issuance fraction. Batches have
    either floor(B*f) or ceil(B*f) positives; quotas are shuffled each epoch.
    Sampling is uniform within classes, without replacement when possible.
    """

    def __init__(self, flags, batch_size, epoch_samples, seed, fraction):
        flags = np.asarray(flags, dtype=bool)
        if batch_size < 2 or epoch_samples < batch_size:
            raise ValueError('batch_size >=2 and a nonempty epoch required')
        self.pos = np.flatnonzero(flags)
        self.neg = np.flatnonzero(~flags)
        if not len(self.pos) or not len(self.neg):
            raise ValueError('Both issuance classes required')
        self.requested_fraction = str(fraction)
        self.fraction = float(flags.mean()) if fraction == 'natural' else float(fraction)
        if not 0 < self.fraction < 1:
            raise ValueError('Fraction must be in (0,1)')
        self.batch_size = int(batch_size)
        self.batches = int(epoch_samples) // self.batch_size
        self.seed = int(seed)
        self.epoch = 0
        self.samples = self.batches * self.batch_size
        self.positive_samples = int(np.floor(self.samples * self.fraction + 0.5))
        self.negative_samples = self.samples - self.positive_samples
        if not self.positive_samples or not self.negative_samples:
            raise ValueError('Epoch must contain both classes')
        self.realized_fraction = self.positive_samples / self.samples

    def __len__(self):
        return self.batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        p = rng.choice(self.pos, self.positive_samples, replace=self.positive_samples > len(self.pos))
        q = rng.choice(self.neg, self.negative_samples, replace=self.negative_samples > len(self.neg))
        base, extra = divmod(self.positive_samples, self.batches)
        quotas = np.full(self.batches, base, dtype=np.int64)
        quotas[:extra] += 1
        rng.shuffle(quotas)
        ip = iq = 0
        for quota in quotas:
            k = int(quota)
            n = self.batch_size - k
            batch = np.concatenate([p[ip:ip + k], q[iq:iq + n]])
            ip += k
            iq += n
            rng.shuffle(batch)
            yield batch.tolist()
        assert ip == self.positive_samples and iq == self.negative_samples

def make_inputs(batch, device, include_mrms):
    x = batch['inputs'].to(device, non_blocking=True)
    if include_mrms:
        valid = batch['mrms_valid_masks'].to(device, non_blocking=True)
        radar = batch['mrms_reflectivity_dbz'].to(device, non_blocking=True)
        radar = radar.clamp(0, 70) / 70 * valid
        x = torch.cat([x, radar, valid], 2)
    if not torch.isfinite(x).all():
        raise RuntimeError('Nonfinite model input')
    return x

@torch.no_grad()
def evaluate(model, loader, device, args):
    model.eval()
    tp = np.zeros(2)
    fp = tp.copy()
    fn = tp.copy()
    brier = valid_total = loss_total = examples = raw_bce_total = 0.0
    for bi, batch in enumerate(loader):
        if args.max_val_batches and bi >= args.max_val_batches:
            break
        x = make_inputs(batch, device, args.include_mrms)
        target = batch['targets'].to(device)
        mask = batch['loss_masks'].to(device)
        h, w = x.shape[-2:]
        prob = torch.zeros_like(target)
        count = torch.zeros((h, w), device=device)
        for y in (0, h - 128):
            for col in (0, w - 128):
                logits = model(x[..., y:y + 128, col:col + 128])
                prob[..., y:y + 128, col:col + 128] += logits.sigmoid()
                count[y:y + 128, col:col + 128] += 1
        if not (count > 0).all():
            raise RuntimeError('Uncovered validation pixel')
        prob /= count
        loss, _, _ = ci_loss(torch.logit(prob.clamp(1e-06, 1 - 1e-06)), target, mask, args.positive_weight, args.dice_weight, args.negative_weight)
        n = len(x)
        loss_total += float(loss) * n
        examples += n
        brier += float(((prob - target).square() * mask).sum())
        valid_total += float(mask.sum())
        raw_bce_total += float((F.binary_cross_entropy(prob.clamp(1e-06, 1 - 1e-06), target, reduction='none') * mask).sum())
        pred = prob >= args.threshold
        obs = target >= 0.5
        valid = mask > 0
        tp += (pred & obs & valid).sum((0, 2, 3)).cpu().numpy()
        fp += (pred & ~obs & valid).sum((0, 2, 3)).cpu().numpy()
        fn += (~pred & obs & valid).sum((0, 2, 3)).cpu().numpy()
    if not examples or not valid_total:
        raise RuntimeError('No valid validation samples')
    csi = tp / np.maximum(tp + fp + fn, 1)
    return dict(validation_loss=loss_total / examples, validation_brier=brier / valid_total, validation_bce_unweighted=raw_bce_total / valid_total, validation_csi_lead_0=float(csi[0]), validation_csi_lead_1=float(csi[1]))

def cohort_check(args):
    result = {}
    keys = {}
    for split in ['train', 'validation', 'test']:
        natural = pd.read_parquet(_ci_resolve(args.dataset_root / f'{split}_issuance_index_natural.parquet'))
        ready = pd.read_parquet(_ci_resolve(args.dataset_root / f'{split}_issuance_index_common.parquet'))
        cols = ['period_id', 'issuance_time']
        if ready.duplicated(cols).any() or not ready.mrms_history_complete_after_repair.astype(bool).all():
            raise RuntimeError('Invalid complete cohort ' + split)
        if len(ready.merge(natural[cols], on=cols, how='inner')) != len(ready):
            raise RuntimeError('Unknown cohort keys')
        result[split] = dict(eligible=len(natural), retained=len(ready), excluded=len(natural) - len(ready), positive_fraction=float(ready.any_usable_positive.mean()))
        keys[split] = set(ready.period_id.astype(str))
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        if keys[a] & keys[b]:
            raise RuntimeError('Period leakage')
    total = sum((v['excluded'] for v in result.values()))
    if total != args.expected_exclusions:
        raise RuntimeError(f'Exclusions changed: {total}, expected {args.expected_exclusions}')
    return result

def selftest():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(2)
    for architecture in ['resunet', 'convlstm']:
        for radar in [False, True]:
            model = V3Model(architecture, radar, width=8).to(device)
            x = torch.randn(1, 4, 9 if radar else 7, 128, 128, device=device, requires_grad=True)
            z = model(x)
            assert z.shape == (1, 2, 128, 128)
            target = torch.zeros_like(z)
            target[:, :, 50:53, 50:53] = 1
            mask = torch.ones_like(z)
            loss, _, _ = ci_loss(z, target, mask, 25, 0)
            loss.backward()
            assert torch.isfinite(loss) and all((torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            assert all((x.grad[:, t].abs().sum() > 0 for t in range(4)))
            print('ARCHITECTURE + FOUR-FRAME GRADIENT PASSED', architecture, radar, flush=True)
    flags = np.array([True] * 74 + [False] * 26)
    sampler = FractionBatchSampler(flags, 8, 80, 123, '0.5')
    batches = list(sampler)
    assert len(batches) == 10
    assert all((flags[b].sum() == 4 for b in batches))
    sampler.epoch = 1
    assert list(sampler) != batches
    z = torch.zeros(1, 2, 8, 8, requires_grad=True)
    mask = torch.ones_like(z)
    mask[:, :, :, 0] = 0
    ci_loss(z, torch.zeros_like(z), mask, 25, 0)[0].backward()
    assert (z.grad[:, :, :, 0] == 0).all()
    natural = FractionBatchSampler(flags, 8, 80, 123, 'natural')
    natural_batches = list(natural)
    assert sum((int(flags[b].sum()) for b in natural_batches)) == 59
    assert all((len(b) == 8 for b in natural_batches))
    assert len(set((i for b in natural_batches for i in b))) == 80
    assert list(natural) == natural_batches
    natural.epoch = 1
    assert list(natural) != natural_batches
    print('SAMPLER / MASK SELFTEST PASSED: 50/50 and natural exact epoch quotas', flush=True)
    from train_ci_v3_four_models import V3Model as ControlModel
    for architecture in ['resunet', 'convlstm']:
        torch.manual_seed(123)
        old = ControlModel(architecture, True, 8)
        torch.manual_seed(123)
        new = V3Model(architecture, True, 8, 'raw')
        assert old.state_dict().keys() == new.state_dict().keys()
        assert all((torch.equal(old.state_dict()[k], new.state_dict()[k]) for k in old.state_dict()))
        print('RAW INITIALIZATION IDENTICAL TO CONTROL', architecture, flush=True)
    for radar in [False, True]:
        model = V3Model('resunet', radar, 8, 'raw_delta').to(device)
        x = torch.randn(1, 4, 9 if radar else 7, 128, 128, device=device)
        if radar:
            x[:, :, 8] = 1
        x.requires_grad_()
        z = model(x)
        assert z.shape == (1, 2, 128, 128)
        ci_loss(z, torch.zeros_like(z), torch.ones_like(z), 1, 0)[0].backward()
        assert all((x.grad[:, t, :7].abs().sum() > 0 for t in range(4)))
        assert all((torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        assert model.enc1.blocks[0].conv1.in_channels == (63 if radar else 49)
        print('RAW + DELTAS FOUR-FRAME GRADIENT PASSED', radar, flush=True)
    x = torch.zeros(1, 4, 9, 2, 2, device=device)
    x[:, :, 8] = 1
    x[:, 0, 8] = 0
    x[:, 1, 7] = 0.5
    x[:, 2, 7] = 0.75
    x[:, 3, 7] = 1
    features = early_fusion_features(x, True, 'raw_delta')
    extra = features[:, 36:].reshape(1, 3, 9, 2, 2)
    assert (extra[:, 0, 7:] == 0).all()
    assert (extra[:, 1:, 7] == 0.25).all() and (extra[:, 1:, 8] == 1).all()
    print('RADAR DELTA CAUSAL VALIDITY GATING PASSED', flush=True)

def atomic_save(path, payload):
    tmp = path.with_suffix('.tmp')
    torch.save(payload, tmp)
    tmp.replace(path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--selftest', action='store_true')
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--dataset-root', type=Path, default=DATA / 'datasets/ci_v3_birth_2x30min_v1')
    p.add_argument('--mrms-input-root', type=Path, default=DATA / 'datasets/ci_v3_mrms_inputs_complete_v1')
    p.add_argument('--normalization-json', type=Path, default=PROJECT / 'experiments/ci_resunet_earlyfusion_v1/normalization.json')
    p.add_argument('--run-dir', type=Path)
    p.add_argument('--architecture', choices=['resunet', 'convlstm'], default='resunet')
    p.add_argument('--include-mrms', action='store_true')
    p.add_argument('--input-mode', choices=['raw', 'raw_delta'], default='raw')
    p.add_argument('--train-positive-fraction', choices=['0.5', 'natural'], default='0.5')
    p.add_argument('--negative-weight', type=float, choices=[1.0, 2.0], default=1.0)
    p.add_argument('--target-kind', choices=['soft', 'binary'], required=False, default='soft')
    p.add_argument('--resume', action='store_true')
    for name, default in [('epochs', 20), ('batch-size', 8), ('num-workers', 2), ('width', 48), ('epoch-samples', 12456), ('seed', 20260930), ('expected-exclusions', 106), ('max-train-batches', 0), ('max-val-batches', 0)]:
        p.add_argument('--' + name, type=int, default=default)
    for name, default in [('learning-rate', 0.0002), ('weight-decay', 0.0001), ('positive-weight', 25.0), ('dice-weight', 0.0), ('threshold', 0.3)]:
        p.add_argument('--' + name, type=float, default=default)
    args = p.parse_args()
    if args.selftest:
        selftest()
        z = torch.tensor([[[[0.0, 0.0, 0.0]]]], requires_grad=True)
        y = torch.tensor([[[[0.0, 1.0, 0.25]]]])
        mask = torch.ones_like(z)
        a = ci_loss(z, y, mask, 1, 0, 1)[0]
        expected = F.binary_cross_entropy_with_logits(z, y)
        assert torch.allclose(a, expected)
        b = ci_loss(z, y, mask, 1, 0, 2)[0]
        b.backward()
        g = z.grad.detach().ravel()
        assert torch.allclose(g[0].abs(), 2 * g[1].abs())
        assert torch.allclose(g[2], g[0] / 4)
        z = torch.zeros_like(z, requires_grad=True)
        mask[..., 0] = 0
        ci_loss(z, y, mask, 1, 0, 2)[0].backward()
        assert z.grad[..., 0].eq(0).all()
        print('PAIRED BACKGROUND-ONLY WEIGHT / SOFT HALO / IGNORE GRADIENT SELFTEST PASSED', flush=True)
        return
    if args.width % 8 or args.width < 8:
        raise ValueError('Width must be a positive multiple of 8')
    if args.architecture != 'resunet' or args.input_mode != 'raw' or (not args.include_mrms):
        raise ValueError('This controlled four-run comparison uses raw ResUNet GOES+MRMS only')
    if args.positive_weight != 1 or args.dice_weight != 0 or args.train_positive_fraction != '0.5':
        raise ValueError('This comparison fixes PW1, Dice0 and issuance sampling 50/50')
    data_ready = json.loads((args.dataset_root / 'READY.json').read_text())
    if data_ready['target_kind'] != args.target_kind:
        raise RuntimeError('Requested target kind differs from prepared dataset')
    report = cohort_check(args)
    from ci_v4_training_support import CIGoesV4Dataset as CIGoesMRMSV3Dataset, sha as dataset_sha
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train = CIGoesMRMSV3Dataset(args.dataset_root, args.mrms_input_root, split='train', normalization=args.normalization_json, include_mrms=args.include_mrms, seed=args.seed)
    val = CIGoesMRMSV3Dataset(args.dataset_root, args.mrms_input_root, split='validation', normalization=args.normalization_json, include_mrms=args.include_mrms, seed=args.seed)
    if args.preflight:
        for split in ['train', 'validation', 'test']:
            ds = CIGoesMRMSV3Dataset(args.dataset_root, args.mrms_input_root, split=split, normalization=args.normalization_json, include_mrms=True, seed=args.seed)
            indices = [int(np.flatnonzero(ds.issuances.any_usable_positive.to_numpy(bool))[0]), int(np.flatnonzero(~ds.issuances.any_usable_positive.to_numpy(bool))[0])]
            for batch in DataLoader(Subset(ds, indices), batch_size=2, num_workers=args.num_workers):
                x = make_inputs(batch, 'cpu', True)
                assert x.shape[:3] == (2, 4, 9) and batch['targets'].shape[1] == 2
                assert ((batch['targets'] >= 0) & (batch['targets'] <= 1)).all()
                assert (batch['targets'][batch['loss_masks'] == 0] == 0).all()
            print('COHORT LOADER PASSED', split, report[split], flush=True)
        args.dataset_root.joinpath('COHORT_VERIFIED.json').write_text(json.dumps(dict(status='ready_with_explicit_exclusions', splits=report, causal_tolerance_seconds=150), indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    if args.run_dir is None:
        raise ValueError('--run-dir required')
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if (args.run_dir / 'completed.json').exists():
        raise RuntimeError('Run already completed; queue should skip it')
    existing = any(((args.run_dir / f).exists() for f in ['config.json', 'metrics.csv', 'last.pt']))
    if existing and (not args.resume):
        raise RuntimeError('Run exists: use --resume to continue a saved epoch')
    if args.resume and (not (args.run_dir / 'last.pt').is_file()):
        raise RuntimeError('No last.pt to resume; preserve this run and choose a new directory')
    device = torch.device('cuda')
    model = V3Model(args.architecture, args.include_mrms, args.width, args.input_mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    sampler = FractionBatchSampler(train.issuances.any_usable_positive.to_numpy(bool), args.batch_size, args.epoch_samples, args.seed, args.train_positive_fraction)
    options = dict(num_workers=args.num_workers, pin_memory=True)
    if args.num_workers:
        options.update(persistent_workers=False, prefetch_factor=2)
    train_loader = DataLoader(train, batch_sampler=sampler, **options)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, **options)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(cohort=report, train_positive_fraction=sampler.realized_fraction, sampling_fraction_requested=args.train_positive_fraction, sampling_fraction_resolved=sampler.fraction, positive_samples_per_epoch=sampler.positive_samples, negative_samples_per_epoch=sampler.negative_samples, input_mode=args.input_mode, history_offsets=[-30, -20, -10, 0], output_windows=['(0,30]', '(30,60]'], precision='float32', parameters=sum((p.numel() for p in model.parameters())), actual_epoch_samples=len(sampler) * args.batch_size, updates_per_epoch=len(sampler), normalization_sha256=hashlib.sha256(args.normalization_json.read_bytes()).hexdigest(), trainer_sha256=hashlib.sha256(Path(_ci_resolve(__file__)).read_bytes()).hexdigest(), validation='natural common cohort, full domain, four fixed tiles, probability averaging', pair_identity=data_ready['pair_identity'], dataset_ready_sha256=dataset_sha(args.dataset_root / 'READY.json'), support_sha256=dataset_sha(Path(_ci_resolve(__file__)).parent / 'ci_v4_training_support.py'), background_weight_rule='Exact zero-target pixels only; soft positive halo unchanged')
    best = math.inf
    best_csi = -1.0
    steps = 0
    start_epoch = 0
    if args.resume:
        try:
            ck = torch.load(args.run_dir / 'last.pt', map_location='cpu', weights_only=False)
        except TypeError:
            ck = torch.load(args.run_dir / 'last.pt', map_location='cpu')
        previous = ck['config']
        for key in config:
            if key not in {'resume', 'selftest', 'preflight'} and config[key] != previous.get(key):
                raise RuntimeError('Resume configuration differs: ' + key)
        model.load_state_dict(ck['model_state'], strict=True)
        optimizer.load_state_dict(ck['optimizer_state'])
        scheduler.load_state_dict(ck['scheduler_state'])
        start_epoch = int(ck['epoch'])
        steps = int(ck['metrics']['optimizer_steps'])
        best = float(ck['best_validation_loss'])
        best_csi = float(ck['best_mean_csi'])
        if 'torch_rng_state' in ck:
            torch.set_rng_state(ck['torch_rng_state'])
        if 'cuda_rng_states' in ck:
            torch.cuda.set_rng_state_all(ck['cuda_rng_states'])
        if 'numpy_rng_state' in ck:
            np.random.set_state(ck['numpy_rng_state'])
        if 'python_rng_state' in ck:
            random.setstate(ck['python_rng_state'])
        history = pd.read_csv(_ci_resolve(args.run_dir / 'metrics.csv'))
        history = history[history.epoch <= start_epoch]
        if len(history) != start_epoch or history.epoch.tolist() != list(range(1, start_epoch + 1)):
            raise RuntimeError('Resume metrics/checkpoint epoch mismatch')
        history.to_csv(args.run_dir / 'metrics.csv', index=False)
        print('RESUMING AFTER EPOCH', start_epoch, 'updates', steps, flush=True)
        del ck
    else:
        (args.run_dir / 'config.json').write_text(json.dumps(config, indent=2))
    print(json.dumps(config, indent=2), flush=True)
    for epoch in range(start_epoch, args.epochs):
        train.set_epoch(epoch)
        sampler.epoch = epoch
        model.train()
        loss_sum = 0.0
        batches = 0
        for bi, batch in enumerate(train_loader):
            if args.max_train_batches and bi >= args.max_train_batches:
                break
            x = make_inputs(batch, device, args.include_mrms)
            target = batch['targets'].to(device)
            mask = batch['loss_masks'].to(device)
            optimizer.zero_grad(set_to_none=True)
            z = model(x)
            loss, bce, dice = ci_loss(z, target, mask, args.positive_weight, args.dice_weight, args.negative_weight)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise RuntimeError('Nonfinite FP32 gradient')
            optimizer.step()
            steps += 1
            loss_sum += float(loss)
            batches += 1
            if bi % 50 == 0:
                print(f'epoch={epoch + 1:02d} batch={bi:04d}/{len(sampler)} loss={float(loss):.6f}', flush=True)
        if not batches:
            raise RuntimeError('Empty training epoch')
        metrics = evaluate(model, val_loader, device, args)
        row = dict(epoch=epoch + 1, train_loss=loss_sum / batches, optimizer_steps=steps, learning_rate=optimizer.param_groups[0]['lr'], **metrics)
        row['mean_csi'] = (metrics['validation_csi_lead_0'] + metrics['validation_csi_lead_1']) / 2
        path = args.run_dir / 'metrics.csv'
        with path.open('a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if epoch == 0:
                writer.writeheader()
            writer.writerow(row)
        scheduler.step()
        improved = metrics['validation_loss'] < best
        improved_csi = row['mean_csi'] > best_csi
        best = min(best, metrics['validation_loss'])
        best_csi = max(best_csi, row['mean_csi'])
        payload = dict(model_state=model.state_dict(), optimizer_state=optimizer.state_dict(), scheduler_state=scheduler.state_dict(), config=config, epoch=epoch + 1, metrics=row, best_validation_loss=best, best_mean_csi=best_csi, torch_rng_state=torch.get_rng_state(), cuda_rng_states=torch.cuda.get_rng_state_all(), numpy_rng_state=np.random.get_state(), python_rng_state=random.getstate())
        if improved:
            atomic_save(args.run_dir / 'best.pt', payload)
        if improved_csi:
            atomic_save(args.run_dir / 'best_csi.pt', payload)
        atomic_save(args.run_dir / 'last.pt', payload)
        print('EPOCH COMPLETE', json.dumps(row), flush=True)
    status = 'smoke_completed' if args.max_train_batches or args.max_val_batches else 'completed'
    (args.run_dir / 'completed.json').write_text(json.dumps(dict(status=status, epochs=args.epochs, best_val_loss=best, best_mean_csi=best_csi, optimizer_steps=steps), indent=2))
    print('TRAINING COMPLETED', args.run_dir, flush=True)
if __name__ == '__main__':
    main()
