"""Physical max-dilation of existing soft targets, training split ONLY.

No catalogue, birth core, loss mask, input, issuance flag, or crop anchor changes.
Hard uses exactly the nonzero support of its soft counterpart.
"""
from ci_nowcasting.paths import resolve_path as _ci_resolve, project_root as _ci_project, source_root as _ci_source
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import maximum_filter
ARMS = {'I1_A3_soft_r1': ('soft', 1), 'I2_A3_hard_r1': ('hard', 1), 'I3_A3_soft_r2': ('soft', 2), 'I4_A3_hard_r2': ('hard', 2)}
PROJECT = Path(_ci_resolve(_ci_resolve('/project')))
CONTROL = PROJECT / 'experiments/ci_a_radar_features_v1/A3_mrms35_delta'

def sha(path):
    h = hashlib.sha256()
    with Path(_ci_resolve(path)).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def kernel(pixels, dy, dx):
    if pixels not in (0, 1, 2) or not np.isfinite([dy, dx]).all() or min(dy, dx) <= 0:
        raise ValueError('Invalid inflation radius or physical grid')
    radius = float(pixels) * min(float(dy), float(dx))
    ny = int(np.ceil(radius / dy))
    nx = int(np.ceil(radius / dx))
    yy, xx = np.mgrid[-ny:ny + 1, -nx:nx + 1]
    return ((yy * dy) ** 2 + (xx * dx) ** 2 <= radius ** 2 + 1e-08, radius)

def transform(targets, masks, pixels, dy, dx, kind):
    if kind not in ('soft', 'hard'):
        raise ValueError('Unknown target kind')
    y = np.asarray(targets)
    m = np.asarray(masks)
    if y.shape != m.shape or y.ndim != 3:
        raise ValueError('Expected lead,H,W targets and masks')
    valid = m > 0
    if not np.isfinite(y).all() or (y < 0).any() or (y > 1).any():
        raise RuntimeError('Invalid base soft target')
    if np.any(y[~valid] != 0):
        raise RuntimeError('Original ignored targets must already be zero')
    foot, _ = kernel(pixels, dy, dx)
    out = np.empty_like(y, dtype=np.float32)
    for lead in range(len(y)):
        field = maximum_filter(y[lead], footprint=foot, mode='constant', cval=0)
        field[~valid[lead]] = 0
        out[lead] = (field > 0).astype(np.float32) if kind == 'hard' else field
    if np.any(out[~valid] != 0):
        raise RuntimeError('Inflation entered ignored supervision')
    return out

def _make_dataset_class():
    from ci_v4_training_support import CIGoesV4Dataset

    class InflatedDataset(CIGoesV4Dataset):

        def __init__(self, *args, inflation_kind, inflation_pixels, inflation_dy_km, inflation_dx_km, **kwargs):
            self.inflation_kind = inflation_kind
            self.inflation_pixels = int(inflation_pixels)
            self.inflation_dy_km = float(inflation_dy_km)
            self.inflation_dx_km = float(inflation_dx_km)
            super().__init__(*args, **kwargs)

        def _build_targets(self, rows):
            targets, masks, candidates = super()._build_targets(rows)
            if self.split == 'train':
                targets = transform(targets, masks, self.inflation_pixels, self.inflation_dy_km, self.inflation_dx_km, self.inflation_kind)
            return (targets, masks, candidates)
    InflatedDataset.__module__ = __name__
    InflatedDataset.__qualname__ = 'InflatedDataset'
    return InflatedDataset

def __getattr__(name):
    if name == 'InflatedDataset':
        cls = _make_dataset_class()
        globals()[name] = cls
        return cls
    raise AttributeError(name)

def control_config(control=CONTROL):
    path = Path(_ci_resolve(control)) / 'config.json'
    cfg = json.loads(path.read_text())
    if json.loads((Path(_ci_resolve(control)) / 'completed.json').read_text()).get('status') != 'completed':
        raise RuntimeError('A3 control training is not complete')
    expected = dict(architecture='resunet', include_mrms=True, width=48, input_mode='raw', target_kind='soft', positive_weight=1.0, negative_weight=1.0, dice_weight=0.0, batch_size=8, epoch_samples=12456, radar_features='both', seed=20260930, weight_decay=0.0001, epochs=20, learning_rate=0.0002)
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise RuntimeError('Unexpected A3 control setting: ' + key)
    import train_ci_a_radar_features as T
    if sha(T.__file__) != cfg['trainer_sha256']:
        raise RuntimeError('A3 trainer changed after the control run')
    if sha(cfg['normalization_json']) != cfg['normalization_sha256']:
        raise RuntimeError('Normalization changed')
    if sha(Path(_ci_resolve(cfg['dataset_root'])) / 'READY.json') != cfg['dataset_ready_sha256']:
        raise RuntimeError('Dataset READY changed')
    return cfg

def validate_args(args):
    cfg = control_config(args.control_run)
    for key in ('architecture', 'include_mrms', 'width', 'input_mode', 'target_kind', 'positive_weight', 'negative_weight', 'dice_weight', 'batch_size', 'epoch_samples', 'radar_features', 'seed', 'weight_decay', 'learning_rate'):
        if getattr(args, key) != cfg[key]:
            raise RuntimeError('Not an A3-controlled inflation run: ' + key)
    for key in ('dataset_root', 'mrms_input_root', 'normalization_json'):
        if Path(_ci_resolve(getattr(args, key))).resolve() != Path(_ci_resolve(cfg[key])).resolve():
            raise RuntimeError('Different A3 input/data path: ' + key)
    if str(args.train_positive_fraction) != '0.5':
        raise RuntimeError('Original 50/50 issuance sampler required')
    if args.epochs not in (1, 20):
        raise RuntimeError('Only 1-epoch smoke or 20-epoch full runs allowed')
    limited = bool(args.max_train_batches or args.max_val_batches)
    if limited and (args.epochs != 1 or args.max_train_batches != 3 or args.max_val_batches != 2):
        raise RuntimeError('Smoke must be one epoch, 3 train batches, 2 validation batches')
    if not limited and args.epochs != 20:
        raise RuntimeError('Full run must have exactly 20 epochs')
    if (args.inflation_kind, args.inflation_pixels) != ARMS[args.arm]:
        raise RuntimeError('Arm/radius/type mismatch')
    if args.run_dir is not None:
        out = Path(_ci_resolve(args.run_dir)).resolve()
        original = Path(_ci_resolve(args.control_run)).resolve()
        if out == original or original in out.parents:
            raise RuntimeError('Original control run cannot be modified')
    from build_ci_soft_targets_native_goes import load_and_validate_goes_grid
    grid = load_and_validate_goes_grid()
    args.inflation_dy_km = float(grid['dy_km'])
    args.inflation_dx_km = float(grid['dx_km'])
    _, radius = kernel(args.inflation_pixels, args.inflation_dy_km, args.inflation_dx_km)
    print('CONTROL VERIFIED: fresh A3, target inflation TRAIN ONLY;', args.arm, 'radius_km=', radius, flush=True)
    return cfg

def contract(args):
    _, radius = kernel(args.inflation_pixels, args.inflation_dy_km, args.inflation_dx_km)
    return dict(version=1, arm=args.arm, kind=args.inflation_kind, extra_radius_pixels=args.inflation_pixels, extra_radius_km=radius, dy_km=args.inflation_dy_km, dx_km=args.inflation_dx_km, support_sha256=sha(__file__), control_config_sha256=sha(Path(_ci_resolve(args.control_run)) / 'config.json'), base_trainer_sha256=control_config(args.control_run)['trainer_sha256'], initialization='random_from_scratch; no pretrained weights loaded', spatial_transform='physical circular maximum filter of existing masked soft target BEFORE cropping', hard_definition='1 wherever inflated soft target >0; no intensity-based envelope or .5 cutoff', masks='unchanged; ignored pixels remain zero and unsupervised', crop_anchors='unchanged original valid birth cores; same issuance sampling flags', training_only=True, validation_targets='unchanged original soft targets for all arms', checkpoint_selection='primary epoch20 last.pt; best.pt retained for common original-soft validation loss', verification_reference='unchanged original >=35-dBZ birth cores and loss masks')

def state_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def numerical_selftest():
    y = np.zeros((2, 19, 23), np.float32)
    m = np.ones_like(y)
    m[:, 0] = 0
    y[0, 9, 10] = 1
    y[0, 9, 11] = 0.4
    y[1, 6, 6] = 0.7
    dy = dx = 2.0039658909142015
    soft = transform(y, m, 1, dy, dx, 'soft')
    hard = transform(y, m, 1, dy, dx, 'hard')
    assert np.array_equal(soft > 0, hard > 0) and set(np.unique(hard)) <= {0.0, 1.0}
    assert np.all(soft >= y) and np.all(soft[m == 0] == 0) and np.array_equal(y[0, 9, 10], np.float32(1))
    assert soft[0, 8, 10] == 1 and soft[0, 8, 9] == 0
    assert np.array_equal(transform(y, m, 0, dy, dx, 'soft'), y)
    assert np.all(transform(y, m, 2, dy, dx, 'soft') >= soft)
    assert soft[1, 9, 10] == 0
    empty = transform(np.zeros_like(y), m, 2, dy, dx, 'hard')
    assert not empty.any()
    bad = y.copy()
    bad[0, 0, 0] = 1
    try:
        transform(bad, m, 1, dy, dx, 'soft')
    except RuntimeError:
        pass
    else:
        raise AssertionError('Invalid supervision became a dilation seed')
    z = np.zeros((1, 9, 9), np.float32)
    z[0, 4, 4] = 1
    a = transform(z, np.ones_like(z), 1, 1, 3, 'soft')
    assert a[0, 3, 4] == 1 and a[0, 4, 3] == 0
    z[:] = 0
    z[0, 0, 0] = 1
    a = transform(z, np.ones_like(z), 1, 1, 1, 'soft')
    assert a.sum() == 3
    a = transform(y, m, 2, dy, dx, 'soft')[:, 8:12, 12:16]
    b = transform(y[:, 8:12, 12:16], m[:, 8:12, 12:16], 2, dy, dx, 'soft')
    assert a.any() and (not b.any())
    print('INFLATION NUMERICAL SELFTEST PASSED: soft/hard identical support, real physical disks, masks, anisotropy, empty cases, pre-crop transform', flush=True)

def torch_selftest():
    numerical_selftest()
    import torch
    import train_ci_a_radar_features as T
    torch.set_num_threads(2)
    torch.manual_seed(20260930)
    a = T.V3Model('resunet', True, 8, 'raw', 'both')
    torch.manual_seed(20260930)
    b = T.V3Model('resunet', True, 8, 'raw', 'both')
    assert state_hash(a) == state_hash(b)
    x = torch.randn(1, 4, 9, 32, 32)
    x[:, :, 8] = 1
    y = np.zeros((2, 32, 32), np.float32)
    y[:, 15, 15] = 1
    m = np.ones_like(y)
    m[:, :2] = 0
    for kind in ('soft', 'hard'):
        target = torch.from_numpy(transform(y, m, 1, 2.004, 2.004, kind))[None]
        mask = torch.from_numpy(m)[None]
        b.zero_grad()
        logit = b(x)
        logit.retain_grad()
        loss = T.ci_loss(logit, target, mask, 1, 0, 1)[0]
        loss.backward()
        assert torch.isfinite(loss) and logit.grad[mask == 0].eq(0).all()
        assert all((torch.isfinite(p.grad).all() for p in b.parameters() if p.grad is not None))
    assert b.enc1.blocks[0].conv1.in_channels == 46
    print('A3 INFLATION TORCH SELFTEST PASSED: same random initialization, 46 inputs, two heads, finite gradients, ignored pixels', flush=True)
if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--selftest', action='store_true')
    p.add_argument('--torch-selftest', action='store_true')
    args = p.parse_args()
    if args.torch_selftest:
        torch_selftest()
    else:
        numerical_selftest()
