"""Shared, versioned radar target protocol. No GPU dependency."""
from ci_nowcasting.paths import resolve_path as _ci_resolve, project_root as _ci_project, source_root as _ci_source
import importlib.util
import json
import sys
from collections import OrderedDict
from pathlib import Path
import numpy as np
import pandas as pd
PROJECT = Path(_ci_resolve(_ci_resolve('/project')))
CODE = PROJECT / str(_ci_source() / 'scripts/isolated_ci_binary_v1/goes_ir_v1')
DATA = PROJECT / 'data/ci_isolated_mrms_binary_v1/goes_ir_parallax_native_filled_v1'
SPLIT = DATA / 'splits/period_grouped_80_10_10_v1'
MINUTE = 60000000000
PROTO = dict(version=1, threshold_dbz=35.0, min_valid_dbz=-50.0, minimum_source_coverage=0.75, bin_minutes=10, interval='[start,end)', maximum_gap_seconds=180, maximum_boundary_gap_seconds=150, minimum_scans=4, temporal_target='any threshold exceedance in window; no persistence/isolation requirement', spatial_target='any observed source pixel >=35 dBZ within native GOES cell', valid_mask='at least 75% finite source pixels >=-50 dBZ at EVERY selected scan', data_use='train for gradients; validation for evaluation only; test never read')

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj

def json_write(path, obj):
    path = Path(_ci_resolve(path))
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)

def selected_issuances(split, smoke_samples=0):
    table = pd.read_parquet(_ci_resolve(SPLIT / (split + '_issuance_index_natural.parquet')))
    table['issuance_time'] = pd.to_datetime(table.issuance_time)
    if smoke_samples:
        n = smoke_samples // 2
        yes = table[table.any_usable_positive.astype(bool)].head(n)
        no = table[~table.any_usable_positive.astype(bool)].head(n)
        if len(yes) != n or len(no) != n:
            raise RuntimeError('Insufficient smoke examples')
        table = pd.concat([yes, no]).sort_index()
    return table.reset_index(drop=True)

def valid_window(times, start, end):
    times = np.asarray(times, dtype=np.int64)
    return bool(len(times) >= PROTO['minimum_scans'] and times[0] >= start and (times[-1] < end) and (times[0] - start <= 150000000000) and (end - times[-1] <= 150000000000) and np.all(np.diff(times) <= 180000000000) and np.all(np.diff(times) > 0))

def project_scan(field, mapping, shape):
    values = field[mapping['source_y'], mapping['source_x']]
    good = np.isfinite(values) & (values >= PROTO['min_valid_dbz'])
    flat = np.asarray(mapping['target_y'], np.int64) * shape[1] + mapping['target_x']
    total = np.asarray(mapping['total_count']).reshape(-1)
    count = np.bincount(flat[good], minlength=int(np.prod(shape)))
    valid = (total > 0) & (count >= PROTO['minimum_source_coverage'] * total)
    present = np.bincount(flat[good & (values >= 35)], minlength=len(total)) > 0
    return (present.reshape(shape), valid.reshape(shape))

def atomic_npz(path, **values):
    path = Path(_ci_resolve(path))
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **values)
    tmp.replace(path)

class RadarCache:

    def __init__(self, root, shape):
        self.root = Path(_ci_resolve(root))
        self.shape = tuple(shape)
        self.lru = OrderedDict()
        self.manifest = pd.read_csv(_ci_resolve(self.root / 'manifest.csv'))
        if self.manifest.duplicated(['period_id']).any():
            raise RuntimeError('Duplicate cached period')
        self.lookup = self.manifest.set_index('period_id').path.to_dict()

    def get(self, period, time):
        period = str(period)
        if period not in self.lru:
            with np.load(_ci_resolve(self.lookup[period]), allow_pickle=False) as z:
                if tuple(z['shape']) != self.shape:
                    raise RuntimeError('Radar cache grid mismatch')
                self.lru[period] = {k: z[k] for k in ['start_ns', 'target_bits', 'mask_bits']}
            if len(self.lru) > 8:
                self.lru.popitem(last=False)
        self.lru.move_to_end(period)
        c = self.lru[period]
        starts = pd.Timestamp(time).value + np.arange(6, dtype=np.int64) * 10 * MINUTE
        j = np.searchsorted(c['start_ns'], starts)
        if np.any(j >= len(c['start_ns'])) or not np.array_equal(c['start_ns'][j], starts):
            raise RuntimeError('Missing exact radar windows: ' + period + ' ' + str(time))
        arrays = []
        for key in ['target_bits', 'mask_bits']:
            ar = np.unpackbits(c[key][j], axis=1, count=int(np.prod(self.shape)))
            arrays.append(ar.reshape(6, *self.shape).astype(np.float32))
        return arrays
