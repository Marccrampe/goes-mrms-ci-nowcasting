"""Build radar-only auxiliary targets for existing train/validation periods."""
from ci_nowcasting.paths import resolve_path as _ci_resolve, project_root as _ci_project, source_root as _ci_source
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import zarr
from ci_radar_aux_common_v1 import *

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smoke-samples', type=int, default=0)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tables = {s: selected_issuances(s, args.smoke_samples) for s in ['train', 'validation']}
    if set(tables['train'].period_id) & set(tables['validation'].period_id):
        raise RuntimeError('Train/validation period leakage')
    settings = {**PROTO, 'smoke_samples': args.smoke_samples, 'index_hashes': {s: hashlib.sha256(t.to_csv(index=False).encode()).hexdigest() for s, t in tables.items()}}
    protocol = args.output / 'protocol.json'
    if protocol.exists() and json.loads(protocol.read_text()) != settings:
        raise RuntimeError('Different protocol or indexes: choose a new cache directory')
    json_write(protocol, settings)
    bm = module('radar_aux_projection', CODE / 'build_ci_soft_targets_native_goes.py')
    grid = bm.load_and_validate_goes_grid()
    shape = tuple(grid['shape'])
    mapcache = {}
    records = []
    combined = pd.concat([t.assign(split=s) for s, t in tables.items()], ignore_index=True)
    groups = list(combined.groupby('period_id', sort=True))
    for pi, (period, table) in enumerate(groups):
        year = int(table.iloc[0].year)
        if table.year.nunique() != 1 or table.split.nunique() != 1:
            raise RuntimeError('Mixed period metadata')
        starts = np.unique((table.issuance_time.astype('int64').to_numpy()[:, None] + np.arange(6, dtype=np.int64)[None, :] * 10 * MINUTE).ravel())
        output = args.output / (str(period) + '.npz')
        if output.exists():
            with np.load(_ci_resolve(output), allow_pickle=False) as z:
                if not np.array_equal(z['start_ns'], starts) or tuple(z['shape']) != shape:
                    raise RuntimeError('Stale cached period: ' + period)
                record = json.loads(str(z['summary']))
            records.append(record)
            print('CACHE', pi + 1, '/', len(groups), period, flush=True)
            continue
        periodroot = DATA.parent / f'year_{year}_strict_v2_001' / 'periods' / period
        mr = zarr.open_group(_ci_resolve(str(periodroot / 'reflectivity.zarr')), mode='r')
        times = np.asarray(mr['time_ns'][:], dtype=np.int64)
        if len(times) < 1 or np.any(np.diff(times) <= 0):
            raise RuntimeError('Invalid radar times: ' + period)
        mapping = bm.build_mapping(periodroot, grid, mapcache)
        if tuple(mr['reflectivity_dbz'].shape[1:]) != tuple(mapping['expected_shape']):
            raise RuntimeError('Radar/mapping shape mismatch')
        packed_y = []
        packed_m = []
        valid_pixels = positive_pixels = valid_windows = 0
        if len(starts) > 1 and np.any(np.diff(starts) < 10 * MINUTE):
            raise RuntimeError('Overlapping auxiliary windows; check issuance cadence')
        for wi, start in enumerate(starts):
            end = start + 10 * MINUTE
            left, right = np.searchsorted(times, [start, end], side='left')
            y = np.zeros(shape, bool)
            mask = np.ones(shape, bool)
            if valid_window(times[left:right], start, end):
                for ti in range(left, right):
                    field = np.asarray(mr['reflectivity_dbz'][ti], dtype=np.float32)
                    yes, valid = project_scan(field, mapping, shape)
                    y |= yes
                    mask &= valid
                valid_windows += 1
            else:
                mask[:] = False
            y &= mask
            valid_pixels += int(mask.sum())
            positive_pixels += int(y.sum())
            packed_y.append(np.packbits(y.reshape(-1)))
            packed_m.append(np.packbits(mask.reshape(-1)))
            if wi and wi % 100 == 0:
                print('BUILD', period, wi, '/', len(starts), 'windows', flush=True)
        record = dict(period_id=period, year=year, split=str(table.iloc[0].split), path=str(output.resolve()), windows=len(starts), temporally_complete_windows=valid_windows, valid_pixels=valid_pixels, positive_pixels=positive_pixels, valid_fraction=valid_pixels / (len(starts) * np.prod(shape)))
        atomic_npz(output, start_ns=starts, target_bits=np.stack(packed_y), mask_bits=np.stack(packed_m), shape=np.array(shape, dtype=np.int32), summary=json.dumps(record, default=int))
        records.append(record)
        print('BUILT', pi + 1, '/', len(groups), period, 'valid_fraction', round(record['valid_fraction'], 4), flush=True)
    frame = pd.DataFrame(records)
    temporary = args.output / 'manifest.csv.tmp'
    frame.to_csv(temporary, index=False)
    temporary.replace(args.output / 'manifest.csv')
    for split, g in frame.groupby('split'):
        if int(g.valid_pixels.sum()) == 0 or int(g.positive_pixels.sum()) == 0:
            raise RuntimeError('No valid radar supervision/positives in ' + split)
        print(split, 'windows', int(g.windows.sum()), 'positive_fraction', g.positive_pixels.sum() / g.valid_pixels.sum(), flush=True)
    json_write(args.output / 'completed.json', dict(periods=len(frame), shape=shape, settings=settings))
    print('RADAR AUXILIARY CACHE COMPLETE', args.output, flush=True)
if __name__ == '__main__':
    main()
