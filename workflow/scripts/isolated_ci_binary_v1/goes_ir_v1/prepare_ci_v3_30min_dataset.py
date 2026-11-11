"""V3 birth targets in (0,30] and (30,60]; independent native coverage masks."""
from ci_nowcasting.paths import resolve_path as _ci_resolve, project_root as _ci_project, source_root as _ci_source
import argparse
import hashlib
import importlib.util
import json
import sys
from collections import OrderedDict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
MINUTE = 60000000000
PROJECT = Path(_ci_resolve(_ci_resolve('/crest/students/mcrampe00/PDE_CNN_precip')))

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj

def write_json(path, value):
    path = Path(_ci_resolve(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str))
    tmp.replace(path)

def frame_write(frame, path):
    tmp = path.with_name(path.name + '.tmp')
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)

def atomic_npz(path, **arrays):
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    tmp.replace(path)

def temporal_indices(times, start, end):
    i = np.searchsorted(times, start - int(2.5 * MINUTE))
    j = np.searchsorted(times, end + int(2.5 * MINUTE), side='right')
    selected = times[i:j]
    ok = bool(len(selected) >= 2 and selected[0] <= start and (selected[-1] >= end) and np.all(np.diff(selected) > 0) and np.all(np.diff(selected) <= 5 * MINUTE))
    return (int(i), int(j), ok)

def event_selection(times, issuance, lead):
    start = issuance + lead * 30 * MINUTE
    return (times > start) & (times <= start + 30 * MINUTE)

def scope_covers(scope, start, end):
    cursor = start
    for a, b in scope:
        if b < cursor:
            continue
        if a > cursor:
            return False
        cursor = max(cursor, b)
        if cursor >= end:
            return True
    return False

def prior_radar(times, anchors):
    indices = np.searchsorted(times, anchors, side='right') - 1
    safe = np.clip(indices, 0, max(0, len(times) - 1))
    ages = anchors - times[safe]
    valid = (indices >= 0) & (ages >= 0) & (ages <= int(2.5 * MINUTE))
    return (np.where(valid, times[safe], -1), valid)

def projected_valid(field, mapping, flat, count):
    values = field[mapping['source_y'], mapping['source_x']]
    good = np.isfinite(values) & (values >= -50.0)
    found = np.bincount(flat[good], minlength=len(count))
    return (count > 0) & (found >= 0.75 * count)

def uncertain_by_scan(root, lookup, shape, spacing):
    result = {}
    for folder in sorted(root.glob('chunk_*')):
        if not (folder / 'completed.json').exists():
            raise RuntimeError('Incomplete V3 chunk: ' + str(folder))
        audit = pd.read_csv(_ci_resolve(folder / 'audit.csv'), usecols=['birth_time'])
        with np.load(_ci_resolve(folder / 'uncertain_footprints.npz'), allow_pickle=False) as z:
            if tuple(z['shape']) != lookup.shape:
                raise RuntimeError('Uncertainty native grid mismatch')
            rows = z['audit_row'].astype(np.int64)
            offsets = z['offsets'].astype(np.int64)
            if len(offsets) != len(rows) + 1 or offsets[-1] != len(z['y']):
                raise RuntimeError('Corrupt uncertainty archive')
            if len(rows) == 0:
                continue
            stamps = pd.DatetimeIndex(pd.to_datetime(audit.iloc[rows].birth_time)).asi8
            for timestamp in np.unique(stamps):
                pieces = []
                for index in np.flatnonzero(stamps == timestamp):
                    a, b = offsets[index:index + 2]
                    pieces.append(lookup[z['y'][a:b], z['x'][a:b]])
                cells = np.concatenate(pieces)
                cells = cells[cells >= 0]
                if len(cells) == 0:
                    continue
                seed = np.zeros(shape, bool)
                seed.flat[np.unique(cells)] = True
                expanded = distance_transform_edt(~seed, sampling=spacing) <= 8.0
                packed = np.packbits(expanded.ravel())
                if int(timestamp) in result:
                    result[int(timestamp)] |= packed
                else:
                    result[int(timestamp)] = packed
    return result

def selftest():
    t = np.array([0, 30, 60, 61], np.int64) * MINUTE
    assert event_selection(t, 0, 0).tolist() == [False, True, False, False]
    assert event_selection(t, 0, 1).tolist() == [False, False, True, False]
    times = np.arange(-2, 65, 2, dtype=np.int64) * MINUTE
    assert temporal_indices(times, 0, 30 * MINUTE)[2]
    assert not temporal_indices(np.array([0, 10, 30]) * MINUTE, 0, 30 * MINUTE)[2]
    chosen, valid = prior_radar(np.array([-1, 1]) * MINUTE, np.array([0]) * MINUTE)
    assert chosen[0] == -MINUTE and valid[0]
    assert scope_covers([(0, 30), (30, 60)], 0, 60)
    assert not scope_covers([(0, 29), (30, 60)], 0, 60)
    positive = np.array([False, True, False])
    uncertain = np.array([True, True, False])
    assert (~uncertain | positive).tolist() == [False, True, True]
    print('TWO-WINDOW SELFTEST PASSED: endpoints, gaps, causal radar, uncertainty', flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=PROJECT)
    parser.add_argument('--catalogue-root', type=Path, required=False)
    parser.add_argument('--output-root', type=Path, required=False)
    parser.add_argument('--selftest', action='store_true')
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    if args.catalogue_root is None or args.output_root is None:
        parser.error('--catalogue-root and --output-root are required')
    import zarr
    project = args.project
    catalogue = args.catalogue_root
    output = args.output_root
    code = project / str(_ci_source() / 'scripts/isolated_ci_binary_v1/goes_ir_v1')
    data = project / 'data/ci_isolated_mrms_binary_v1/goes_ir_parallax_native_filled_v1'
    marker = catalogue / 'completed.json'
    if not marker.exists():
        raise RuntimeError('V3 is not complete; cannot build training targets')
    v3 = json.loads(marker.read_text())
    if v3.get('periods') != 548 or not str(v3.get('status', '')).startswith('all_scans_completed'):
        raise RuntimeError('Unexpected V3 completion status')
    sources = [catalogue / 'ci_v3_event_catalogue.parquet', catalogue / 'v1_temporal_scope.csv', catalogue / 'build_config.json', data / 'state/ci_goes_lead_index.parquet', Path(_ci_resolve(__file__)), code / 'build_ci_soft_targets_native_goes.py', code / 'ci_goes_patch_dataset.py']
    config = dict(version=1, windows_minutes=[[0, 30], [30, 60]], interval='(start,end]', history_offsets_minutes=[-30, -20, -10, 0], minimum_source_coverage=0.75, maximum_gap_minutes=5, endpoint_tolerance_minutes=2.5, uncertain_buffer_km=8, positive_core_overrides_spatial_uncertainty=True, positive_core_does_not_override_missing_temporal_coverage=True, target_semantics='canonical birth only, not ongoing convection occupancy', sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'build_config.json').exists():
        if json.loads((output / 'build_config.json').read_text()) != config:
            raise RuntimeError('Build source/config changed: use a new output version')
    else:
        write_json(output / 'build_config.json', config)
    if (output / 'completed.json').exists():
        print('DATASET ALREADY COMPLETED:', output)
        return
    builder = module('v3_targets_soft', code / 'build_ci_soft_targets_native_goes.py')
    loader = module('v3_targets_goes', code / 'ci_goes_patch_dataset.py')
    grid = builder.load_and_validate_goes_grid()
    shape = tuple(grid['shape'])
    size = int(np.prod(shape))
    packed_size = (size + 7) // 8
    events = pd.read_parquet(_ci_resolve(catalogue / 'ci_v3_event_catalogue.parquet'))
    events['birth_time'] = pd.to_datetime(events.birth_time)
    if len(events):
        declared = events[['period_id', 'split']].drop_duplicates()
        if declared.period_id.duplicated().any():
            raise RuntimeError('Event split inconsistency')
    if events.event_id.duplicated().any():
        raise RuntimeError('Duplicate canonical event IDs')
    scope = pd.read_csv(_ci_resolve(catalogue / 'v1_temporal_scope.csv'))
    assignments = scope[['period_id', 'year', 'split']].drop_duplicates()
    if assignments.period_id.duplicated().any() or len(assignments) != 548:
        raise RuntimeError('Period split assignment is not unique')
    if not set(assignments.split).issubset({'train', 'validation', 'test'}):
        raise RuntimeError('Unknown split names')
    if len(events):
        check = events[['period_id', 'split']].drop_duplicates().merge(assignments[['period_id', 'split']], on='period_id', suffixes=('_event', '_scope'), validate='one_to_one')
        if not (check.split_event == check.split_scope).all():
            raise RuntimeError('Event/scope split mismatch')
    if not set(events.period_id).issubset(set(assignments.period_id)):
        raise RuntimeError('Event outside retained periods')
    old = pd.read_parquet(_ci_resolve(data / 'state/ci_goes_lead_index.parquet'))
    issuance = old[['period_id', 'issuance_time']].drop_duplicates()
    issuance['issuance_time'] = pd.to_datetime(issuance.issuance_time)
    issuance = issuance.merge(assignments, on='period_id', validate='many_to_one')
    goes_times = set()
    for store in sorted(data.glob('year=*/month=*/goes_ir.zarr')):
        group = zarr.open_group(_ci_resolve(str(store)), mode='r')
        if tuple(np.asarray(group['channel'][:], int)) != tuple(range(8, 15)):
            raise RuntimeError('GOES channels changed: ' + str(store))
        times = loader.decode_time_array(group['time'][:], group['time'].attrs.asdict())
        if len(set(pd.DatetimeIndex(times).asi8)) != len(times):
            raise RuntimeError('Duplicate GOES times')
        goes_times.update(map(int, pd.DatetimeIndex(times).asi8))
    issue_ns = pd.DatetimeIndex(issuance.issuance_time).asi8
    issuance['goes_history_complete'] = [all((int(t + o * MINUTE) in goes_times for o in (-30, -20, -10, 0))) for t in issue_ns]
    records = []
    for number, assignment in enumerate(assignments.sort_values(['year', 'period_id']).itertuples(index=False), 1):
        pid = str(assignment.period_id)
        year = int(assignment.year)
        folder = output / 'periods' / pid
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / 'completed.json').exists():
            records.append(json.loads((folder / 'completed.json').read_text()))
            print(f'SKIP {number}/548 {pid}', flush=True)
            continue
        subset = issuance[issuance.period_id == pid].sort_values('issuance_time').reset_index(drop=True)
        radar_path = data.parent / f'year_{year}_strict_v2_001' / 'periods' / pid / 'reflectivity.zarr'
        radar = zarr.open_group(_ci_resolve(str(radar_path)), mode='r')
        times = np.asarray(radar['time_ns'][:], np.int64)
        if len(times) == 0 or np.any(np.diff(times) <= 0):
            raise RuntimeError('Invalid MRMS times: ' + pid)
        mapping = builder.build_mapping(radar_path.parent, grid, {})
        flat = np.asarray(mapping['target_y'], np.int64) * shape[1] + mapping['target_x']
        total = np.asarray(mapping['total_count']).ravel()
        lookup = np.full(tuple(mapping['expected_shape']), -1, np.int32)
        lookup[mapping['source_y'], mapping['source_x']] = flat
        uncertain = uncertain_by_scan(catalogue / 'periods' / pid, lookup, shape, (grid['dy_km'], grid['dx_km']))
        un_times = np.array(sorted(uncertain), np.int64)
        local_events = events[events.period_id == pid].sort_values('birth_time').reset_index(drop=True)
        birth_ns = pd.DatetimeIndex(local_events.birth_time).asi8
        local_scope = sorted(((pd.Timestamp(a).value, pd.Timestamp(b).value) for a, b in scope.loc[scope.period_id == pid, ['window_start', 'window_end']].itertuples(index=False, name=None)))
        mask_bits = np.zeros((len(subset), 2, packed_size), np.uint8)
        radar_anchors = np.full((len(subset), 4), -1, np.int64)
        leads = []
        issues = []
        cache = OrderedDict()

        def get_valid(index):
            if index not in cache:
                field = np.asarray(radar['reflectivity_dbz'][index], np.float32)
                if field.shape != lookup.shape:
                    raise RuntimeError('MRMS orientation mismatch')
                cache[index] = np.packbits(projected_valid(field, mapping, flat, total))
            cache.move_to_end(index)
            if len(cache) > 80:
                cache.popitem(last=False)
            return cache[index]
        for row_index, row in enumerate(subset.itertuples(index=False)):
            t = pd.Timestamp(row.issuance_time).value
            radar_anchors[row_index], ready = prior_radar(times, t + np.array([-30, -20, -10, 0]) * MINUTE)
            selected_counts = []
            usable_counts = []
            event_ids_all = []
            for lead in range(2):
                start = t + lead * 30 * MINUTE
                end = start + 30 * MINUTE
                i, j, complete = temporal_indices(times, start, end)
                complete = complete and scope_covers(local_scope, start, end)
                selected = local_events[event_selection(birth_ns, t, lead)]
                paths = sorted({p for value in selected.soft_target_paths for p in json.loads(value)})
                for path in paths:
                    if not Path(_ci_resolve(path)).is_file():
                        raise FileNotFoundError(path)
                core = np.zeros(shape, bool)
                for path in paths:
                    with np.load(_ci_resolve(path), allow_pickle=False) as z:
                        if tuple(z['shape']) != shape:
                            raise RuntimeError('Soft grid mismatch')
                        if not np.all(np.isfinite(z['soft_value'])):
                            raise RuntimeError('Nonfinite soft values')
                        core[z['core_y'], z['core_x']] = True
                pixels = 0
                uncertain_pixels = 0
                if complete and row.goes_history_complete:
                    bits = np.full(packed_size, 255, np.uint8)
                    for scan in range(i, j):
                        bits &= get_valid(scan)
                    ui, uj = np.searchsorted(un_times, [start, end], side='right')
                    uncertain_bits = np.zeros(packed_size, np.uint8)
                    for stamp in un_times[ui:uj]:
                        uncertain_bits |= uncertain[int(stamp)]
                    base = np.unpackbits(bits, count=size).astype(bool).reshape(shape)
                    unknown = np.unpackbits(uncertain_bits, count=size).astype(bool).reshape(shape)
                    uncertain_pixels = int((base & unknown).sum())
                    loss = base & ~unknown | core
                    pixels = int(loss.sum())
                    mask_bits[row_index, lead] = np.packbits(loss.ravel())
                usable = bool(complete and row.goes_history_complete and (pixels > 0))
                ids = selected.event_id.astype(str).tolist()
                selected_counts.append(len(ids))
                usable_counts.append(usable)
                if usable:
                    event_ids_all.extend(ids)
                leads.append(dict(period_id=pid, year=year, split=assignment.split, issuance_time=row.issuance_time, lead_index=lead, lead_start_minutes=lead * 30, lead_end_minutes=(lead + 1) * 30, positive_event_ids=json.dumps(ids), positive_count=len(ids), soft_target_paths=json.dumps(paths), target_temporally_valid=bool(complete), goes_history_complete=bool(row.goes_history_complete), usable_lead=usable, supervised_positive=usable and bool(ids), loss_mask_path=str(folder / 'window_masks.npz'), mask_row=row_index, loss_pixels=pixels, uncertain_pixels_removed=uncertain_pixels, label_kind='ignored' if not usable else 'positive' if ids else 'negative'))
            issues.append(dict(period_id=pid, year=year, split=assignment.split, issuance_time=row.issuance_time, any_target_usable=any(usable_counts), all_two_targets_usable=all(usable_counts), usable_lead_count=sum(usable_counts), any_usable_positive=bool(event_ids_all), usable_positive_lead_count=sum((bool(c) and u for c, u in zip(selected_counts, usable_counts))), total_positive_events=len(set(event_ids_all)), goes_history_complete=bool(row.goes_history_complete), mrms_history_complete=bool(ready.all()), radar_path=str(radar_path), loss_mask_path=str(folder / 'window_masks.npz'), mask_row=row_index))
        atomic_npz(folder / 'window_masks.npz', shape=np.array(shape, np.int32), mask_bits=mask_bits, issuance_time_ns=pd.DatetimeIndex(subset.issuance_time).asi8, mrms_input_times_ns=radar_anchors)
        frame_write(pd.DataFrame(leads), folder / 'lead_index.parquet')
        frame_write(pd.DataFrame(issues), folder / 'issuance_index.parquet')
        record = dict(period_id=pid, split=assignment.split, issuances=len(issues), v3_events=len(local_events))
        write_json(folder / 'completed.json', record)
        records.append(record)
        write_json(output / 'progress.json', dict(completed_periods=len(records), total_periods=548))
        print(f'DONE {number}/548 {pid} issuances={len(issues)} events={len(local_events)}', flush=True)
    all_leads = pd.concat([pd.read_parquet(_ci_resolve(output / 'periods' / p / 'lead_index.parquet')) for p in assignments.period_id], ignore_index=True)
    all_issues = pd.concat([pd.read_parquet(_ci_resolve(output / 'periods' / p / 'issuance_index.parquet')) for p in assignments.period_id], ignore_index=True)
    if all_leads.duplicated(['period_id', 'issuance_time', 'lead_index']).any():
        raise RuntimeError('Duplicate training lead keys')
    if not (all_leads.groupby(['period_id', 'issuance_time']).size() == 2).all():
        raise RuntimeError('Each issuance must have two windows')
    summaries = []
    for split in ['train', 'validation', 'test']:
        issues = all_issues[(all_issues.split == split) & all_issues.any_target_usable].copy()
        keys = issues[['period_id', 'issuance_time']]
        leads = all_leads.merge(keys, on=['period_id', 'issuance_time'], validate='many_to_one')
        frame_write(issues, output / f'{split}_issuance_index_natural.parquet')
        frame_write(leads, output / f'{split}_lead_index_natural.parquet')
        frame_write(issues[issues.mrms_history_complete], output / f'{split}_issuance_index_mrms_ready.parquet')
        summaries.append(dict(split=split, issuances=len(issues), positive_issuances=int(issues.any_usable_positive.sum()), positive_issuance_fraction=float(issues.any_usable_positive.mean()), usable_leads=int(leads.usable_lead.sum()), positive_usable_leads=int(leads.supervised_positive.sum()), mrms_history_ready=int(issues.mrms_history_complete.sum()), supervised_grid_fraction=float(leads.loss_pixels.sum() / (max(1, len(leads)) * size)), uncertain_pixels_removed=int(leads.uncertain_pixels_removed.sum())))
    all_leads[(all_leads.positive_count > 0) & ~all_leads.usable_lead].to_csv(output / 'ignored_positive_windows.csv', index=False)
    all_issues[~all_issues.any_target_usable].to_csv(output / 'excluded_issuances.csv', index=False)
    all_issues[~all_issues.mrms_history_complete].to_csv(output / 'mrms_history_gaps.csv', index=False)
    all_issues[~all_issues.goes_history_complete].to_csv(output / 'goes_history_gaps.csv', index=False)
    pd.DataFrame(summaries).to_csv(output / 'split_summary.csv', index=False)
    write_json(output / 'completed.json', dict(status='indexes_completed_requires_loader_smoke', periods=548, grid_shape=shape, summary=summaries, target=config['target_semantics'], excluded_issuances=int((~all_issues.any_target_usable).sum()), mrms_history_missing=int((~all_issues.mrms_history_complete).sum()), downloads_performed=False, old_dataset_modified=False))
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)
    print('INDEX BUILD COMPLETED:', output, flush=True)
if __name__ == '__main__':
    main()
