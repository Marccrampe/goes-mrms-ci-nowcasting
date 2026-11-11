"""Build tested contour20 V3 rules on retained native MRMS scans.
Separate catalogue/soft targets; does not modify V1 or train any model.
"""
from ci_nowcasting.paths import resolve_path as _ci_resolve, project_root as _ci_project, source_root as _ci_source
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy.ndimage import find_objects, distance_transform_edt, minimum, maximum
PROJECT = Path(_ci_resolve(_ci_resolve('/crest/students/mcrampe00/PDE_CNN_precip')))
CTX = None
AUDIT_COLUMNS = ['object_id', 'period_id', 'year', 'birth_time', 'component_id', 'pixels', 'native_y', 'native_x', 'lineage', 'status', 'split', 'canonical_event_id', 'canonical_birth_time', 'duplicate_of', 'duplicate_review_status', 'birth_mask_path', 'soft_target_path', 'isolation_band_path', 'native_dy_km', 'native_dx_km', 'native_extent_bound_km', 'selected_offset_km', 'group_pixels', 'group_components', 'area_km2']

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str))
    tmp.replace(path)

def write_frame(frame, path):
    tmp = path.with_name(path.name + '.tmp')
    if path.suffix == '.parquet':
        frame.to_parquet(tmp, index=False)
    else:
        frame.to_csv(tmp, index=False)
    tmp.replace(path)

def combine(paths, columns):
    frames = [pd.read_parquet(_ci_resolve(p)) for p in paths]
    nonempty = [f for f in frames if len(f)]
    return pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame(columns=columns)

def points_of(lab, objects):
    boxes = find_objects(lab)
    result = {}
    for k in objects:
        box = boxes[int(k) - 1]
        p = np.argwhere(lab[box] == k)
        p += np.array([box[0].start, box[1].start])
        result[int(k)] = p
    return result

def fast_lineage(current, previous, spacing, gap):
    ids = np.unique(current)
    ids = ids[ids > 0]
    if previous is None or gap > 5:
        return {int(k): 'unknown_previous_scan' for k in ids}
    distance = distance_transform_edt(previous == 0, sampling=spacing) if np.any(previous) else np.full(previous.shape, np.inf)
    near = minimum(distance, labels=current, index=ids)
    overlaps = maximum((previous > 0).astype(np.uint8), labels=current, index=ids)
    gate = 1.0 + 40.0 * gap * 60 / 1000
    return {int(k): 'continuing_or_split' if overlap else 'ambiguous_motion_or_nearby_birth' if d <= gate else 'new_candidate' for k, overlap, d in zip(ids, overlaps, near)}

def contour_context(labels, points, seed, spacing, maximum_reach):
    p = points[seed]
    pad = np.ceil(maximum_reach / np.asarray(spacing)).astype(int) + 2
    low = np.maximum(0, p.min(axis=0) - pad)
    high = np.minimum(labels.shape, p.max(axis=0) + pad + 1)
    y0, x0 = map(int, low)
    y1, x1 = map(int, high)
    roi = np.s_[y0:y1, x0:x1]
    distance = distance_transform_edt(labels[roi] != seed, sampling=spacing)
    edge = min((p[:, 0].min() + 0.5) * spacing[0], (labels.shape[0] - 0.5 - p[:, 0].max()) * spacing[0], (p[:, 1].min() + 0.5) * spacing[1], (labels.shape[1] - 0.5 - p[:, 1].max()) * spacing[1])
    return (roi, distance, float(edge), np.array([y0, x0]))

def test_contour(frame, valid, labels, points, seed, fresh, offset, spacing, cache):
    roi, distance, edge, origin = cache
    inside = distance <= offset
    outer = distance <= offset + 1.0
    band = outer & ~inside
    metrics = dict(band_pixels=int(band.sum()), band_valid_fraction=float(valid[roi][band].mean()) if band.any() else np.nan)
    for z in (10, 20, 35):
        metrics[f'band_fraction_ge{z}'] = float(np.mean(frame[roi][band] >= z)) if band.any() else np.nan

    def reject(reason):
        return (None, reason, metrics)
    if edge < offset + 1 or not band.any():
        return reject('boundary')
    if not valid[roi][band].all():
        return reject('unknown_annulus_pixels')
    if valid[roi][outer].mean() < 0.99:
        return reject('invalid_disk')
    if np.any(frame[roi][band] >= 20):
        return reject('echo_ge20_in_annulus')
    members = sorted(set(map(int, np.unique(labels[roi][inside]))) - {0})
    if seed not in members:
        return reject('seed_not_enclosed')
    for k in members:
        p = points[k] - origin
        if not ((p >= 0).all() and (p < np.asarray(inside.shape)).all()):
            return reject('component_crosses_annulus')
        if not inside[p[:, 0], p[:, 1]].all():
            return reject('component_crosses_annulus')
        if k not in fresh:
            return reject('enclosed_existing_or_ambiguous_core')
    return (dict(members=members, selected_offset_km=float(offset), roi=roi, outer=outer, band_points=np.argwhere(band) + origin), 'accepted_geometry', metrics)

def analyse(pilot, core, fields, times, spacing, lo, hi, offsets, eligible=None):
    valid = np.isfinite(fields) & (fields >= -50)
    labs, objs, pts, states = ([], [], [], [])
    for i, f in enumerate(fields):
        lab, obj = core.segment(f, valid[i], 35.0, np.nextafter(3 * np.prod(spacing), 0.0), spacing)
        labs.append(lab)
        objs.append(obj)
        pts.append(points_of(lab, obj))
        if any((len(p) < 3 for p in pts[-1].values())):
            raise RuntimeError('Minimum native pixel count failed')
        gap = (times[i] - times[i - 1]).total_seconds() / 60 if i else np.inf
        state = fast_lineage(lab, labs[i - 1] if i else None, spacing, gap)
        prior = np.flatnonzero((times >= times[i] - pd.Timedelta(minutes=30)) & (times < times[i]))
        for k in state:
            p = pts[-1][k]
            if state[k] == 'new_candidate' and any((np.any(labs[j][p[:, 0], p[:, 1]] > 0) for j in prior)):
                state[k] = 'ambiguous_prior_echo_at_birth_location'
        states.append(state)
    cfg = dict(persistence_minutes=10, persistence_tolerance_minutes=2, maximum_tracking_gap_minutes=5, tracking_padding_km=1, maximum_tracking_speed_m_s=40, maximum_missing_native_scans=1)
    audit, events, uncertain = ([], [], [])
    persistence_cache = {}
    for i in range(lo, hi):
        if eligible is not None and i not in eligible:
            continue
        t = times[i]
        fresh = {k for k, s in states[i].items() if s == 'new_candidate'}
        used = set()
        context = np.flatnonzero((times >= t - pd.Timedelta(minutes=30)) & (times <= t + pd.Timedelta(minutes=12)))
        complete = core.temporal_window_complete(times, t - pd.Timedelta(minutes=30), t + pd.Timedelta(minutes=12), 5, 2.5)
        for k, state in states[i].items():
            if state == 'continuing_or_split' or k in used:
                continue
            p = pts[i][k]
            row = dict(birth_time=t, component_id=k, pixels=len(p), native_y=float(p[:, 0].mean()), native_x=float(p[:, 1].mean()), lineage=state, isolation_kind='contour', band_echo_threshold_dbz=20.0, band_width_km=1.0, native_dy_km=float(spacing[0]), native_dx_km=float(spacing[1]))
            accepted = None
            if not complete or state == 'unknown_previous_scan':
                row['status'] = 'insufficient_temporal_coverage'
            elif state != 'new_candidate':
                row['status'] = state
            else:
                cache = contour_context(labs[i], pts[i], k, spacing, max(offsets) + 1)
                for offset in offsets:
                    group, reason, metrics = test_contour(fields[i], valid[i], labs[i], pts[i], k, fresh, offset, spacing, cache)
                    key = f'offset_{offset:g}'
                    row[key + '_reason'] = reason
                    if not group:
                        continue
                    roi = group['roi']
                    context_valid = valid[context, roi[0], roi[1]]
                    if np.any(context_valid[:, group['outer']].mean(axis=1) < 0.99):
                        row[key + '_reason'] = 'invalid_context_disk'
                        continue
                    for m in group['members']:
                        if (i, m) not in persistence_cache:
                            persistence_cache[i, m] = core.match_persistence(objs[i][m], i, times, labs, objs, spacing, cfg) is not None
                    if not all((persistence_cache[i, m] for m in group['members'])):
                        row[key + '_reason'] = 'persistence_failed'
                        continue
                    accepted = group
                    row.update(metrics)
                    break
                if accepted is None:
                    row['status'] = 'geometry_quality_or_persistence_rejected'
                elif used.intersection(accepted['members']):
                    row['status'] = 'overlapping_group_requires_review'
                else:
                    used.update(accepted['members'])
                    q = np.concatenate([pts[i][m] for m in accepted['members']])
                    row.update(status='v3_accepted_requires_review', selected_offset_km=accepted['selected_offset_km'], outer_reach_km=accepted['selected_offset_km'] + 1.0, group_components=json.dumps(accepted['members']), group_pixels=len(q), area_km2=len(q) * np.prod(spacing))
                    events.append((row, q, accepted['band_points']))
            if row['status'].startswith('ambiguous') or row['status'] in {'insufficient_temporal_coverage', 'overlapping_group_requires_review'}:
                uncertain.append((len(audit), p))
            audit.append(row)
    return (audit, events, uncertain)

def object_id(period_id, birth, component):
    token = f'{period_id}|{pd.Timestamp(birth).value}|{int(component)}'
    return 'ICIV3_' + hashlib.sha256(token.encode()).hexdigest()[:24]

def reviewed_event_metadata(row):
    row.update(canonical_event_id=row['object_id'], canonical_birth_time=row['birth_time'], duplicate_of='', duplicate_review_status='not_manually_linked')
    pid = 'PERIOD_cdd25d789f4df191a34ee013'
    if row['period_id'] != pid:
        return
    first = pd.Timestamp('2021-06-28 17:48:41')
    second = pd.Timestamp('2021-06-28 17:52:39')
    matched = pd.Timestamp(row['birth_time']) == first and int(row['component_id']) == 16 or (pd.Timestamp(row['birth_time']) == second and int(row['component_id']) == 18)
    if matched:
        canonical = object_id(pid, first, 16)
        row.update(canonical_event_id=canonical, canonical_birth_time=first, duplicate_of=canonical if row['object_id'] != canonical else '', duplicate_review_status='user_reviewed_same_event_2026_10_07')

def final_event_tables(cat, soft, output):
    ids = set(cat.object_id)
    if not set(cat.canonical_event_id).issubset(ids):
        raise RuntimeError('A reviewed event is missing its first detection; inspect the pair')
    events = []
    for eid, f in cat.groupby('canonical_event_id', sort=True):
        first = f[f.object_id == eid].iloc[0]
        row = first.to_dict()
        row.update(event_id=eid, birth_time=pd.Timestamp(first.canonical_birth_time), detection_count=len(f), object_ids=json.dumps(f.object_id.astype(str).tolist()), detection_birth_times=json.dumps(f.birth_time.astype(str).tolist()), birth_mask_paths=json.dumps(f.birth_mask_path.astype(str).tolist()), soft_target_paths=json.dumps(f.soft_target_path.astype(str).tolist()), target_rule='pixelwise maximum of all referenced detection soft targets', time_rule='canonical first birth; later reviewed fragments do not create a second birth')
        events.append(row)
    table = pd.DataFrame(events, columns=list(cat.columns) + ['event_id', 'detection_count', 'object_ids', 'detection_birth_times', 'birth_mask_paths', 'soft_target_paths', 'target_rule', 'time_rule'])
    write_frame(table, output / 'ci_v3_event_catalogue.parquet')
    metadata = cat[['object_id', 'canonical_event_id', 'canonical_birth_time', 'duplicate_review_status']]
    for c in metadata.columns[1:]:
        if c in soft:
            soft = soft.drop(columns=[c])
    write_frame(soft.merge(metadata, on='object_id', validate='one_to_one'), output / 'soft_target_build_manifest.parquet')
    flags = []
    from scipy.spatial import cKDTree
    from functools import lru_cache

    @lru_cache(maxsize=256)
    def points(path, dy, dx):
        with np.load(_ci_resolve(path), allow_pickle=False) as z:
            return np.column_stack([z['y'] * dy, z['x'] * dx])
    for pid, f in cat.groupby('period_id', sort=True):
        f = f.sort_values('birth_time').reset_index(drop=True)
        ns = pd.DatetimeIndex(f.birth_time).asi8
        centers = f[['native_y', 'native_x']].to_numpy() * f[['native_dy_km', 'native_dx_km']].to_numpy()
        offsets = f.native_extent_bound_km.to_numpy(float)
        for j in range(len(f)):
            start = int(np.searchsorted(ns, ns[j] - 30 * 60000000000.0))
            for i in range(start, j):
                dt = (ns[j] - ns[i]) / 60000000000.0
                if dt < 1.5:
                    continue
                gate = min(10.0, 1.0 + 2.4 * dt)
                if np.linalg.norm(centers[j] - centers[i]) > gate + offsets[i] + offsets[j]:
                    continue
                a, b = (f.iloc[i], f.iloc[j])
                q = points(str(a.birth_mask_path), float(a.native_dy_km), float(a.native_dx_km))
                r = points(str(b.birth_mask_path), float(b.native_dy_km), float(b.native_dx_km))
                distance = float(cKDTree(q).query(r)[0].min())
                if distance <= gate:
                    flags.append(dict(period_id=pid, first_id=a.object_id, second_id=b.object_id, first_birth=a.birth_time, second_birth=b.birth_time, gap_minutes=dt, core_edge_distance_km=distance, already_manually_linked=a.canonical_event_id == b.canonical_event_id, note='Proximity flag only; not an automatic duplicate decision'))
    columns = ['period_id', 'first_id', 'second_id', 'first_birth', 'second_birth', 'gap_minutes', 'core_edge_distance_km', 'already_manually_linked', 'note']
    write_frame(pd.DataFrame(flags, columns=columns), output / 'nearby_birth_flags.csv')
    write_frame(cat[cat.duplicate_of != ''], output / 'reviewed_duplicate_links.csv')
    return (table, len(flags))

def initialise(project):
    global CTX
    import zarr
    from pyproj import Geod
    project = Path(_ci_resolve(project))
    code = project / str(_ci_source() / 'scripts/isolated_ci_binary_v1/goes_ir_v1')
    pilot = load('v3_pilot', code / 'ci_label_pilot_v2.py')
    core = load('v3_core', project / str(_ci_source() / 'scripts/isolated_ci_binary_v1/core.py'))
    builder = load('v3_soft', code / 'build_ci_soft_targets_native_goes.py')
    CTX = (pilot, core, builder, builder.load_and_validate_goes_grid(), zarr, Geod(ellps='WGS84'))

def period_job(job):
    pilot, core, builder, grid, zarr, geod = CTX
    period, year, out, minutes, offsets, windows, split = job
    period, out = (Path(_ci_resolve(period)), Path(_ci_resolve(out)))
    out.mkdir(parents=True, exist_ok=True)
    finished = out / 'completed.json'
    if finished.exists():
        result = json.loads(finished.read_text())
        print('SKIP', period.name, flush=True)
        return result
    radar = zarr.open_group(_ci_resolve(str(period / 'reflectivity.zarr')), mode='r')
    times = pd.DatetimeIndex(pd.to_datetime(np.asarray(radar['time_ns'][:], np.int64)))
    if len(times) == 0 or times.has_duplicates or (not times.is_monotonic_increasing):
        raise RuntimeError('Empty, duplicated or unordered times: ' + str(period))
    lat, lon = (np.asarray(radar['latitude'][:]), np.asarray(radar['longitude'][:]))
    spacing = (abs(geod.inv(lon.mean(), lat[0], lon.mean(), lat[1])[2]) / 1000, abs(geod.inv(lon[0], lat.mean(), lon[1], lat.mean())[2]) / 1000)
    shape = (len(lat), len(lon))
    mapping = builder.build_mapping(period, grid, {})
    lookup = np.full(shape, -1, np.int32)
    lookup[mapping['source_y'], mapping['source_x']] = mapping['target_y'] * grid['shape'][1] + mapping['target_x']
    allowed = np.zeros(len(times), bool)
    for start, finish in windows:
        allowed |= (times >= pd.Timestamp(start)) & (times <= pd.Timestamp(finish))
    owned = np.flatnonzero(allowed)
    blocks = []
    position = 0
    while position < len(owned):
        stop = int(times.searchsorted(times[owned[position]] + pd.Timedelta(minutes=minutes)))
        next_position = int(np.searchsorted(owned, stop))
        next_position = max(next_position, position + 1)
        block = owned[position:next_position]
        blocks.append((int(block[0]), int(block[-1]) + 1, set(map(int, block))))
        position = next_position
    for chunk, (begin, end, active) in enumerate(blocks):
        folder = out / f'chunk_{chunk:05d}'
        folder.mkdir(exist_ok=True)
        if (folder / 'completed.json').exists():
            continue
        a = max(0, int(times.searchsorted(times[begin] - pd.Timedelta(minutes=45))))
        b = int(times.searchsorted(times[end - 1] + pd.Timedelta(minutes=15), side='right'))
        fields = np.asarray(radar['reflectivity_dbz'][a:b, :, :], np.float32)
        if fields.shape[1:] != shape:
            raise RuntimeError('Native grid orientation mismatch')
        audit, events, uncertain = analyse(pilot, core, fields, times[a:b], spacing, begin - a, end - a, offsets, {i - a for i in active})
        catalogue, soft_rows = ([], [])
        for row in audit:
            row.update(object_id=object_id(period.name, row['birth_time'], row['component_id']), period_id=period.name, year=year, split=split)
            reviewed_event_metadata(row)
        for row, p, band in events:
            gc = pilot.projected_core(p, lookup, grid['shape'])
            if not gc.any():
                row['status'] = 'outside_goes_grid'
                continue
            native_path = out / 'birth_masks' / f"{row['object_id']}.npz"
            builder.atomic_save_npz(native_path, y=p[:, 0].astype(np.uint16), x=p[:, 1].astype(np.uint16), shape=np.asarray(shape, np.int32))
            band_path = out / 'isolation_bands' / f"{row['object_id']}.npz"
            builder.atomic_save_npz(band_path, y=band[:, 0].astype(np.uint16), x=band[:, 1].astype(np.uint16), shape=np.asarray(shape, np.int32), offset_km=np.float32(row['selected_offset_km']), width_km=np.float32(1), echo_threshold_dbz=np.float32(20))
            center = np.array([row['native_y'], row['native_x']])
            row['native_extent_bound_km'] = float(np.linalg.norm((p - center) * np.asarray(spacing), axis=1).max())
            entry = {**row, 'birth_mask_path': str(native_path), 'isolation_band_path': str(band_path), 'reference_count': 0, 'goes_core_y': float(np.where(gc)[0].mean()), 'goes_core_x': float(np.where(gc)[1].mean())}
            soft_path = out / 'soft_targets' / f"{row['object_id']}.npz"
            soft_rows.append(builder.build_object_target(SimpleNamespace(**entry), mapping, grid, soft_path))
            entry['soft_target_path'] = str(soft_path)
            catalogue.append(entry)
        packed_offsets, arrays, indices = ([0], [], [])
        for index, p in uncertain:
            indices.append(index)
            arrays.append(p)
            packed_offsets.append(packed_offsets[-1] + len(p))
        p = np.concatenate(arrays) if arrays else np.empty((0, 2), np.int32)
        builder.atomic_save_npz(folder / 'uncertain_footprints.npz', y=p[:, 0].astype(np.uint16), x=p[:, 1].astype(np.uint16), offsets=np.asarray(packed_offsets, np.int64), audit_row=np.asarray(indices, np.int64), shape=np.asarray(shape, np.int32))
        write_frame(pd.DataFrame(audit) if audit else pd.DataFrame(columns=AUDIT_COLUMNS), folder / 'audit.csv')
        write_frame(pd.DataFrame(catalogue) if catalogue else pd.DataFrame(columns=AUDIT_COLUMNS), folder / 'catalogue.parquet')
        write_frame(pd.DataFrame(soft_rows) if soft_rows else pd.DataFrame(columns=['object_id']), folder / 'soft_manifest.parquet')
        write_json(folder / 'completed.json', dict(scans=len(active), accepted=len(catalogue), statuses=dict(Counter((r['status'] for r in audit)))))
        print(f'{period.name} chunk={chunk + 1}/{len(blocks)} accepted={len(catalogue)} last_scan={times[end - 1]}', flush=True)
        del fields
    chunks = [out / f'chunk_{i:05d}' for i in range(len(blocks))]
    cat = combine([p / 'catalogue.parquet' for p in chunks], AUDIT_COLUMNS)
    soft = combine([p / 'soft_manifest.parquet' for p in chunks], ['object_id'])
    if cat.object_id.duplicated().any():
        raise RuntimeError('Duplicated V3 IDs across chunks')
    write_frame(cat, out / 'catalogue.parquet')
    write_frame(soft, out / 'soft_manifest.parquet')
    counts = Counter()
    for p in chunks:
        counts.update(json.loads((p / 'completed.json').read_text())['statuses'])
    result = dict(period_id=period.name, year=year, split=split, scans=len(owned), stored_scans=len(times), accepted=len(cat), statuses=dict(counts), mrms_spacing_km=spacing, output_dir=str(out))
    write_json(finished, result)
    return result

def geometry_equivalence_selftest():
    from scipy.ndimage import label
    test_path = Path(_ci_resolve(__file__)).with_name('test_ci_v2_geometry_three_cases.py')
    if not test_path.is_file():
        raise RuntimeError('Missing tested geometry script: ' + str(test_path))
    reference = load('v3_geometry_reference', test_path)
    reference.selftest()
    rng = np.random.default_rng(187)
    n = 0
    for spacing in ((1.0, 1.0), (1.11, 0.835)):
        for trial in range(30):
            shape = (91, 103)
            f = np.zeros(shape, np.float32)
            f[44:47, 48] = 40
            if trial % 3 == 0:
                f[44:48, 53] = 40
            f[rng.random(shape) < 0.003] = 25
            f[rng.random(shape) < 0.003] = 15
            f[44:47, 48] = 40
            valid = rng.random(shape) > 0.0005
            labs, _ = label(f >= 35, np.ones((3, 3)))
            points = {int(k): np.argwhere(labs == k) for k in np.unique(labs) if k}
            seed = int(labs[45, 48])
            fresh = set(points)
            if trial % 4 == 0:
                fresh = {seed}
            cache = contour_context(labs, points, seed, spacing, 15)
            for offset in (1, 2, 3, 5, 8, 10, 14):
                a, ra, ma = test_contour(f, valid, labs, points, seed, fresh, offset, spacing, cache)
                b, rb, mb = reference.geometry(f, valid, labs, points, seed, fresh, 'contour', 20, offset, spacing)
                assert ra == rb, (ra, rb, offset)
                assert (a is None) == (b is None)
                for key in ma:
                    assert np.isclose(ma[key], mb[key], equal_nan=True)
                if a is not None:
                    assert a['members'] == b['members']
                    assert np.array_equal(a['band_points'], np.argwhere(b['band']))
                n += 1
    row = dict(object_id=object_id('PERIOD_cdd25d789f4df191a34ee013', '2021-06-28 17:52:39', 18), period_id='PERIOD_cdd25d789f4df191a34ee013', birth_time=pd.Timestamp('2021-06-28 17:52:39'), component_id=18)
    reviewed_event_metadata(row)
    assert row['canonical_birth_time'] == pd.Timestamp('2021-06-28 17:48:41')
    assert row['duplicate_of'] == object_id(row['period_id'], '2021-06-28 17:48:41', 16)
    print('V3 ROI/GLOBAL GEOMETRY EQUIVALENCE:', n, 'checks PASSED', flush=True)

def selftest(pilot):
    from scipy.ndimage import label

    class TestCore:

        @staticmethod
        def segment(f, valid, threshold, area, spacing):
            lab, n = label(valid & (f >= threshold), np.ones((3, 3)))
            return (lab, {i: i for i in range(1, n + 1)})

        @staticmethod
        def temporal_window_complete(times, start, end, gap, tolerance):
            return times.min() <= start and times.max() >= end

        @staticmethod
        def match_persistence(obj, i, times, labs, objects, spacing, cfg):
            return obj
    pilot.selftest()
    geometry_equivalence_selftest()
    rng = np.random.default_rng(12)
    for gap in [2.0, 5.0, 6.0]:
        for _ in range(12):
            current, _ = label(rng.random((51, 51)) > 0.995, np.ones((3, 3)))
            previous, _ = label(rng.random((51, 51)) > 0.995, np.ones((3, 3)))
            assert fast_lineage(current, previous, (1.1, 0.83), gap) == pilot.lineage(current, previous, (1.1, 0.83), gap)
    assert fast_lineage(np.zeros((10, 10), int), np.zeros((10, 10), int), (1, 1), 2) == {}
    times = pd.date_range('2021-01-01', periods=70, freq='2min')
    fields = np.zeros((70, 51, 51), np.float32)
    fields[30:40, 24:27, 25] = 40
    one = analyse(pilot, TestCore, fields, times, (1.0, 1.0), 23, 48, [5, 8, 10, 15])
    left = analyse(pilot, TestCore, fields[:38], times[:38], (1.0, 1.0), 23, 31, [5, 8, 10, 15])
    right = analyse(pilot, TestCore, fields[8:], times[8:], (1.0, 1.0), 23, 40, [5, 8, 10, 15])
    assert len(one[1]) == len(left[1]) + len(right[1]) == 1
    assert np.array_equal(one[1][0][1], left[1][0][1])
    assert one[1][0][0]['birth_time'] == times[30]
    print('FULL V3 SELFTEST PASSED: chunk boundary and native birth mask', flush=True)

def merge_windows(pairs):
    merged = []
    for a, b in sorted(set(((pd.Timestamp(a), pd.Timestamp(b)) for a, b in pairs))):
        if pd.isna(a) or pd.isna(b) or b < a:
            raise RuntimeError('Invalid V1 target window')
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(b, merged[-1][1])
        else:
            merged.append([a, b])
    return [[str(a), str(b)] for a, b in merged]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=PROJECT)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--lead-index', type=Path)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--chunk-minutes', type=int, default=60)
    parser.add_argument('--offsets', nargs='+', type=float, default=[1, 2, 3, 5, 8, 10, 14])
    parser.add_argument('--limit-periods', type=int, default=0)
    parser.add_argument('--selftest', action='store_true')
    args = parser.parse_args()
    code = args.project / str(_ci_source() / 'scripts/isolated_ci_binary_v1/goes_ir_v1')
    pilot_path = code / 'ci_label_pilot_v2.py'
    pilot = load('selftest_pilot', pilot_path)
    if args.selftest:
        selftest(pilot)
        return
    if args.workers < 1 or args.chunk_minutes < 10 or (not np.isfinite(args.offsets).all()) or (min(args.offsets) < 1) or (max(args.offsets) > 14):
        parser.error('Invalid workers, chunk size or offsets')
    native = args.project / 'data/ci_isolated_mrms_binary_v1'
    output = args.output_root or native / 'goes_ir_parallax_native_filled_v1/labels/ci_catalogue_v3_contour20_full_scans_v1'
    output.mkdir(parents=True, exist_ok=True)
    data = native / 'goes_ir_parallax_native_filled_v1'
    lead_path = args.lead_index or data / 'state/ci_goes_lead_index.parquet'
    lead = pd.read_parquet(_ci_resolve(lead_path))
    required = {'period_id', 'year', 'issuance_time', 'lead_start_minutes', 'lead_end_minutes'}
    if not required.issubset(lead.columns):
        raise RuntimeError('V1 lead index missing columns: ' + str(required - set(lead.columns)))
    start_times = pd.to_datetime(lead.issuance_time) + pd.to_timedelta(lead.lead_start_minutes, unit='min')
    end_times = pd.to_datetime(lead.issuance_time) + pd.to_timedelta(lead.lead_end_minutes, unit='min')
    lead = lead.assign(window_start=start_times, window_end=end_times)
    assignments = {}
    splits = data / 'splits/period_grouped_80_10_10_v1'
    for split in ['train', 'validation', 'test']:
        path = splits / f'{split}_issuance_index_natural.parquet'
        if not path.exists():
            raise RuntimeError('Missing original split index: ' + str(path))
        for pid in pd.read_parquet(_ci_resolve(path), columns=['period_id']).period_id.astype(str).unique():
            if pid in assignments and assignments[pid] != split:
                raise RuntimeError('V1 period split leakage: ' + pid)
            assignments[pid] = split
    plan = []
    for pid, frame in lead.groupby('period_id', sort=True):
        if frame.year.nunique() != 1:
            raise RuntimeError('Period has several years: ' + str(pid))
        year = int(frame.year.iloc[0])
        period = native / f'year_{year}_strict_v2_001/periods' / str(pid)
        if not (period / 'reflectivity.zarr').is_dir():
            raise RuntimeError('Missing retained V1 MRMS store: ' + str(period))
        windows = merge_windows(frame[['window_start', 'window_end']].itertuples(index=False, name=None))
        plan.append(dict(period=str(period), year=year, windows=windows, split=assignments.get(str(pid), 'unassigned_v1_no_usable_issuance')))
    v2_scope = data / 'labels/ci_catalogue_v2_full_scans_v1/v1_temporal_scope.csv'
    reference = pd.read_csv(_ci_resolve(v2_scope))
    expected = {(str(r.period_id), int(r.year), str(r.split), pd.Timestamp(r.window_start), pd.Timestamp(r.window_end)) for r in reference.itertuples()}
    proposed = {(Path(_ci_resolve(p['period'])).name, p['year'], p['split'], pd.Timestamp(a), pd.Timestamp(b)) for p in plan for a, b in p['windows']}
    if expected != proposed or len(plan) != 548:
        raise RuntimeError('V1 retained scope changed: expected exactly the existing 548 periods/windows')
    if args.limit_periods:
        plan = plan[:args.limit_periods]
    if not plan:
        raise RuntimeError('No periods in V1 lead index')
    sources = [Path(_ci_resolve(__file__)).resolve(), pilot_path, code / 'build_ci_soft_targets_native_goes.py', args.project / str(_ci_source() / 'scripts/isolated_ci_binary_v1/core.py'), lead_path, v2_scope, code / 'test_ci_v2_geometry_three_cases.py']
    config = dict(offsets=sorted(set(args.offsets)), chunk_minutes=args.chunk_minutes, sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, plan=plan, rules='tested contour_ge20; >=3 native pixels >=35; 1-km band <20; offsets recorded in config; persistence 10 +/-2', canonical_events='one user-reviewed pair linked to earliest birth; other proximity flags not merged', caveats='pixel-centre bands; no proof of continuous barrier; conservative lineage is not a full tracker; no retry of rejected first birth', scope='only original V1 lead-index target windows, all stored native scans within them', purpose='V3 catalogue and soft labels; training indexes NOT rebuilt')
    config_path = output / 'build_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError('Changed rules/code/windows: use a new output root, do not mix builds')
    write_json(config_path, config)
    write_frame(pd.DataFrame([dict(period_id=Path(_ci_resolve(p['period'])).name, year=p['year'], split=p['split'], window_start=a, window_end=b) for p in plan for a, b in p['windows']]), output / 'v1_temporal_scope.csv')
    jobs = [(p['period'], p['year'], str(output / 'periods' / Path(_ci_resolve(p['period'])).name), args.chunk_minutes, config['offsets'], p['windows'], p['split']) for p in plan]
    print(f'FULL V3: {len(jobs)} periods, {args.workers} CPU workers; no GPU; output={output}', flush=True)
    results, errors = ([], [])
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialise, initargs=(str(args.project),)) as pool:
        futures = {pool.submit(period_job, job): job[0] for job in jobs}
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                print(f"DONE {len(results)}/{len(jobs)} {result['period_id']} scans={result['scans']} groups={result['accepted']}", flush=True)
            except Exception as exc:
                traceback.print_exc()
                errors.append(dict(period=futures[future], error=repr(exc)))
                print('FAILED', futures[future], repr(exc), flush=True)
            write_json(output / 'progress.json', dict(completed=len(results), total=len(jobs), errors=errors))
    write_frame(pd.DataFrame(results), output / 'period_summary.csv')
    if errors:
        write_json(output / 'failed_periods.json', errors)
        raise RuntimeError(f'{len(errors)} periods failed; fix and rerun same command to resume')
    cat = combine([Path(_ci_resolve(r['output_dir'])) / 'catalogue.parquet' for r in results], AUDIT_COLUMNS)
    soft = combine([Path(_ci_resolve(r['output_dir'])) / 'soft_manifest.parquet' for r in results], ['object_id'])
    if cat.object_id.duplicated().any() or len(cat) != len(soft):
        raise RuntimeError('Final catalogue/soft manifest audit failed')
    cat = cat.sort_values(['year', 'period_id', 'birth_time', 'object_id'])
    write_frame(cat, output / 'ci_v3_object_catalogue.parquet')
    write_frame(soft, output / 'soft_target_build_manifest.parquet')
    old_path = native / 'goes_ir_parallax_native_filled_v1/labels/ci_soft_targets_v1/state/soft_target_build_manifest.parquet'
    old = pd.read_parquet(_ci_resolve(old_path))
    keep = np.zeros(len(old), bool)
    births = pd.to_datetime(old.birth_time)
    for p in plan:
        same_period = old.period_id.astype(str).to_numpy() == Path(_ci_resolve(p['period'])).name
        for a, b in p['windows']:
            keep |= same_period & (births >= pd.Timestamp(a)) & (births <= pd.Timestamp(b))
    old = old.loc[keep]
    write_frame(old, output / 'v1_objects_in_same_windows.parquet')
    event_table, flag_count = final_event_tables(cat, soft, output)
    event_counts = event_table.groupby(['year', 'period_id']).size().rename('v3_events')
    new_counts = cat.groupby(['year', 'period_id']).size().rename('v3_detections')
    old_counts = old.groupby(['year', 'period_id']).size().rename('v1_objects')
    comparison = pd.DataFrame(results)[['year', 'period_id', 'scans']].set_index(['year', 'period_id'])
    comparison = comparison.join(old_counts).join(new_counts).join(event_counts).fillna(0).reset_index()
    write_frame(comparison, output / 'v1_v3_counts_by_period.csv')
    write_frame(comparison.groupby('year')[['scans', 'v1_objects', 'v3_detections', 'v3_events']].sum().reset_index(), output / 'v1_v3_counts_by_year.csv')
    summary = dict(status='all_scans_completed_requires_review', periods=len(results), scans=sum((r['scans'] for r in results)), v3_detections=len(cat), v3_events=len(event_table), nearby_birth_flags=flag_count, v1_objects_in_same_periods=int(comparison.v1_objects.sum()), unique_v3_detection_ids=cat.object_id.nunique(), statuses=dict(sum((Counter(r['statuses']) for r in results), Counter())), comparison='Counts, not recall: V3 groups and V1 cells are different units', uncertain='per-chunk uncertain_footprints.npz keyed to audit.csv; NOT negative labels', training_indexes_rebuilt=False)
    write_json(output / 'completed.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    print('ALL V3 SCANS COMPLETED:', output, flush=True)
if __name__ == '__main__':
    main()
