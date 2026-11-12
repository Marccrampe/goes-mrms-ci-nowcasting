"""Build reviewable V4 per-detection targets; never change V3 or train models.

Geometry reproduces the 20-dBZ pilot: seeded 8-connected component, clipped
to distance <= selected_offset_km from the native birth core and excluding
the saved isolation band. Native MRMS and exact birth times are retained.
This changes supervision geometry, not the event detection definition.
"""
from ci_nowcasting.paths import resolve_path as _ci_resolve, project_root as _ci_project, source_root as _ci_source
import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
PROJECT = Path(_ci_resolve(_ci_resolve('/project')))
DATA = PROJECT / 'data/ci_isolated_mrms_binary_v1/goes_ir_parallax_native_filled_v1'
CODE = PROJECT / str(_ci_source() / 'scripts/isolated_ci_binary_v1/goes_ir_v1')
CAT = DATA / 'labels/ci_catalogue_v3_contour20_full_scans_v1'
DS = DATA / 'datasets/ci_v3_birth_2x30min_v1'
DEFAULT_OUT = DATA / 'labels/ci_v4_binary20_birth_envelopes_v1'
SUSPECT_REVIEW_IDS = {'ICIV3_2ec6365b216d25397f65aea9'}

def sha(path):
    digest = hashlib.sha256()
    with Path(_ci_resolve(path)).open('rb') as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()

def atomic_json(path, value):
    temporary = Path(_ci_resolve(str(path) + '.tmp'))
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)

def atomic_parquet(path, value):
    temporary = Path(_ci_resolve(str(path) + '.tmp'))
    value.to_parquet(temporary, index=False)
    temporary.replace(path)

def atomic_npz(path, value):
    temporary = Path(_ci_resolve(str(path) + '.tmp'))
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **value)
    temporary.replace(path)

def existing_path(value):
    if value is None or pd.isna(value) or (not str(value).strip()):
        return None
    path = Path(_ci_resolve(str(value)))
    return path if path.is_absolute() else PROJECT / path

def load_pilot(path):
    spec = importlib.util.spec_from_file_location('v4_binary20_pilot', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if 'thresholds' not in inspect.signature(module.envelopes).parameters:
        raise RuntimeError('Pilot is too old: install the version supporting 20 dBZ.')
    return module

def make_target(pilot, field, row, shape, mapping):
    seed = pilot.native_mask(row.birth_mask_path, field.shape, int(row.group_pixels))
    band = pilot.native_mask(row.isolation_band_path, field.shape)
    masks, reports = pilot.envelopes(field, seed, float(row.native_dy_km), float(row.native_dx_km), float(row.selected_offset_km), band, np.zeros(field.shape, bool), thresholds=(35, 20))
    envelope = masks[20]
    projected = pilot.project_mask(envelope, mapping, shape)
    projected_core = pilot.project_mask(seed, mapping, shape)
    old_path = existing_path(row.soft_target_path)
    if old_path is None or not old_path.is_file():
        raise RuntimeError(f'Missing original projected target for {row.object_id}: {old_path}')
    with np.load(_ci_resolve(old_path), allow_pickle=False) as stored:
        old = {key: stored[key].copy() for key in stored.files}
    if tuple(np.asarray(old['shape']).ravel()) != shape:
        raise RuntimeError('Original target grid mismatch: ' + str(old_path))
    old_core = np.zeros(shape, bool)
    old_core[old['core_y'], old['core_x']] = True
    if not np.all(projected[old_core]):
        raise RuntimeError('V4 does not contain the saved GOES birth core: ' + str(row.object_id))
    ys, xs = np.nonzero(projected)
    nys, nxs = np.nonzero(envelope)
    old_soft = np.zeros(shape, np.float32)
    np.maximum.at(old_soft, (old['soft_y'], old['soft_x']), old['soft_value'])
    old.update(soft_y=ys.astype(np.uint16), soft_x=xs.astype(np.uint16), soft_value=np.ones(len(ys), np.float32), hard_y=ys.astype(np.uint16), hard_x=xs.astype(np.uint16), sigma_km=np.float32(0), hard_buffer_km=np.float32(0), cutoff_km=np.float32(0), native_envelope_y=nys.astype(np.uint16), native_envelope_x=nxs.astype(np.uint16), native_shape=np.asarray(field.shape, np.int32), birth_time_ns=np.int64(pd.Timestamp(row.birth_time).value), envelope_threshold_dbz=np.float32(20), target_kind=np.asarray('binary_seeded_native_birth_envelope'))
    report = next((item for item in reports if item['threshold_dbz'] == 20)).copy()
    for key in list(report):
        if 'prior' in key:
            del report[key]
    report.update(object_id=str(row.object_id), canonical_event_id=str(row.canonical_event_id), period_id=str(row.period_id), split=str(row.split), year=int(row.year), birth_time=pd.Timestamp(row.birth_time), selected_offset_km=float(row.selected_offset_km), old_soft_target_path=str(old_path), goes_positive_pixels=int(projected.sum()), goes_native_core_pixels=int(projected_core.sum()), original_goes_core_pixels=int(old_core.sum()), old_soft_nonzero_pixels=int((old_soft > 0).sum()), old_target_ge_half_pixels=int((old_soft >= 0.5).sum()), old_target_mass=float(old_soft.sum()), v4_target_mass=int(projected.sum()), native_core_median_dbz=float(np.median(field[seed])), native_core_max_dbz=float(np.max(field[seed])), suspect_case_manual_review=bool(str(row.object_id) in SUSPECT_REVIEW_IDS), radar_quality_review_status='pending', action='retain_pending_review')
    if not len(ys):
        raise RuntimeError('Empty projected V4 target: ' + str(row.object_id))
    if not np.isfinite(old['soft_value']).all() or not (old['soft_value'] == 1).all():
        raise RuntimeError('V4 target is not binary')
    return (old, report)

def selftest(pilot):
    pilot.selftest()
    field = np.full((13, 15), -10.0, np.float32)
    field[5:8, 5:8] = 25.0
    field[6, 6] = 40.0
    field[0:2, 0:2] = 40.0
    seed = np.zeros(field.shape, bool)
    seed[6, 6] = True
    band = np.zeros(field.shape, bool)
    band[:, 9] = True
    masks, _ = pilot.envelopes(field, seed, 1.0, 0.8, 2.0, band, np.zeros(field.shape, bool), thresholds=(35, 20))
    assert masks[20].sum() == 9 and (not masks[20][0, 0])
    mapping = dict(source_y=np.array([5, 6, 7]), source_x=np.array([5, 6, 7]), target_y=np.array([1, 1, 2]), target_x=np.array([1, 1, 2]))
    target = pilot.project_mask(masks[20], mapping, (4, 4))
    assert target.dtype == bool and target.sum() == 2
    print('V4 BINARY20 BUILD SELFTEST PASSED', flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-script', type=Path, default=PROJECT / str(_ci_source() / 'scripts/pilot_ci_v4_binary_envelopes.py'))
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--catalogue-root', type=Path, default=CAT)
    parser.add_argument('--dataset-root', type=Path, default=DS)
    parser.add_argument('--limit-periods', type=int, default=0)
    parser.add_argument('--selftest', action='store_true')
    args = parser.parse_args()
    pilot = load_pilot(args.pilot_script)
    if args.selftest:
        selftest(pilot)
        return
    if args.limit_periods < 0:
        parser.error('--limit-periods must be nonnegative')
    source = args.catalogue_root / 'ci_v3_object_catalogue.parquet'
    out = args.output_dir.resolve()
    if out == args.catalogue_root.resolve() or out == args.dataset_root.resolve():
        raise RuntimeError('Choose a separate V4 output directory')
    out.mkdir(parents=True, exist_ok=True)
    config = dict(version=1, catalogue=str(source), catalogue_sha256=sha(source), pilot_sha256=sha(args.pilot_script), builder_sha256=sha(Path(_ci_resolve(__file__))), threshold_dbz=20, source_dataset=str(args.dataset_root.resolve()), geometry='Pilot-exact seeded 8-connected native echo; clipped to selected_offset_km from core; saved isolation band excluded', projection='Any mapped native positive pixel; binary output; original >=35 core retained', temporal_rule='Exact birth scan; original V3 detection/event identity retained', radar_quality_filter='None; every accepted V3 detection retained pending review', limit_periods=args.limit_periods)
    config_path = out / 'build_config.json'
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise RuntimeError('Build configuration changed; choose a new output directory')
    else:
        atomic_json(config_path, config)
    if (out / 'completed.json').exists():
        print('ALREADY COMPLETED:', out, flush=True)
        return
    sys.path.insert(0, str(CODE))
    from build_ci_soft_targets_native_goes import load_and_validate_goes_grid, build_mapping
    from ci_goes_v3_30min_dataset import base as original_loader
    shape = tuple(original_loader.GRID_SHAPE)
    grid = load_and_validate_goes_grid()
    table = pd.read_parquet(_ci_resolve(source))
    table = table.loc[table.status == 'v3_accepted_requires_review'].copy()
    if table.object_id.duplicated().any() or table.empty:
        raise RuntimeError('Empty/duplicate accepted detection catalogue')
    table.birth_time = pd.to_datetime(table.birth_time)
    paths = {}
    for split in ['train', 'validation', 'test']:
        index = pd.read_parquet(_ci_resolve(args.dataset_root / f'{split}_issuance_index_natural.parquet'))
        for pid, frame in index.groupby('period_id'):
            values = frame.radar_path.astype(str).unique()
            if len(values) != 1 or (str(pid) in paths and paths[str(pid)] != values[0]):
                raise RuntimeError('Conflicting radar paths for ' + str(pid))
            paths[str(pid)] = values[0]
    periods = sorted(table.period_id.astype(str).unique())
    if args.limit_periods:
        periods = periods[:args.limit_periods]
    all_reports = []
    for number, pid in enumerate(periods, 1):
        rows = table.loc[table.period_id.astype(str) == pid].sort_values(['birth_time', 'object_id'])
        period_dir = out / 'periods' / pid
        period_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = period_dir / 'target_manifest.parquet'
        done = period_dir / 'completed.json'
        if done.exists():
            reports = pd.read_parquet(_ci_resolve(manifest_path))
            if reports.object_id.tolist() != rows.object_id.astype(str).tolist():
                raise RuntimeError('Resume catalogue mismatch: ' + pid)
            if not all((Path(_ci_resolve(path)).is_file() for path in reports.v4_target_path)):
                raise RuntimeError('Resume target file missing: ' + pid)
            print(f'SKIP {number}/{len(periods)} {pid} detections={len(reports)}', flush=True)
        else:
            print(f'START {number}/{len(periods)} {pid} detections={len(rows)}', flush=True)
            radar_path = paths.get(pid)
            if radar_path is None:
                year = int(rows.iloc[0].year)
                radar_path = str(PROJECT / 'data/ci_isolated_mrms_binary_v1' / f'year_{year}_strict_v2_001/periods' / pid / 'reflectivity.zarr')
            reader = pilot.Radar(radar_path)
            mapping = build_mapping(Path(_ci_resolve(radar_path)).parent, grid, {})
            records = []
            for birth, birth_rows in rows.groupby('birth_time', sort=True):
                field = reader.exact(birth)
                for row in birth_rows.itertuples(index=False):
                    payload, report = make_target(pilot, field, row, shape, mapping)
                    destination = period_dir / (str(row.object_id) + '.npz')
                    atomic_npz(destination, payload)
                    report['v4_target_path'] = str(destination)
                    records.append(report)
            reports = pd.DataFrame(records).sort_values(['birth_time', 'object_id']).reset_index(drop=True)
            atomic_parquet(manifest_path, reports)
            atomic_json(done, dict(status='built_requires_radar_quality_review', detections=len(reports)))
            print(f'DONE {number}/{len(periods)} {pid} detections={len(reports)}', flush=True)
        all_reports.append(reports)
        atomic_json(out / 'progress.json', dict(periods_completed=number, periods_total=len(periods), detections_completed=sum((len(part) for part in all_reports))))
    report = pd.concat(all_reports, ignore_index=True)
    if not args.limit_periods and len(report) != len(table):
        raise RuntimeError('V4 count does not match accepted V3 detections')
    if report.object_id.duplicated().any():
        raise RuntimeError('Duplicate V4 object ID')
    if report.old_soft_target_path.duplicated().any():
        raise RuntimeError('Multiple detections share an original target path; inspect before remapping windows')
    atomic_parquet(out / 'target_manifest.parquet', report)
    report.to_csv(out / 'target_manifest.csv', index=False)
    report[['object_id', 'canonical_event_id', 'period_id', 'split', 'birth_time', 'suspect_case_manual_review', 'requires_geometry_review', 'radar_quality_review_status', 'action']].to_csv(out / 'quality_review.csv', index=False)
    enriched = table.merge(report[['object_id', 'v4_target_path', 'radar_quality_review_status']], on='object_id', how='inner', validate='one_to_one')
    atomic_parquet(out / 'ci_v4_detection_catalogue.parquet', enriched)
    stats = report.groupby(['year', 'split']).agg(detections=('object_id', 'size'), native_envelope_pixels=('candidate_pixels', 'sum'), goes_positive_pixels=('goes_positive_pixels', 'sum'), geometry_review_flags=('requires_geometry_review', 'sum')).reset_index()
    stats.to_csv(out / 'summary_by_year_split.csv', index=False)
    status = dict(status='partial_smoke_completed' if args.limit_periods else 'all_v4_targets_built_requires_quality_review', periods=len(periods), detections=len(report), canonical_events=int(report.canonical_event_id.nunique()), threshold_dbz=20, target_values=[0, 1], geometry_review_flags=int(report.requires_geometry_review.sum()), manually_identified_suspects=int(report.suspect_case_manual_review.sum()), radar_quality_review='pending; all detections retained', window_indexes_rebuilt=False, training_started=False, next_step='Review suspect detections, then remap V3 window references to approved V4 targets and recompute supervised positivity.', note='No V3 files changed. No event recall claim. V4 can be smaller than the old 4-km buffered soft target.')
    atomic_json(out / 'completed.json', status)
    print(stats.to_string(index=False), flush=True)
    print(json.dumps(status, indent=2), flush=True)
    print('V4 TARGET BUILD COMPLETED:', out, flush=True)
if __name__ == '__main__':
    main()
