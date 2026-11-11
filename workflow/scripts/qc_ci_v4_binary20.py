"""Read-only V4 audit + TRAIN review GIFs. Flags do not classify radar noise.

Audits every saved binary target. CH13 statistics use the saved >=35-dBZ
core pixels, at floor(birth, 10 min) and 10 min later. Future observations
are visual QC only. Source GOES before NaN filling is compared for selected
review cases, with exact times/channels and explicit grid checks.
No catalogue, targets, split, loss mask or training index is changed.
"""
from ci_nowcasting.paths import resolve_path as _ci_resolve, project_root as _ci_project, source_root as _ci_source
import argparse
import hashlib
import importlib.util
import json
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
PROJECT = Path(_ci_resolve(_ci_resolve('/crest/students/mcrampe00/PDE_CNN_precip')))
DATA = PROJECT / 'data/ci_isolated_mrms_binary_v1/goes_ir_parallax_native_filled_v1'
CODE = PROJECT / str(_ci_source() / 'scripts/isolated_ci_binary_v1/goes_ir_v1')
V4 = DATA / 'labels/ci_v4_binary20_birth_envelopes_v1'
DS = DATA / 'datasets/ci_v3_birth_2x30min_v1'
OUT = PROJECT / 'experiments/ci_v4_binary20_quality_review_v1'
KNOWN = 'ICIV3_2ec6365b216d25397f65aea9'

def json_write(path, value):
    temporary = Path(_ci_resolve(str(path) + '.tmp'))
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)

def file_sha(path):
    h = hashlib.sha256()
    with Path(_ci_resolve(path)).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def coords(z, prefix, shape):
    y, x = (np.asarray(z[prefix + '_y']), np.asarray(z[prefix + '_x']))
    if y.ndim != 1 or x.ndim != 1 or len(y) != len(x):
        raise ValueError('Invalid sparse coordinate dimensions: ' + prefix)
    if not np.issubdtype(y.dtype, np.integer) or not np.issubdtype(x.dtype, np.integer):
        raise ValueError('Noninteger sparse coordinates: ' + prefix)
    if np.any(y < 0) or np.any(y >= shape[0]) or np.any(x < 0) or np.any(x >= shape[1]):
        raise ValueError('Out-of-grid sparse coordinates: ' + prefix)
    return (y.astype(np.int64), x.astype(np.int64))

def check_target(path, birth, expected_pixels):
    with np.load(_ci_resolve(path), allow_pickle=False) as z:
        shape = tuple(np.asarray(z['shape']).ravel().astype(int))
        if shape != (212, 230):
            raise ValueError('Unexpected GOES grid: ' + str(shape))
        y, x = coords(z, 'soft', shape)
        cy, cx = coords(z, 'core', shape)
        hy, hx = coords(z, 'hard', shape)
        v = np.asarray(z['soft_value'])
        if not len(y) or v.shape != y.shape or (not np.isfinite(v).all()) or (not (v == 1).all()):
            raise ValueError('Empty or nonbinary V4 supervision')
        target = np.zeros(shape, bool)
        target[y, x] = True
        hard = np.zeros(shape, bool)
        hard[hy, hx] = True
        if target.sum() != len(y) or len(y) != int(expected_pixels):
            raise ValueError('Repeated pixels or manifest count mismatch')
        if not len(cy) or not target[cy, cx].all() or (not np.array_equal(target, hard)):
            raise ValueError('Birth core outside target / inconsistent hard mask')
        if int(z['birth_time_ns']) != pd.Timestamp(birth).value or float(z['envelope_threshold_dbz']) != 20:
            raise ValueError('Birth time or threshold mismatch')
        if any((float(z[key]) != 0 for key in ['sigma_km', 'hard_buffer_km', 'cutoff_km'])):
            raise ValueError('Residual soft/buffer parameter')
        native_shape = tuple(np.asarray(z['native_shape']).astype(int))
        ny, nx = coords(z, 'native_envelope', native_shape)
        if not len(ny):
            raise ValueError('Empty native envelope')
    return (cy, cx)

class Goes:

    def __init__(self, dataset):
        self.dataset = dataset
        self.cache = OrderedDict()
        self.raw = OrderedDict()

    def frame(self, stamp):
        stamp = pd.Timestamp(stamp)
        key = int(stamp.value)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        month = self.dataset._month_data(stamp)
        index = month['time_lookup'].get(key)
        if index is None:
            raise ValueError('Exact GOES time absent: ' + str(stamp))
        group = month['group']
        channels = np.asarray(group['channel'][:]).astype(int)
        indices = np.flatnonzero(channels == 13)
        if len(indices) != 1 or group['CMI'].attrs.get('units') != 'K':
            raise ValueError('CH13 channel coordinate or units mismatch')
        frame = np.asarray(group['CMI'][int(index), int(indices[0])], np.float32)
        self.cache[key] = frame
        if len(self.cache) > 12:
            self.cache.popitem(last=False)
        while len(self.dataset.month_cache) > 2:
            self.dataset.month_cache.pop(next(iter(self.dataset.month_cache)))
        return frame

    def source(self, stamp):
        import xarray as xr
        stamp = pd.Timestamp(stamp)
        filled = self.frame(stamp)
        group = self.dataset._month_data(stamp)['group']
        path = group.attrs.get('source_store')
        if not path or not Path(_ci_resolve(path)).is_dir():
            raise ValueError('Unfilled source_store unavailable')
        if path not in self.raw:
            self.raw[path] = xr.open_zarr(path, consolidated=False, chunks=None)
            if len(self.raw) > 2:
                _, old = self.raw.popitem(last=False)
                old.close()
        self.raw.move_to_end(path)
        source = self.raw[path]
        for coordinate in ['x', 'y', 'channel']:
            if coordinate not in source.coords or coordinate not in group:
                raise ValueError('Cannot verify raw/filled coordinate: ' + coordinate)
            left = np.asarray(source[coordinate].values)
            right = np.asarray(group[coordinate][:])
            if left.shape != right.shape or not np.allclose(left, right, rtol=0, atol=1e-06, equal_nan=True):
                raise ValueError('Raw/filled coordinate mismatch: ' + coordinate)
        cmi = source['CMI']
        if cmi.attrs.get('units') != 'K':
            raise ValueError('Unfilled CMI units are not K')
        frame = np.asarray(cmi.sel(time=stamp.to_datetime64(), channel=13).transpose('y', 'x').values, np.float32)
        if frame.shape != filled.shape:
            raise ValueError('Raw/filled frame shape mismatch')
        return (frame, str(path))

def summary_values(frame, cy, cx, prefix):
    values = frame[cy, cx]
    good = values[np.isfinite(values)]
    return {prefix + '_finite_fraction': float(np.isfinite(values).mean()), prefix + '_median_k': float(np.median(good)) if len(good) else np.nan, prefix + '_min_k': float(np.min(good)) if len(good) else np.nan}

def audit(args, manifest, goes):
    records = []
    for number, row in enumerate(manifest.sort_values('birth_time').itertuples(index=False), 1):
        info = dict(object_id=row.object_id, canonical_event_id=row.canonical_event_id, split=row.split, year=int(row.year), period_id=row.period_id, birth_time=str(row.birth_time), file_error='', goes_errors='', geometry_review=bool(row.requires_geometry_review), native_pixels=int(row.candidate_pixels), goes_pixels=int(row.goes_positive_pixels), old_soft_nonzero_pixels=int(row.old_soft_nonzero_pixels), old_target_ge_half_pixels=int(row.old_target_ge_half_pixels), known_suspect_review=bool(row.object_id == KNOWN))
        try:
            cy, cx = check_target(row.v4_target_path, row.birth_time, row.goes_positive_pixels)
            info['core_pixels'] = len(cy)
        except Exception as exc:
            info['file_error'] = type(exc).__name__ + ': ' + str(exc)
            records.append(info)
            continue
        base = pd.Timestamp(row.birth_time).floor('10min')
        info['goes_prebirth_time'] = str(base)
        info['goes_prebirth_age_seconds'] = (pd.Timestamp(row.birth_time) - base).total_seconds()
        errors = []
        for offset, prefix in [(0, 'prebirth'), (10, 'next_scan')]:
            try:
                info.update(summary_values(goes.frame(base + pd.Timedelta(minutes=offset)), cy, cx, prefix))
            except Exception as exc:
                errors.append(prefix + ': ' + str(exc))
        info['goes_errors'] = '; '.join(errors)
        info['warm_both_scans_review'] = bool(info.get('prebirth_median_k', -np.inf) >= 285 and info.get('next_scan_median_k', -np.inf) >= 285)
        records.append(info)
        if number == 1 or number % 500 == 0 or number == len(manifest):
            print(f'AUDIT {number}/{len(manifest)}', flush=True)
            json_write(args.output_dir / 'progress.json', dict(stage='audit', objects=number, total=len(manifest)))
    result = pd.DataFrame(records)
    result.to_csv(args.output_dir / 'object_qc.csv', index=False)
    summary = dict(detections=len(result), file_errors=int(result.file_error.ne('').sum()), goes_lookup_errors=int(result.goes_errors.ne('').sum()), geometry_review_flags=int(result.geometry_review.sum()), warm_both_scans_review=int(result.get('warm_both_scans_review', pd.Series(False, index=result.index)).fillna(False).sum()), known_suspects=int(result.known_suspect_review.sum()), threshold_k_for_review_only=285, note='Warm CH13 is a review flag, not a noise verdict. Next GOES scan is QC only. No automatic exclusion.')
    json_write(args.output_dir / 'audit_summary.json', summary)
    result.groupby(['year', 'split']).agg(detections=('object_id', 'size'), median_native_pixels=('native_pixels', 'median'), median_goes_pixels=('goes_pixels', 'median'), median_old_hard_pixels=('old_target_ge_half_pixels', 'median')).to_csv(args.output_dir / 'size_comparison.csv')
    print(json.dumps(summary, indent=2), flush=True)
    if summary['file_errors']:
        raise RuntimeError('Target integrity errors: inspect object_qc.csv before continuing')
    return result

def select_cases(qc, count, seed):
    eligible = qc.loc[(qc.split == 'train') & qc.file_error.fillna('').eq('')].copy()
    rng = np.random.default_rng(seed)
    selected, seen = ([], set())

    def take(frame, reason, limit, repeat_period=False):
        for row in frame.itertuples(index=False):
            if any((item['object_id'] == row.object_id for item in selected)):
                continue
            if not repeat_period and row.period_id in seen:
                continue
            selected.append(dict(object_id=row.object_id, reason=reason))
            seen.add(row.period_id)
            if len(selected) >= count:
                return
            limit -= 1
            if limit <= 0:
                break
    take(eligible.loc[eligible.object_id == KNOWN], 'known_24July_suspect', 1)
    warm = eligible.get('warm_both_scans_review', pd.Series(False, index=eligible.index)).fillna(False)
    if 'next_scan_median_k' in eligible:
        take(eligible.loc[warm].sort_values('next_scan_median_k', ascending=False), 'warm_two_GOES_scans_review', 3)
    if len(selected) < count:
        take(eligible.loc[eligible.geometry_review].sort_values('native_pixels', ascending=False), 'geometry_clip_review', 2)
    if len(selected) < count:
        take(eligible.sort_values('native_pixels', ascending=False), 'large_envelope', 2)
    if len(selected) < count:
        take(eligible.sort_values('native_pixels'), 'small_envelope', 1)
    for year, group in eligible.groupby('year'):
        if len(selected) >= count:
            break
        take(group.iloc[rng.permutation(len(group))], f'random_control_{year}', 1)
    if len(selected) < count:
        take(eligible.iloc[rng.permutation(len(eligible))], 'random_control', count - len(selected))
    return pd.DataFrame(selected[:count])

def raw_comparison(args, row, goes, folder, pilot):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.ndimage import distance_transform_edt
    with np.load(_ci_resolve(row.v4_target_path), allow_pickle=False) as stored:
        shape = tuple(stored['shape'])
        cy, cx = coords(stored, 'core', shape)
    core = np.zeros(shape, bool)
    core[cy, cx] = True
    region = pilot.origin(core, 30, 30)
    rows, errors = ([], [])
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    cmap = plt.get_cmap('turbo').copy()
    cmap.set_bad('#999999')
    for line, offset in enumerate([0, 10]):
        stamp = pd.Timestamp(row.birth_time).floor('10min') + pd.Timedelta(minutes=offset)
        record = dict(object_id=row.object_id, time=str(stamp), raw_error='')
        filled, raw = (np.full(shape, np.nan), np.full(shape, np.nan))
        try:
            filled = goes.frame(stamp)
            raw, source_path = goes.source(stamp)
            record['source_store'] = source_path
            record['raw_core_missing_fraction'] = float((~np.isfinite(raw[cy, cx])).mean())
            if np.isfinite(raw).any():
                distance = distance_transform_edt(~np.isfinite(raw))
                record['core_fill_distance_pixels_max'] = float(distance[cy, cx].max())
            record.update(summary_values(raw, cy, cx, 'raw_core'))
            valid_pair = np.isfinite(raw) & np.isfinite(filled)
            record['max_change_at_originally_finite_pixels_k'] = float(np.max(np.abs(raw[valid_pair] - filled[valid_pair]))) if valid_pair.any() else np.nan
        except Exception as exc:
            record['raw_error'] = type(exc).__name__ + ': ' + str(exc)
            errors.append(record['raw_error'])
        for col, data, title in [(0, raw, 'CH13 before NaN filling'), (1, filled, 'CH13 after NaN filling')]:
            im = axes[line, col].imshow(data[region], cmap=cmap, vmin=190, vmax=310, interpolation='nearest')
            if core[region].any() and (not core[region].all()):
                axes[line, col].contour(core[region], levels=[0.5], colors='magenta', linewidths=0.8)
            axes[line, col].set_title(title + '\n' + str(stamp), fontsize=9)
        difference = filled - raw
        imd = axes[line, 2].imshow(difference[region], cmap='RdBu_r', vmin=-5, vmax=5, interpolation='nearest')
        axes[line, 2].set_title('Filled minus original finite values\nGray: original missing / unavailable', fontsize=9)
        rows.append(record)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes[:, :2].ravel().tolist(), label='K', shrink=0.6)
    fig.colorbar(imd, ax=axes[:, 2].ravel().tolist(), label='K', shrink=0.6)
    fig.suptitle(str(row.object_id) + '\nMagenta: saved >=35-dBZ birth core; gray: missing. ' + ('SOURCE LOOKUP ERROR: see raw_comparison.json' if errors else 'Raw/filled grid checked.'), fontsize=10)
    fig.savefig(folder / 'CH13_before_after_fill.png', dpi=140)
    plt.close(fig)
    json_write(folder / 'raw_comparison.json', rows)
    return rows

def review(args, manifest, qc, goes, dataset):
    path = PROJECT / str(_ci_source() / 'scripts/pilot_ci_v4_binary_envelopes.py')
    spec = importlib.util.spec_from_file_location('v4_quality_pilot', path)
    pilot = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pilot
    spec.loader.exec_module(pilot)
    from build_ci_soft_targets_native_goes import load_and_validate_goes_grid, build_mapping
    grid = load_and_validate_goes_grid()
    cat = pd.read_parquet(_ci_resolve(args.v4_root / 'ci_v4_detection_catalogue.parquet'))
    cat.birth_time = pd.to_datetime(cat.birth_time)
    cases = select_cases(qc, args.cases, args.seed)
    cases.to_csv(args.output_dir / 'selected_cases.csv', index=False)
    visual_dir = args.output_dir / 'visuals'
    visual_dir.mkdir(exist_ok=True)
    render_args = SimpleNamespace(output_dir=visual_dir, thresholds=[35, 20], patch_km=60.0, goes_dy=float(grid['dy_km']), goes_dx=float(grid['dx_km']), before=args.before, after=args.after, step_minutes=2, fps=4)
    raw_reports, failures = ([], [])
    for number, selected in enumerate(cases.itertuples(index=False), 1):
        row = cat.loc[cat.object_id == selected.object_id].iloc[0]
        print(f'REVIEW {number}/{len(cases)} {row.object_id} {selected.reason}', flush=True)
        radar_path = PROJECT / 'data/ci_isolated_mrms_binary_v1' / f'year_{int(row.year)}_strict_v2_001/periods' / str(row.period_id) / 'reflectivity.zarr'
        try:
            reader = pilot.Radar(radar_path)
            mapping = build_mapping(radar_path.parent, grid, {})
            info = dict(row=row, nearby_mature_pixels=0, group_pixels=int(row.group_pixels), group_components=1)
            pilot.render(render_args, info, radar_path, reader, mapping, dataset, (212, 230), number)
            folder = visual_dir / f'case_{number:02d}_{row.object_id}'
            with np.load(_ci_resolve(folder / 'candidate_masks_native.npz')) as shown, np.load(_ci_resolve(row.v4_target_path)) as saved:
                displayed = np.column_stack([shown['y_ge20'], shown['x_ge20']])
                actual = np.column_stack([saved['native_envelope_y'], saved['native_envelope_x']])
                if not np.array_equal(displayed, actual):
                    raise RuntimeError('Rendered envelope differs from saved V4 target')
            raw_reports.extend(raw_comparison(args, row, goes, folder, pilot))
        except Exception as exc:
            failures.append(dict(object_id=row.object_id, error=type(exc).__name__ + ': ' + str(exc)))
            print('REVIEW FAILED:', failures[-1], flush=True)
    pd.DataFrame(raw_reports).to_csv(args.output_dir / 'raw_GOES_review.csv', index=False)
    json_write(args.output_dir / 'render_report.json', dict(selected=len(cases), failures=failures, scope='Curated TRAIN visual review, not precision/recall. No automatic noise classification.'))
    links = []
    import html
    for folder in sorted(visual_dir.glob('case_*')):
        name = html.escape(folder.name)
        links.append(f'<h2>{name}</h2><a href="visuals/{name}/comparison.gif">Radar/CH13 animation</a> | <a href="visuals/{name}/CH13_before_after_fill.png">GOES before/after fill</a>')
    (args.output_dir / 'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>V4 quality review</h1>' + ''.join(links), encoding='utf-8')
    return failures

def selftest():
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as temporary:
        path = Path(_ci_resolve(temporary)) / 'test.npz'
        base = dict(shape=np.array([212, 230]), soft_y=np.array([2, 3]), soft_x=np.array([2, 3]), soft_value=np.ones(2), core_y=np.array([2]), core_x=np.array([2]), hard_y=np.array([2, 3]), hard_x=np.array([2, 3]), native_shape=np.array([654, 622]), native_envelope_y=np.array([5]), native_envelope_x=np.array([5]), birth_time_ns=np.int64(pd.Timestamp('2022-07-24').value), envelope_threshold_dbz=20, sigma_km=0, hard_buffer_km=0, cutoff_km=0)
        np.savez(path, **base)
        cy, cx = check_target(path, '2022-07-24', 2)
        assert cy.tolist() == [2] and cx.tolist() == [2]
        base['soft_value'] = np.array([1.0, 0.2])
        np.savez(path, **base)
        try:
            check_target(path, '2022-07-24', 2)
        except ValueError:
            pass
        else:
            raise AssertionError('Soft value was not rejected')
    print('V4 TARGET INTEGRITY SELFTEST PASSED', flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selftest', action='store_true')
    parser.add_argument('--mode', choices=['audit', 'review', 'all'], default='all')
    parser.add_argument('--v4-root', type=Path, default=V4)
    parser.add_argument('--dataset-root', type=Path, default=DS)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    parser.add_argument('--cases', type=int, default=16)
    parser.add_argument('--before', type=int, default=10)
    parser.add_argument('--after', type=int, default=30)
    parser.add_argument('--seed', type=int, default=20261015)
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    if args.cases < 1 or args.before < 0 or args.after < 0:
        parser.error('Invalid review settings')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = dict(manifest_sha256=file_sha(args.v4_root / 'target_manifest.parquet'), cases=args.cases, before=args.before, after=args.after, seed=args.seed)
    config_path = args.output_dir / 'qc_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError('QC input/settings changed: choose a new output directory')
    json_write(config_path, config)
    status_path = args.output_dir / 'completed.json'
    if args.mode == 'all' and status_path.exists():
        print('ALREADY COMPLETED:', status_path, flush=True)
        return
    source_status = json.loads((args.v4_root / 'completed.json').read_text())
    if source_status['status'] != 'all_v4_targets_built_requires_quality_review':
        raise RuntimeError('Full V4 target build is not complete')
    manifest = pd.read_parquet(_ci_resolve(args.v4_root / 'target_manifest.parquet'))
    manifest.birth_time = pd.to_datetime(manifest.birth_time)
    if len(manifest) != int(source_status['detections']) or manifest.object_id.duplicated().any():
        raise RuntimeError('V4 manifest identity/count mismatch')
    sys.path.insert(0, str(CODE))
    from ci_goes_v3_30min_dataset import CIGoesV3Dataset
    dataset = CIGoesV3Dataset(args.dataset_root, split='train', mode='full', normalization=None, include_mrms=False)
    goes = Goes(dataset)
    qc_path = args.output_dir / 'object_qc.csv'
    if args.mode != 'review':
        qc = audit(args, manifest, goes)
    else:
        qc = pd.read_csv(_ci_resolve(qc_path))
        if len(qc) != len(manifest) or set(qc.object_id) != set(manifest.object_id):
            raise RuntimeError('Audit incomplete or stale')
    decisions = manifest[['object_id', 'canonical_event_id', 'period_id', 'split', 'birth_time']].copy()
    decisions['decision'] = 'unreviewed'
    decisions['reason'] = ''
    decisions_path = args.output_dir / 'manual_decisions_template.csv'
    if not decisions_path.exists():
        decisions.to_csv(decisions_path, index=False)
    failures = review(args, manifest, qc, goes, dataset) if args.mode != 'audit' else []
    if args.mode != 'audit':
        json_write(status_path, dict(status='quality_diagnostics_generated_requires_human_review', detections=len(manifest), selected_train_cases=args.cases, render_failures=failures, exclusions_applied=0, source_GOES_errors_report='raw_GOES_review.csv', note='Diagnostics completed does not mean labels approved. No production files changed.'))
    print('QUALITY DIAGNOSTIC FINISHED:', args.output_dir, flush=True)
if __name__ == '__main__':
    main()
