"""Three-window geometry pilot; production catalogues are read-only."""
from ci_nowcasting.paths import resolve_path as _ci_resolve, project_root as _ci_project, source_root as _ci_source
import argparse
import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, minimum, maximum, label
PROJECT = Path(_ci_resolve(_ci_resolve('/crest/students/mcrampe00/PDE_CNN_precip')))
VARIANTS = {'circle_V2_ge10': ('circle', 10, [5, 8, 10, 15]), 'circle_small_ge10': ('circle', 10, [2, 3, 5, 8, 10, 15]), 'contour_ge10': ('contour', 10, [1, 2, 3, 5, 8, 10, 14]), 'contour_ge20': ('contour', 20, [1, 2, 3, 5, 8, 10, 14]), 'contour_ge35': ('contour', 35, [1, 2, 3, 5, 8, 10, 14])}
CASES = [('case07_2025', 2025, 'PERIOD_58b97d09918d77406e338c80', '2025-06-07 17:26:38', 'ICI_f3e4070ed40c6a8782b1ba21'), ('case08_2022', 2022, 'PERIOD_041ee0775b3552047fec8938', '2022-06-04 20:28:43', 'ICI_76c59e816f072c26b13cd573'), ('case15_near_storm', 2021, 'PERIOD_cdd25d789f4df191a34ee013', '2021-06-28 17:44:40', None)]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj

def save_json(path, value):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str))
    tmp.replace(path)

def lineage(current, previous, spacing, gap):
    ids = np.unique(current)
    ids = ids[ids > 0]
    if previous is None or gap > 5:
        return {int(k): 'unknown_previous_scan' for k in ids}
    d = distance_transform_edt(previous == 0, sampling=spacing) if previous.any() else np.full(previous.shape, np.inf)
    near = minimum(d, labels=current, index=ids)
    overlap = maximum((previous > 0).astype(np.uint8), labels=current, index=ids)
    return {int(k): 'continuing_or_split' if hit else 'ambiguous_motion_or_nearby_birth' if distance <= 1 + 2.4 * gap else 'new_candidate' for k, hit, distance in zip(ids, overlap, near)}

def geometry(frame, valid, labs, points, seed, fresh, kind, threshold, parameter, spacing):
    """Circle radius OR outward distance from the complete >=35 core.
    All bands are 1 km thick. The contour's outer reach is parameter+1 km.
    Distance is measured between native pixel centres, as in the existing pilot.
    """
    p = points[seed]
    cy, cx = p.mean(axis=0)
    yy, xx = np.ogrid[:frame.shape[0], :frame.shape[1]]
    if kind == 'circle':
        d = np.hypot((yy - cy) * spacing[0], (xx - cx) * spacing[1])
        inside, outer = (d < parameter - 1, d <= parameter)
        edge = min((cy + 0.5) * spacing[0], (frame.shape[0] - 0.5 - cy) * spacing[0], (cx + 0.5) * spacing[1], (frame.shape[1] - 0.5 - cx) * spacing[1])
        boundary = edge < parameter
    else:
        seed_mask = labs == seed
        d = distance_transform_edt(~seed_mask, sampling=spacing)
        inside, outer = (d <= parameter, d <= parameter + 1)
        edge = min((p[:, 0].min() + 0.5) * spacing[0], (frame.shape[0] - 0.5 - p[:, 0].max()) * spacing[0], (p[:, 1].min() + 0.5) * spacing[1], (frame.shape[1] - 0.5 - p[:, 1].max()) * spacing[1])
        boundary = edge < parameter + 1
    band = outer & ~inside
    details = {'band_pixels': int(band.sum()), 'band_valid_fraction': float(valid[band].mean()) if band.any() else np.nan}
    for z in [10, 20, 35]:
        details[f'band_fraction_ge{z}'] = float(np.mean(frame[band] >= z)) if band.any() else np.nan

    def reject(reason):
        return (None, reason, details)
    if boundary or not band.any():
        return reject('boundary')
    if not valid[band].all():
        return reject('unknown_annulus_pixels')
    if valid[outer].mean() < 0.99:
        return reject('invalid_disk')
    if np.any(frame[band] >= threshold):
        return reject(f'echo_ge{threshold}_in_annulus')
    members = sorted(set(map(int, np.unique(labs[inside]))) - {0})
    if seed not in members:
        return reject('seed_not_enclosed')
    for k in members:
        q = points[k]
        if not inside[q[:, 0], q[:, 1]].all():
            return reject('component_crosses_annulus')
        if k not in fresh:
            return reject('enclosed_existing_or_ambiguous_core')
    return ({'members': members, 'outer': outer, 'band': band}, 'accepted_geometry', details)

def selftest():
    f = np.zeros((61, 61), np.float32)
    f[30, 20:41] = 40
    valid = np.ones(f.shape, bool)
    labs, _ = label(f >= 35, np.ones((3, 3)))
    points = {1: np.argwhere(labs == 1)}
    assert geometry(f, valid, labs, points, 1, {1}, 'circle', 10, 5, (1, 1))[0] is None
    assert geometry(f, valid, labs, points, 1, {1}, 'contour', 10, 1, (1, 1))[0]
    f[30, 42] = 15
    assert geometry(f, valid, labs, points, 1, {1}, 'contour', 10, 1, (1, 1))[0] is None
    assert geometry(f, valid, labs, points, 1, {1}, 'contour', 20, 1, (1, 1))[0]
    valid[30, 42] = False
    assert geometry(f, valid, labs, points, 1, {1}, 'contour', 20, 1, (1, 1))[1] == 'unknown_annulus_pixels'
    assert lineage(labs, labs, (1, 1), 2)[1] == 'continuing_or_split'
    f = np.zeros((61, 61), np.float32)
    f[30:33, 30] = 40
    f[30:33, 32] = 40
    labs, _ = label(f >= 35, np.ones((3, 3)))
    points = {k: np.argwhere(labs == k) for k in (1, 2)}
    answer = geometry(f, np.ones_like(f, bool), labs, points, 1, {1}, 'contour', 35, 5, (1, 1))
    assert answer[1] == 'enclosed_existing_or_ambiguous_core'
    print('GEOMETRY SELFTEST PASSED', flush=True)

def render(destination, name, times, fields, spacing, bounds, events, reference, lookup, renderer, old, saved_new):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    y0, y1, x0, x1 = bounds
    crop = np.s_[y0:y1, x0:x1]
    extent = (x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5)
    reference_points = np.argwhere(reference)
    images, timing = ([], [])
    birth = reference_time = pd.Timestamp(dict(((c[0], c[3]) for c in CASES))[name])
    nearest = int(np.abs(times.asi8 - birth.value).argmin())
    for i, t in enumerate(times):
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout='constrained')
        ax = axes.ravel()
        radar = np.where(np.isfinite(fields[i]) & (fields[i] >= -50), fields[i], np.nan)
        for panel, variant in enumerate(VARIANTS):
            ax[panel].imshow(radar[crop], extent=extent, vmin=0, vmax=60, cmap='turbo')
            recent = [e for e in events[variant] if t - pd.Timedelta(minutes=10) < e['time'] <= t]
            for e in recent:
                p = e['points']
                ax[panel].scatter(p[:, 1], p[:, 0], s=10, facecolors='none', edgecolors='magenta', linewidths=0.8)
                band = e['band'][crop]
                if band.any() and (not band.all()):
                    gy, gx = np.mgrid[y0:y1, x0:x1]
                    ax[panel].contour(gx, gy, band.astype(float), [0.5], colors=['white'], linewidths=0.6)
            ax[panel].set_title(variant + '\nNative MRMS; births in preceding 10 min', fontsize=9)
        error, source = ('', 'MISSING')
        try:
            goes, source = renderer.goes_frame(t)
            native_goes = np.full(fields[i].shape, np.nan)
            inside = lookup >= 0
            native_goes[inside] = goes.ravel()[lookup[inside]]
        except Exception as exc:
            native_goes = np.full(fields[i].shape, np.nan)
            error = repr(exc)
        ax[5].imshow(native_goes[crop], extent=extent, vmin=200, vmax=300, cmap='turbo')
        for frame, color, version in [(old, 'cyan', 'v1'), (saved_new, 'magenta', 'v2')]:
            rows = frame[(frame.birth_time > t - pd.Timedelta(minutes=10)) & (frame.birth_time <= t)]
            for r in rows.itertuples():
                path = str(r.output_path) if version == 'v1' else renderer.soft_paths[str(r.object_id)]
                mask = renderer_review.sparse(path)[0]
                pick = lookup >= 0
                overlay = np.zeros(reference.shape, bool)
                overlay[pick] = mask.ravel()[lookup[pick]]
                if overlay[crop].any() and (not overlay[crop].all()):
                    gy, gx = np.mgrid[y0:y1, x0:x1]
                    ax[5].contour(gx, gy, overlay[crop].astype(float), [0.5], colors=[color], linewidths=0.8)
        ax[5].set_title('C13 mapped onto native MRMS grid\nCyan: saved V1; magenta: saved V2', fontsize=9)
        for a in ax:
            a.plot(reference_points[:, 1].mean(), reference_points[:, 0].mean(), '+', color='white', ms=8)
            a.set_xlim(x0 - 0.5, x1 - 0.5)
            a.set_ylim(y1 - 0.5, y0 - 0.5)
        fig.suptitle(f'{name} | MRMS UTC {t} | reference birth {reference_time}\nBands are 1 km thick. Contour = distance from complete >=35-dBZ core. Magenta = recent births, not ongoing storm presence.\nMRMS colors: 0-60 dBZ; C13: 200-300 K. Future frames are QC only.', fontsize=9)
        fig.canvas.draw()
        image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        plt.close(fig)
        images.append(image)
        if i == nearest:
            image.save(destination / 'birth_comparison.png')
        timing.append(dict(mrms_time=str(t), goes_source=str(source), goes_error=error))
    tmp = destination / 'comparison.tmp.gif'
    images[0].save(tmp, save_all=True, append_images=images[1:], duration=700, loop=0)
    tmp.replace(destination / 'comparison.gif')
    for im in images:
        im.close()
    pd.DataFrame(timing).to_csv(destination / 'frame_times.csv', index=False)

def run(args):
    import zarr
    from pyproj import Geod
    global renderer_review
    code = PROJECT / str(_ci_source() / 'scripts/isolated_ci_binary_v1/goes_ir_v1')
    data = PROJECT / 'data/ci_isolated_mrms_binary_v1/goes_ir_parallax_native_filled_v1'
    catalogue = data / 'labels/ci_catalogue_v2_full_scans_v1'
    core = load('geometry_core', PROJECT / str(_ci_source() / 'scripts/isolated_ci_binary_v1/core.py'))
    renderer_review = load('geometry_review', code / 'review_ci_v2_extremes.py')
    old = pd.read_parquet(_ci_resolve(catalogue / 'v1_objects_in_same_windows.parquet'))
    saved_new = pd.read_parquet(_ci_resolve(catalogue / 'ci_v2_object_catalogue.parquet'))
    soft = pd.read_parquet(_ci_resolve(catalogue / 'soft_target_build_manifest.parquet'))
    for f in [old, saved_new]:
        f['birth_time'] = pd.to_datetime(f.birth_time)
    paths = dict(zip(soft.object_id.astype(str), soft.output_path.astype(str)))
    renderer = renderer_review.Renderer(SimpleNamespace(project=PROJECT), old, saved_new, paths)
    builder, grid = (renderer.builder, renderer.grid)
    manifest = pd.read_parquet(_ci_resolve(data / 'state/ci_soft_target_object_manifest.parquet'))
    scope = pd.read_csv(_ci_resolve(catalogue / 'v1_temporal_scope.csv'))
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    config = dict(variants=VARIANTS, cases=CASES, band_width_km=1, patch_km=args.patch_km, history='unchanged: previous-scan overlap/motion and prior-30min footprint screen', persistence='unchanged: 10 +/- 2 min, 40 m/s + 1 km, max gap 5 min, one missing scan', scope='three visual-QC windows; no production labels modified; no downloads', caveat='Pixel-centre bands, not proof of a continuous echo-free geometric barrier. No full motion-compensated tracker.', hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(_ci_resolve(__file__)).resolve(), code / 'ci_label_pilot_v2.py', PROJECT / str(_ci_source() / 'scripts/isolated_ci_binary_v1/core.py'), code / 'review_ci_v2_extremes.py', catalogue / 'v1_temporal_scope.csv', catalogue / 'ci_v2_object_catalogue.parquet', catalogue / 'v1_objects_in_same_windows.parquet', catalogue / 'soft_target_build_manifest.parquet', data / 'state/ci_soft_target_object_manifest.parquet']})
    config = json.loads(json.dumps(config, default=str))
    if (out / 'config.json').exists() and json.loads((out / 'config.json').read_text()) != config:
        raise RuntimeError('Changed settings/source: choose another output directory')
    save_json(out / 'config.json', config)
    all_summary = []
    cfg = dict(persistence_minutes=10, persistence_tolerance_minutes=2, maximum_tracking_gap_minutes=5, tracking_padding_km=1, maximum_tracking_speed_m_s=40, maximum_missing_native_scans=1)
    for name, year, pid, when, oid in CASES:
        dest = out / name
        dest.mkdir(exist_ok=True)
        if (dest / 'completed.json').exists():
            all_summary.extend(json.loads((dest / 'completed.json').read_text())['summary'])
            print('SKIP', name, flush=True)
            continue
        print('START', name, pid, flush=True)
        birth = pd.Timestamp(when)
        windows = scope[scope.period_id == pid]
        if windows.empty:
            raise RuntimeError('Missing original V1 time scope: ' + pid)
        start, end = (birth - pd.Timedelta(minutes=30), birth + pd.Timedelta(minutes=40))
        period = data.parent / f'year_{year}_strict_v2_001/periods' / pid
        radar = zarr.open_group(_ci_resolve(str(period / 'reflectivity.zarr')), mode='r')
        all_times = pd.DatetimeIndex(pd.to_datetime(np.asarray(radar['time_ns'][:], np.int64)))
        if all_times.has_duplicates or not all_times.is_monotonic_increasing:
            raise RuntimeError('Unordered MRMS times')
        take = np.flatnonzero((all_times >= start - pd.Timedelta(minutes=45)) & (all_times <= end + pd.Timedelta(minutes=15)))
        times = all_times[take]
        fields = np.asarray(radar['reflectivity_dbz'].oindex[take, :, :], np.float32)
        lat, lon = (np.asarray(radar['latitude'][:]), np.asarray(radar['longitude'][:]))
        geod = Geod(ellps='WGS84')
        spacing = (abs(geod.inv(lon.mean(), lat[0], lon.mean(), lat[1])[2]) / 1000, abs(geod.inv(lon[0], lat.mean(), lon[1], lat.mean())[2]) / 1000)
        valid = np.isfinite(fields) & (fields >= -50)
        labs, objs, points, states = ([], [], [], [])
        for i, f in enumerate(fields):
            lab, obj = core.segment(f, valid[i], 35.0, np.nextafter(3 * np.prod(spacing), 0.0), spacing)
            labs.append(lab)
            objs.append(obj)
            points.append({k: np.column_stack([v.ys, v.xs]) for k, v in obj.items()})
            gap = (times[i] - times[i - 1]).total_seconds() / 60 if i else np.inf
            state = lineage(lab, labs[i - 1] if i else None, spacing, gap)
            prior = np.flatnonzero((times >= times[i] - pd.Timedelta(minutes=30)) & (times < times[i]))
            for k, status in list(state.items()):
                if status == 'new_candidate':
                    p = points[-1][k]
                    if any((np.any(labs[j][p[:, 0], p[:, 1]] > 0) for j in prior)):
                        state[k] = 'ambiguous_prior_echo_at_birth_location'
            states.append(state)
        reference = np.zeros(fields.shape[1:], bool)
        if oid:
            rows = manifest[manifest.object_id.astype(str) == oid]
            if len(rows) != 1:
                raise RuntimeError('Cannot locate exact V1 native birth mask: ' + oid)
            reference = builder.decode_mask(rows.iloc[0].birth_mask_path, fields.shape[1:])
        else:
            j = int(np.abs(times.asi8 - birth.value).argmin())
            if abs((times[j] - birth).total_seconds()) > 1:
                raise RuntimeError('Missing exact near-storm reference scan')
            for k in [17, 19]:
                if k not in points[j] or len(points[j][k]) != 3:
                    raise RuntimeError('Reference labels changed; inspect the native scan')
                p = points[j][k]
                if not (95 <= p[:, 0].mean() <= 110 and 170 <= p[:, 1].mean() <= 185):
                    raise RuntimeError('Reference position changed')
                reference[p[:, 0], p[:, 1]] = True
        ref_labels, ref_count = label(reference, np.ones((3, 3)))
        py, px = np.where(reference)
        cy, cx = (py.mean(), px.mean())
        y0, y1 = (max(0, int(cy - args.patch_km / 2 / spacing[0])), min(reference.shape[0], int(cy + args.patch_km / 2 / spacing[0]) + 1))
        x0, x1 = (max(0, int(cx - args.patch_km / 2 / spacing[1])), min(reference.shape[1], int(cx + args.patch_km / 2 / spacing[1]) + 1))
        mapping = builder.build_mapping(period, grid, {})
        lookup = np.full(reference.shape, -1, np.int32)
        lookup[mapping['source_y'], mapping['source_x']] = mapping['target_y'] * grid['shape'][1] + mapping['target_x']
        events = {v: [] for v in VARIANTS}
        audit = []
        persistence_cache = {}
        for i, t in enumerate(times):
            if not start - pd.Timedelta(minutes=10) <= t <= end:
                continue
            if not any((pd.Timestamp(r.window_start) <= t <= pd.Timestamp(r.window_end) for r in windows.itertuples())):
                continue
            fresh = {k for k, s in states[i].items() if s == 'new_candidate'}
            context = np.flatnonzero((times >= t - pd.Timedelta(minutes=30)) & (times <= t + pd.Timedelta(minutes=12)))
            complete = core.temporal_window_complete(times, t - pd.Timedelta(minutes=30), t + pd.Timedelta(minutes=12), 5, 2.5)
            all_valid = np.all(valid[context], axis=0)
            for variant, (kind, threshold, parameters) in VARIANTS.items():
                used = set()
                for k, status in states[i].items():
                    p = points[i][k]
                    if not (y0 <= p[:, 0].mean() < y1 and x0 <= p[:, 1].mean() < x1):
                        continue
                    if status == 'continuing_or_split' or k in used:
                        continue
                    base = dict(case=name, variant=variant, birth_time=str(t), component_id=int(k), seed_pixels=len(p), native_y=float(p[:, 0].mean()), native_x=float(p[:, 1].mean()), lineage=status, kind=kind, band_echo_threshold_dbz=threshold)
                    if not complete or status != 'new_candidate':
                        audit.append(dict(**base, reason='insufficient_temporal_coverage' if not complete else status, accepted=False))
                        continue
                    for parameter in parameters:
                        group, reason, metrics = geometry(fields[i], valid[i], labs[i], points[i], k, fresh, kind, threshold, parameter, spacing)
                        accepted = False
                        if group:
                            if not all_valid[group['outer']].all() and any((valid[j][group['outer']].mean() < 0.99 for j in context)):
                                reason = 'invalid_context_disk'
                            else:
                                for m in group['members']:
                                    key = (i, m)
                                    if key not in persistence_cache:
                                        persistence_cache[key] = core.match_persistence(objs[i][m], i, times, labs, objs, spacing, cfg) is not None
                                if not all((persistence_cache[i, m] for m in group['members'])):
                                    reason = 'persistence_failed'
                                elif used.intersection(group['members']):
                                    reason = 'overlapping_group_requires_review'
                                else:
                                    q = np.concatenate([points[i][m] for m in group['members']])
                                    if not np.any(lookup[q[:, 0], q[:, 1]] >= 0):
                                        reason = 'outside_goes_grid'
                                    else:
                                        accepted, reason = (True, 'accepted_requires_review')
                        row = dict(**base, parameter_km=parameter, accepted=accepted, reason=reason, **metrics)
                        if accepted:
                            used.update(group['members'])
                            q = np.concatenate([points[i][m] for m in group['members']])
                            row.update(group_components=json.dumps(group['members']), group_pixels=len(q))
                            events[variant].append(dict(time=t, points=q, band=group['band'], record=row))
                        audit.append(row)
                        if accepted or reason == 'overlapping_group_requires_review':
                            break
        table = pd.DataFrame(audit)
        table.to_csv(dest / 'geometry_decisions.csv', index=False)
        summary = []
        for variant in VARIANTS:
            nearby = [e for e in events[variant] if abs((e['time'] - birth).total_seconds()) <= 600]
            covered = set()
            for e in nearby:
                q = e['points']
                covered.update(set(map(int, np.unique(ref_labels[q[:, 0], q[:, 1]]))) - {0})
            summary.append(dict(case=name, variant=variant, accepted_groups_in_patch=len(events[variant]), reference_components=int(ref_count), reference_components_touched_within_10min=len(covered), note='Overlap screen on three positive review cases; not dataset recall or precision.'))
            pd.DataFrame([e['record'] for e in events[variant]]).to_csv(dest / (variant + '_accepted.csv'), index=False)
        pd.DataFrame(summary).to_csv(dest / 'summary.csv', index=False)
        display = np.flatnonzero((times >= start) & (times <= end))
        print('RENDER', name, 'frames=', len(display), flush=True)
        render(dest, name, times[display], fields[display], spacing, (y0, y1, x0, x1), events, reference, lookup, renderer, old[old.period_id == pid], saved_new[saved_new.period_id == pid])
        save_json(dest / 'completed.json', dict(summary=summary, frames=len(display), mrms_spacing_km=spacing, reference_birth=str(birth), reference_pixels=int(reference.sum()), reference_components=int(ref_count)))
        all_summary.extend(summary)
        print(pd.DataFrame(summary).to_string(index=False), flush=True)
        print('DONE', name, flush=True)
        del fields, valid, labs, objs, points, states, events
    pd.DataFrame(all_summary).to_csv(out / 'summary.csv', index=False)
    save_json(out / 'completed.json', dict(status='three_geometry_tests_completed_requires_review', cases=3))
    with zipfile.ZipFile(out / 'geometry_test_bundle.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.suffix in {'.csv', '.json', '.png', '.gif'}:
                z.write(p, str(p.relative_to(out)))
    print('ALL THREE GEOMETRY TESTS COMPLETED:', out, flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=PROJECT / 'experiments/ci_v2_geometry_three_cases_v1')
    parser.add_argument('--patch-km', type=float, default=80)
    parser.add_argument('--selftest', action='store_true')
    args = parser.parse_args()
    if args.selftest:
        selftest()
    else:
        run(args)
if __name__ == '__main__':
    main()
