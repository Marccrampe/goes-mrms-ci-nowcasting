"""One-period label pilot. Production data are read-only.
Candidates require visual review, especially motion, splits and merges.
"""
from ci_nowcasting.paths import resolve_path as _ci_resolve, project_root as _ci_project, source_root as _ci_source
import argparse
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, label

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj

def json_save(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str))

def lineage(current, previous, spacing, gap, speed=40.0, padding=1.0):
    """Conservative continuity screen, not a full storm tracker."""
    result = {}
    if previous is None or gap > 5:
        return {int(k): 'unknown_previous_scan' for k in np.unique(current) if k}
    gate = padding + speed * gap * 60 / 1000
    distance = distance_transform_edt(previous == 0, sampling=spacing) if np.any(previous) else np.full(previous.shape, np.inf)
    for k in np.unique(current):
        if not k:
            continue
        take = current == k
        if np.any(previous[take] > 0):
            status = 'continuing_or_split'
        elif distance[take].min() <= gate:
            status = 'ambiguous_motion_or_nearby_birth'
        else:
            status = 'new_candidate'
        result[int(k)] = status
    return result

def test_ring(frame, valid, labels, points, seed, fresh, radius, thickness, spacing, minimum_valid):
    """Fixed seed centroid; test the complete enclosed >=35-dBZ group."""
    cy, cx = points[seed].mean(axis=0)
    yy, xx = np.ogrid[:frame.shape[0], :frame.shape[1]]
    distance = np.hypot((yy - cy) * spacing[0], (xx - cx) * spacing[1])
    inner = radius - thickness
    inside = distance < inner
    annulus = (distance >= inner) & (distance <= radius)
    edge = min((cy + 0.5) * spacing[0], (frame.shape[0] - 0.5 - cy) * spacing[0], (cx + 0.5) * spacing[1], (frame.shape[1] - 0.5 - cx) * spacing[1])
    if edge < radius or not annulus.any():
        return (None, 'boundary')
    if not valid[annulus].all():
        return (None, 'unknown_annulus_pixels')
    if valid[distance <= radius].mean() < minimum_valid:
        return (None, 'invalid_disk')
    if np.any(frame[annulus] >= 10):
        return (None, 'echo_ge10_in_annulus')
    members = set(map(int, np.unique(labels[inside]))) - {0}
    if seed not in members:
        return (None, 'seed_not_enclosed')
    for k in members:
        p = points[k]
        if np.any(distance[p[:, 0], p[:, 1]] >= inner):
            return (None, 'component_crosses_annulus')
        if k not in fresh:
            return (None, 'enclosed_existing_or_ambiguous_core')
    return ({'members': sorted(members), 'cy': float(cy), 'cx': float(cx), 'radius_km': float(radius)}, 'accepted_geometry')

def projected_core(points, lookup, shape):
    out = np.zeros(shape, bool)
    q = lookup[points[:, 0], points[:, 1]]
    q = q[q >= 0]
    out.ravel()[q] = True
    return out

def soft_from_core(core, spacing):
    if not core.any():
        return np.zeros(core.shape, np.float32)
    d = distance_transform_edt(~core, sampling=spacing)
    result = np.exp(-0.5 * (d / 3.0) ** 2).astype(np.float32)
    result[d > 8.0] = 0
    result[core] = 1
    return result

def selftest():
    shape = (51, 51)
    frame = np.zeros(shape, np.float32)
    frame[24:27, 25] = 40
    labs, _ = label(frame >= 35, np.ones((3, 3)))
    pts = {1: np.argwhere(labs == 1)}
    valid = np.ones(shape, bool)
    item, _ = test_ring(frame, valid, labs, pts, 1, {1}, 5, 1, (1, 1), 0.99)
    assert item and item['members'] == [1]
    frame[25, 30] = 15
    assert test_ring(frame, valid, labs, pts, 1, {1}, 5, 1, (1, 1), 0.99)[0] is None
    frame[25, 30] = 0
    valid[25, 30] = False
    assert test_ring(frame, valid, labs, pts, 1, {1}, 5, 1, (1, 1), 0.99)[0] is None
    valid[:] = True
    frame[24:27, 27] = 40
    labs, _ = label(frame >= 35, np.ones((3, 3)))
    pts = {k: np.argwhere(labs == k) for k in (1, 2)}
    assert test_ring(frame, valid, labs, pts, 1, {1, 2}, 5, 1, (1, 1), 0.99)[0]['members'] == [1, 2]
    assert test_ring(frame, valid, labs, pts, 1, {1}, 5, 1, (1, 1), 0.99)[0] is None
    assert lineage(labs, labs, (1, 1), 2)[1] == 'continuing_or_split'
    assert lineage(labs, np.zeros(shape, int), (1, 1), 2)[1] == 'new_candidate'
    assert not soft_from_core(np.zeros((10, 10), bool), (2, 2)).any()
    print('LABEL PILOT SELFTEST PASSED', flush=True)

def run(args):
    import zarr
    from pyproj import Geod
    project = args.project
    code = project / str(_ci_source() / 'scripts/isolated_ci_binary_v1/goes_ir_v1')
    data = project / 'data/ci_isolated_mrms_binary_v1/goes_ir_parallax_native_filled_v1'
    builder = module('pilot_builder', code / 'build_ci_soft_targets_native_goes.py')
    dm = module('pilot_dataset', code / 'ci_goes_patch_dataset.py')
    core = module('pilot_core', project / str(_ci_source() / 'scripts/isolated_ci_binary_v1/core.py'))
    lead = pd.read_parquet(_ci_resolve(data / 'state/ci_goes_training_lead_index.parquet'))
    rows = lead[lead.period_id.astype(str) == args.period_id]
    if rows.empty:
        raise RuntimeError('Period missing from lead index: ' + args.period_id)
    year = int(rows.year.iloc[0])
    period = data.parent / f'year_{year}_strict_v2_001/periods' / args.period_id
    radar = zarr.open_group(_ci_resolve(str(period / 'reflectivity.zarr')), mode='r')
    times_all = pd.DatetimeIndex(pd.to_datetime(np.asarray(radar['time_ns'][:], np.int64)))
    if times_all.has_duplicates or not times_all.is_monotonic_increasing:
        raise RuntimeError('MRMS times must be ordered and unique')
    print('PERIOD:', args.period_id, flush=True)
    print('MRMS AVAILABLE UTC:', times_all.min(), 'to', times_all.max(), flush=True)
    centre = pd.Timestamp(args.center)
    if centre.tzinfo is not None:
        centre = centre.tz_convert('UTC').tz_localize(None)
    start = centre - pd.Timedelta(minutes=args.before)
    end = centre + pd.Timedelta(minutes=args.after)
    if start < times_all.min() or end > times_all.max():
        raise RuntimeError(f'Chosen display window {start} to {end} is outside MRMS bounds above')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / 'completed.json').exists():
        raise RuntimeError('This pilot already completed; choose another output directory')
    lat = np.asarray(radar['latitude'][:], float)
    lon = np.asarray(radar['longitude'][:], float)
    geod = Geod(ellps='WGS84')
    dy = abs(geod.inv(lon.mean(), lat[0], lon.mean(), lat[1])[2]) / 1000
    dx = abs(geod.inv(lon[0], lat.mean(), lon[1], lat.mean())[2]) / 1000
    spacing = (dy, dx)
    settings = {**vars(args), 'display_start_utc': start, 'display_end_utc': end, 'mrms_spacing_km': spacing, 'minimum_pixels': 3, 'threshold_dbz': 35, 'ring_echo_threshold_dbz': 10, 'ring_thickness_km': 1, 'tracking': 'conservative adjacent-scan motion gate, 40 m/s + 1 km; ambiguous links excluded', 'births': 'first >=35-dBZ component meeting >=3 native MRMS pixels', 'historical_context_minutes': 30, 'historical_reappearance': 'prior >=35 echo at birth footprint marks candidate ambiguous; not a 30-km exclusion', 'persistence': 'existing core.match_persistence, 10 +/- 2 minutes; merges possible', 'projection': 'same builder mapping; no new parallax correction', 'geometry': 'local metric using measured MRMS spacing, distance from fixed seed centroid', 'soft_kernel': 'unchanged: sigma 3 km, cutoff 8 km', 'target_display': 'births in preceding 10 minutes, equal rule for old and pilot', 'status': 'PILOT_REQUIRES_VISUAL_REVIEW; no production labels modified'}
    json_save(args.output_dir / 'settings.json', settings)
    take = np.flatnonzero((times_all >= start - pd.Timedelta(minutes=45)) & (times_all <= end + pd.Timedelta(minutes=15)))
    times = times_all[take]
    fields = np.asarray(radar['reflectivity_dbz'].oindex[take, :, :], np.float32)
    valid = np.isfinite(fields) & (fields >= -50)
    if fields.shape[1:] != (len(lat), len(lon)):
        raise RuntimeError('Unexpected native MRMS grid orientation')
    labels, objects, points, states = ([], [], [], [])
    min_area = np.nextafter(3 * dy * dx, 0.0)
    for i, f in enumerate(fields):
        lab, obj = core.segment(f, valid[i], 35.0, min_area, spacing)
        labels.append(lab)
        objects.append(obj)
        points.append({k: np.argwhere(lab == k) for k in obj})
        if any((len(p) < 3 for p in points[-1].values())):
            raise RuntimeError('Native minimum-pixel check failed')
        gap = (times[i] - times[i - 1]).total_seconds() / 60 if i else np.inf
        status = lineage(lab, labels[i - 1] if i else None, spacing, gap)
        prior = np.flatnonzero((times >= times[i] - pd.Timedelta(minutes=30)) & (times < times[i]))
        for k, state in list(status.items()):
            if state == 'new_candidate':
                p = points[-1][k]
                if any((np.any(labels[j][p[:, 0], p[:, 1]] > 0) for j in prior)):
                    status[k] = 'ambiguous_prior_echo_at_birth_location'
        states.append(status)
    grid = builder.load_and_validate_goes_grid()
    mapping = builder.build_mapping(period, grid, {})
    lookup = np.full(fields.shape[1:], -1, np.int32)
    lookup[mapping['source_y'], mapping['source_x']] = mapping['target_y'] * grid['shape'][1] + mapping['target_x']
    cfg = {'persistence_minutes': 10, 'persistence_tolerance_minutes': 2, 'maximum_tracking_gap_minutes': 5, 'tracking_padding_km': 1, 'maximum_tracking_speed_m_s': 40, 'maximum_missing_native_scans': 1}
    records, events, ambiguous = ([], [], [])
    for i, t in enumerate(times):
        if not start - pd.Timedelta(minutes=10) <= t <= end:
            continue
        fresh = {k for k, s in states[i].items() if s == 'new_candidate'}
        used = set()
        for k, state in states[i].items():
            if state == 'continuing_or_split' or k in used:
                continue
            p = points[i][k]
            row = {'birth_time': str(t), 'component_id': int(k), 'pixels': len(p), 'area_km2': len(p) * dy * dx, 'lineage': state, 'native_y': float(p[:, 0].mean()), 'native_x': float(p[:, 1].mean())}
            context = np.flatnonzero((times >= t - pd.Timedelta(minutes=30)) & (times <= t + pd.Timedelta(minutes=12)))
            complete = core.temporal_window_complete(times, t - pd.Timedelta(minutes=30), t + pd.Timedelta(minutes=12), 5, 2.5)
            if not complete or state == 'unknown_previous_scan':
                row['status'] = 'insufficient_temporal_coverage'
            elif state != 'new_candidate':
                row['status'] = state
                ambiguous.append({'time': t, 'core': projected_core(p, lookup, grid['shape'])})
            else:
                accepted = None
                for radius in args.radii:
                    group, reason = test_ring(fields[i], valid[i], labels[i], points[i], k, fresh, radius, 1.0, spacing, 0.99)
                    row[f'radius_{radius:g}_reason'] = reason
                    if group:
                        cy, cx = (group['cy'], group['cx'])
                        yy, xx = np.ogrid[:fields.shape[1], :fields.shape[2]]
                        disk = np.hypot((yy - cy) * dy, (xx - cx) * dx) <= radius
                        if any((valid[j][disk].mean() < 0.99 for j in context)):
                            row[f'radius_{radius:g}_reason'] = 'invalid_context_disk'
                            continue
                        confirmed = [core.match_persistence(objects[i][m], i, times, labels, objects, spacing, cfg) for m in group['members']]
                        if any((v is None for v in confirmed)):
                            row[f'radius_{radius:g}_reason'] = 'persistence_failed'
                            continue
                        accepted = group
                        break
                if accepted:
                    members = accepted['members']
                    all_pts = np.concatenate([points[i][m] for m in members])
                    gc = projected_core(all_pts, lookup, grid['shape'])
                    if not gc.any():
                        row['status'] = 'outside_goes_grid'
                    elif used.intersection(members):
                        row['status'] = 'overlapping_group_requires_review'
                    else:
                        used.update(members)
                        row.update(status='pilot_accepted_requires_review', selected_radius_km=accepted['radius_km'], group_components=json.dumps(members), group_pixels=len(all_pts))
                        event_id = f'PILOT_{i:04d}_{k:04d}'
                        row['event_id'] = event_id
                        yy, xx = np.ogrid[:fields.shape[1], :fields.shape[2]]
                        d = np.hypot((yy - accepted['cy']) * dy, (xx - accepted['cx']) * dx)
                        ring_points = np.argwhere((d >= accepted['radius_km'] - 1.0) & (d <= accepted['radius_km']))
                        events.append({'time': t, 'core': gc, 'soft': soft_from_core(gc, (grid['dy_km'], grid['dx_km'])), 'ring': projected_core(ring_points, lookup, grid['shape']), 'event_id': event_id, 'radius': accepted['radius_km']})
                else:
                    row['status'] = 'geometry_quality_or_persistence_rejected'
            records.append(row)
    audit = pd.DataFrame(records)
    audit['human_review'] = ''
    audit['review_notes'] = ''
    audit.to_csv(args.output_dir / 'candidate_audit.csv', index=False)
    if 'status' in audit:
        audit[audit.status == 'pilot_accepted_requires_review'].to_csv(args.output_dir / 'accepted_pilot_groups.csv', index=False)
    manifest = pd.read_parquet(_ci_resolve(data / 'labels/ci_soft_targets_v1/state/soft_target_build_manifest.parquet'))
    old = []
    for r in manifest[manifest.period_id.astype(str) == args.period_id].itertuples():
        t = pd.Timestamp(r.birth_time)
        if start - pd.Timedelta(minutes=10) <= t <= end:
            with np.load(_ci_resolve(r.output_path), allow_pickle=False) as ar:
                soft = np.zeros(grid['shape'], np.float32)
                soft[ar['soft_y'].astype(int), ar['soft_x'].astype(int)] = ar['soft_value']
                c = np.zeros(grid['shape'], bool)
                c[ar['core_y'].astype(int), ar['core_x'].astype(int)] = True
            old.append({'time': t, 'soft': soft, 'core': c, 'event_id': r.object_id})
    pd.DataFrame([{'object_id': e['event_id'], 'birth_time': str(e['time'])} for e in old], columns=['object_id', 'birth_time']).to_csv(args.output_dir / 'old_births.csv', index=False)
    ds = dm.CIGoesPatchDataset.__new__(dm.CIGoesPatchDataset)
    ds.goes_root, ds.month_cache = (data, {})
    frames, timing = ([], [])
    for requested in pd.date_range(start, end, freq=f'{args.frame_step_minutes}min'):
        mi = int(np.abs(times.asi8 - requested.value).argmin())
        rt = times[mi]
        if abs((rt - requested).total_seconds()) > 150:
            raise RuntimeError('No MRMS scan within 2.5 min of display time ' + str(requested))
        item = {'requested': str(requested), 'radar_time': str(rt), 'error': ''}
        c13 = np.full(grid['shape'], np.nan, np.float32)
        try:
            month = ds._month_data(requested)
            nominal = pd.to_datetime(sorted(month['time_lookup']))
            gi_time = nominal[int(np.abs(nominal.asi8 - rt.value).argmin())]
            if abs((gi_time - rt).total_seconds()) > 360:
                raise RuntimeError('No GOES nominal frame within 6 minutes')
            gi = month['time_lookup'][gi_time.value]
            group = month['group']
            source = gi_time
            if 'source_scan_end' in group:
                ar = group['source_scan_end']
                source = dm.decode_time_array(np.asarray(ar[gi:gi + 1, 5]), ar.attrs.asdict())[0]
            if pd.isna(source) or abs((source - rt).total_seconds()) > 360:
                raise RuntimeError('GOES source timestamp missing or more than 6 min from MRMS')
            c13 = np.asarray(group['CMI'][gi, 5], np.float32)
            item.update(goes_nominal=str(gi_time), goes_source=str(source))
        except Exception as exc:
            item['error'] = str(exc)
        radar_goes = np.full(grid['shape'], -np.inf, np.float32)
        vals = fields[mi][mapping['source_y'], mapping['source_x']]
        good = valid[mi][mapping['source_y'], mapping['source_x']]
        np.maximum.at(radar_goes, (mapping['target_y'][good], mapping['target_x'][good]), vals[good])
        radar_goes[~np.isfinite(radar_goes)] = np.nan

        def active(event_list):
            return [e for e in event_list if rt - pd.Timedelta(minutes=10) < e['time'] <= rt]
        oe, ne, ae = (active(old), active(events), active(ambiguous))

        def maximum(evs, name):
            if evs:
                return np.maximum.reduce([e[name] for e in evs])
            return np.zeros(grid['shape'], np.float32)
        frames.append({'time': rt, 'c13': c13, 'radar': radar_goes, 'old': maximum(oe, 'soft'), 'new': maximum(ne, 'soft'), 'old_core': maximum(oe, 'core').astype(bool), 'new_core': maximum(ne, 'core').astype(bool), 'new_ring': maximum(ne, 'ring').astype(bool), 'ambiguous_core': maximum(ae, 'core').astype(bool), 'timing': item, 'new_ids': [e['event_id'] for e in ne]})
        timing.append(item)
    json_save(args.output_dir / 'frame_times.json', timing)
    render(args, frames, grid)
    counts = pd.Series([r['status'] for r in records]).value_counts().to_dict()
    summary = {'status': 'completed_requires_visual_review', 'period_id': args.period_id, 'old_births_in_display_interval': sum((start <= e['time'] <= end for e in old)), 'pilot_groups_in_display_interval': sum((start <= e['time'] <= end for e in events)), 'candidate_status_counts': counts, 'missing_goes_frames': sum((bool(t['error']) for t in timing)), 'five_patches_overlap': True, 'five_gifs': 5}
    json_save(args.output_dir / 'completed.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    print('Saved:', args.output_dir, flush=True)

def render(args, frames, grid):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    size = 128
    h, w = grid['shape']
    if min(h, w) < size:
        raise RuntimeError('Grid smaller than 128 pixels')
    origins = [(0, 0), (0, w - size), (h - size, 0), (h - size, w - size), ((h - size) // 2, (w - size) // 2)]
    json_save(args.output_dir / 'patches.json', [{'patch': i + 1, 'origin_yx': p, 'size': size} for i, p in enumerate(origins)])
    for patch, (y0, x0) in enumerate(origins, 1):
        images = []
        sl = np.s_[y0:y0 + size, x0:x0 + size]
        for f in frames:
            fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout='constrained')
            ax = axes.ravel()
            ax[0].imshow(f['c13'][sl], vmin=200, vmax=300, cmap='turbo', origin='upper')
            ax[0].set_title('GOES C13 (K)\nsource ' + f['timing'].get('goes_source', 'MISSING'))
            im = ax[1].imshow(f['radar'][sl], vmin=0, vmax=60, cmap='turbo', origin='upper')
            ax[1].set_title('MRMS max projected to GOES (dBZ)\n' + str(f['time']))
            fig.colorbar(im, ax=ax[1], fraction=0.035)
            for j, name, title in [(2, 'old', 'OLD accepted births'), (3, 'new', 'V2 PILOT births: review required')]:
                im = ax[j].imshow(f[name][sl], vmin=0, vmax=1, cmap='magma', origin='upper')
                c = f[name + '_core'][sl]
                if c.any():
                    ax[j].contour(c.astype(float), levels=[0.5], colors=['cyan'], linewidths=0.8)
                ax[j].set_title(title)
                fig.colorbar(im, ax=ax[j], fraction=0.035)
            for j in (0, 1):
                for name, color in [('old_core', 'cyan'), ('new_core', 'magenta')]:
                    c = f[name][sl]
                    if c.any():
                        ax[j].contour(c.astype(float), [0.5], colors=[color], linewidths=0.8)
            ring = f['new_ring'][sl]
            if ring.any():
                ax[1].contour(ring.astype(float), [0.5], colors=['white'], linewidths=0.6)
            c = f['ambiguous_core'][sl]
            if c.any():
                yy, xx = np.where(c)
                ax[3].scatter(xx, yy, c='orange', s=9, marker='x')
            fig.suptitle(f"Patch {patch}/5 | UTC {f['time']} | origin ({y0}, {x0})\nTargets = births in previous 10 min; not ongoing echo. Orange = ambiguous lineage.", fontsize=10)
            for a in ax:
                a.set_xlim(-0.5, size - 0.5)
                a.set_ylim(size - 0.5, -0.5)
                a.set_xlabel('GOES x pixel')
                a.set_ylabel('GOES y pixel')
            fig.canvas.draw()
            image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
            images.append(image)
            plt.close(fig)
        path = args.output_dir / f'patch_{patch:02d}.gif'
        images[0].save(path, save_all=True, append_images=images[1:], duration=650, loop=0)
        for im in images:
            im.close()
        print('GIF', patch, '/ 5:', path, flush=True)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, default=Path(_ci_resolve(_ci_resolve('/project'))))
    p.add_argument('--period-id')
    p.add_argument('--center', help='UTC centre of display window')
    p.add_argument('--before', type=int, default=40)
    p.add_argument('--after', type=int, default=40)
    p.add_argument('--radii', nargs='+', type=float, default=[5, 8, 10, 15])
    p.add_argument('--frame-step-minutes', type=int, default=2)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--selftest', action='store_true')
    args = p.parse_args()
    if args.selftest:
        selftest()
        return
    if not args.period_id or not args.center or (not args.output_dir):
        p.error('--period-id, --center and --output-dir are required')
    args.radii = sorted(set(args.radii))
    if args.before < 0 or args.after < 0 or args.frame_step_minutes <= 0 or (not args.radii) or (min(args.radii) <= 1):
        p.error('Invalid time window, frame step or radii')
    run(args)
if __name__ == '__main__':
    main()
