"""Native-MRMS visual pilot: V3 birth cores and seeded >=30/25 dBZ envelopes.

Reads V3 detections; default case selection uses TRAIN only. No production
labels/indexes are changed. Fixed birth-time targets remain fixed in animations.
Bounds and prior-echo flags are review hypotheses, not new CI acceptance rules.
"""
from ci_nowcasting.paths import resolve_path as _ci_resolve, project_root as _ci_project, source_root as _ci_source
import argparse
import html
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import label, distance_transform_edt
PROJECT = Path(_ci_resolve(_ci_resolve('/project')))
CODE = PROJECT / str(_ci_source() / 'scripts/isolated_ci_binary_v1/goes_ir_v1')
DATA = PROJECT / 'data/ci_isolated_mrms_binary_v1/goes_ir_parallax_native_filled_v1'
CAT = DATA / 'labels/ci_catalogue_v3_contour20_full_scans_v1'
DS = DATA / 'datasets/ci_v3_birth_2x30min_v1'
CONNECTIVITY = np.ones((3, 3), np.uint8)

def native_mask(path, shape, expected_pixels=None):
    """Decode known sparse/dense mask forms; reject ambiguous/invalid schemas."""
    with np.load(_ci_resolve(path), allow_pickle=False) as z:
        keys = set(z.files)
        for key in ['native_shape', 'shape']:
            if key in keys and tuple(np.asarray(z[key]).ravel()) != tuple(shape):
                raise RuntimeError(f'Native mask shape mismatch: {path}, {z[key]}, {shape}')
        result = None
        for ykey, xkey in [('native_y', 'native_x'), ('y', 'x'), ('core_y', 'core_x'), ('mask_y', 'mask_x'), ('birth_y', 'birth_x'), ('band_y', 'band_x'), ('group_y', 'group_x'), ('ys', 'xs'), ('y_indices', 'x_indices'), ('y_idx', 'x_idx'), ('rows', 'cols'), ('row_indices', 'col_indices'), ('y_global', 'x_global'), ('native_mask_y', 'native_mask_x')]:
            if ykey not in keys or xkey not in keys:
                continue
            ys, xs = (np.asarray(z[ykey]), np.asarray(z[xkey]))
            if ys.ndim != 1 or xs.ndim != 1 or len(ys) != len(xs):
                continue
            if not np.issubdtype(ys.dtype, np.integer) or not np.issubdtype(xs.dtype, np.integer):
                continue
            if np.any((ys < 0) | (ys >= shape[0])) or np.any((xs < 0) | (xs >= shape[1])):
                raise RuntimeError('Native coordinate outside grid: ' + str(path))
            result = np.zeros(shape, bool)
            result[ys, xs] = True
            break
        if result is None:
            for key in ['mask', 'birth_mask', 'native_mask', 'band_mask']:
                if key in keys and z[key].shape == tuple(shape):
                    result = np.asarray(z[key], bool)
                    break
        if result is None:
            for key in ['mask_bits', 'birth_mask_bits', 'band_bits', 'bits']:
                if key in keys:
                    result = np.unpackbits(np.asarray(z[key], np.uint8).ravel(), count=int(np.prod(shape))).reshape(shape).astype(bool)
                    break
        if result is None:
            for key in ['flat_indices', 'flat_index']:
                if key in keys:
                    indices = np.asarray(z[key], np.int64).ravel()
                    if np.any((indices < 0) | (indices >= np.prod(shape))):
                        raise RuntimeError('Invalid native flat indices')
                    result = np.zeros(int(np.prod(shape)), bool)
                    result[indices] = True
                    result = result.reshape(shape)
                    break
        if result is None:
            raise RuntimeError(f'Unsupported native mask schema: {path}; keys={sorted(keys)}')
    if expected_pixels is not None and int(result.sum()) != int(expected_pixels):
        raise RuntimeError(f'Native birth count mismatch: {path}: {result.sum()} != {expected_pixels}')
    return result

def envelopes(field, seed, dy, dx, offset, band, prior, thresholds=(35, 30, 25)):
    valid = np.isfinite(field) & (field >= -50)
    if not seed.any() or not valid[seed].all() or (not (field[seed] >= 35).all()):
        raise RuntimeError('Stored birth seed is not valid >=35 dBZ at exact birth scan')
    distance = distance_transform_edt(~seed, sampling=(dy, dx))
    inside = (distance < offset + 1e-08) & ~band
    result, reports = ({}, [])
    for threshold in thresholds:
        if threshold == 35:
            raw = seed.copy()
            candidate = raw.copy()
        else:
            lab, _ = label(valid & (field >= threshold), CONNECTIVITY)
            ids = np.unique(lab[seed])
            ids = ids[ids > 0]
            raw = np.isin(lab, ids)
            bounded, _ = label(raw & inside, CONNECTIVITY)
            selected = np.unique(bounded[seed])
            selected = selected[selected > 0]
            candidate = np.isin(bounded, selected)
        if not candidate[seed].all():
            raise RuntimeError('Envelope lost birth seed')
        result[threshold] = candidate
        reports.append(dict(threshold_dbz=threshold, birth_core_pixels=int(seed.sum()), raw_connected_pixels=int(raw.sum()), candidate_pixels=int(candidate.sum()), candidate_area_km2=float(candidate.sum() * dy * dx), raw_pixels_outside_isolation_interior=int((raw & ~inside).sum()), raw_pixels_in_stored_isolation_band=int((raw & band).sum()), prior_ge35_overlap_pixels=int((candidate & prior).sum()), extension_prior_ge35_overlap_pixels=int((candidate & ~seed & prior).sum()), requires_geometry_review=bool((raw & ~inside).any() or (raw & band).any()), requires_prior_echo_review=bool((candidate & prior).any())))
    ordered = sorted(result, reverse=True)
    assert all((np.all(result[a] <= result[b]) for a, b in zip(ordered, ordered[1:])))
    return (result, reports)

class Radar:

    def __init__(self, path):
        import zarr
        self.path = Path(_ci_resolve(path))
        self.group = zarr.open_group(_ci_resolve(str(path)), mode='r')
        self.times = np.asarray(self.group['time_ns'][:], np.int64)
        if len(self.times) == 0 or np.any(np.diff(self.times) <= 0):
            raise RuntimeError('Radar timestamps must be unique and increasing')
        self.shape = tuple(self.group['reflectivity_dbz'].shape[-2:])

    def exact(self, time):
        t = int(pd.Timestamp(time).value)
        i = int(np.searchsorted(self.times, t))
        if i == len(self.times) or int(self.times[i]) != t:
            raise RuntimeError('Exact birth scan absent: ' + str(time))
        return np.asarray(self.group['reflectivity_dbz'][i], np.float32)

    def past(self, time, region):
        t = int(pd.Timestamp(time).value)
        i = int(np.searchsorted(self.times, t, side='right')) - 1
        if i < 0 or t - int(self.times[i]) > 150000000000:
            return (None, None)
        return (np.asarray(self.group['reflectivity_dbz'][i, region[0], region[1]], np.float32), str(pd.Timestamp(int(self.times[i]), unit='ns')))

    def prior_screen(self, birth, region):
        end = int(pd.Timestamp(birth).value)
        start = end - 30 * 60000000000
        indices = np.flatnonzero((self.times >= start) & (self.times < end))
        shape = (region[0].stop - region[0].start, region[1].stop - region[1].start)
        prior = np.zeros(shape, bool)
        observed = np.zeros(shape, np.int32)
        for i in indices:
            field = np.asarray(self.group['reflectivity_dbz'][int(i), region[0], region[1]], np.float32)
            valid = np.isfinite(field) & (field >= -50)
            prior |= valid & (field >= 35)
            observed += valid
        coverage = observed / max(len(indices), 1)
        gap = float(np.diff(np.r_[start, self.times[indices], end]).max() / 1000000000.0)
        return (prior, coverage, dict(prior_scans=len(indices), prior_max_gap_seconds=gap, history_temporal_coverage_ok=bool(len(indices) and gap <= 150)))

def project_mask(mask, mapping, shape):
    selected = mask[mapping['source_y'], mapping['source_x']]
    result = np.zeros(shape, bool)
    result[mapping['target_y'][selected], mapping['target_x'][selected]] = True
    return result

def origin(mask, size_y, size_x):
    points = np.argwhere(mask)
    cy, cx = np.rint(points.mean(axis=0)).astype(int)
    h, w = mask.shape
    sy, sx = (min(size_y, h), min(size_x, w))
    y = min(max(cy - sy // 2, 0), h - sy)
    x = min(max(cx - sx // 2, 0), w - sx)
    return np.s_[y:y + sy, x:x + sx]

def choose_candidates(table, count, seed, ids):
    train = table[table.split == 'train'].copy()
    train = train[train.status == 'v3_accepted_requires_review']
    train = train[train.birth_time == train.canonical_birth_time]
    train = train.drop_duplicates('canonical_event_id')
    if ids:
        picked = train[train.object_id.isin(ids)]
        if len(picked) != len(set(ids)):
            raise RuntimeError('Some --object-ids are not canonical accepted TRAIN births')
        return picked
    if 'reference_count' in train and (train.reference_count > 0).any():
        train = train[train.reference_count > 0]
    if not len(train):
        raise RuntimeError('No eligible train birth')
    rng = np.random.default_rng(seed)
    picks = []
    for _, group in train.groupby('year'):
        picks.extend(group.iloc[rng.permutation(len(group))].head(4).index.tolist())
    picks.extend(train.sort_values('group_pixels', ascending=False).head(4).index.tolist())
    multi = train[train.group_components.map(lambda value: len(json.loads(value)) if isinstance(value, str) else len(value)) > 1]
    picks.extend(multi.sort_values('group_pixels', ascending=False).head(4).index.tolist())
    picks.extend(train.sort_values('selected_offset_km').head(8).index.tolist())
    picks = list(dict.fromkeys(picks))
    return train.loc[picks].head(count)

def goe(ds, time, region):
    t = pd.Timestamp(time).floor('10min')
    for age in [0, 10]:
        stamp = t - pd.Timedelta(minutes=age)
        try:
            month = ds._month_data(stamp)
            i = month['time_lookup'].get(int(stamp.value))
            if i is not None:
                return (np.asarray(month['group']['CMI'][i, 5, region[0], region[1]], np.float32), str(stamp))
        except (FileNotFoundError, KeyError, OSError, RuntimeError):
            pass
    return (None, None)

def render(args, info, radar_path, reader, mapping, ds, grid_shape, number):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    row = info['row']
    field = reader.exact(row.birth_time)
    dy, dx = (float(row.native_dy_km), float(row.native_dx_km))
    seed = native_mask(row.birth_mask_path, reader.shape, int(row.group_pixels))
    band = native_mask(row.isolation_band_path, reader.shape)
    region = origin(seed, int(np.ceil(args.patch_km / dy)), int(np.ceil(args.patch_km / dx)))
    target, _ = envelopes(field, seed, dy, dx, float(row.selected_offset_km), band, np.zeros(reader.shape, bool), args.thresholds)
    points = np.argwhere(target[min(args.thresholds)])
    lower, upper = (points.min(axis=0), points.max(axis=0) + 1)
    history_region = np.s_[lower[0]:upper[0], lower[1]:upper[1]]
    previous, coverage, history_report = reader.prior_screen(row.birth_time, history_region)
    prior = np.zeros(reader.shape, bool)
    prior[history_region] = previous
    native_coverage = np.zeros(reader.shape, np.float32)
    native_coverage[history_region] = coverage
    local_prior = prior[region]
    target, reports = envelopes(field, seed, dy, dx, float(row.selected_offset_km), band, prior, args.thresholds)
    projected = {threshold: project_mask(mask, mapping, grid_shape) for threshold, mask in target.items()}
    if not projected[35].any():
        raise RuntimeError('Birth seed not projected inside GOES grid')
    goes_region = origin(projected[35], int(np.ceil(args.patch_km / args.goes_dy)), int(np.ceil(args.patch_km / args.goes_dx)))
    folder = args.output_dir / f'case_{number:02d}_{row.object_id}'
    folder.mkdir(exist_ok=True)
    np.savez_compressed(folder / 'candidate_masks_native.npz', shape=np.array(reader.shape), birth_time_ns=np.int64(pd.Timestamp(row.birth_time).value), **{f'y_ge{t}': np.where(mask)[0].astype(np.int32) for t, mask in target.items()}, **{f'x_ge{t}': np.where(mask)[1].astype(np.int32) for t, mask in target.items()})
    for report in reports:
        t = report['threshold_dbz']
        report.update(case=number, object_id=row.object_id, canonical_event_id=row.canonical_event_id, period_id=row.period_id, birth_time=str(row.birth_time), selected_offset_km=float(row.selected_offset_km), goes_target_pixels=int(projected[t].sum()), display_clips_candidate=bool(target[t].sum() != target[t][region].sum()), candidate_history_valid_fraction=float(native_coverage[target[t]].mean()), requires_history_coverage_review=bool(not history_report['history_temporal_coverage_ok'] or (native_coverage[target[t]] < 0.9).any()), **history_report)
    pd.DataFrame(reports).to_csv(folder / 'envelope_metrics.csv', index=False)
    fig, axes = plt.subplots(2, 1 + len(args.thresholds), figsize=(4 * (1 + len(args.thresholds)), 8))
    radar_cmap = plt.get_cmap('turbo').copy()
    radar_cmap.set_bad('#999999')
    radar_cmap.set_under('#111111')
    goes_cmap = plt.get_cmap('turbo').copy()
    goes_cmap.set_bad('#999999')
    local_shape = seed[region].shape
    gm_shape = projected[35][goes_region].shape
    rad_im = axes[0, 0].imshow(np.full(local_shape, np.nan), cmap=radar_cmap, vmin=0, vmax=60)
    goes_im = axes[1, 0].imshow(np.full(gm_shape, np.nan), cmap=goes_cmap, vmin=190, vmax=310)
    fig.colorbar(rad_im, ax=axes[0, 0], shrink=0.7, label='dBZ')
    fig.colorbar(goes_im, ax=axes[1, 0], shrink=0.7, label='K')
    for col, threshold in enumerate(args.thresholds, 1):
        ax = axes[0, col]
        f = field[region]
        ax.imshow(np.ma.masked_where(~np.isfinite(f) | (f < -50), f), cmap='Greys', vmin=0, vmax=60)
        candidate = target[threshold][region]
        overlay = np.zeros((*candidate.shape, 4), float)
        overlay[candidate] = [0, 0.85, 0.9, 0.65]
        ax.imshow(overlay)
        if band[region].any() and (not band[region].all()):
            ax.contour(band[region], levels=[0.5], colors=['magenta'], linewidths=0.7)
        suspect = candidate & local_prior
        if suspect.any():
            ys, xs = np.where(suspect)
            ax.scatter(xs, ys, s=8, color='red')
        ax.set_title(f"Birth mask >= {threshold} dBZ\n{reports[col - 1]['candidate_pixels']} native pixels")
        ax = axes[1, col]
        ax.imshow(projected[threshold][goes_region], cmap='Blues', vmin=0, vmax=1)
        ax.set_title(f"GOES-grid binary target >= {threshold}\n{reports[col - 1]['goes_target_pixels']} pixels")
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    frames, timestamps = ([], [])
    offsets = list(range(-args.before, args.after + 1, args.step_minutes))
    if 0 not in offsets:
        offsets = sorted(set(offsets + [0]))
    title = fig.suptitle('', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for offset in offsets:
        requested = pd.Timestamp(row.birth_time) + pd.Timedelta(minutes=offset)
        rf, rt = reader.past(requested, region)
        gf, gt = goe(ds, requested, goes_region)
        rad_im.set_data(np.full(local_shape, np.nan) if rf is None else np.ma.masked_where(~np.isfinite(rf) | (rf < -50), rf))
        goes_im.set_data(np.full(gm_shape, np.nan) if gf is None else gf)
        axes[0, 0].set_title('Observed MRMS\n' + (rt or 'Missing scan'))
        axes[1, 0].set_title('Observed CH13\n' + (gt or 'Missing scan'))
        title.set_text(f'{row.period_id} / {row.object_id}\nBirth {row.birth_time} UTC; display T{offset:+d} min. Targets fixed at birth.\nCyan: candidate envelope; magenta: isolation band; red: local prior >=35 echo (review flag only).')
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()))
        timestamps.append(dict(offset_minutes=offset, requested=str(requested), radar_scan=rt, goes_scan=gt))
        if offset in [0, 10, 20]:
            fig.savefig(folder / f'birth_plus_{offset:02d}min.png', dpi=150)
    plt.close(fig)
    frames[0].save(folder / 'comparison.gif', save_all=True, append_images=frames[1:], duration=int(1000 / args.fps), loop=0, optimize=False)
    (folder / 'times.json').write_text(json.dumps(dict(frames=timestamps, birth=str(row.birth_time), radar_path=str(radar_path), note='Future observations are visual QC only; fixed masks use exact birth scan.'), indent=2))
    print('RENDERED', folder, 'frames=', len(frames), flush=True)
    return reports

def selftest():
    import tempfile
    field = np.full((20, 30), 0.0, np.float32)
    field[10, 10:12] = 40
    field[10, 12:14] = 32
    field[10, 14:16] = 27
    seed = np.zeros_like(field, bool)
    seed[10, 10:12] = True
    band = np.zeros_like(seed)
    prior = np.zeros_like(seed)
    targets, reports = envelopes(field, seed, 1, 2, 20, band, prior)
    assert [int(targets[t].sum()) for t in [35, 30, 25]] == [2, 4, 6]
    prior[10, 15] = True
    targets, reports = envelopes(field, seed, 1, 2, 5, band, prior)
    assert reports[2]['requires_geometry_review'] and (not targets[25][10, 15])
    field[1, 1] = 30
    targets, reports = envelopes(field, seed, 1, 2, 20, band, prior)
    assert not targets[25][1, 1] and reports[2]['requires_prior_echo_review']
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(_ci_resolve(tmp)) / 'birth.npz'
        np.savez(path, shape=seed.shape, y=np.where(seed)[0], x=np.where(seed)[1])
        assert np.array_equal(native_mask(path, seed.shape, 2), seed)
    group_field = np.zeros((15, 15), np.float32)
    group_field[3, 3] = group_field[11, 11] = 40
    group_field[3, 4] = group_field[11, 12] = 28
    group_seed = group_field >= 35
    grouped, _ = envelopes(group_field, group_seed, 1, 1, 3, np.zeros_like(group_seed), np.zeros_like(group_seed))
    assert grouped[25].sum() == 4 and grouped[30].sum() == 2
    print('NATIVE SEED / ANISOTROPIC ENVELOPE / ISOLATION LIMIT / PRIOR FLAG SELFTEST PASSED', flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selftest', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--cases', type=int, default=6)
    parser.add_argument('--candidates', type=int, default=36)
    parser.add_argument('--object-ids', nargs='+')
    parser.add_argument('--case-manifest', type=Path)
    parser.add_argument('--thresholds', nargs='+', type=int, default=[35, 30, 25], choices=[35, 30, 25, 20])
    parser.add_argument('--seed', type=int, default=20261014)
    parser.add_argument('--before', type=int, default=20)
    parser.add_argument('--after', type=int, default=30)
    parser.add_argument('--step-minutes', type=int, default=2)
    parser.add_argument('--patch-km', type=float, default=60)
    parser.add_argument('--fps', type=int, default=4)
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    if args.output_dir is None:
        parser.error('--output-dir required')
    if 35 not in args.thresholds or len(set(args.thresholds)) != len(args.thresholds):
        parser.error('Thresholds must include 35, without duplicates')
    args.thresholds = sorted(args.thresholds, reverse=True)
    if args.case_manifest:
        if args.object_ids:
            parser.error('Use case-manifest OR object-ids')
        args.object_ids = pd.read_csv(_ci_resolve(args.case_manifest)).object_id.astype(str).tolist()
    if args.cases < 1 or args.candidates < args.cases or args.step_minutes < 1 or (args.fps < 1) or (args.patch_km <= 0):
        parser.error('Invalid sampling/render arguments')
    if (args.output_dir / 'completed.json').exists():
        raise RuntimeError('Review already complete; choose another output directory')
    sys.path.insert(0, str(CODE))
    from ci_goes_v3_30min_dataset import CIGoesV3Dataset
    from build_ci_soft_targets_native_goes import load_and_validate_goes_grid, build_mapping
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_parquet(_ci_resolve(CAT / 'ci_v3_object_catalogue.parquet'))
    for key in ['birth_time', 'canonical_birth_time']:
        table[key] = pd.to_datetime(table[key])
    candidates = choose_candidates(table, args.candidates, args.seed, args.object_ids)
    issuances = pd.read_parquet(_ci_resolve(DS / 'train_issuance_index_natural.parquet'))
    paths = issuances.groupby('period_id').radar_path.first().to_dict()
    ds = CIGoesV3Dataset(DS, split='train', mode='full', normalization=None, include_mrms=False)
    grid = load_and_validate_goes_grid()
    args.goes_dy, args.goes_dx = (float(grid['dy_km']), float(grid['dx_km']))
    grid_shape = (212, 230)
    ranked = []
    for number, row in enumerate(candidates.itertuples(index=False), 1):
        if row.period_id not in paths:
            print('SKIP no train issuance radar path:', row.object_id, flush=True)
            continue
        path = paths[row.period_id]
        reader = Radar(path)
        field = reader.exact(row.birth_time)
        seed = native_mask(row.birth_mask_path, reader.shape, int(row.group_pixels))
        dy, dx = (float(row.native_dy_km), float(row.native_dx_km))
        distance = distance_transform_edt(~seed, sampling=(dy, dx))
        nearby = np.isfinite(field) & (field >= 35) & (distance > 1) & (distance <= 15)
        ranked.append(dict(row=row, nearby_mature_pixels=int(nearby.sum()), group_pixels=int(seed.sum()), group_components=len(json.loads(row.group_components)) if isinstance(row.group_components, str) else len(row.group_components)))
        print(f'SCORE {number}/{len(candidates)} {row.object_id} nearby>=35={nearby.sum()}', flush=True)
    if not ranked:
        raise RuntimeError('No usable case')
    selected, used = ([], set())

    def take(items, count):
        for info in items:
            if info['row'].period_id in used:
                continue
            selected.append(info)
            used.add(info['row'].period_id)
            count -= 1
            if count == 0:
                break
    if args.object_ids:
        selected = ranked[:args.cases]
    else:
        take(sorted(ranked, key=lambda r: r['nearby_mature_pixels'], reverse=True), 2)
        take(sorted(ranked, key=lambda r: r['group_pixels'], reverse=True), 1)
        take(sorted(ranked, key=lambda r: r['group_components'], reverse=True), 1)
        take(sorted(ranked, key=lambda r: r['nearby_mature_pixels']), args.cases)
        selected = selected[:args.cases]
    pd.DataFrame([dict(object_id=i['row'].object_id, period_id=i['row'].period_id, birth_time=str(i['row'].birth_time), year=i['row'].year, split=i['row'].split, nearby_mature_pixels=i['nearby_mature_pixels'], group_pixels=i['group_pixels'], group_components=i['group_components']) for i in selected]).to_csv(args.output_dir / 'case_manifest.csv', index=False)
    all_reports = []
    for number, info in enumerate(selected, 1):
        row = info['row']
        path = paths[row.period_id]
        reader = Radar(path)
        mapping = build_mapping(Path(_ci_resolve(path)).parent, grid, {})
        all_reports.extend(render(args, info, path, reader, mapping, ds, grid_shape, number))
    report = pd.DataFrame(all_reports)
    report.to_csv(args.output_dir / 'envelope_comparison.csv', index=False)
    links = [f'<li><a href="{html.escape(p.parent.name)}/comparison.gif">{html.escape(p.parent.name)}</a></li>' for p in sorted(args.output_dir.glob('case_*/comparison.gif'))]
    (args.output_dir / 'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>V4 binary envelope pilot</h1><ul>' + ''.join(links) + '</ul>')
    (args.output_dir / 'completed.json').write_text(json.dumps(dict(status='completed_requires_visual_review', cases=len(selected), thresholds_dbz=args.thresholds, split='train', target_time='exact native birth scan', target_geometry='Seeded 8-connected threshold component, within distance selected_offset_km of saved V3 core; stored isolation band excluded.', prior_screen='Local co-location with >=35 echo in preceding 30 min: review flag, NOT advection-aware lineage or confirmed contamination.', note='Pilot only; production labels, catalogue and indexes unchanged; GOES projection uses any sampled native positive pixel.', flags=['Review geometry clipping, prior echo, radar gaps, missing GOES frames and display cropping before full V4 generation.']), indent=2))
    print('\n' + report.to_string(index=False), flush=True)
    print('\nDONE:', args.output_dir, flush=True)
if __name__ == '__main__':
    main()
