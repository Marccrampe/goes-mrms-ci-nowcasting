"""Review saved V1/V2 labels: extrema, missing counterparts and nearby births.
No catalogue rebuild, no model, no download. Matching is a QC screen.
"""
from ci_nowcasting.paths import resolve_path as _ci_resolve, project_root as _ci_project, source_root as _ci_source
import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
import traceback
import zipfile
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
PROJECT = Path(_ci_resolve(_ci_resolve('/project')))

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj

def save_json(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, default=str))
    temp.replace(path)

@lru_cache(maxsize=256)
def sparse(path):
    with np.load(_ci_resolve(path), allow_pickle=False) as z:
        a = {k: np.asarray(z[k]).copy() for k in z.files}
    shape = tuple(map(int, a['shape']))
    core = np.zeros(shape, bool)
    core[a['core_y'].astype(int), a['core_x'].astype(int)] = True
    target = np.zeros(shape, np.float32)
    target[a['soft_y'].astype(int), a['soft_x'].astype(int)] = a['soft_value']
    dx, dy = (float(a['dx_km']), float(a['dy_km']))
    points = np.column_stack([a['core_y'] * dy, a['core_x'] * dx])
    if not len(points):
        raise RuntimeError('Empty saved core: ' + path)
    return (core, target, points)

def distance_km(path1, path2):
    p, q = (sparse(str(path1))[2], sparse(str(path2))[2])
    return float(cKDTree(q).query(p)[0].min())

def correspondences(old, new, soft_paths, max_minutes, spatial_km):
    grouped = {str(pid): f.sort_values('birth_time') for pid, f in new.groupby('period_id')}
    records = []
    tolerances = sorted(set([5, 10, 20, max_minutes]))
    for count, r in enumerate(old.itertuples(), 1):
        row = dict(old_id=str(r.object_id), period_id=str(r.period_id), year=int(r.year), old_birth_time=r.birth_time, old_path=str(r.output_path))
        choices = []
        f = grouped.get(str(r.period_id))
        if f is not None:
            dt = (f.birth_time - pd.Timestamp(r.birth_time)).dt.total_seconds() / 60
            for n, gap in zip(f.loc[dt.abs() <= max_minutes].itertuples(), dt[dt.abs() <= max_minutes]):
                d = distance_km(r.output_path, soft_paths[str(n.object_id)])
                choices.append((abs(float(gap)), d, float(gap), str(n.object_id)))
        valid = [c for c in choices if c[1] <= spatial_km]
        for tolerance in tolerances:
            row[f'candidate_within_{tolerance:g}min'] = any((c[0] <= tolerance for c in valid))
        row['candidate_count'] = len(valid)
        row['candidate_v2_id'] = ''
        row['delta_minutes_v2_minus_v1'] = np.nan
        row['core_distance_km'] = np.nan
        row['nearest_v2_id_in_time_window'] = ''
        row['nearest_core_distance_km'] = np.nan
        if valid:
            best = min(valid)
            row.update(candidate_v2_id=best[3], delta_minutes_v2_minus_v1=best[2], core_distance_km=best[1])
        if choices:
            nearest = min(choices, key=lambda c: (c[1], c[0]))
            row.update(nearest_v2_id_in_time_window=nearest[3], nearest_core_distance_km=nearest[1])
        row['qc_status'] = 'no_candidate_counterpart' if not valid else 'candidate_within_10min' if any((c[0] <= 10 for c in valid)) else 'candidate_with_larger_time_shift'
        records.append(row)
        if count % 200 == 0:
            print(f'V1/V2 matching {count}/{len(old)}', flush=True)
    return pd.DataFrame(records)

def nearby_births(new, spacing, max_minutes=30.0, radius_km=10.0):
    records = []
    for pid, f in new.groupby('period_id'):
        f = f.sort_values('birth_time').reset_index(drop=True)
        coords = f[['goes_core_y', 'goes_core_x']].to_numpy(float) * np.asarray(spacing)
        times = pd.DatetimeIndex(f.birth_time).asi8
        for j in range(len(f)):
            start = int(np.searchsorted(times, times[j] - max_minutes * 60000000000.0))
            for i in range(start, j):
                dt = (times[j] - times[i]) / 60000000000.0
                if dt < 1.5:
                    continue
                distance = float(np.linalg.norm(coords[j] - coords[i]))
                if distance <= min(radius_km, 1.0 + 2.4 * dt):
                    records.append(dict(period_id=str(pid), first_id=str(f.object_id.iloc[i]), second_id=str(f.object_id.iloc[j]), first_birth=f.birth_time.iloc[i], second_birth=f.birth_time.iloc[j], gap_minutes=float(dt), centroid_distance_km=distance))
    columns = ['period_id', 'first_id', 'second_id', 'first_birth', 'second_birth', 'gap_minutes', 'centroid_distance_km']
    return pd.DataFrame(records, columns=columns)

def select_cases(old, new, matches, repeats, number):
    old_lookup = old.set_index('object_id')
    new_lookup = new.set_index('object_id')
    cases, used = ([], set())

    def add(category, source, oid, other=''):
        key = (source, str(oid))
        if key in used:
            return False
        used.add(key)
        r = (new_lookup if source == 'v2' else old_lookup).loc[oid]
        cases.append(dict(category=category, source=source, object_id=str(oid), other_v2_id=str(other), period_id=str(r.period_id), year=int(r.year), birth_time=str(r.birth_time), group_pixels=int(r.group_pixels) if source == 'v2' else None, radius_km=float(r.selected_radius_km) if source == 'v2' else None))
        return True
    shuffled = new.sample(frac=1, random_state=20261007)
    minima = shuffled.sort_values('group_pixels', kind='stable')
    minimum_size = new.group_pixels.min()
    taken_periods = set()
    for r in minima.itertuples():
        if r.group_pixels != minimum_size or len(taken_periods) >= number:
            break
        if r.period_id not in taken_periods and add('minimum', 'v2', r.object_id):
            taken_periods.add(r.period_id)
    selected = 0
    for r in new.sort_values('group_pixels', ascending=False).itertuples():
        if add('maximum', 'v2', r.object_id):
            selected += 1
        if selected >= number:
            break
    selected = 0
    unmatched = matches[matches.qc_status == 'no_candidate_counterpart'].sample(frac=1, random_state=20261007)
    for r in unmatched.itertuples():
        if add('v1_without_v2', 'v1', r.old_id, r.nearest_v2_id_in_time_window):
            selected += 1
        if selected >= number:
            break
    selected = 0
    shifted = matches[matches.qc_status == 'candidate_with_larger_time_shift'].copy()
    shifted['absolute_shift'] = shifted.delta_minutes_v2_minus_v1.abs()
    for r in shifted.sort_values('absolute_shift', ascending=False).itertuples():
        if add('v1_shifted_birth', 'v1', r.old_id, r.candidate_v2_id):
            selected += 1
        if selected >= number:
            break
    selected = 0
    if len(repeats):
        ranked = repeats.sort_values(['exact_centroid_distance_km', 'gap_minutes'])
        for r in ranked.itertuples():
            if add('nearby_repeated_births', 'v2', r.second_id, r.first_id):
                cases[-1]['pair_gap_minutes'] = r.gap_minutes
                cases[-1]['pair_distance_km'] = r.exact_centroid_distance_km
                selected += 1
            if selected >= number:
                break
    for i, case in enumerate(cases, 1):
        case['case_id'] = f"{i:02d}_{case['category']}"
    return cases

class Renderer:

    def __init__(self, args, old, new, soft_paths):
        import zarr
        from pyproj import Geod
        self.args, self.zarr = (args, zarr)
        self.old, self.new, self.soft_paths = (old, new, soft_paths)
        code = args.project / str(_ci_source() / 'scripts/isolated_ci_binary_v1/goes_ir_v1')
        self.builder = module('review_builder', code / 'build_ci_soft_targets_native_goes.py')
        self.dm = module('review_dataset', code / 'ci_goes_patch_dataset.py')
        self.grid = self.builder.load_and_validate_goes_grid()
        self.data = args.project / 'data/ci_isolated_mrms_binary_v1/goes_ir_parallax_native_filled_v1'
        self.ds = self.dm.CIGoesPatchDataset.__new__(self.dm.CIGoesPatchDataset)
        self.ds.goes_root, self.ds.month_cache = (self.data, {})
        self.geod = Geod(ellps='WGS84')

    def goes_frame(self, timestamp):
        candidates = []
        seen = set()
        for t in [timestamp, timestamp - pd.Timedelta(minutes=8), timestamp + pd.Timedelta(minutes=8)]:
            key = (t.year, t.month)
            if key in seen:
                continue
            seen.add(key)
            try:
                month = self.ds._month_data(t)
                values = np.asarray(sorted(month['time_lookup']), np.int64)
                if len(values):
                    nearest = values[int(np.abs(values - timestamp.value).argmin())]
                    candidates.append((abs(nearest - timestamp.value), month, int(nearest)))
            except FileNotFoundError:
                continue
        if not candidates:
            raise RuntimeError('No local GOES month available')
        _, month, nominal = min(candidates, key=lambda item: item[0])
        if abs(nominal - timestamp.value) > 360000000000.0:
            raise RuntimeError('No GOES nominal frame within 6 min')
        i = month['time_lookup'][nominal]
        group = month['group']
        source = pd.Timestamp(nominal)
        if 'source_scan_end' in group:
            a = group['source_scan_end']
            source = self.dm.decode_time_array(np.asarray(a[i:i + 1, 5]), a.attrs.asdict())[0]
        if pd.isna(source) or abs((source - timestamp).total_seconds()) > 360:
            raise RuntimeError('Missing/distant GOES source timestamp')
        return (np.asarray(group['CMI'][i, 5], np.float32), source)

    def prepare(self, case):
        period = self.data.parent / f"year_{case['year']}_strict_v2_001/periods" / case['period_id']
        radar = self.zarr.open_group(str(period / 'reflectivity.zarr'), mode='r')
        times = pd.DatetimeIndex(pd.to_datetime(np.asarray(radar['time_ns'][:], np.int64)))
        if times.has_duplicates or not times.is_monotonic_increasing:
            raise RuntimeError('Unordered radar times')
        birth = pd.Timestamp(case['birth_time'])
        take = np.flatnonzero((times >= birth - pd.Timedelta(minutes=self.args.before)) & (times <= birth + pd.Timedelta(minutes=self.args.after)))
        if not len(take):
            raise RuntimeError('No radar frames in review window')
        if np.min(np.abs(times[take].asi8 - birth.value)) > 150000000000.0:
            raise RuntimeError('No MRMS scan within 2.5 minutes of selected birth')
        fields = np.asarray(radar['reflectivity_dbz'].oindex[take, :, :], np.float32)
        times = times[take]
        lat, lon = (np.asarray(radar['latitude'][:]), np.asarray(radar['longitude'][:]))
        dy = abs(self.geod.inv(lon.mean(), lat[0], lon.mean(), lat[1])[2]) / 1000
        dx = abs(self.geod.inv(lon[0], lat.mean(), lon[1], lat.mean())[2]) / 1000
        mapping = self.builder.build_mapping(period, self.grid, {})
        lookup = np.full(fields.shape[1:], -1, np.int32)
        lookup[mapping['source_y'], mapping['source_x']] = mapping['target_y'] * self.grid['shape'][1] + mapping['target_x']
        if case['source'] == 'v2':
            row = self.new[self.new.object_id == case['object_id']].iloc[0]
            with np.load(_ci_resolve(row.birth_mask_path), allow_pickle=False) as z:
                p = np.column_stack([z['y'], z['x']]).astype(int)
            path = self.soft_paths[case['object_id']]
        else:
            row = self.old[self.old.object_id == case['object_id']].iloc[0]
            path = str(row.output_path)
            core = sparse(path)[0]
            selected = core[mapping['target_y'], mapping['target_x']]
            p = np.column_stack([mapping['source_y'][selected], mapping['source_x'][selected]]).astype(int)
        if not len(p):
            raise RuntimeError('Selected core is not mapped to native radar')
        cy, cx = p.mean(axis=0)
        gc = sparse(path)[0]
        gy, gx = np.argwhere(gc).mean(axis=0)
        gh, gw = self.grid['shape']
        size = min(gh, gw, int(np.ceil(self.args.patch_km / min(self.grid['dx_km'], self.grid['dy_km']))))
        y0 = int(np.clip(round(gy - size / 2), 0, gh - size))
        x0 = int(np.clip(round(gx - size / 2), 0, gw - size))
        gs = np.s_[y0:y0 + size, x0:x0 + size]
        ry0, ry1 = (max(0, int(cy - self.args.patch_km / 2 / dy)), min(fields.shape[1], int(cy + self.args.patch_km / 2 / dy) + 1))
        rx0, rx1 = (max(0, int(cx - self.args.patch_km / 2 / dx)), min(fields.shape[2], int(cx + self.args.patch_km / 2 / dx) + 1))
        native = np.s_[ry0:ry1, rx0:rx1]
        values = fields[:, p[:, 0], p[:, 1]]
        ok = np.isfinite(values) & (values >= -50)
        peak = np.max(np.where(ok, values, -np.inf), axis=1)
        peak[~np.isfinite(peak)] = np.nan
        fraction = np.divide((ok & (values >= 35)).sum(axis=1), ok.sum(axis=1), out=np.full(len(times), np.nan), where=ok.sum(axis=1) > 0)
        old = self.old[(self.old.period_id == case['period_id']) & (self.old.birth_time > times.min() - pd.Timedelta(minutes=10)) & (self.old.birth_time <= times.max())]
        new = self.new[(self.new.period_id == case['period_id']) & (self.new.birth_time > times.min() - pd.Timedelta(minutes=10)) & (self.new.birth_time <= times.max())]
        return dict(case=case, birth=birth, fields=fields, times=times, spacing=(dy, dx), mapping=mapping, lookup=lookup, native=native, gs=gs, native_bounds=(ry0, ry1, rx0, rx1), goes_origin=(y0, x0), point=(cy, cx), old=old, new=new, peak=peak, fraction=fraction)

    def draw(self, prepared, i):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
        from PIL import Image
        z = prepared
        case, t = (z['case'], z['times'][i])
        dy, dx = z['spacing']
        f = z['fields'][i]
        valid = np.isfinite(f) & (f >= -50)
        projected = np.full(self.grid['shape'], -np.inf, np.float32)
        m = z['mapping']
        good = valid[m['source_y'], m['source_x']]
        np.maximum.at(projected, (m['target_y'][good], m['target_x'][good]), f[m['source_y'][good], m['source_x'][good]])
        projected[~np.isfinite(projected)] = np.nan
        error = ''
        try:
            c13, source = self.goes_frame(t)
        except Exception as exc:
            c13, source = (np.full(self.grid['shape'], np.nan), 'MISSING')
            error = str(exc)
        old = z['old'][(z['old'].birth_time > t - pd.Timedelta(minutes=10)) & (z['old'].birth_time <= t)]
        new = z['new'][(z['new'].birth_time > t - pd.Timedelta(minutes=10)) & (z['new'].birth_time <= t)]

        def target(rows, version):
            soft = np.zeros(self.grid['shape'], np.float32)
            core = np.zeros(self.grid['shape'], bool)
            for r in rows.itertuples():
                path = str(r.output_path) if version == 'v1' else self.soft_paths[str(r.object_id)]
                c, s, _ = sparse(path)
                soft = np.maximum(soft, s)
                core |= c
            return (core, soft)
        old_core, old_soft = target(old, 'v1')
        new_core, new_soft = target(new, 'v2')
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout='constrained')
        ax = axes.ravel()
        gs = z['gs']
        ax[0].imshow(c13[gs], vmin=200, vmax=300, cmap='turbo')
        ax[0].set_title('GOES C13 (K)\nsource UTC ' + str(source))
        ry0, ry1, rx0, rx1 = z['native_bounds']
        radar_native = np.where(valid[z['native']], f[z['native']], np.nan)
        extent = (rx0 - 0.5, rx1 - 0.5, ry1 - 0.5, ry0 - 0.5)
        im = ax[1].imshow(radar_native, extent=extent, vmin=0, vmax=60, cmap='turbo')
        ax[1].set_title('Native MRMS (dBZ): actual scan\n' + str(t))
        fig.colorbar(im, ax=ax[1], fraction=0.035)
        cy, cx = z['point']
        ax[1].plot(cx, cy, marker='+', color='white', ms=10, mew=1.4)
        for r in new.itertuples():
            with np.load(_ci_resolve(r.birth_mask_path), allow_pickle=False) as ar:
                yy, xx = (ar['y'].astype(int), ar['x'].astype(int))
            ax[1].scatter(xx, yy, s=9, facecolors='none', edgecolors='magenta', linewidths=0.8)
            for radius in [r.selected_radius_km, r.selected_radius_km - 1.0]:
                ax[1].add_patch(Ellipse((r.native_x, r.native_y), 2 * radius / dx, 2 * radius / dy, fill=False, edgecolor='white', linewidth=0.65))
        ax[1].set_xlim(rx0 - 0.5, rx1 - 0.5)
        ax[1].set_ylim(ry1 - 0.5, ry0 - 0.5)
        ax[2].imshow(projected[gs], vmin=0, vmax=60, cmap='turbo')
        ax[2].set_title('MRMS max projected onto GOES grid')
        for a, field, title in [(ax[3], old_soft, 'V1 saved birth targets'), (ax[4], new_soft, 'V2 saved birth targets')]:
            im = a.imshow(field[gs], vmin=0, vmax=1, cmap='magma')
            a.set_title(title + '\nBirths in preceding 10 minutes')
            fig.colorbar(im, ax=a, fraction=0.035)
        for j in [0, 2, 3, 4]:
            overlays = [(old_core, 'cyan')] if j == 3 else [(new_core, 'magenta')] if j == 4 else [(old_core, 'cyan'), (new_core, 'magenta')]
            for core, color in overlays:
                c = core[gs]
                if c.any() and (not c.all()):
                    ax[j].contour(c.astype(float), [0.5], colors=[color], linewidths=0.8)
            ax[j].set_xlabel('GOES crop x pixel')
            ax[j].set_ylabel('GOES crop y pixel')
        ax[1].set_xlabel('Native MRMS x pixel')
        ax[1].set_ylabel('Native MRMS y pixel')
        minutes = (z['times'] - z['birth']).total_seconds() / 60
        ax[5].plot(minutes, z['peak'], color='black', label='Max dBZ in fixed birth footprint')
        ax[5].axhline(35, color='red', linestyle='--', lw=0.8)
        ax[5].axhline(20, color='grey', linestyle=':', lw=0.8)
        ax[5].axvline(0, color='blue', linestyle='--', label='Selected birth')
        ax[5].axvline((t - z['birth']).total_seconds() / 60, color='magenta', label='Current frame')
        ax[5].set_ylim(-5, 65)
        ax[5].set_xlabel('Minutes relative to selected birth')
        ax[5].set_ylabel('dBZ')
        ax[5].set_title('Fixed core location: not a tracked cell' + ('\nV1 GOES core mapped back to MRMS' if case['source'] == 'v1' else ''))
        ax[5].legend(fontsize=7, loc='lower left')
        ax[5].grid(alpha=0.2)
        if case.get('other_v2_id'):
            other = self.new[self.new.object_id == case['other_v2_id']]
            if len(other):
                delta = (pd.Timestamp(other.birth_time.iloc[0]) - z['birth']).total_seconds() / 60
                ax[5].axvline(delta, color='orange', linestyle=':', lw=1)
        fig.suptitle(f"{case['case_id']} | {case['period_id']}\nSelected {case['source']}: {case['object_id']} | birth UTC {z['birth']}\nCyan = V1; magenta = V2; White annulus = V2 isolation test. Future observations are visual QC only.", fontsize=9)
        fig.canvas.draw()
        image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        plt.close(fig)
        return (image, dict(radar_time=str(t), goes_source=str(source), goes_error=error, minutes_from_birth=float((t - z['birth']).total_seconds() / 60), native_fixed_footprint_max_dbz=z['peak'][i], native_fixed_footprint_fraction_ge35=z['fraction'][i]))

    def render(self, case, output):
        destination = output / case['case_id']
        destination.mkdir(exist_ok=True)
        if (destination / 'completed.json').exists():
            return json.loads((destination / 'completed.json').read_text())
        prepared = self.prepare(case)
        nearest = int(np.abs(prepared['times'].asi8 - prepared['birth'].value).argmin())
        images, timing = ([], [])
        for i in range(len(prepared['times'])):
            image, info = self.draw(prepared, i)
            images.append(image)
            timing.append(info)
            if i == nearest:
                image.save(destination / 'birth.png')
        temporary = destination / 'animation.tmp.gif'
        images[0].save(temporary, save_all=True, append_images=images[1:], duration=650, loop=0)
        temporary.replace(destination / 'animation.gif')
        for image in images:
            image.close()
        pd.DataFrame(timing).to_csv(destination / 'frame_times.csv', index=False)
        prepared['old'].to_csv(destination / 'v1_births_in_window.csv', index=False)
        prepared['new'].to_csv(destination / 'v2_births_in_window.csv', index=False)
        result = dict(**case, frames=len(timing), missing_goes=sum((bool(t['goes_error']) for t in timing)), maximum_radar_gap_minutes=float(np.diff(prepared['times'].asi8).max() / 60000000000.0) if len(timing) > 1 else None, selected_birth_radar_offset_seconds=float((prepared['times'][nearest] - prepared['birth']).total_seconds()), actual_window_start=str(prepared['times'].min()), actual_window_end=str(prepared['times'].max()))
        save_json(destination / 'completed.json', result)
        return result

def selftest():
    with tempfile.TemporaryDirectory() as d:
        paths = {}
        for name, x in [('old', 10), ('new', 10), ('far', 30)]:
            path = Path(_ci_resolve(d)) / (name + '.npz')
            np.savez(path, shape=[40, 40], core_y=[10], core_x=[x], soft_y=[10], soft_x=[x], soft_value=[1.0], dx_km=2.0, dy_km=2.0)
            paths[name] = str(path)
        old = pd.DataFrame([dict(object_id='O', period_id='P', year=2021, birth_time=pd.Timestamp('2021-01-01 12:00'), output_path=paths['old'])])
        new = pd.DataFrame([dict(object_id='N', period_id='P', birth_time=pd.Timestamp('2021-01-01 11:40')), dict(object_id='F', period_id='P', birth_time=pd.Timestamp('2021-01-01 12:00')), dict(object_id='OTHER', period_id='Q', birth_time=pd.Timestamp('2021-01-01 12:00'))])
        result = correspondences(old, new, {'N': paths['new'], 'F': paths['far'], 'OTHER': paths['new']}, 30, 5)
        assert not result.candidate_within_10min.iloc[0]
        assert result.candidate_within_30min.iloc[0]
        assert result.candidate_v2_id.iloc[0] == 'N'
        assert result.delta_minutes_v2_minus_v1.iloc[0] == -20
        assert distance_km(paths['old'], paths['far']) == 40
    sparse.cache_clear()
    print('REVIEW SELFTEST PASSED: birth shift, spatial mismatch, period separation', flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=PROJECT)
    parser.add_argument('--catalogue-root', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--per-category', type=int, default=3)
    parser.add_argument('--before', type=int, default=30)
    parser.add_argument('--after', type=int, default=40)
    parser.add_argument('--patch-km', type=float, default=80.0)
    parser.add_argument('--spatial-km', type=float, default=5.0)
    parser.add_argument('--time-tolerance-minutes', type=int, default=30)
    parser.add_argument('--selftest', action='store_true')
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    if args.per_category < 1 or args.before < 0 or args.after < 0 or (args.patch_km < 20) or (args.time_tolerance_minutes < 20):
        parser.error('Invalid review parameters')
    root = args.catalogue_root or args.project / 'data/ci_isolated_mrms_binary_v1/goes_ir_parallax_native_filled_v1/labels/ci_catalogue_v2_full_scans_v1'
    out = args.output_dir or args.project / 'experiments/ci_v2_extremes_review_v1'
    out.mkdir(parents=True, exist_ok=True)
    files = [root / 'ci_v2_object_catalogue.parquet', root / 'v1_objects_in_same_windows.parquet', root / 'soft_target_build_manifest.parquet']
    config = json.loads(json.dumps(dict(arguments=vars(args), hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files + [Path(_ci_resolve(__file__)).resolve()]}, correspondence='QC candidate, minimum GOES core distance, same period; not a recall estimate'), default=str))
    if (out / 'config.json').exists() and json.loads((out / 'config.json').read_text()) != config:
        raise RuntimeError('Changed inputs/settings: use another output directory')
    save_json(out / 'config.json', config)
    new, old, soft = [pd.read_parquet(_ci_resolve(p)) for p in files]
    for f in [old, new]:
        f['birth_time'] = pd.to_datetime(f.birth_time)
        f['object_id'] = f.object_id.astype(str)
        f['period_id'] = f.period_id.astype(str)
        if f.object_id.duplicated().any():
            raise RuntimeError('Duplicated object identifiers')
    soft_paths = dict(zip(soft.object_id.astype(str), soft.output_path.astype(str)))
    if set(new.object_id) - set(soft_paths):
        raise RuntimeError('Missing V2 soft targets')
    print('Matching saved V1/V2 objects...', flush=True)
    matches = correspondences(old, new, soft_paths, args.time_tolerance_minutes, args.spatial_km)
    matches.to_csv(out / 'v1_v2_correspondences.csv', index=False)
    print(matches.qc_status.value_counts().to_string(), flush=True)
    tolerance_summary = pd.DataFrame([dict(tolerance_minutes=t, spatial_km=args.spatial_km, old_objects=len(matches), with_candidate=int(matches[f'candidate_within_{t:g}min'].sum()), without_candidate=int((~matches[f'candidate_within_{t:g}min']).sum())) for t in sorted(set([5, 10, 20, args.time_tolerance_minutes]))])
    tolerance_summary.to_csv(out / 'match_tolerance_summary.csv', index=False)
    print(tolerance_summary.to_string(index=False), flush=True)
    print('Searching nearby repeated births (flags, not confirmed duplicates)...', flush=True)
    with np.load(_ci_resolve(soft_paths[str(new.object_id.iloc[0])]), allow_pickle=False) as ar:
        spacing = (float(ar['dy_km']), float(ar['dx_km']))
    repeats = nearby_births(new, spacing)
    exact = []
    for r in repeats.itertuples():
        p, q = (sparse(soft_paths[r.first_id])[2], sparse(soft_paths[r.second_id])[2])
        d = float(np.linalg.norm(p.mean(axis=0) - q.mean(axis=0)))
        exact.append(d)
    repeats['exact_centroid_distance_km'] = exact
    repeats = repeats[repeats.exact_centroid_distance_km <= 10.0].copy()
    repeats.to_csv(out / 'nearby_birth_pairs.csv', index=False)
    cases = select_cases(old, new, matches, repeats, args.per_category)
    pd.DataFrame(cases).to_csv(out / 'selected_cases.csv', index=False)
    renderer = Renderer(args, old, new, soft_paths)
    results, failures = ([], [])
    for i, case in enumerate(cases, 1):
        print(f"RENDER {i}/{len(cases)} {case['case_id']} {case['object_id']}", flush=True)
        try:
            result = renderer.render(case, out)
            results.append(result)
            print(f"DONE frames={result['frames']} missing_GOES={result['missing_goes']}", flush=True)
        except Exception as exc:
            traceback.print_exc()
            failures.append(dict(case_id=case['case_id'], error=repr(exc)))
    summary = dict(selected=len(cases), rendered=len(results), failures=failures, selected_categories=pd.Series([c['category'] for c in cases]).value_counts().to_dict(), missing_goes_frames=sum((r['missing_goes'] for r in results)), v1_qc_status_counts=matches.qc_status.value_counts().to_dict(), nearby_birth_pairs=len(repeats), note='Orange time marker = other V2 birth. Near pairs require visual review; not confirmed duplicates.')
    save_json(out / 'render_report.json', summary)
    import html
    cards = []
    for r in results:
        name = r['case_id']
        cards.append(f'''<h2>{html.escape(name)}</h2><p>{html.escape(str(r['birth_time']))} UTC | {html.escape(r['object_id'])}</p><a href="{name}/birth.png">Birth PNG</a> | <a href="{name}/frame_times.csv">Frame timestamps</a><br><img style="max-width:100%" src="{name}/animation.gif">''')
    (out / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>V2 label review</title><h1>Saved V1/V2 label review</h1>' + ''.join(cards))
    with zipfile.ZipFile(out / 'review_bundle.zip', 'w', zipfile.ZIP_STORED) as bundle:
        for p in out.rglob('*'):
            if p.is_file() and p.suffix in {'.gif', '.png', '.csv', '.json', '.html'}:
                bundle.write(p, str(p.relative_to(out)))
    print(json.dumps(summary, indent=2), flush=True)
    if failures:
        raise RuntimeError('Some cases failed; inspect render_report.json and rerun to resume')
    print('ALL EXTREMES REVIEW CASES COMPLETED:', out, flush=True)
if __name__ == '__main__':
    main()
