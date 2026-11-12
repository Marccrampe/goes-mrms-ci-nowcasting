from ci_nowcasting.paths import resolve_path as _ci_resolve, project_root as _ci_project, source_root as _ci_source
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm
import zarr
DATA = Path(_ci_resolve(_ci_resolve('/project/data/ci_isolated_mrms_binary_v1')))
FILLED = DATA / 'goes_ir_parallax_native_filled_v1'
STATE = FILLED / 'state'
OBJECT_MANIFEST = STATE / 'ci_soft_target_object_manifest.parquet'
SIGMA_KM = 3.0
HARD_BUFFER_KM = 4.0
CUTOFF_KM = 8.0

def decode_mask(path, expected_shape):
    with np.load(_ci_resolve(path), allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    if {'y', 'x'}.issubset(arrays):
        shape = expected_shape
        if 'shape' in arrays:
            shape = tuple((int(value) for value in arrays['shape'].ravel()))
        mask = np.zeros(shape, dtype=bool)
        yy = arrays['y'].astype(np.int64)
        xx = arrays['x'].astype(np.int64)
        mask[yy, xx] = True
        if mask.shape == expected_shape:
            return mask
        if mask.T.shape == expected_shape:
            return mask.T
    for key in ('mask', 'birth_mask', 'valid_mask', 'array', 'arr_0'):
        if key not in arrays:
            continue
        value = arrays[key]
        if value.shape == expected_shape:
            return value.astype(bool)
        if value.T.shape == expected_shape:
            return value.T.astype(bool)
    raise RuntimeError(f'Unsupported mask format: {path}; keys={list(arrays)}')

def nearest_axis_indices(axis, values):
    axis = np.asarray(axis)
    values = np.asarray(values)
    descending = axis[0] > axis[-1]
    working = axis[::-1] if descending else axis
    right = np.searchsorted(working, values)
    right = np.clip(right, 1, len(working) - 1)
    left = right - 1
    use_right = np.abs(working[right] - values) < np.abs(working[left] - values)
    indices = np.where(use_right, right, left)
    if descending:
        indices = len(axis) - 1 - indices
    return indices.astype(np.int64)

def load_and_validate_goes_grid():
    stores = sorted(FILLED.glob('year=*/month=*/goes_ir.zarr'))
    if len(stores) != 59:
        raise RuntimeError(f'Expected 59 GOES stores, found {len(stores)}')
    reference = zarr.open_group(_ci_resolve(str(stores[0])), mode='r')
    x_rad = np.asarray(reference['x'][:], dtype=np.float64)
    y_rad = np.asarray(reference['y'][:], dtype=np.float64)
    projection = reference['goes_imager_projection'].attrs.asdict()
    projection_keys = ['perspective_point_height', 'longitude_of_projection_origin', 'semi_major_axis', 'semi_minor_axis', 'sweep_angle_axis']
    for store in stores[1:]:
        group = zarr.open_group(_ci_resolve(str(store)), mode='r')
        if not np.array_equal(np.asarray(group['x'][:], dtype=np.float64), x_rad):
            raise RuntimeError(f'Different GOES x grid: {store}')
        if not np.array_equal(np.asarray(group['y'][:], dtype=np.float64), y_rad):
            raise RuntimeError(f'Different GOES y grid: {store}')
        other_projection = group['goes_imager_projection'].attrs.asdict()
        for key in projection_keys:
            if other_projection[key] != projection[key]:
                raise RuntimeError(f'Different projection {key}: {store}')
    height = float(projection['perspective_point_height'])
    x_m = x_rad * height
    y_m = y_rad * height
    dx_km = float(np.nanmedian(np.abs(np.diff(x_m)))) / 1000.0
    dy_km = float(np.nanmedian(np.abs(np.diff(y_m)))) / 1000.0
    geos_crs = CRS.from_proj4(' '.join(['+proj=geos', f'+h={height}', '+lon_0=' + str(float(projection['longitude_of_projection_origin'])), f"+a={float(projection['semi_major_axis'])}", f"+b={float(projection['semi_minor_axis'])}", f"+sweep={projection['sweep_angle_axis']}", '+units=m', '+no_defs']))
    transformer = Transformer.from_crs('EPSG:4326', geos_crs, always_xy=True)
    return {'stores': stores, 'x_m': x_m, 'y_m': y_m, 'shape': (len(y_m), len(x_m)), 'dx_km': dx_km, 'dy_km': dy_km, 'projection': projection, 'transformer': transformer}

def build_mapping(period_root, goes_grid, cache):
    mrms = zarr.open_group(_ci_resolve(str(period_root / 'reflectivity.zarr')), mode='r')
    latitude = np.asarray(mrms['latitude'][:], dtype=np.float64)
    longitude = np.asarray(mrms['longitude'][:], dtype=np.float64)
    expected_shape = (len(latitude), len(longitude))
    digest = hashlib.sha256()
    digest.update(latitude.tobytes())
    digest.update(longitude.tobytes())
    grid_hash = digest.hexdigest()
    if grid_hash in cache:
        return cache[grid_hash]
    lon_2d, lat_2d = np.meshgrid(longitude, latitude)
    projected_x, projected_y = goes_grid['transformer'].transform(lon_2d, lat_2d)
    projected_x = np.asarray(projected_x, dtype=np.float64)
    projected_y = np.asarray(projected_y, dtype=np.float64)
    dx_m = goes_grid['dx_km'] * 1000.0
    dy_m = goes_grid['dy_km'] * 1000.0
    inside = np.isfinite(projected_x) & np.isfinite(projected_y) & (projected_x >= np.min(goes_grid['x_m']) - dx_m / 2) & (projected_x <= np.max(goes_grid['x_m']) + dx_m / 2) & (projected_y >= np.min(goes_grid['y_m']) - dy_m / 2) & (projected_y <= np.max(goes_grid['y_m']) + dy_m / 2)
    source_y, source_x = np.where(inside)
    target_x = nearest_axis_indices(goes_grid['x_m'], projected_x[inside])
    target_y = nearest_axis_indices(goes_grid['y_m'], projected_y[inside])
    total_count = np.zeros(goes_grid['shape'], dtype=np.int32)
    np.add.at(total_count, (target_y, target_x), 1)
    mapping = {'grid_hash': grid_hash, 'expected_shape': expected_shape, 'source_y': source_y, 'source_x': source_x, 'target_y': target_y, 'target_x': target_x, 'total_count': total_count}
    cache[grid_hash] = mapping
    return mapping

def atomic_save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)

def build_object_target(row, mapping, goes_grid, output_path):
    core_mrms = decode_mask(Path(_ci_resolve(row.birth_mask_path)), mapping['expected_shape'])
    positive_count = np.zeros(goes_grid['shape'], dtype=np.int32)
    np.add.at(positive_count, (mapping['target_y'], mapping['target_x']), core_mrms[mapping['source_y'], mapping['source_x']].astype(np.int32))
    core_goes = positive_count > 0
    if not np.any(core_goes):
        raise RuntimeError(f'Core disappeared for {row.object_id}')
    core_coverage = np.zeros(goes_grid['shape'], dtype=np.float32)
    core_coverage[core_goes] = positive_count[core_goes] / mapping['total_count'][core_goes]
    distance_km = distance_transform_edt(~core_goes, sampling=(goes_grid['dy_km'], goes_grid['dx_km']))
    hard_buffer = distance_km <= HARD_BUFFER_KM
    soft = np.exp(-0.5 * (distance_km / SIGMA_KM) ** 2).astype(np.float32)
    soft[distance_km > CUTOFF_KM] = 0.0
    soft[core_goes] = 1.0
    soft_y, soft_x = np.where(soft > 0)
    core_y, core_x = np.where(core_goes)
    hard_y, hard_x = np.where(hard_buffer)
    if len(soft_y) == 0:
        raise RuntimeError(f'Empty soft target for {row.object_id}')
    atomic_save_npz(output_path, shape=np.asarray(goes_grid['shape'], dtype=np.int32), soft_y=soft_y.astype(np.uint16), soft_x=soft_x.astype(np.uint16), soft_value=soft[soft_y, soft_x].astype(np.float16), core_y=core_y.astype(np.uint16), core_x=core_x.astype(np.uint16), core_coverage=core_coverage[core_y, core_x].astype(np.float16), hard_y=hard_y.astype(np.uint16), hard_x=hard_x.astype(np.uint16), sigma_km=np.asarray(SIGMA_KM, dtype=np.float32), hard_buffer_km=np.asarray(HARD_BUFFER_KM, dtype=np.float32), cutoff_km=np.asarray(CUTOFF_KM, dtype=np.float32), dx_km=np.asarray(goes_grid['dx_km'], dtype=np.float32), dy_km=np.asarray(goes_grid['dy_km'], dtype=np.float32))
    return {'object_id': row.object_id, 'period_id': row.period_id, 'year': int(row.year), 'birth_time': pd.Timestamp(row.birth_time), 'output_path': str(output_path), 'grid_hash': mapping['grid_hash'], 'mrms_core_pixels': int(core_mrms.sum()), 'goes_core_pixels': int(core_goes.sum()), 'hard_buffer_pixels': int(hard_buffer.sum()), 'soft_target_pixels': int(len(soft_y)), 'soft_y_min': int(soft_y.min()), 'soft_y_max': int(soft_y.max()), 'soft_x_min': int(soft_x.min()), 'soft_x_max': int(soft_x.max()), 'core_coverage_mean': float(core_coverage[core_goes].mean()), 'reference_count': int(row.reference_count)}

def main():
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--object-id')
    selection.add_argument('--all', action='store_true')
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    objects = pd.read_parquet(_ci_resolve(OBJECT_MANIFEST))
    if args.object_id:
        objects = objects[objects['object_id'] == args.object_id].copy()
        if len(objects) != 1:
            raise RuntimeError(f'Expected one object for {args.object_id}, found {len(objects)}')
    objects = objects.sort_values(['year', 'period_id', 'object_id']).reset_index(drop=True)
    goes_grid = load_and_validate_goes_grid()
    mapping_cache = {}
    result_rows = []
    for row in tqdm(list(objects.itertuples(index=False)), desc='Build CI soft targets'):
        period_root = Path(_ci_resolve(row.period_root))
        mapping = build_mapping(period_root, goes_grid, mapping_cache)
        output_path = args.output_root / f'year={int(row.year):04d}' / f'{row.object_id}.npz'
        result_rows.append(build_object_target(row, mapping, goes_grid, output_path))
    result = pd.DataFrame(result_rows)
    state = args.output_root / 'state'
    state.mkdir(parents=True, exist_ok=True)
    manifest_path = state / 'soft_target_build_manifest.parquet'
    result.to_parquet(manifest_path, index=False)
    metadata = {'objects_built': int(len(result)), 'grid_shape': list(goes_grid['shape']), 'dx_km': float(goes_grid['dx_km']), 'dy_km': float(goes_grid['dy_km']), 'sigma_km': SIGMA_KM, 'hard_buffer_km': HARD_BUFFER_KM, 'cutoff_km': CUTOFF_KM, 'unique_mrms_grids': len(mapping_cache), 'all_targets_nonempty': bool(result['soft_target_pixels'].gt(0).all())}
    metadata_path = state / 'build_summary.json'
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print('\n===== CI SOFT TARGET BUILD =====')
    for key, value in metadata.items():
        print(f'{key}: {value}')
    print('\nTarget statistics:')
    print(result[['object_id', 'mrms_core_pixels', 'goes_core_pixels', 'hard_buffer_pixels', 'soft_target_pixels', 'core_coverage_mean']].to_string(index=False))
    print('\nManifest:', manifest_path)
    print('Summary:', metadata_path)
if __name__ == '__main__':
    main()
