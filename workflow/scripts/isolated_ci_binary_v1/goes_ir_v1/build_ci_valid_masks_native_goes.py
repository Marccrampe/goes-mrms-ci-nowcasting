from ci_nowcasting.paths import resolve_path as _ci_resolve, project_root as _ci_project, source_root as _ci_source
import argparse
import importlib.util
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
MIN_VALID_FRACTION = 0.75

def load_soft_target_module(path):
    spec = importlib.util.spec_from_file_location('soft_target_builder', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def atomic_save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(_ci_resolve(str(path) + '.tmp'))
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)

def decode_valid_mask(builder, path, expected_shape):
    """Decode packed valid masks or fall back to the object-mask decoder."""
    path = Path(_ci_resolve(path))
    expected_shape = tuple((int(v) for v in expected_shape))
    with np.load(_ci_resolve(path), allow_pickle=False) as archive:
        keys = set(archive.files)
        if 'packed' in keys:
            if 'shape' not in keys:
                raise RuntimeError(f'Packed mask has no shape metadata: {path}')
            stored_shape = tuple((int(value) for value in np.asarray(archive['shape']).ravel()))
            packed = np.asarray(archive['packed'], dtype=np.uint8).ravel()
            pixel_count = int(np.prod(stored_shape))
            unpacked = np.unpackbits(packed)
            if unpacked.size < pixel_count:
                raise RuntimeError(f'Packed mask is too short: {path}; bits={unpacked.size}, expected={pixel_count}')
            mask = unpacked[:pixel_count].reshape(stored_shape).astype(bool)
            if mask.shape == expected_shape:
                return mask
            if mask.T.shape == expected_shape:
                return mask.T
            raise RuntimeError(f'Packed-mask shape mismatch: {path}; stored={mask.shape}, expected={expected_shape}')
    return builder.decode_mask(path, expected_shape)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--soft-target-builder', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--min-valid-fraction', type=float, default=MIN_VALID_FRACTION)
    args = parser.parse_args()
    manifest = pd.read_parquet(_ci_resolve(args.manifest))
    required_columns = {'year', 'period_id', 'valid_mask_id', 'valid_mask_path', 'valid_mask_exists'}
    missing_columns = required_columns - set(manifest.columns)
    if missing_columns:
        raise RuntimeError(f'Manifest missing columns: {sorted(missing_columns)}')
    if len(manifest) != 599:
        raise RuntimeError(f'Expected 599 valid masks, found {len(manifest)}')
    if not manifest['valid_mask_exists'].all():
        missing = manifest.loc[~manifest['valid_mask_exists'], 'valid_mask_path']
        raise FileNotFoundError(f'Missing valid-mask files:\n{missing.to_string(index=False)}')
    duplicate_columns = ['year', 'period_id', 'valid_mask_id']
    if manifest.duplicated(duplicate_columns).any():
        raise RuntimeError('Duplicated period/valid-mask combinations')
    builder = load_soft_target_module(args.soft_target_builder)
    print('Loading and validating native GOES grid...')
    goes_grid = builder.load_and_validate_goes_grid()
    if 'shape' not in goes_grid:
        raise RuntimeError(f'GOES grid lacks shape; keys={sorted(goes_grid)}')
    goes_shape = tuple((int(value) for value in goes_grid['shape']))
    mapping_cache = {}
    output_rows = []
    for row in tqdm(manifest.itertuples(index=False), total=len(manifest), desc='Project valid masks'):
        source_path = Path(_ci_resolve(row.valid_mask_path))
        period_root = source_path.parent.parent
        mapping = builder.build_mapping(period_root, goes_grid, mapping_cache)
        expected_mapping_keys = {'source_y', 'source_x', 'target_y', 'target_x'}
        missing_mapping_keys = expected_mapping_keys - set(mapping)
        if missing_mapping_keys:
            raise RuntimeError(f'Projection mapping lacks keys {sorted(missing_mapping_keys)}; available={sorted(mapping)}')
        shape_key = 'mrms_shape' if 'mrms_shape' in mapping else 'expected_shape' if 'expected_shape' in mapping else None
        if shape_key is None:
            raise RuntimeError(f'Projection mapping contains no MRMS shape; available={sorted(mapping)}')
        mrms_shape = tuple((int(value) for value in mapping[shape_key]))
        valid_mrms = decode_valid_mask(builder, source_path, mrms_shape).astype(bool)
        source_y = np.asarray(mapping['source_y'], dtype=np.int64)
        source_x = np.asarray(mapping['source_x'], dtype=np.int64)
        target_y = np.asarray(mapping['target_y'], dtype=np.int64)
        target_x = np.asarray(mapping['target_x'], dtype=np.int64)
        if 'total_count' in mapping:
            total_count = np.asarray(mapping['total_count'], dtype=np.int32).copy()
        else:
            total_count = np.zeros(goes_shape, dtype=np.int32)
            np.add.at(total_count, (target_y, target_x), 1)
        valid_count = np.zeros(goes_shape, dtype=np.int32)
        np.add.at(valid_count, (target_y, target_x), valid_mrms[source_y, source_x].astype(np.int32))
        has_source = total_count > 0
        valid_fraction = np.zeros(goes_shape, dtype=np.float32)
        valid_fraction[has_source] = valid_count[has_source] / total_count[has_source]
        valid_goes = has_source & (valid_fraction >= float(args.min_valid_fraction))
        projected_mask_empty = not bool(valid_goes.any())
        output_path = args.output_root / f'year={int(row.year):04d}' / str(row.period_id) / f'{row.valid_mask_id}.npz'
        atomic_save_npz(output_path, valid_mask=valid_goes.astype(np.uint8), valid_fraction=valid_fraction.astype(np.float16), shape=np.asarray(goes_shape, dtype=np.int32))
        output_rows.append({'year': int(row.year), 'period_id': str(row.period_id), 'valid_mask_id': str(row.valid_mask_id), 'source_valid_mask_path': str(source_path), 'projected_valid_mask_path': str(output_path), 'valid_pixels': int(valid_goes.sum()), 'total_pixels': int(valid_goes.size), 'valid_fraction_of_grid': float(valid_goes.mean()), 'minimum_source_coverage': float(args.min_valid_fraction), 'source_valid_pixels': int(valid_mrms.sum()), 'projected_mask_empty': bool(projected_mask_empty), 'output_exists': output_path.exists()})
    output_manifest = pd.DataFrame(output_rows)
    state_root = args.output_root / 'state'
    state_root.mkdir(parents=True, exist_ok=True)
    output_manifest_path = state_root / 'projected_valid_mask_manifest.parquet'
    output_manifest.to_parquet(output_manifest_path, index=False)
    if len(output_manifest) != 599:
        raise RuntimeError('Incorrect output-manifest length')
    if not output_manifest['output_exists'].all():
        raise RuntimeError('Some output masks are missing')
    summary = {'passed': True, 'masks_built': int(len(output_manifest)), 'empty_source_masks': int((output_manifest['source_valid_pixels'] == 0).sum()), 'empty_projected_masks': int(output_manifest['projected_mask_empty'].sum()), 'unique_periods': int(output_manifest['period_id'].nunique()), 'grid_shape': list(goes_shape), 'minimum_source_coverage': float(args.min_valid_fraction), 'valid_pixels_min': int(output_manifest['valid_pixels'].min()), 'valid_pixels_median': float(output_manifest['valid_pixels'].median()), 'valid_pixels_max': int(output_manifest['valid_pixels'].max()), 'valid_grid_fraction_min': float(output_manifest['valid_fraction_of_grid'].min()), 'valid_grid_fraction_median': float(output_manifest['valid_fraction_of_grid'].median()), 'valid_grid_fraction_max': float(output_manifest['valid_fraction_of_grid'].max())}
    summary_path = state_root / 'build_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2))
    print('\n===== PROJECTED VALID MASKS =====')
    for key, value in summary.items():
        print(f'{key}: {value}')
    print('\nManifest:', output_manifest_path)
    print('Summary:', summary_path)
if __name__ == '__main__':
    main()
