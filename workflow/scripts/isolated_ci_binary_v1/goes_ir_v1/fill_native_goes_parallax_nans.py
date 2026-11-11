"""Nearest-fill spatial NaN holes in monthly native-grid GOES stores.

The original stores are never modified. Timestamps containing an entirely
invalid channel are excluded; no temporal fallback is permitted.
"""
from __future__ import annotations
from ci_nowcasting.paths import resolve_path as _ci_resolve, project_root as _ci_project, source_root as _ci_source
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import zarr
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm
DEFAULT_INPUT_ROOT = Path(_ci_resolve(_ci_resolve('/crest/students/mcrampe00/PDE_CNN_precip/data/ci_isolated_mrms_binary_v1/goes_ir_parallax_native_v1')))
DEFAULT_OUTPUT_ROOT = Path(_ci_resolve(_ci_resolve('/crest/students/mcrampe00/PDE_CNN_precip/data/ci_isolated_mrms_binary_v1/goes_ir_parallax_native_filled_v1')))

def copy_attrs(source, destination):
    for key, value in source.attrs.asdict().items():
        destination.attrs[key] = value

def create_like(group, name, source, shape):
    chunks = source.chunks
    if chunks and shape:
        chunks = tuple((min(int(chunk), int(size)) if int(size) > 0 else int(chunk) for chunk, size in zip(chunks, shape)))
    kwargs = {'shape': shape, 'dtype': source.dtype, 'chunks': chunks, 'compressor': source.compressor, 'fill_value': source.fill_value, 'order': source.order, 'overwrite': True}
    if source.filters:
        kwargs['filters'] = source.filters
    output = group.create_dataset(name, **kwargs)
    copy_attrs(source, output)
    return output

def fill_frame(frame):
    """Fill spatial holes without using another timestamp."""
    frame = np.asarray(frame, dtype=np.float32)
    if frame.ndim != 3:
        raise ValueError(f'Expected (channel,y,x), got {frame.shape}')
    n_channels, height, width = frame.shape
    valid = np.isfinite(frame)
    channel_has_data = valid.reshape(n_channels, -1).any(axis=1)
    if not channel_has_data.all():
        return None
    filled = frame.copy()
    missing_bits = np.zeros((height, width), dtype=np.uint8)
    filled_fraction = np.zeros(n_channels, dtype=np.float32)
    max_distance = np.zeros(n_channels, dtype=np.float32)
    mask_groups = {}
    for channel_index in range(n_channels):
        invalid = ~valid[channel_index]
        if not invalid.any():
            continue
        missing_bits[invalid] |= np.uint8(1 << channel_index)
        filled_fraction[channel_index] = float(invalid.mean())
        key = np.packbits(invalid.ravel()).tobytes()
        mask_groups.setdefault(key, []).append(channel_index)
    for channel_indices in mask_groups.values():
        reference_channel = channel_indices[0]
        invalid = ~valid[reference_channel]
        distances, nearest_indices = distance_transform_edt(invalid, return_distances=True, return_indices=True)
        group_max_distance = float(distances[invalid].max())
        for channel_index in channel_indices:
            channel_invalid = ~valid[channel_index]
            if not np.array_equal(channel_invalid, invalid):
                distances_c, indices_c = distance_transform_edt(channel_invalid, return_distances=True, return_indices=True)
                nearest_values = frame[channel_index][tuple(indices_c)]
                filled[channel_index, channel_invalid] = nearest_values[channel_invalid]
                max_distance[channel_index] = float(distances_c[channel_invalid].max())
            else:
                nearest_values = frame[channel_index][tuple(nearest_indices)]
                filled[channel_index, invalid] = nearest_values[invalid]
                max_distance[channel_index] = group_max_distance
    if not np.isfinite(filled).all():
        raise RuntimeError('Spatial filling left non-finite values')
    if not np.array_equal(filled[valid], frame[valid]):
        raise RuntimeError('Filling modified originally valid values')
    return (filled, missing_bits, filled_fraction, max_distance)

def process_month(args):
    input_root = Path(_ci_resolve(args.input_root)).resolve()
    output_root = Path(_ci_resolve(args.output_root)).resolve()
    if input_root == output_root:
        raise ValueError('Input and output roots must differ')
    source_path = input_root / f'year={args.year:04d}' / f'month={args.month:02d}' / 'goes_ir.zarr'
    destination = output_root / f'year={args.year:04d}' / f'month={args.month:02d}' / 'goes_ir.zarr'
    partial = destination.with_name('goes_ir.zarr.partial')
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if destination.exists():
        if not args.overwrite:
            print(f'Already exists: {destination}')
            return
        shutil.rmtree(destination)
    if partial.exists():
        shutil.rmtree(partial)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = zarr.open_group(_ci_resolve(str(source_path)), mode='r')
    source_cmi = source['CMI']
    if source_cmi.ndim != 4:
        raise ValueError(f'Expected CMI(time,channel,y,x), got {source_cmi.shape}')
    n_time, n_channels, height, width = source_cmi.shape
    if n_channels != 7:
        raise ValueError(f'Expected seven channels, got {n_channels}')
    output = zarr.open_group(_ci_resolve(str(partial)), mode='w')
    copy_attrs(source, output)
    output.attrs.update({'nan_filled': True, 'nan_fill_method': '2-D Euclidean nearest finite neighbour on native GOES grid', 'temporal_fallback_used': False, 'source_store': str(source_path)})
    output_cmi = output.create_dataset('CMI', shape=(n_time, n_channels, height, width), dtype=source_cmi.dtype, chunks=source_cmi.chunks, compressor=source_cmi.compressor, filters=source_cmi.filters, fill_value=source_cmi.fill_value, order=source_cmi.order, overwrite=True)
    copy_attrs(source_cmi, output_cmi)
    output_cmi.attrs['nan_filled'] = True
    missing_bits = output.create_dataset('CMI_missing_bits', shape=(n_time, height, width), dtype='uint8', chunks=(1, height, width), compressor=source_cmi.compressor, fill_value=0, overwrite=True)
    missing_bits.attrs.update({'_ARRAY_DIMENSIONS': ['time', 'y', 'x'], 'long_name': 'Original CMI missingness before spatial filling; bits 0-6 correspond to channel indices 0-6', 'flag_meanings': 'bit0_C08 bit1_C09 bit2_C10 bit3_C11 bit4_C12 bit5_C13 bit6_C14'})
    metric_chunks = (min(256, max(1, n_time)), n_channels)
    filled_fraction_array = output.create_dataset('filled_fraction', shape=(n_time, n_channels), dtype='float32', chunks=metric_chunks, compressor=source_cmi.compressor, fill_value=np.nan, overwrite=True)
    filled_fraction_array.attrs.update({'_ARRAY_DIMENSIONS': ['time', 'channel'], 'long_name': 'Fraction of native GOES pixels spatially filled', 'units': '1'})
    max_distance_array = output.create_dataset('max_fill_distance_pixels', shape=(n_time, n_channels), dtype='float32', chunks=metric_chunks, compressor=source_cmi.compressor, fill_value=np.nan, overwrite=True)
    max_distance_array.attrs.update({'_ARRAY_DIMENSIONS': ['time', 'channel'], 'long_name': 'Maximum nearest-fill distance', 'units': 'native GOES pixels'})
    source_time_index = output.create_dataset('source_time_index', shape=(n_time,), dtype='int64', chunks=(min(1024, max(1, n_time)),), compressor=source_cmi.compressor, fill_value=-1, overwrite=True)
    source_time_index.attrs.update({'_ARRAY_DIMENSIONS': ['time'], 'long_name': 'Time index in the unfilled source monthly store'})
    retained_indices = []
    excluded_all_invalid_indices = []
    excluded_quality_indices = []
    output_index = 0
    total_missing_pixels = 0
    maximum_distance_seen = 0.0
    source_times_for_sort = np.asarray(source['time'][:], dtype=np.int64)
    if len(np.unique(source_times_for_sort)) != n_time:
        raise RuntimeError('Source monthly store contains duplicate timestamps')
    chronological_input_indices = np.argsort(source_times_for_sort, kind='stable')
    for input_index in tqdm(chronological_input_indices.tolist(), desc=f'Fill {args.year:04d}-{args.month:02d}'):
        raw_frame = np.asarray(source_cmi[input_index], dtype=np.float32)
        result = fill_frame(raw_frame)
        if result is None:
            excluded_all_invalid_indices.append(input_index)
            continue
        filled, frame_missing_bits, filled_fraction, max_distance = result
        if float(np.max(filled_fraction)) > float(args.max_filled_fraction) or float(np.max(max_distance)) > float(args.max_fill_distance_pixels):
            excluded_quality_indices.append(input_index)
            continue
        filled_for_storage = filled.astype(source_cmi.dtype, copy=False)
        if not np.isfinite(filled_for_storage).all():
            raise RuntimeError(f'Non-finite output at source index {input_index}')
        output_cmi[output_index] = filled_for_storage
        missing_bits[output_index] = frame_missing_bits
        filled_fraction_array[output_index] = filled_fraction
        max_distance_array[output_index] = max_distance
        source_time_index[output_index] = input_index
        retained_indices.append(input_index)
        total_missing_pixels += int(np.count_nonzero(frame_missing_bits))
        maximum_distance_seen = max(maximum_distance_seen, float(max_distance.max()))
        output_index += 1
    if output_index == 0:
        raise RuntimeError('No usable timestamps were retained')
    output_cmi.resize(output_index, n_channels, height, width)
    missing_bits.resize(output_index, height, width)
    filled_fraction_array.resize(output_index, n_channels)
    max_distance_array.resize(output_index, n_channels)
    source_time_index.resize(output_index)
    retained_indices_np = np.asarray(retained_indices, dtype=np.int64)
    for name, source_array in source.arrays():
        if name == 'CMI':
            continue
        dimensions = list(source_array.attrs.get('_ARRAY_DIMENSIONS', []))
        if 'time' in dimensions and (not dimensions or dimensions[0] != 'time'):
            raise RuntimeError(f'Unsupported non-leading time dimension for {name}: {dimensions}')
        if dimensions and dimensions[0] == 'time':
            new_shape = (output_index, *source_array.shape[1:])
            destination_array = create_like(output, name, source_array, new_shape)
            all_values = np.asarray(source_array[:])
            destination_array[:] = all_values[retained_indices_np]
        else:
            destination_array = create_like(output, name, source_array, source_array.shape)
            if source_array.shape == ():
                destination_array[()] = source_array[()]
            else:
                destination_array[:] = source_array[:]
    zarr.consolidate_metadata(str(partial))
    verification = zarr.open_group(_ci_resolve(str(partial)), mode='r')
    if verification['CMI'].shape[0] != output_index:
        raise RuntimeError('Final CMI length verification failed')
    if verification['time'].shape[0] != output_index:
        raise RuntimeError('Final time length verification failed')
    output_times = np.asarray(verification['time'][:])
    if len(np.unique(output_times)) != len(output_times):
        raise RuntimeError('Output contains duplicate timestamps')
    if len(output_times) > 1 and np.any(np.diff(output_times.astype(np.int64)) <= 0):
        raise RuntimeError('Output timestamps are not strictly increasing')
    partial.replace(destination)
    excluded_indices = sorted(excluded_all_invalid_indices + excluded_quality_indices)
    report = {'input_store': str(source_path), 'output_store': str(destination), 'source_timestamps': int(n_time), 'retained_timestamps': int(output_index), 'excluded_all_invalid_channel_timestamps': int(len(excluded_all_invalid_indices)), 'excluded_quality_threshold_timestamps': int(len(excluded_quality_indices)), 'excluded_source_indices': excluded_indices, 'excluded_all_invalid_source_indices': excluded_all_invalid_indices, 'excluded_quality_source_indices': excluded_quality_indices, 'max_allowed_filled_fraction': float(args.max_filled_fraction), 'max_allowed_fill_distance_pixels': float(args.max_fill_distance_pixels), 'channels': int(n_channels), 'grid_shape': [int(height), int(width)], 'output_dtype': str(source_cmi.dtype), 'temporal_fallback_used': False, 'all_retained_frames_verified_finite': True, 'total_pixel_locations_with_any_missing_channel': int(total_missing_pixels), 'maximum_fill_distance_pixels': float(maximum_distance_seen)}
    report_path = destination.parent / 'nan_fill_report.json'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('\n===== NAN-FILL COMPLETE =====')
    print('Input:', source_path)
    print('Output:', destination)
    print('Source timestamps:', n_time)
    print('Retained timestamps:', output_index)
    print('Excluded all-invalid timestamps:', len(excluded_all_invalid_indices))
    print('Excluded by quality thresholds:', len(excluded_quality_indices))
    print('Maximum fill distance:', round(maximum_distance_seen, 3), 'pixels')
    print('Report:', report_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--month', type=int, required=True)
    parser.add_argument('--input-root', default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument('--output-root', default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--max-filled-fraction', type=float, default=0.08)
    parser.add_argument('--max-fill-distance-pixels', type=float, default=4.0)
    args = parser.parse_args()
    process_month(args)
if __name__ == '__main__':
    main()
