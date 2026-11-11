"""Resumable MRMS-only binary isolated-CI catalogue construction."""
from __future__ import annotations
from ci_nowcasting.paths import resolve_path as _ci_resolve, project_root as _ci_project, source_root as _ci_source
import argparse
import hashlib
import json
import os
import shutil
import time
from requests.exceptions import RequestException
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import zarr
from scipy import ndimage as ndi
from core import ALGORITHM_VERSION, build_periods, bounds, classify_birth, edge_distance_km, forecast_intervals, grid_spacing_km, lead_index, local_valid_fraction, match_persistence, minimum_distance_between_components, segment, stable_id, temporal_window_complete
from mrms_access import decode_downloaded, list_period_files, load_legacy
from strict_isolation import STRICT_AUDIT_COLUMNS, strict_filter_accepted

def decode_downloaded_with_retry(legacy, archive, source, temp, bbox, attempts=8):
    """Retry transient MRMS download failures with exponential backoff."""
    for attempt in range(1, attempts + 1):
        try:
            return decode_downloaded(legacy, archive, source, temp, bbox)
        except (OSError, RequestException) as exc:
            retryable = isinstance(exc, RequestException) or 'empty download' in str(exc).lower()
            if not retryable or attempt >= attempts:
                raise
            for candidate in Path(_ci_resolve(temp)).glob('*'):
                try:
                    if candidate.is_file() and candidate.stat().st_size == 0:
                        candidate.unlink()
                except FileNotFoundError:
                    pass
            delay = min(60, 5 * 2 ** (attempt - 1))
            print(f'Transient empty download for {source.filename}; retry {attempt + 1}/{attempts} in {delay}s', flush=True)
            time.sleep(delay)
    raise RuntimeError('Unreachable download retry state')

def dump(path, value):
    temporary = Path(_ci_resolve(str(path) + '.tmp'))
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)

def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()

def checked_output(cfg):
    root = Path(_ci_resolve(cfg['output_root'])).resolve()
    allowed = (Path(_ci_resolve(cfg['project_root'])) / 'data/ci_isolated_mrms_binary_v1').resolve()
    if root != allowed and allowed not in root.parents:
        raise ValueError(f'Output must stay under {allowed}')
    return root

def scan_bbox(cfg):
    aoi = cfg['aoi']
    middle = (aoi['lat_min'] + aoi['lat_max']) / 2
    lat_margin = cfg['isolation_margin_km'] / 111.19508
    lon_margin = cfg['isolation_margin_km'] / (111.19508 * np.cos(np.deg2rad(middle)))
    return [aoi['lat_min'] - lat_margin, aoi['lat_max'] + lat_margin, aoi['lon_min'] - lon_margin, aoi['lon_max'] + lon_margin]

def _expand_period(record, cfg):
    """Add issuance, detection, and source-context bounds to one period."""
    item = dict(record)
    step = f"{int(cfg['issuance_step_minutes'])}min"
    reserved = pd.Timedelta(minutes=float(cfg['reserved_forecast_minutes']))
    past = pd.Timedelta(minutes=float(cfg['past_isolation_minutes']))
    future = pd.Timedelta(minutes=float(cfg['future_confirmation_minutes']))
    candidate_start = pd.Timestamp(item['candidate_start'])
    candidate_end = pd.Timestamp(item['candidate_end'])
    reserved_issuance_start = (candidate_start - reserved).floor(step)
    issuance_end = candidate_end.floor(step)
    detection_start = reserved_issuance_start
    detection_end = issuance_end + reserved
    scan_start = detection_start - past
    scan_end = detection_end + future
    item.update(candidate_start=candidate_start, candidate_end=candidate_end, reserved_issuance_start=reserved_issuance_start, issuance_end=issuance_end, detection_start=detection_start, detection_end=detection_end, scan_start=scan_start, scan_end=scan_end)
    item['period_id'] = stable_id('PERIOD', int(item.get('year', candidate_start.year)), candidate_start.isoformat(), candidate_end.isoformat(), scan_start.isoformat(), scan_end.isoformat(), cfg.get('algorithm_version', ALGORITHM_VERSION))
    item['status'] = 'PLANNED'
    return item

def expand_and_merge_periods(periods, cfg):
    """Expand periods and merge any resulting overlapping scan intervals."""
    if periods.empty:
        return periods.copy()
    expanded = [_expand_period(record, cfg) for record in periods.to_dict(orient='records')]
    merged = []
    for year in sorted({int(record['year']) for record in expanded}):
        yearly = sorted((record for record in expanded if int(record['year']) == year), key=lambda record: pd.Timestamp(record['scan_start']))
        current = None
        for record in yearly:
            if current is None:
                current = record
                continue
            if pd.Timestamp(record['scan_start']) <= pd.Timestamp(current['scan_end']):
                current['candidate_start'] = min(pd.Timestamp(current['candidate_start']), pd.Timestamp(record['candidate_start']))
                current['candidate_end'] = max(pd.Timestamp(current['candidate_end']), pd.Timestamp(record['candidate_end']))
                left_count = current.get('candidate_timestamp_count', 0)
                right_count = record.get('candidate_timestamp_count', 0)
                left_count = 0 if pd.isna(left_count) else int(left_count)
                right_count = 0 if pd.isna(right_count) else int(right_count)
                current['candidate_timestamp_count'] = left_count + right_count
                current = _expand_period(current, cfg)
            else:
                merged.append(current)
                current = record
        if current is not None:
            merged.append(current)
    return pd.DataFrame(merged).sort_values(['year', 'candidate_start', 'candidate_end']).reset_index(drop=True)

def plan(cfg, write=True):
    source = Path(_ci_resolve(cfg['candidate_catalog']))
    catalog = pd.read_parquet(_ci_resolve(source), columns=['ci_time', 'latitude', 'longitude'])
    periods = build_periods(catalog, cfg)
    periods = expand_and_merge_periods(periods, cfg)
    years = cfg.get('years')
    if years:
        periods = periods.loc[periods.year.isin(years)].reset_index(drop=True)
    exclusion_path = Path(_ci_resolve(__file__)).with_name('excluded_periods.json')
    exclusions = json.loads(exclusion_path.read_text()) if exclusion_path.is_file() else []
    active_exclusions = []
    for exclusion in exclusions:
        excluded_id = str(exclusion['period_id'])
        excluded_year = int(exclusion['year'])
        excluded_start = pd.Timestamp(exclusion['candidate_start'])
        excluded_end = pd.Timestamp(exclusion['candidate_end'])
        match = periods['period_id'].astype(str).eq(excluded_id) | periods['year'].astype(int).eq(excluded_year) & pd.to_datetime(periods['candidate_start']).eq(excluded_start) & pd.to_datetime(periods['candidate_end']).eq(excluded_end)
        matched_count = int(match.sum())
        if matched_count:
            print(f'Excluding {matched_count} period(s): {excluded_id}')
            periods = periods.loc[~match].reset_index(drop=True)
            active_exclusions.append(exclusion)
    if cfg.get('maximum_periods'):
        periods = periods.head(int(cfg['maximum_periods'])).copy()
    periods['scan_bbox'] = json.dumps(scan_bbox(cfg))
    root = checked_output(cfg)
    if write:
        if root.exists():
            existing = json.loads((root / 'config.json').read_text())
            if existing != cfg:
                raise ValueError('Existing output has a different configuration')
        else:
            root.mkdir(parents=True)
            dump(root / 'config.json', cfg)
        if active_exclusions:
            dump(root / 'excluded_periods.json', active_exclusions)
        temporary = root / 'period_manifest.parquet.tmp'
        periods.to_parquet(temporary, index=False)
        temporary.replace(root / 'period_manifest.parquet')
    return periods

def validity_record(valid, directory):
    packed = np.packbits(np.asarray(valid, np.uint8), axis=None)
    mask_id = stable_id('VALID', list(valid.shape), hashlib.sha256(packed.tobytes()).hexdigest())
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{mask_id}.npz'
    if not path.exists():
        temporary = directory / f'.{mask_id}.{os.getpid()}.npz'
        np.savez_compressed(temporary, packed=packed, shape=np.asarray(valid.shape, np.int32))
        temporary.replace(path)
    return (mask_id, path)

def save_birth_mask(directory, object_id, component, shape):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{object_id}.npz'
    np.savez_compressed(path, y=component.ys.astype(np.int32), x=component.xs.astype(np.int32), shape=np.asarray(shape, np.int32))
    return path

def coordinate(latitude, longitude, y, x):
    iy, ix = (int(round(y)), int(round(x)))
    if np.asarray(latitude).ndim == 1:
        return (float(latitude[iy]), float(longitude[ix]))
    return (float(latitude[iy, ix]), float(longitude[iy, ix]))

def process_period(row, cfg, root, resume=False):
    final = root / 'periods' / row.period_id
    marker = final / '_SUCCESS.json'
    if marker.exists():
        recorded = json.loads(marker.read_text())
        if recorded['config_sha256'] != config_hash(cfg):
            raise ValueError(f'Resume configuration mismatch for {row.period_id}')
        return recorded
    if final.exists():
        raise RuntimeError(f'Incomplete final period exists and will not be overwritten: {final}')
    resumable_stages = list((root / 'staging').glob(f'{row.period_id}_*')) if resume else []
    resumable_stages = [value for value in resumable_stages if (value / 'download_inventory.parquet').is_file()]
    if resumable_stages:
        stage = max(resumable_stages, key=lambda value: len(pd.read_parquet(_ci_resolve(value / 'download_inventory.parquet'))))
    else:
        stage = root / 'staging' / f'{row.period_id}_{os.getpid()}'
        stage.mkdir(parents=True, exist_ok=False)
    temp = Path(_ci_resolve(cfg.get('temporary_root', os.environ.get('TMPDIR', '/tmp')))) / 'isolated_ci_binary_v1' / row.period_id
    temp.mkdir(parents=True, exist_ok=True)
    legacy = load_legacy(cfg['project_root'])
    archive = legacy.S3MRMSArchive(timeout=cfg['network_timeout_seconds'])
    listed = list_period_files(archive, row.scan_start, row.scan_end)
    if not listed:
        raise RuntimeError(f'No MRMS files listed for {row.period_id}')
    frames, valid_frames, times, inventory = ([], [], [], [])
    try:
        latitude = longitude = None
        reusable = None
        checkpoint = stage / 'download_inventory.parquet'
        if checkpoint.is_file():
            saved = pd.read_parquet(_ci_resolve(checkpoint)).sort_values('reflectivity_row')
            expected = [value.filename for value in listed[:len(saved)]]
            if saved.source_filename.tolist() != expected:
                raise ValueError('Staging checkpoint does not match current native-file listing')
            reflectivity = zarr.open_group(_ci_resolve(str(stage / 'reflectivity.zarr')), mode='r+')
            latitude = np.asarray(reflectivity['latitude'][:])
            longitude = np.asarray(reflectivity['longitude'][:])
            shape = tuple(reflectivity['reflectivity_dbz'].shape[1:])
            for record in saved.to_dict('records'):
                mask_data = np.load(_ci_resolve(record['staging_valid_mask_path']))
                valid = np.unpackbits(mask_data['packed'])[:int(np.prod(mask_data['shape']))].reshape(tuple(mask_data['shape'])).astype(bool)
                frames.append(np.asarray(reflectivity['reflectivity_dbz'][record['reflectivity_row']], np.float32))
                valid_frames.append(valid)
                times.append(pd.Timestamp(record['actual_time']))
                record.pop('staging_valid_mask_path')
                inventory.append(record)
            reusable = 'partial'
        if resume and reusable is None:
            for candidate in (root / 'staging').glob(f'{row.period_id}_*'):
                store_path = candidate / 'reflectivity.zarr'
                mask_paths = list((candidate / 'valid_masks').glob('*.npz'))
                try:
                    store = zarr.open_group(_ci_resolve(str(store_path)), mode='r')
                    exact_times = np.array_equal(np.asarray(store['time_ns'][:], np.int64), np.asarray([value.valid_time.value for value in listed], np.int64))
                    complete = store['reflectivity_dbz'].shape[0] == len(listed) and len(list(store_path.joinpath('reflectivity_dbz').glob('[0-9]*'))) == len(listed)
                    if exact_times and complete and (len(mask_paths) == 1):
                        reusable = (candidate, store, mask_paths[0])
                        break
                except Exception:
                    continue
        if isinstance(reusable, tuple):
            stage, reflectivity, mask_path = reusable
            checkpoint = stage / 'download_inventory.parquet'
            latitude = np.asarray(reflectivity['latitude'][:])
            longitude = np.asarray(reflectivity['longitude'][:])
            shape = tuple(reflectivity['reflectivity_dbz'].shape[1:])
            packed = np.load(_ci_resolve(mask_path))
            valid = np.unpackbits(packed['packed'])[:int(np.prod(packed['shape']))].reshape(tuple(packed['shape'])).astype(bool)
            mask_id = mask_path.stem
            for index, source in enumerate(listed):
                dbz = np.asarray(reflectivity['reflectivity_dbz'][index], np.float32)
                frames.append(dbz)
                valid_frames.append(valid)
                times.append(source.valid_time)
                inventory.append(dict(period_id=row.period_id, actual_time=source.valid_time, source_filename=source.filename, source_url=source.url, valid_mask_id=mask_id, valid_mask_path=str((final / 'valid_masks' / mask_path.name).resolve()), global_valid_fraction=float(valid.mean()), temporal_context_complete=False, grid_height=dbz.shape[0], grid_width=dbz.shape[1], dx_km=np.nan, dy_km=np.nan, reflectivity_path=str((final / 'reflectivity.zarr').resolve()), reflectivity_row=index))
            reusable = 'complete'
        valid_dir = stage / 'valid_masks'
        for index, source in enumerate(listed[len(inventory):], start=len(inventory)):
            source_available = True
            download_error = None
            try:
                dbz, valid, lat, lon = decode_downloaded_with_retry(legacy, archive, source, temp, scan_bbox(cfg))
            except (OSError, RequestException) as exc:
                if latitude is None:
                    raise RuntimeError(f'Cannot represent a missing MRMS scan before the reconstruction grid has been initialized: {source.filename}') from exc
                source_available = False
                download_error = f'{type(exc).__name__}: {exc}'
                dbz = np.full(shape, -999.0, dtype=np.float32)
                valid = np.zeros(shape, dtype=bool)
                lat = latitude
                lon = longitude
                print(f'Permanent MRMS download failure for {source.filename}; recording an invalid scan and continuing', flush=True)
            if latitude is None:
                latitude, longitude = (lat, lon)
                shape = dbz.shape
                reflectivity = zarr.open_group(_ci_resolve(str(stage / 'reflectivity.zarr')), mode='w')
                reflectivity.create_dataset('reflectivity_dbz', shape=(len(listed), *shape), chunks=(1, *shape), dtype='f2')
                reflectivity.create_dataset('time_ns', data=np.asarray([value.valid_time.value for value in listed], np.int64))
                reflectivity.create_dataset('latitude', data=lat)
                reflectivity.create_dataset('longitude', data=lon)
            elif dbz.shape != shape or not np.allclose(lat, latitude) or (not np.allclose(lon, longitude)):
                raise ValueError('MRMS grid changed within a reconstruction period')
            reflectivity['reflectivity_dbz'][index] = dbz.astype(np.float16)
            mask_id, mask_path = validity_record(valid, valid_dir)
            frames.append(dbz)
            valid_frames.append(valid)
            times.append(source.valid_time)
            record = dict(period_id=row.period_id, actual_time=source.valid_time, source_filename=source.filename, source_url=source.url, source_available=source_available, download_error=download_error, valid_mask_id=mask_id, valid_mask_path=str((final / 'valid_masks' / mask_path.name).resolve()), global_valid_fraction=float(valid.mean()), temporal_context_complete=False, grid_height=dbz.shape[0], grid_width=dbz.shape[1], dx_km=np.nan, dy_km=np.nan, reflectivity_path=str((final / 'reflectivity.zarr').resolve()), reflectivity_row=index)
            inventory.append(record)
            saved = pd.DataFrame([dict(value, staging_valid_mask_path=str(stage / 'valid_masks' / f"{value['valid_mask_id']}.npz")) for value in inventory])
            temporary_checkpoint = Path(_ci_resolve(str(checkpoint) + '.tmp'))
            saved.to_parquet(temporary_checkpoint, index=False)
            temporary_checkpoint.replace(checkpoint)
        times = pd.DatetimeIndex(times)
        spacing = grid_spacing_km(latitude, longitude)
        labels, objects = ([], [])
        for dbz, valid in zip(frames, valid_frames):
            lab, obj = segment(dbz, valid, cfg['threshold_dbz'], cfg['minimum_area_km2'], spacing)
            labels.append(lab)
            objects.append(obj)
        unavailable_scans = 0
        for record in inventory:
            value = record.get('source_available', True)
            if not pd.isna(value) and (not bool(value)):
                unavailable_scans += 1
        funnel = dict(native_scans_requested=len(listed), native_scans_complete=len(frames) - unavailable_scans, native_scans_unavailable=unavailable_scans, components_ge_35_and_16km2=sum((len(value) for value in objects)), objects_failing_temporal_or_spatial_qc=0, objects_with_mature_echo_in_past_30min=0, objects_with_other_mature_echo_inside_30km=0, objects_failing_persistence=0, accepted_isolated_births=0)
        audit, accepted = ([], [])
        aoi = cfg['aoi']
        for index, current_objects in enumerate(objects):
            birth_time = times[index]
            past_start = birth_time - pd.Timedelta(minutes=cfg['past_isolation_minutes'])
            confirm_end = birth_time + pd.Timedelta(minutes=cfg['persistence_minutes'] + cfg['persistence_tolerance_minutes'])
            past_indices = [j for j in range(index) if times[j] >= past_start]
            support_indices = [j for j, value in enumerate(times) if past_start <= value <= confirm_end]
            temporal = temporal_window_complete(times, past_start, confirm_end, cfg['maximum_tracking_gap_minutes'], cfg['endpoint_tolerance_minutes'])
            past_echo = np.logical_or.reduce([labels[j] > 0 for j in past_indices]) if past_indices else None
            past_distance_field = ndi.distance_transform_edt(~past_echo, sampling=spacing) if past_echo is not None and past_echo.any() else None
            for component in current_objects.values():
                lat, lon = coordinate(latitude, longitude, component.centroid_y, component.centroid_x)
                past_distance = float('nan') if not past_indices else float('inf') if past_distance_field is None else float(past_distance_field[component.ys, component.xs].min())
                other_components = [other for other in current_objects.values() if other.label_id != component.label_id]
                same_distance = min((minimum_distance_between_components(component, other, spacing) for other in other_components), default=float('inf'))
                local_valid = local_valid_fraction(component, [valid_frames[j] for j in support_indices], cfg['isolation_radius_km'], spacing) >= cfg['minimum_local_valid_fraction'] if support_indices else False
                preliminary, _ = classify_birth(aoi['lat_min'] <= lat <= aoi['lat_max'] and aoi['lon_min'] <= lon <= aoi['lon_max'], temporal, edge_distance_km(component, frames[0].shape, spacing) >= cfg['isolation_radius_km'], local_valid, past_distance, same_distance, True, cfg['isolation_radius_km'])
                confirmation_index = None
                if preliminary:
                    confirmation_index = match_persistence(component, index, times, labels, objects, spacing, cfg)
                accepted_birth, reason = classify_birth(aoi['lat_min'] <= lat <= aoi['lat_max'] and aoi['lon_min'] <= lon <= aoi['lon_max'], temporal, edge_distance_km(component, frames[0].shape, spacing) >= cfg['isolation_radius_km'], local_valid, past_distance, same_distance, confirmation_index is not None, cfg['isolation_radius_km'])
                if reason in ['outside_requested_aoi', 'temporal_or_spatial_support', 'local_radar_missingness']:
                    funnel['objects_failing_temporal_or_spatial_qc'] += 1
                elif reason == 'mature_echo_in_past_30min':
                    funnel['objects_with_mature_echo_in_past_30min'] += 1
                elif reason == 'other_mature_echo_inside_30km':
                    funnel['objects_with_other_mature_echo_inside_30km'] += 1
                elif reason == 'persistence_not_confirmed':
                    funnel['objects_failing_persistence'] += 1
                object_id = stable_id('ICI', ALGORITHM_VERSION, birth_time.isoformat(), hashlib.sha256(np.stack([component.ys, component.xs], axis=1).tobytes()).hexdigest(), list(frames[0].shape), spacing)
                audit.append(dict(period_id=row.period_id, object_id=object_id, birth_time=birth_time, birth_y_px=component.centroid_y, birth_x_px=component.centroid_x, latitude=lat, longitude=lon, area_km2=component.area_km2, area_px=component.area_px, max_dbz=component.max_dbz, mean_dbz=component.mean_dbz, past_min_mature_distance_km=past_distance, same_time_min_mature_distance_km=same_distance, binary_target=1 if accepted_birth else 0, diagnostic_rejection_reason=reason))
                if not accepted_birth:
                    continue
                confirmation_time = times[confirmation_index]
                path = save_birth_mask(stage / 'birth_masks', object_id, component, frames[0].shape)
                accepted.append(dict(object_id=object_id, period_id=row.period_id, birth_time=birth_time, confirmation_time=confirmation_time, birth_y_px=component.centroid_y, birth_x_px=component.centroid_x, latitude=lat, longitude=lon, area_km2=component.area_km2, area_px=component.area_px, max_dbz=component.max_dbz, mean_dbz=component.mean_dbz, past_min_mature_distance_km=past_distance, same_time_min_mature_distance_km=same_distance, source_birth=listed[index].filename, source_confirmation=listed[confirmation_index].filename, native_birth_mask_path=str((final / 'birth_masks' / path.name).resolve()), algorithm_version=ALGORITHM_VERSION))
                funnel['accepted_isolated_births'] += 1
        for record in inventory:
            center = pd.Timestamp(record['actual_time'])
            record['temporal_context_complete'] = temporal_window_complete(times, center - pd.Timedelta(minutes=cfg['past_isolation_minutes']), center + pd.Timedelta(minutes=cfg['reserved_forecast_minutes']), cfg['maximum_tracking_gap_minutes'], cfg['endpoint_tolerance_minutes'])
            record['dy_km'], record['dx_km'] = spacing
        accepted, strict_audit = strict_filter_accepted(frames=frames, valid_frames=valid_frames, times=times, spacing_km=spacing, accepted=accepted, cfg=cfg, period_id=row.period_id, funnel=funnel)
        strict_by_id = {value['object_id']: value for value in strict_audit}
        for candidate in audit:
            object_id = candidate.get('object_id')
            if object_id in strict_by_id:
                candidate.update(strict_by_id[object_id])
        pd.DataFrame(strict_audit, columns=STRICT_AUDIT_COLUMNS).to_parquet(stage / 'strict_isolation_audit.parquet', index=False)
        pd.DataFrame(inventory).to_parquet(stage / 'frame_inventory.parquet', index=False)
        pd.DataFrame(accepted, columns=cfg['catalog_columns']).to_parquet(stage / 'isolated_ci_catalog.parquet', index=False)
        pd.DataFrame(audit).to_parquet(stage / 'candidate_audit.parquet', index=False)
        dump(stage / 'detection_funnel.json', funnel)
        marker_value = dict(period_id=row.period_id, config_sha256=config_hash(cfg), completed_utc=datetime.now(timezone.utc).isoformat(), **funnel)
        dump(stage / '_SUCCESS.json', marker_value)
        final.parent.mkdir(parents=True, exist_ok=True)
        stage.replace(final)
        return marker_value
    finally:
        shutil.rmtree(temp, ignore_errors=True)

def required_frame_rows(frames, start, end, cfg):
    """Return all radar-validity masks needed to verify one target interval."""
    times = pd.DatetimeIndex(frames.actual_time)
    before = np.flatnonzero(times <= start)
    after = np.flatnonzero(times >= end)
    if not len(before) or not len(after):
        return (frames.iloc[0:0], False)
    lo, hi = (int(before[-1]), int(after[0]))
    selected = frames.iloc[lo:hi + 1]
    complete = temporal_window_complete(times, start, end, cfg['maximum_tracking_gap_minutes'], cfg['endpoint_tolerance_minutes'])
    return (selected, complete)

def issuance_bounds(period, cfg):
    """Return globally aligned issuance bounds for the requested horizon."""
    step = f"{int(cfg['issuance_step_minutes'])}min"
    horizon = pd.Timedelta(minutes=float(cfg['max_forecast_minutes']))
    start = (pd.Timestamp(period.candidate_start) - horizon).floor(step)
    end = pd.Timestamp(period.candidate_end).floor(step)
    return (start, end)

def label_required_bounds(issuance, lead_start, lead_end, cfg):
    """Return the complete context needed to prove a lead-bin target."""
    birth_window_start = issuance + pd.Timedelta(minutes=lead_start)
    birth_window_end = issuance + pd.Timedelta(minutes=lead_end)
    required_start = birth_window_start - pd.Timedelta(minutes=float(cfg['past_isolation_minutes']))
    required_end = birth_window_end + pd.Timedelta(minutes=float(cfg['future_confirmation_minutes']))
    return (required_start, required_end)

def build_label_index(cfg, frames, objects, periods):
    """Build forecast intervals from an immutable physical-object catalogue."""
    labels = []
    completed = set(frames.period_id.unique())
    for period in periods.loc[periods.period_id.isin(completed)].itertuples(index=False):
        pf = frames.loc[frames.period_id.eq(period.period_id)].sort_values('actual_time')
        po = objects.loc[objects.period_id.eq(period.period_id)]
        start, end = issuance_bounds(period, cfg)
        for issuance in pd.date_range(start, end, freq=f"{cfg['issuance_step_minutes']}min"):
            for index, lead_start, lead_end in forecast_intervals(cfg['forecast_bin_minutes'], cfg['max_forecast_minutes']):
                interval_start = issuance + pd.Timedelta(minutes=lead_start)
                interval_end = issuance + pd.Timedelta(minutes=lead_end)
                required_start, required_end = label_required_bounds(issuance, lead_start, lead_end, cfg)
                required, temporal = required_frame_rows(pf, required_start, required_end, cfg)
                hit = po.loc[(po.birth_time > interval_start) & (po.birth_time <= interval_end), 'object_id'].tolist()
                required_mask_ids = list(dict.fromkeys(required.valid_mask_id.astype(str).tolist()))
                labels.append(dict(period_id=period.period_id, issuance_time=issuance, lead_index=index, lead_start_minutes=lead_start, lead_end_minutes=lead_end, positive_object_ids=json.dumps(hit), positive_count=len(hit), target_temporally_valid=bool(temporal), required_valid_mask_ids=json.dumps(required_mask_ids)))
    return pd.DataFrame(labels)

def aggregate(cfg, root, periods):
    successful = []
    for row in periods.itertuples(index=False):
        directory = root / 'periods' / row.period_id
        if (directory / '_SUCCESS.json').exists():
            successful.append((row, directory))
    if not successful:
        raise RuntimeError('No completed periods to aggregate')
    if len(successful) != len(periods):
        completed_ids = {row.period_id for row, _ in successful}
        missing_ids = [str(period_id) for period_id in periods.period_id if period_id not in completed_ids]
        preview = ', '.join(missing_ids[:10])
        raise RuntimeError(f'Refusing to finalize incomplete dataset: {len(successful)}/{len(periods)} periods completed; {len(missing_ids)} missing. First missing: {preview}')
    frames = pd.concat([pd.read_parquet(_ci_resolve(directory / 'frame_inventory.parquet')) for _, directory in successful], ignore_index=True)
    objects = pd.concat([pd.read_parquet(_ci_resolve(directory / 'isolated_ci_catalog.parquet')) for _, directory in successful], ignore_index=True)
    audits = pd.concat([pd.read_parquet(_ci_resolve(directory / 'candidate_audit.parquet')) for _, directory in successful], ignore_index=True)
    if objects.object_id.duplicated().any():
        raise ValueError('Duplicate accepted physical birth ID across periods')
    label_index = build_label_index(cfg, frames, objects, periods)
    frames.to_parquet(root / 'frame_inventory.parquet', index=False)
    objects.to_parquet(root / 'isolated_ci_catalog.parquet', index=False)
    audits.to_parquet(root / 'candidate_audit.parquet', index=False)
    label_index.to_parquet(root / 'label_index.parquet', index=False)
    totals = {key: 0 for key in ['native_scans_requested', 'native_scans_complete', 'native_scans_unavailable', 'components_ge_35_and_16km2', 'objects_failing_temporal_or_spatial_qc', 'objects_with_mature_echo_in_past_30min', 'objects_with_other_mature_echo_inside_30km', 'objects_failing_persistence', 'accepted_isolated_births']}
    for _, directory in successful:
        value = json.loads((directory / 'detection_funnel.json').read_text())
        for key in totals:
            totals[key] += int(value.get(key, 0))
    totals.update(completed_periods=len(successful), planned_periods=len(periods))
    dump(root / 'detection_funnel.json', totals)
    updated = periods.copy()
    updated['status'] = updated.period_id.map({row.period_id: 'COMPLETE' for row, _ in successful}).fillna('PLANNED')
    updated.to_parquet(root / 'period_manifest.parquet', index=False)
    card = ['# MRMS-only binary isolated-CI ground truth', '', f'Algorithm: `{ALGORITHM_VERSION}`', '', 'Target 1 is the observed birth core of an accepted isolated CI. Every other correctly observed pixel is target 0. Radar missingness is represented by referenced validity masks and never silently converted to zero.', '', 'The physical catalogue is independent of GOES, HRRR, forecast patches, lead-bin geometry, and train/validation/test splits. The legacy potential-CI catalogue is used only to recover periods for native-cadence MRMS rescanning.', '', f'Completed periods: {len(successful)}/{len(periods)}; native frames: {len(frames)}; accepted isolated births: {len(objects)}.', '', f"Labels use {cfg['forecast_bin_minutes']}-minute intervals through {cfg['max_forecast_minutes']} minutes. They can be regenerated through 120 minutes from the same object catalogue when the reserved source support exists.", '', 'MRMS product: MergedReflectivityQCComposite_00.50 with the tested fallback. Threshold: valid pixels >=35 dBZ; minimum physical area: 16 km2; baseline isolation: 30 km over the preceding 30 minutes.', '', 'No model training tensors, GOES histories, HRRR predictors, or dataset splits are created by this workflow.']
    (root / 'DATASET_CARD.md').write_text('\n'.join(card) + '\n')
    return totals

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--year', type=int)
    parser.add_argument('--period-id')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--finalize', action='store_true')
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    root = checked_output(cfg)
    if not args.plan and (not args.dry_run) and (root / 'period_manifest.parquet').exists():
        if json.loads((root / 'config.json').read_text()) != cfg:
            raise ValueError('Existing output has a different configuration')
        periods = pd.read_parquet(_ci_resolve(root / 'period_manifest.parquet'))
    else:
        periods = plan(cfg, write=not args.dry_run)
    if args.dry_run:
        print(periods.to_string(index=False))
        print(f'planned_periods={len(periods)}')
        return
    if args.plan and (not (args.year or args.period_id or args.finalize or args.validate)):
        print(f"Wrote {len(periods)} periods to {root / 'period_manifest.parquet'}")
        return
    if args.validate:
        from validate import validate
        validate(root, cfg)
        return
    selected = periods
    if args.year is not None:
        selected = selected.loc[selected.year.eq(args.year)]
    if args.period_id:
        selected = selected.loc[selected.period_id.eq(args.period_id)]
    if not args.finalize:
        for row in selected.itertuples(index=False):
            print(json.dumps(process_period(row, cfg, root, args.resume), indent=2), flush=True)
    if args.finalize or (args.year is None and (not args.period_id)):
        totals = aggregate(cfg, root, periods)
        from validate import validate
        validate(root, cfg)
        print(json.dumps(totals, indent=2))
if __name__ == '__main__':
    main()
