"""Pure scientific primitives for the MRMS-only binary isolated-CI catalogue."""
from __future__ import annotations
from ci_nowcasting.paths import resolve_path as _ci_resolve, project_root as _ci_project, source_root as _ci_source
import hashlib
import json
import math
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
ALGORITHM_VERSION = 'isolated_ci_binary_mrms_v2_strict'

def stable_id(prefix, *values):
    payload = json.dumps(values, sort_keys=True, default=str).encode()
    return f'{prefix}_{hashlib.sha256(payload).hexdigest()[:24]}'

def grid_spacing_km(latitude, longitude):
    lat, lon = (np.asarray(latitude), np.asarray(longitude))
    if lat.ndim == 1 and lon.ndim == 1:
        dy = float(np.nanmedian(np.abs(np.diff(lat))) * 111.19508)
        dx = float(np.nanmedian(np.abs(np.diff(lon))) * 111.19508 * np.cos(np.deg2rad(np.nanmedian(lat))))
    elif lat.ndim == 2 and lon.ndim == 2:
        dy = float(np.nanmedian(np.abs(np.diff(lat, axis=0))) * 111.19508)
        dx = float(np.nanmedian(np.abs(np.diff(lon, axis=1))) * 111.19508 * np.cos(np.deg2rad(np.nanmedian(lat))))
    else:
        raise ValueError('Latitude and longitude must both be 1-D or both be 2-D')
    if not (np.isfinite(dx) and np.isfinite(dy) and (dx > 0) and (dy > 0)):
        raise ValueError(f'Invalid physical grid spacing dx={dx}, dy={dy}')
    return (dy, dx)

@dataclass
class Component:
    label_id: int
    ys: np.ndarray
    xs: np.ndarray
    area_px: int
    area_km2: float
    centroid_y: float
    centroid_x: float
    max_dbz: float
    mean_dbz: float

def segment(frame, valid, threshold_dbz, minimum_area_km2, spacing_km):
    """Segment 8-connected valid pixels with the explicit >= operator."""
    frame, valid = (np.asarray(frame), np.asarray(valid, dtype=bool))
    dy, dx = spacing_km
    minimum_pixels = int(math.ceil(float(minimum_area_km2) / (dy * dx)))
    raw, number = ndi.label(valid & np.isfinite(frame) & (frame >= threshold_dbz), np.ones((3, 3), np.uint8))
    labels = np.zeros(raw.shape, np.int32)
    objects = {}
    next_id = 1
    for raw_id, sl in enumerate(ndi.find_objects(raw), 1):
        if sl is None:
            continue
        yy, xx = np.where(raw[sl] == raw_id)
        yy, xx = (yy + sl[0].start, xx + sl[1].start)
        if len(yy) < minimum_pixels:
            continue
        values = frame[yy, xx]
        labels[yy, xx] = next_id
        objects[next_id] = Component(next_id, yy.astype(np.int32), xx.astype(np.int32), len(yy), float(len(yy) * dy * dx), float(yy.mean()), float(xx.mean()), float(values.max()), float(values.mean()))
        next_id += 1
    return (labels, objects)

def bounds(component, shape, radius_km, spacing_km):
    dy, dx = spacing_km
    ry, rx = (int(math.ceil(radius_km / dy)), int(math.ceil(radius_km / dx)))
    return (max(0, int(component.ys.min()) - ry), min(shape[0], int(component.ys.max()) + ry + 1), max(0, int(component.xs.min()) - rx), min(shape[1], int(component.xs.max()) + rx + 1))

def edge_distance_km(component, shape, spacing_km):
    dy, dx = spacing_km
    return float(min(component.ys.min() * dy, (shape[0] - 1 - component.ys.max()) * dy, component.xs.min() * dx, (shape[1] - 1 - component.xs.max()) * dx))

def minimum_distance_to_labels(component, labels, spacing_km, exclude_label=None):
    other = labels > 0
    if exclude_label is not None:
        other &= labels != exclude_label
    if not other.any():
        return float('inf')
    distance = ndi.distance_transform_edt(~other, sampling=spacing_km)
    return float(distance[component.ys, component.xs].min())

def minimum_past_distance(component, past_labels, spacing_km):
    if not past_labels:
        return float('nan')
    answer = float('inf')
    for labels in past_labels:
        answer = min(answer, minimum_distance_to_labels(component, labels, spacing_km))
    return answer

def temporal_window_complete(times, start, end, max_gap_minutes, endpoint_tolerance_minutes):
    times = pd.DatetimeIndex(times)
    selected = times[(times >= start - pd.Timedelta(minutes=endpoint_tolerance_minutes)) & (times <= end + pd.Timedelta(minutes=endpoint_tolerance_minutes))]
    if not len(selected):
        return False
    before = selected[selected <= start]
    after = selected[selected >= end]
    if not len(before) or not len(after):
        return False
    relevant = selected[(selected >= before[-1]) & (selected <= after[0])]
    return len(relevant) < 2 or relevant.to_series().diff().dropna().max() <= pd.Timedelta(minutes=max_gap_minutes)

def local_valid_fraction(component, valid_frames, radius_km, spacing_km):
    y0, y1, x0, x1 = bounds(component, valid_frames[0].shape, radius_km, spacing_km)
    return float(min((np.asarray(frame[y0:y1, x0:x1], bool).mean() for frame in valid_frames)))

def match_persistence(component, birth_index, times, labels, objects, spacing_km, cfg):
    """Track only through the short confirmation window; tolerate one missing scan."""
    birth_time = pd.Timestamp(times[birth_index])
    current = component
    last_index = birth_index
    missing = 0
    upper = cfg['persistence_minutes'] + cfg['persistence_tolerance_minutes']
    lower = cfg['persistence_minutes'] - cfg['persistence_tolerance_minutes']
    for index in range(birth_index + 1, len(times)):
        elapsed = (pd.Timestamp(times[index]) - birth_time).total_seconds() / 60
        if elapsed > upper:
            break
        gap = (pd.Timestamp(times[index]) - pd.Timestamp(times[last_index])).total_seconds() / 60
        if gap > cfg['maximum_tracking_gap_minutes']:
            return None
        gate = cfg.get('tracking_padding_km', 1.0) + cfg['maximum_tracking_speed_m_s'] * gap * 60 / 1000
        candidates = []
        for candidate in objects[index].values():
            distance = minimum_distance_between_components(current, candidate, spacing_km)
            if distance <= gate:
                candidates.append((distance, candidate.label_id, candidate))
        if not candidates:
            missing += 1
            if missing > cfg['maximum_missing_native_scans']:
                return None
            continue
        current = min(candidates, key=lambda item: (item[0], item[1]))[2]
        last_index = index
        if lower <= elapsed <= upper:
            return index
    return None

def minimum_distance_between_components(first, second, spacing_km):
    y0 = min(int(first.ys.min()), int(second.ys.min()))
    y1 = max(int(first.ys.max()), int(second.ys.max())) + 1
    x0 = min(int(first.xs.min()), int(second.xs.min()))
    x1 = max(int(first.xs.max()), int(second.xs.max())) + 1
    seed = np.zeros((y1 - y0, x1 - x0), bool)
    seed[second.ys - y0, second.xs - x0] = True
    distance = ndi.distance_transform_edt(~seed, sampling=spacing_km)
    return float(distance[first.ys - y0, first.xs - x0].min())

def forecast_intervals(bin_minutes, max_minutes):
    if bin_minutes <= 0 or max_minutes <= 0 or max_minutes % bin_minutes:
        raise ValueError('max_forecast_minutes must be a positive multiple of forecast_bin_minutes')
    return [(index, index * bin_minutes, (index + 1) * bin_minutes) for index in range(max_minutes // bin_minutes)]

def lead_index(birth_time, issuance_time, bin_minutes, max_minutes):
    delta = (pd.Timestamp(birth_time) - pd.Timestamp(issuance_time)).total_seconds() / 60
    for index, start, end in forecast_intervals(bin_minutes, max_minutes):
        if start < delta <= end:
            return index
    return None

def build_periods(catalog, cfg):
    """Use legacy candidates only to recover padded continuous MRMS periods."""
    aoi = cfg['aoi']
    mid_lat = (aoi['lat_min'] + aoi['lat_max']) / 2
    margin_lat = cfg['isolation_margin_km'] / 111.19508
    margin_lon = cfg['isolation_margin_km'] / (111.19508 * np.cos(np.deg2rad(mid_lat)))
    selected = catalog.loc[catalog.latitude.between(aoi['lat_min'] - margin_lat, aoi['lat_max'] + margin_lat) & catalog.longitude.between(aoi['lon_min'] - margin_lon, aoi['lon_max'] + margin_lon)].copy()
    times = pd.DatetimeIndex(sorted(pd.to_datetime(selected.ci_time).unique()))
    groups = []
    for value in times:
        if groups and value - groups[-1][-1] <= pd.Timedelta(minutes=cfg['candidate_merge_gap_minutes']):
            groups[-1].append(value)
        else:
            groups.append([value])
    padded = []
    future = cfg['reserved_forecast_minutes'] + cfg['future_confirmation_minutes']
    for group in groups:
        padded.append({'candidate_start': group[0], 'candidate_end': group[-1], 'scan_start': group[0] - pd.Timedelta(minutes=cfg['past_isolation_minutes']), 'scan_end': group[-1] + pd.Timedelta(minutes=future), 'candidate_timestamp_count': len(group)})
    merged = []
    for period in padded:
        if merged and period['scan_start'] <= merged[-1]['scan_end']:
            merged[-1]['candidate_end'] = max(merged[-1]['candidate_end'], period['candidate_end'])
            merged[-1]['scan_end'] = max(merged[-1]['scan_end'], period['scan_end'])
            merged[-1]['candidate_timestamp_count'] += period['candidate_timestamp_count']
        else:
            merged.append(period)
    rows = []
    for period in merged:
        year = int(period['candidate_start'].year)
        period_id = stable_id('PERIOD', period['scan_start'].isoformat(), period['scan_end'].isoformat(), aoi)
        rows.append(dict(period_id=period_id, year=year, status='PLANNED', **period))
    return pd.DataFrame(rows)

def classify_birth(in_aoi, temporal_complete, boundary_supported, local_radar_valid, past_distance_km, same_time_distance_km, persistence_confirmed, isolation_radius_km=30.0):
    """Binary scientific decision; diagnostic reasons never become target classes."""
    if not in_aoi:
        return (False, 'outside_requested_aoi')
    if not temporal_complete or not boundary_supported:
        return (False, 'temporal_or_spatial_support')
    if not local_radar_valid:
        return (False, 'local_radar_missingness')
    if past_distance_km < isolation_radius_km:
        return (False, 'mature_echo_in_past_30min')
    if same_time_distance_km < isolation_radius_km:
        return (False, 'other_mature_echo_inside_30km')
    if not persistence_confirmed:
        return (False, 'persistence_not_confirmed')
    return (True, '')

def supervision_value(has_accepted_birth, radar_valid):
    """Return binary target and validity weight; missingness is never target zero."""
    return (1 if has_accepted_birth else 0, 1 if radar_valid else 0)
