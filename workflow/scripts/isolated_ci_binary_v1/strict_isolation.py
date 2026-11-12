"""Strict radar-context filter for genuinely isolated convective initiation."""
from __future__ import annotations
from ci_nowcasting.paths import resolve_path as _ci_resolve, project_root as _ci_project, source_root as _ci_source
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
STRICT_AUDIT_COLUMNS = ['object_id', 'period_id', 'birth_time', 'strict_isolated', 'strict_rejection_reason', 'background_max_fraction_ge10', 'background_mean_fraction_ge10', 'background_min_valid_fraction', 'connected20_total_area_km2', 'connected20_outer_area_km2', 'strict_frames_evaluated']

@dataclass(frozen=True)
class StrictMetrics:
    strict_isolated: bool
    strict_rejection_reason: str
    background_max_fraction_ge10: float
    background_mean_fraction_ge10: float
    background_min_valid_fraction: float
    connected20_total_area_km2: float
    connected20_outer_area_km2: float
    strict_frames_evaluated: int

    def as_dict(self):
        return {'strict_isolated': bool(self.strict_isolated), 'strict_rejection_reason': self.strict_rejection_reason, 'background_max_fraction_ge10': float(self.background_max_fraction_ge10), 'background_mean_fraction_ge10': float(self.background_mean_fraction_ge10), 'background_min_valid_fraction': float(self.background_min_valid_fraction), 'connected20_total_area_km2': float(self.connected20_total_area_km2), 'connected20_outer_area_km2': float(self.connected20_outer_area_km2), 'strict_frames_evaluated': int(self.strict_frames_evaluated)}

def _local_geometry(shape, center_y, center_x, spacing_km, inner_radius_km, outer_radius_km):
    dy_km, dx_km = map(float, spacing_km)
    height, width = shape
    y_radius = int(np.ceil(outer_radius_km / dy_km))
    x_radius = int(np.ceil(outer_radius_km / dx_km))
    y0 = max(0, int(np.floor(center_y)) - y_radius - 1)
    y1 = min(height, int(np.ceil(center_y)) + y_radius + 2)
    x0 = max(0, int(np.floor(center_x)) - x_radius - 1)
    x1 = min(width, int(np.ceil(center_x)) + x_radius + 2)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    distance = np.sqrt(((yy - float(center_y)) * dy_km) ** 2 + ((xx - float(center_x)) * dx_km) ** 2)
    inner = distance <= float(inner_radius_km)
    outer_disk = distance <= float(outer_radius_km)
    annulus = (distance > float(inner_radius_km)) & outer_disk
    boundary_supported = center_y * dy_km >= outer_radius_km and (height - 1 - center_y) * dy_km >= outer_radius_km and (center_x * dx_km >= outer_radius_km) and ((width - 1 - center_x) * dx_km >= outer_radius_km)
    return (slice(y0, y1), slice(x0, x1), distance, inner, annulus, boundary_supported)

def _nearest_birth_index(times, birth_time, tolerance_minutes):
    times = pd.DatetimeIndex(pd.to_datetime(times))
    birth_time = pd.Timestamp(birth_time)
    if not len(times):
        return None
    difference = np.abs((times - birth_time).total_seconds().to_numpy(dtype=float))
    index = int(np.argmin(difference))
    if difference[index] > float(tolerance_minutes) * 60.0:
        return None
    return index

def strict_isolation_metrics(frames, valid_frames, times, record, spacing_km, cfg):
    """Evaluate clear background and low-echo storm connectivity."""
    birth_time = pd.Timestamp(record['birth_time'])
    center_y = float(record['birth_y_px'])
    center_x = float(record['birth_x_px'])
    background_threshold = float(cfg.get('strict_background_threshold_dbz', 10.0))
    connection_threshold = float(cfg.get('strict_connection_threshold_dbz', 20.0))
    inner_radius = float(cfg.get('strict_inner_radius_km', 10.0))
    outer_radius = float(cfg.get('isolation_radius_km', 30.0))
    past_minutes = float(cfg.get('past_isolation_minutes', 30.0))
    maximum_background_fraction = float(cfg.get('strict_maximum_background_fraction_ge10', 0.05))
    maximum_connected_outer_area = float(cfg.get('strict_maximum_connected20_outer_area_km2', 16.0))
    minimum_valid_fraction = float(cfg.get('minimum_local_valid_fraction', 0.99))
    endpoint_tolerance = float(cfg.get('endpoint_tolerance_minutes', 2.5))
    birth_index = _nearest_birth_index(times, birth_time, endpoint_tolerance)
    empty = StrictMetrics(False, 'strict_birth_frame_missing', np.nan, np.nan, 0.0, np.nan, np.nan, 0)
    if birth_index is None:
        return empty
    birth_frame = np.asarray(frames[birth_index])
    birth_valid = np.asarray(valid_frames[birth_index], dtype=bool)
    ys, xs, distance, inner, annulus, boundary_supported = _local_geometry(birth_frame.shape, center_y, center_x, spacing_km, inner_radius, outer_radius)
    if not boundary_supported or not np.any(annulus):
        return StrictMetrics(False, 'strict_30km_boundary_support_missing', np.nan, np.nan, 0.0, np.nan, np.nan, 0)
    time_index = pd.DatetimeIndex(pd.to_datetime(times))
    start_time = birth_time - pd.Timedelta(minutes=past_minutes)
    context_indices = np.flatnonzero((time_index >= start_time) & (time_index <= birth_time + pd.Timedelta(minutes=endpoint_tolerance)))
    if not len(context_indices):
        return empty
    fractions = []
    valid_fractions = []
    for index in context_indices:
        frame = np.asarray(frames[index])[ys, xs]
        valid = np.asarray(valid_frames[index], dtype=bool)[ys, xs]
        annulus_count = int(np.count_nonzero(annulus))
        valid_count = int(np.count_nonzero(valid & annulus))
        valid_fraction = valid_count / annulus_count if annulus_count else 0.0
        valid_fractions.append(valid_fraction)
        if valid_count:
            echo_count = int(np.count_nonzero(valid & annulus & (frame >= background_threshold)))
            fractions.append(echo_count / valid_count)
        else:
            fractions.append(np.nan)
    finite_fractions = np.asarray(fractions, dtype=float)
    finite_fractions = finite_fractions[np.isfinite(finite_fractions)]
    background_max = float(np.max(finite_fractions)) if len(finite_fractions) else np.nan
    background_mean = float(np.mean(finite_fractions)) if len(finite_fractions) else np.nan
    background_min_valid = float(np.min(valid_fractions)) if valid_fractions else 0.0
    local_birth = birth_frame[ys, xs]
    local_valid = birth_valid[ys, xs]
    low_echo = local_valid & (local_birth >= connection_threshold)
    low_labels, _ = ndi.label(low_echo, structure=np.ones((3, 3), dtype=np.uint8))
    birth_threshold = float(cfg.get('threshold_dbz', 35.0))
    core = local_valid & inner & (local_birth >= birth_threshold)
    overlapping_labels = low_labels[core]
    overlapping_labels = overlapping_labels[overlapping_labels > 0]
    connected_total_area = 0.0
    connected_outer_area = 0.0
    pixel_area = float(spacing_km[0]) * float(spacing_km[1])
    if len(overlapping_labels):
        ids, counts = np.unique(overlapping_labels, return_counts=True)
        selected_label = int(ids[np.argmax(counts)])
        connected = low_labels == selected_label
        connected_total_area = float(np.count_nonzero(connected)) * pixel_area
        connected_outer_area = float(np.count_nonzero(connected & annulus)) * pixel_area
    reason = ''
    if background_min_valid < minimum_valid_fraction:
        reason = 'strict_background_radar_invalid'
    elif not np.isfinite(background_max):
        reason = 'strict_background_radar_invalid'
    elif background_max > maximum_background_fraction:
        reason = 'background_echo_ge10_in_10_30km'
    elif connected_outer_area > maximum_connected_outer_area:
        reason = 'connected_echo_ge20_beyond_10km'
    return StrictMetrics(strict_isolated=reason == '', strict_rejection_reason=reason, background_max_fraction_ge10=background_max, background_mean_fraction_ge10=background_mean, background_min_valid_fraction=background_min_valid, connected20_total_area_km2=connected_total_area, connected20_outer_area_km2=connected_outer_area, strict_frames_evaluated=len(context_indices))

def strict_filter_accepted(frames, valid_frames, times, spacing_km, accepted, cfg, period_id, funnel=None):
    """Apply strict isolation after the original physical birth checks."""
    kept = []
    audit = []
    rejected_background = 0
    rejected_connection = 0
    rejected_invalid = 0
    for original in accepted:
        record = dict(original)
        metrics = strict_isolation_metrics(frames, valid_frames, times, record, spacing_km, cfg)
        values = metrics.as_dict()
        audit.append({'object_id': record['object_id'], 'period_id': period_id, 'birth_time': record['birth_time'], **values})
        if metrics.strict_isolated:
            record.update(values)
            kept.append(record)
        elif metrics.strict_rejection_reason == 'background_echo_ge10_in_10_30km':
            rejected_background += 1
        elif metrics.strict_rejection_reason == 'connected_echo_ge20_beyond_10km':
            rejected_connection += 1
        else:
            rejected_invalid += 1
    if funnel is not None:
        funnel['accepted_before_strict_isolation'] = len(accepted)
        funnel['objects_with_background_echo_ge10'] = rejected_background
        funnel['objects_connected_to_ge20_echo'] = rejected_connection
        funnel['objects_failing_strict_radar_support'] = rejected_invalid
        funnel['accepted_isolated_births'] = len(kept)
    return (kept, audit)
