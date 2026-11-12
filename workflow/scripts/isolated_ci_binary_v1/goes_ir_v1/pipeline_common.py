"""Shared, testable primitives for the strict-CI GOES IR pipeline.

The geometry in this module is intentionally derived from the already
validated ``build_goes_parallax_mrms_zarr_fast_v2.py`` implementation.  The
important convention is that Satpy's corrected swath is a *target sampling
geometry* for an inverse mapping from the regular ABI grid.
"""
from __future__ import annotations
from ci_nowcasting.paths import resolve_path as _ci_resolve, project_root as _ci_project, source_root as _ci_source
import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
import dask.array as da
import numpy as np
import pandas as pd
import requests
import xarray as xr
import zarr
from numcodecs import Blosc
from pyproj import CRS, Transformer
from pyresample.geometry import AreaDefinition, SwathDefinition
from pyresample.kd_tree import get_neighbour_info, get_sample_from_neighbour_info
HERE = Path(_ci_resolve(__file__)).resolve().parent
PROJECT_ROOT = _ci_project()
DATA_ROOT = PROJECT_ROOT / 'data' / 'ci_isolated_mrms_binary_v1'
YEAR_DIR_TEMPLATE = 'year_{year}_strict_v2_001'
OUTPUT_ROOT = DATA_ROOT / 'goes_ir_parallax_native_v1'
MANIFEST_ROOT = OUTPUT_ROOT / 'manifests'
STATE_ROOT = OUTPUT_ROOT / 'state'
TEMP_ROOT = OUTPUT_ROOT / 'temporary_raw'
CHANNELS = tuple(range(8, 15))
CMIPC_PRODUCT = 'ABI-L2-CMIPC'
ACHA_PRODUCT = 'ABI-L2-ACHAC'
GOES19_TRANSITION = pd.Timestamp('2025-04-07 15:10:00')
BUCKETS = {16: 'noaa-goes16', 19: 'noaa-goes19'}
AOI = {'lat_min': 38.5, 'lat_max': 44.5, 'lon_min': -76.5, 'lon_max': -71.0}
PARENT_MARGIN_KM = 300.0
TIME_UNITS = 'milliseconds since 1970-01-01 00:00:00'
TIME_CALENDAR = 'proleptic_gregorian'
MIN_FREE_BYTES = 20 * 1024 ** 3
GOES_NAME_RE = re.compile('^OR_(?P<product>ABI-L2-(?:CMIPC|ACHAC))-M\\d(?:C(?P<channel>\\d{2}))?_G(?P<satellite>\\d{2})_s(?P<start>\\d{13,16})_e(?P<end>\\d{13,16})_c(?P<created>\\d{13,16})\\.nc$')
PROGRESS_COLUMNS = ['requested_time', 'year', 'month', 'status', 'attempts', 'updated_at', 'satellite', 'output_store', 'finite_fraction', 'message']

def utc_naive(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert('UTC').tz_localize(None)
    return stamp

def parse_goes_timestamp(token: str) -> pd.Timestamp:
    """Parse GOES ``YYYYJJJhhmmss[fraction]`` tokens without rounding."""
    token = str(token)
    if len(token) < 13:
        raise ValueError(f'Invalid GOES time token: {token!r}')
    base = pd.to_datetime(token[:13], format='%Y%j%H%M%S', errors='raise')
    fraction = token[13:]
    if fraction:
        base += pd.to_timedelta(int(fraction) / 10 ** len(fraction), unit='s')
    return pd.Timestamp(base)

def parse_goes_filename(path: str | Path) -> dict[str, Any]:
    name = Path(_ci_resolve(str(path))).name
    match = GOES_NAME_RE.match(name)
    if match is None:
        raise ValueError(f'Unrecognized GOES ABI filename: {name}')
    values = match.groupdict()
    channel = int(values['channel']) if values['channel'] else -1
    return {'product': values['product'], 'channel': channel, 'satellite': int(values['satellite']), 'scan_start': parse_goes_timestamp(values['start']), 'scan_end': parse_goes_timestamp(values['end']), 'created_time': parse_goes_timestamp(values['created']), 'filename': name}

def satellite_for_time(value: Any) -> int:
    return 19 if utc_naive(value) >= GOES19_TRANSITION else 16

def year_label_path(year: int) -> Path:
    return DATA_ROOT / YEAR_DIR_TEMPLATE.format(year=int(year)) / 'label_index.parquet'

def monthly_store(value: Any, output_root: Path=OUTPUT_ROOT) -> Path:
    stamp = utc_naive(value)
    return output_root / f'year={stamp:%Y}' / f'month={stamp:%m}' / 'goes_ir.zarr'

def atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path = Path(_ci_resolve(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)

def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path = Path(_ci_resolve(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)

def human_bytes(value: float | int) -> str:
    number = float(value)
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if abs(number) < 1024.0 or unit == 'TiB':
            return f'{number:.2f} {unit}'
        number /= 1024.0
    return f'{number:.2f} TiB'

def _parse_size(value: str) -> int | None:
    match = re.fullmatch('([0-9.]+)([KMGTP]?)', value.strip(), re.IGNORECASE)
    if not match:
        return None
    scale = {'': 1024, 'K': 1024, 'M': 1024 ** 2, 'G': 1024 ** 3, 'T': 1024 ** 4, 'P': 1024 ** 5}[match.group(2).upper()]
    return int(float(match.group(1)) * scale)

def quota_available_bytes(path: Path) -> int | None:
    """Best-effort user quota headroom; returns None when quota is unavailable."""
    if shutil.which('quota') is None:
        return None
    try:
        result = subprocess.run(['quota', '-w', '-s'], capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    for line in reversed((result.stdout + '\n' + result.stderr).splitlines()):
        fields = line.split()
        if len(fields) < 3 or not (fields[0].startswith('/') or ':' in fields[0]):
            continue
        used = _parse_size(fields[1].rstrip('*'))
        limit = _parse_size(fields[2].rstrip('*'))
        if used is not None and limit is not None and (limit > 0):
            return max(0, limit - used)
    return None

def available_bytes(path: Path) -> tuple[int, int | None, int]:
    path = Path(_ci_resolve(path))
    path.mkdir(parents=True, exist_ok=True)
    filesystem_free = int(shutil.disk_usage(path).free)
    quota_free = quota_available_bytes(path)
    effective = min(filesystem_free, quota_free) if quota_free is not None else filesystem_free
    return (filesystem_free, quota_free, effective)

def wait_for_disk(path: Path, minimum_bytes: int=MIN_FREE_BYTES, poll_seconds: int=300, no_wait: bool=False) -> None:
    while True:
        df_free, quota_free, effective = available_bytes(path)
        if effective >= int(minimum_bytes):
            return
        message = f"Disk guard: effective free {human_bytes(effective)} < {human_bytes(minimum_bytes)} (filesystem={human_bytes(df_free)}, quota={(human_bytes(quota_free) if quota_free is not None else 'unavailable')})"
        if no_wait:
            raise RuntimeError(message)
        print(message + f'; pausing {poll_seconds}s', flush=True)
        time.sleep(poll_seconds)

def select_causal_source(available: pd.DataFrame, requested_time: Any, max_age_seconds: float, *, reference_start: Any | None=None) -> pd.Series | None:
    """Choose a completed scan without ever considering future scan ends.

    With ``reference_start``, exact scan-cycle alignment is preferred; ties are
    resolved toward the most recently completed source.  This is intentionally
    not a nearest-time merge against the request.
    """
    if available.empty:
        return None
    requested = utc_naive(requested_time)
    work = available.copy()
    work['scan_start'] = pd.to_datetime(work['scan_start'], errors='coerce')
    work['scan_end'] = pd.to_datetime(work['scan_end'], errors='coerce')
    work = work.loc[work['scan_end'].notna() & (work['scan_end'] <= requested)].copy()
    if work.empty:
        return None
    work['age_seconds'] = (requested - work['scan_end']).dt.total_seconds()
    work = work.loc[(work['age_seconds'] >= 0) & (work['age_seconds'] <= float(max_age_seconds))].copy()
    if work.empty:
        return None
    if reference_start is None:
        work = work.sort_values(['scan_end', 'scan_start', 'created_time'], ascending=[False, False, False])
    else:
        reference = utc_naive(reference_start)
        work['alignment_seconds'] = (work['scan_start'] - reference).abs().dt.total_seconds()
        work = work.sort_values(['alignment_seconds', 'scan_end', 'created_time'], ascending=[True, False, False])
    return work.iloc[0]

def source_rows_for_request(inventory: pd.DataFrame, requested_time: Any) -> list[dict[str, Any]]:
    requested = utc_naive(requested_time)
    satellite = satellite_for_time(requested)
    candidates = inventory.loc[inventory['satellite'].eq(satellite)].copy()
    c13 = select_causal_source(candidates.loc[(candidates['product'] == CMIPC_PRODUCT) & (candidates['channel'] == 13)], requested, 600)
    reference_start = None if c13 is None else c13['scan_start']
    rows: list[dict[str, Any]] = []
    for product, channel, maximum_age in [*((CMIPC_PRODUCT, channel, 600) for channel in CHANNELS), (ACHA_PRODUCT, -1, 900)]:
        subset = candidates.loc[(candidates['product'] == product) & (candidates['channel'] == channel)]
        reference = reference_start if product == CMIPC_PRODUCT else None
        selected = select_causal_source(subset, requested, maximum_age, reference_start=reference)
        base = {'requested_time': requested, 'satellite': satellite, 'product': product, 'channel': int(channel)}
        if selected is None:
            rows.append({**base, 'scan_start': pd.NaT, 'scan_end': pd.NaT, 'age_seconds': np.nan, 'remote_path': None, 'filename': None, 'matched': False})
        else:
            rows.append({**base, 'scan_start': utc_naive(selected['scan_start']), 'scan_end': utc_naive(selected['scan_end']), 'age_seconds': float((requested - utc_naive(selected['scan_end'])).total_seconds()), 'remote_path': str(selected['remote_path']), 'filename': str(selected['filename']), 'matched': True})
    return rows

def remote_to_https(remote_path: str) -> str:
    value = str(remote_path).removeprefix('s3://')
    bucket, key = value.split('/', 1)
    return f'https://{bucket}.s3.amazonaws.com/{key}'

def download_file(remote_path: str, destination: Path, retries: int=4, timeout: int=240) -> Path:
    destination = Path(_ci_resolve(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    partial = destination.with_suffix(destination.suffix + '.part')
    url = remote_to_https(remote_path)
    for attempt in range(retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                with open(_ci_resolve(partial), 'wb') as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if not partial.exists() or partial.stat().st_size == 0:
                raise RuntimeError('Downloaded file is empty')
            os.replace(partial, destination)
            return destination
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt >= retries:
                raise
            time.sleep(min(120, 3 * 2 ** attempt))
    raise AssertionError('unreachable')

def sample_aoi_perimeter(lon_min: float, lon_max: float, lat_min: float, lat_max: float, points_per_edge: int=200) -> tuple[np.ndarray, np.ndarray]:
    horizontal = np.linspace(lon_min, lon_max, points_per_edge)
    vertical = np.linspace(lat_min, lat_max, points_per_edge)
    return (np.concatenate([horizontal, horizontal, np.full(points_per_edge, lon_min), np.full(points_per_edge, lon_max)]), np.concatenate([np.full(points_per_edge, lat_min), np.full(points_per_edge, lat_max), vertical, vertical]))

def geos_crs(attrs: dict[str, Any]) -> CRS:
    return CRS.from_proj4(' '.join(['+proj=geos', f"+h={float(attrs['perspective_point_height'])}", f"+lon_0={float(attrs['longitude_of_projection_origin'])}", f"+a={float(attrs['semi_major_axis'])}", f"+b={float(attrs['semi_minor_axis'])}", f"+sweep={attrs['sweep_angle_axis']}", '+units=m', '+no_defs']))

def projected_crop_bounds(attrs: dict[str, Any], *, lon_min: float, lon_max: float, lat_min: float, lat_max: float, margin_km: float) -> tuple[float, float, float, float]:
    transformer = Transformer.from_crs('EPSG:4326', geos_crs(attrs), always_xy=True)
    lons, lats = sample_aoi_perimeter(lon_min, lon_max, lat_min, lat_max)
    xx, yy = transformer.transform(lons, lats)
    valid = np.isfinite(xx) & np.isfinite(yy)
    if not np.any(valid):
        raise RuntimeError('AOI does not project onto the GOES fixed grid')
    margin = float(margin_km) * 1000.0
    return (float(np.min(np.asarray(xx)[valid]) - margin), float(np.max(np.asarray(xx)[valid]) + margin), float(np.min(np.asarray(yy)[valid]) - margin), float(np.max(np.asarray(yy)[valid]) + margin))

def crop_slices(x: np.ndarray, y: np.ndarray, attrs: dict[str, Any], margin_km: float) -> tuple[slice, slice]:
    h = float(attrs['perspective_point_height'])
    xmin, xmax, ymin, ymax = projected_crop_bounds(attrs, **AOI, margin_km=margin_km)
    xi = np.where((np.asarray(x) * h >= xmin) & (np.asarray(x) * h <= xmax))[0]
    yi = np.where((np.asarray(y) * h >= ymin) & (np.asarray(y) * h <= ymax))[0]
    if not len(xi) or not len(yi):
        raise RuntimeError('No GOES pixels intersect the requested AOI')
    return (slice(int(yi.min()), int(yi.max()) + 1), slice(int(xi.min()), int(xi.max()) + 1))

def _python_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value

def read_native_crop(path: Path, variable: str, margin_km: float=PARENT_MARGIN_KM, max_dqf: int=1) -> xr.Dataset:
    """Read and crop one CMI or ACHA file, keeping only in-memory parent data."""
    with xr.open_dataset(_ci_resolve(path), engine='netcdf4', mask_and_scale=True, decode_times=True) as source:
        if variable not in source or 'goes_imager_projection' not in source:
            raise KeyError(f'{path.name}: missing {variable} or projection')
        attrs = dict(source['goes_imager_projection'].attrs)
        ys, xs = crop_slices(source['x'].values, source['y'].values, attrs, margin_km)
        values = np.asarray(source[variable].isel(y=ys, x=xs).values, dtype=np.float32)
        valid = np.isfinite(values)
        if 'DQF' in source:
            dqf = np.asarray(source['DQF'].isel(y=ys, x=xs).values)
            valid &= np.isfinite(dqf) & (dqf <= int(max_dqf))
        values = np.where(valid, values, np.nan).astype(np.float32)
        x = np.asarray(source['x'].isel(x=xs).values, dtype=np.float32)
        y = np.asarray(source['y'].isel(y=ys).values, dtype=np.float32)
        output = xr.Dataset({variable: (('y', 'x'), values)}, coords={'x': x, 'y': y})
        output[variable].attrs['units'] = str(source[variable].attrs.get('units', ''))
        output['x'].attrs.update({_key: _python_value(_value) for _key, _value in source['x'].attrs.items()})
        output['y'].attrs.update({_key: _python_value(_value) for _key, _value in source['y'].attrs.items()})
        output.attrs['satellite'] = int(str(source.attrs.get('platform_ID', 'G16')).upper().replace('GOES-', '').replace('G', ''))
        output.attrs['projection_attrs'] = attrs
        return output.load()

@dataclass(frozen=True)
class NativeGrid:
    area: AreaDefinition
    flip_x: bool
    flip_y: bool
    dx_m: float
    dy_m: float
    satellite: int
    projection_attrs: dict[str, Any]

def geos_proj_dict(attrs: dict[str, Any]) -> dict[str, Any]:
    return {'proj': 'geos', 'h': float(attrs['perspective_point_height']), 'lon_0': float(attrs['longitude_of_projection_origin']), 'lat_0': float(attrs.get('latitude_of_projection_origin', 0.0)), 'a': float(attrs['semi_major_axis']), 'b': float(attrs['semi_minor_axis']), 'sweep': str(attrs['sweep_angle_axis']), 'units': 'm'}

def build_native_grid(dataset: xr.Dataset, area_id: str) -> NativeGrid:
    attrs = dict(dataset.attrs['projection_attrs'])
    height = float(attrs['perspective_point_height'])
    x_m = np.asarray(dataset.x.values, dtype=np.float64) * height
    y_m = np.asarray(dataset.y.values, dtype=np.float64) * height
    flip_x = bool(x_m[0] > x_m[-1])
    flip_y = bool(y_m[0] < y_m[-1])
    x_can = x_m[::-1] if flip_x else x_m
    y_can = y_m[::-1] if flip_y else y_m
    dx = float(np.nanmedian(np.diff(x_can)))
    dy = float(np.nanmedian(np.abs(np.diff(y_can))))
    extent = (float(x_can[0] - dx / 2), float(y_can[-1] - dy / 2), float(x_can[-1] + dx / 2), float(y_can[0] + dy / 2))
    area = AreaDefinition(area_id, area_id, area_id, geos_proj_dict(attrs), len(x_can), len(y_can), extent)
    return NativeGrid(area, flip_x, flip_y, dx, dy, int(dataset.attrs['satellite']), attrs)

def orient_native(values: np.ndarray, grid: NativeGrid) -> np.ndarray:
    result = np.asarray(values)
    if grid.flip_y:
        result = result[::-1, :]
    if grid.flip_x:
        result = result[:, ::-1]
    return result

def native_grids_equivalent(first: NativeGrid, second: NativeGrid) -> bool:
    return bool(first.area.shape == second.area.shape and first.flip_x == second.flip_x and (first.flip_y == second.flip_y) and (first.satellite == second.satellite) and np.allclose(first.area.area_extent, second.area.area_extent, rtol=0, atol=0.001) and CRS.from_user_input(first.area.crs).equals(CRS.from_user_input(second.area.crs)))

def neighbour_info(source: Any, target: Any, radius_m: float, nprocs: int=1):
    return get_neighbour_info(source, target, radius_of_influence=float(radius_m), neighbours=1, epsilon=0.0, reduce_data=True, nprocs=int(nprocs))

def sample_with_info(values: np.ndarray, output_shape: tuple[int, int], info: Any) -> np.ndarray:
    valid_input, valid_output, indices, _ = info
    sampled = get_sample_from_neighbour_info('nn', output_shape, np.asarray(values, dtype=np.float32), valid_input, valid_output, indices, fill_value=np.nan)
    if np.ma.isMaskedArray(sampled):
        sampled = sampled.filled(np.nan)
    return np.asarray(sampled, dtype=np.float32)

def normalize_cth(height: np.ndarray, units: str) -> np.ndarray:
    result = np.asarray(height, dtype=np.float32)
    if 'km' in str(units).strip().lower():
        result *= 1000.0
    return np.where((result >= 0) & (result <= 25000), result, np.nan).astype(np.float32)

def native_grid_key(dataset: xr.Dataset) -> str:
    """Stable identity for a cropped fixed grid, including platform/projection."""
    digest = hashlib.sha1()
    digest.update(np.asarray(dataset.x.values, dtype=np.float32).tobytes())
    digest.update(np.asarray(dataset.y.values, dtype=np.float32).tobytes())
    digest.update(str(int(dataset.attrs['satellite'])).encode())
    attrs = dataset.attrs['projection_attrs']
    for name in sorted(attrs):
        digest.update(name.encode())
        digest.update(repr(_python_value(attrs[name])).encode())
    return digest.hexdigest()

@dataclass
class ParallaxStaticCache:
    """Reusable fixed-grid objects; dynamic parallax remains once per timestamp."""
    grids: dict[str, NativeGrid] = field(default_factory=dict)
    acha_to_ir: dict[tuple[str, str, int], Any] = field(default_factory=dict)
    correctors: dict[str, Any] = field(default_factory=dict)
    aoi_slices: dict[str, tuple[slice, slice]] = field(default_factory=dict)

    def grid(self, dataset: xr.Dataset, label: str) -> tuple[str, NativeGrid]:
        key = native_grid_key(dataset)
        if key not in self.grids:
            self.grids[key] = build_native_grid(dataset, label)
        return (key, self.grids[key])

def corrected_native_fields(cmi: dict[int, xr.Dataset], acha: xr.Dataset, timestamp: Any, nprocs: int=1, static_cache: ParallaxStaticCache | None=None) -> tuple[dict[int, np.ndarray], NativeGrid]:
    """Apply the verified inverse parallax geometry once and reuse it for 7 bands."""
    from satpy.modifiers.parallax import ParallaxCorrection
    cache = static_cache if static_cache is not None else ParallaxStaticCache()
    shared_key, shared = cache.grid(cmi[13], 'strict_ci_ir')
    for channel, dataset in cmi.items():
        _, channel_grid = cache.grid(dataset, f'C{channel:02d}')
        if not native_grids_equivalent(shared, channel_grid):
            raise RuntimeError(f'C{channel:02d} does not share the C13 native grid')
    acha_key, acha_grid = cache.grid(acha, 'strict_ci_acha')
    if acha_grid.satellite != shared.satellite:
        raise RuntimeError('ACHA and CMI satellite mismatch')
    height = normalize_cth(orient_native(acha['HT'].values, acha_grid), acha['HT'].attrs.get('units', 'm'))
    if native_grids_equivalent(acha_grid, shared):
        shared_height = height
    else:
        mapping_key = (acha_key, shared_key, int(nprocs))
        if mapping_key not in cache.acha_to_ir:
            cache.acha_to_ir[mapping_key] = neighbour_info(acha_grid.area, shared.area, 50000.0, nprocs)
        info = cache.acha_to_ir[mapping_key]
        shared_height = sample_with_info(height, shared.area.shape, info)
    cth = xr.DataArray(da.from_array(shared_height, chunks=(min(512, shared_height.shape[0]), min(512, shared_height.shape[1]))), dims=('y', 'x'), name='cloud_top_height', attrs={'area': shared.area, 'orbital_parameters': {'projection_longitude': float(shared.projection_attrs['longitude_of_projection_origin']), 'projection_latitude': float(shared.projection_attrs.get('latitude_of_projection_origin', 0.0)), 'projection_altitude': float(shared.projection_attrs['perspective_point_height'])}, 'platform_name': f'GOES-{shared.satellite}', 'sensor': 'abi', 'start_time': utc_naive(timestamp).to_pydatetime(), 'units': 'm'})
    if shared_key not in cache.correctors:
        cache.correctors[shared_key] = ParallaxCorrection(shared.area)
    corrected_geometry = cache.correctors[shared_key].corrected_area(cth, cth_resampler='nearest', cth_radius_of_influence=50000.0, lonlat_chunks=1024)
    lons, lats = corrected_geometry.get_lonlats()
    if hasattr(lons, 'compute'):
        lons, lats = (lons.compute(), lats.compute())
    corrected_geometry = SwathDefinition(lons=np.asarray(lons), lats=np.asarray(lats))
    mapping = neighbour_info(shared.area, corrected_geometry, 50000.0, nprocs)
    output = {}
    for channel, dataset in cmi.items():
        values = orient_native(dataset['CMI'].values, shared).astype(np.float32)
        output[channel] = sample_with_info(values, shared.area.shape, mapping)
    return (output, shared)

def final_aoi_slices(dataset: xr.Dataset, grid: NativeGrid) -> tuple[slice, slice]:
    x = np.asarray(dataset.x.values)[::-1] if grid.flip_x else np.asarray(dataset.x.values)
    y = np.asarray(dataset.y.values)[::-1] if grid.flip_y else np.asarray(dataset.y.values)
    return crop_slices(x, y, grid.projection_attrs, margin_km=0.0)

def make_output_frame(requested_time: Any, source_rows: pd.DataFrame, corrected: dict[int, np.ndarray], reference: xr.Dataset, grid: NativeGrid, static_cache: ParallaxStaticCache | None=None) -> xr.Dataset:
    requested = utc_naive(requested_time)
    grid_key = native_grid_key(reference)
    if static_cache is not None and grid_key in static_cache.aoi_slices:
        ys, xs = static_cache.aoi_slices[grid_key]
    else:
        ys, xs = final_aoi_slices(reference, grid)
        if static_cache is not None:
            static_cache.aoi_slices[grid_key] = (ys, xs)
    x = np.asarray(reference.x.values)[::-1] if grid.flip_x else np.asarray(reference.x.values)
    y = np.asarray(reference.y.values)[::-1] if grid.flip_y else np.asarray(reference.y.values)
    data = np.stack([corrected[channel][ys, xs] for channel in CHANNELS], axis=0).astype(np.float16)
    cmi_rows = source_rows.loc[source_rows['product'].eq(CMIPC_PRODUCT)].set_index('channel').loc[list(CHANNELS)]
    acha_row = source_rows.loc[source_rows['product'].eq(ACHA_PRODUCT)].iloc[0]
    projection_attrs = {_key: _python_value(_value) for _key, _value in grid.projection_attrs.items()}
    dataset = xr.Dataset(data_vars={'CMI': (('time', 'channel', 'y', 'x'), data[None, ...]), 'source_scan_start': (('time', 'channel'), pd.to_datetime(cmi_rows.scan_start).values[None, :]), 'source_scan_end': (('time', 'channel'), pd.to_datetime(cmi_rows.scan_end).values[None, :]), 'source_age_seconds': (('time', 'channel'), cmi_rows.age_seconds.to_numpy(np.float32)[None, :]), 'acha_scan_start': ('time', np.asarray([pd.Timestamp(acha_row.scan_start).to_datetime64()])), 'acha_scan_end': ('time', np.asarray([pd.Timestamp(acha_row.scan_end).to_datetime64()])), 'acha_age_seconds': ('time', np.asarray([acha_row.age_seconds], dtype=np.float32)), 'satellite': ('time', np.asarray([grid.satellite], dtype=np.int16)), 'goes_imager_projection': ((), np.int8(0))}, coords={'time': ('time', np.asarray([requested.to_datetime64()])), 'channel': ('channel', np.asarray(CHANNELS, dtype=np.int8)), 'x': ('x', x[xs].astype(np.float32)), 'y': ('y', y[ys].astype(np.float32))}, attrs={'title': 'Parallax-corrected GOES ABI IR for strict isolated CI', 'source_products': f'{CMIPC_PRODUCT}, {ACHA_PRODUCT}', 'parallax_algorithm': 'satpy.modifiers.parallax.ParallaxCorrection inverse mapping', 'parallax_corrected': True, 'native_grid': True, 'resampled_to_mrms': False, 'aoi': 'lat 38.5..44.5, lon -76.5..-71.0', 'native_dx_m_approx': float(grid.dx_m), 'native_dy_m_approx': float(grid.dy_m), 'missing_value_policy': 'NaN; missing GOES is never clear sky'})
    dataset['CMI'].attrs.update({'long_name': 'parallax-corrected ABI Cloud and Moisture Imagery', 'units': str(reference['CMI'].attrs.get('units', 'K')), 'grid_mapping': 'goes_imager_projection'})
    dataset['x'].attrs.update(dict(reference.x.attrs))
    dataset['y'].attrs.update(dict(reference.y.attrs))
    dataset['goes_imager_projection'].attrs.update(projection_attrs)
    return dataset

def apply_time_encoding(dataset: xr.Dataset) -> None:
    for name in dataset.variables:
        if np.issubdtype(dataset[name].dtype, np.datetime64):
            dataset[name].encoding.update({'dtype': 'int64', 'units': TIME_UNITS, 'calendar': TIME_CALENDAR})

def output_encoding(dataset: xr.Dataset) -> dict[str, dict[str, Any]]:
    compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
    ny, nx = (int(dataset.sizes['y']), int(dataset.sizes['x']))
    encoding: dict[str, dict[str, Any]] = {'CMI': {'dtype': 'float16', 'compressor': compressor, 'chunks': (1, 7, min(256, ny), min(256, nx)), '_FillValue': np.nan}, 'source_age_seconds': {'dtype': 'float32', 'compressor': compressor, '_FillValue': np.nan}, 'acha_age_seconds': {'dtype': 'float32', 'compressor': compressor, '_FillValue': np.nan}, 'satellite': {'dtype': 'int16', 'compressor': compressor}}
    for name in ('time', 'source_scan_start', 'source_scan_end', 'acha_scan_start', 'acha_scan_end'):
        encoding[name] = {'dtype': 'int64', 'units': TIME_UNITS, 'calendar': TIME_CALENDAR}
    return encoding

def store_times(store: Path) -> pd.DatetimeIndex:
    if not Path(_ci_resolve(store)).exists():
        return pd.DatetimeIndex([])
    with xr.open_zarr(store, consolidated=False) as dataset:
        return pd.DatetimeIndex(pd.to_datetime(dataset.time.values))

def validate_frame_in_store(store: Path, requested_time: Any, expected_shape: tuple[int, int] | None=None) -> dict[str, Any]:
    requested = utc_naive(requested_time)
    with xr.open_zarr(store, consolidated=False) as dataset:
        times = pd.DatetimeIndex(pd.to_datetime(dataset.time.values))
        count = int((times == requested).sum())
        if count != 1:
            raise RuntimeError(f'{requested}: expected exactly once in {store}, found {count}')
        if list(map(int, dataset.channel.values)) != list(CHANNELS):
            raise RuntimeError('Seven-channel coordinate is incomplete or out of order')
        if dataset.CMI.dtype != np.dtype('float16'):
            raise RuntimeError(f'CMI storage dtype is {dataset.CMI.dtype}, not float16')
        shape = (int(dataset.sizes['y']), int(dataset.sizes['x']))
        if expected_shape is not None and tuple(expected_shape) != shape:
            raise RuntimeError(f'Grid changed: expected {expected_shape}, got {shape}')
        index = int(np.flatnonzero(times == requested)[0])
        values = np.asarray(dataset.CMI.isel(time=index).values)
        if values.shape[0] != 7:
            raise RuntimeError('Written frame lacks seven channels')
        ends = pd.to_datetime(dataset.source_scan_end.isel(time=index).values)
        if np.any(ends > requested):
            raise RuntimeError('Written frame contains a future source scan')
        if pd.Timestamp(dataset.acha_scan_end.isel(time=index).values) > requested:
            raise RuntimeError('Written frame contains a future ACHA scan')
        return {'finite_fraction': float(np.isfinite(values).mean()), 'shape': shape, 'time_count': count}

def validate_batch_in_store(store: Path, requested_times: Iterable[Any], expected_shape: tuple[int, int] | None=None) -> dict[pd.Timestamp, float]:
    """Verify a committed batch with one store open and one CMI read."""
    requested = [utc_naive(value) for value in requested_times]
    if len(requested) != len(set(requested)):
        raise RuntimeError('Batch contains duplicate requested timestamps')
    with xr.open_zarr(store, consolidated=False) as dataset:
        times = pd.DatetimeIndex(pd.to_datetime(dataset.time.values))
        if times.has_duplicates:
            raise RuntimeError(f'Duplicate timestamps exist in {store}')
        if list(map(int, dataset.channel.values)) != list(CHANNELS):
            raise RuntimeError('Seven-channel coordinate is incomplete or out of order')
        if dataset.CMI.dtype != np.dtype('float16'):
            raise RuntimeError(f'CMI storage dtype is {dataset.CMI.dtype}, not float16')
        shape = (int(dataset.sizes['y']), int(dataset.sizes['x']))
        if expected_shape is not None and tuple(expected_shape) != shape:
            raise RuntimeError(f'Grid changed: expected {expected_shape}, got {shape}')
        indices = []
        for stamp in requested:
            found = np.flatnonzero(times == stamp)
            if len(found) != 1:
                raise RuntimeError(f'{stamp}: expected exactly once in {store}, found {len(found)}')
            indices.append(int(found[0]))
        subset = dataset.isel(time=indices)
        values = np.asarray(subset.CMI.values)
        if values.shape[1] != 7:
            raise RuntimeError('Written batch lacks seven channels')
        ends = pd.to_datetime(subset.source_scan_end.values.reshape(-1))
        repeated = np.repeat(np.asarray(requested, dtype='datetime64[ns]'), len(CHANNELS))
        if np.any(ends > repeated):
            raise RuntimeError('Written batch contains a future CMI source scan')
        acha_ends = pd.to_datetime(subset.acha_scan_end.values)
        if np.any(acha_ends > np.asarray(requested, dtype='datetime64[ns]')):
            raise RuntimeError('Written batch contains a future ACHA source scan')
        fractions = np.isfinite(values).reshape(len(requested), -1).mean(axis=1)
        return {stamp: float(fraction) for stamp, fraction in zip(requested, fractions)}

@contextlib.contextmanager
def store_lock(store: Path, timeout_seconds: int=600):
    lock = Path(_ci_resolve(str(store) + '.lock'))
    lock.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    descriptor = None
    while descriptor is None:
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(descriptor, f'{os.getpid()}\n'.encode())
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f'Timed out waiting for {lock}')
            time.sleep(2)
    try:
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)
        lock.unlink(missing_ok=True)

def append_frame_verified(frame: xr.Dataset, store: Path) -> dict[str, Any]:
    """Append one frame with duplicate prevention, rollback, and reopen checks."""
    store = Path(_ci_resolve(store))
    requested = utc_naive(frame.time.values[0])
    store.parent.mkdir(parents=True, exist_ok=True)
    apply_time_encoding(frame)
    with store_lock(store):
        existing = store_times(store)
        if int((existing == requested).sum()) == 1:
            result = validate_frame_in_store(store, requested)
            result['already_present'] = True
            return result
        if int((existing == requested).sum()) > 1:
            raise RuntimeError(f'Duplicate timestamp already present: {requested}')
        if store.exists():
            with xr.open_zarr(store, consolidated=False) as current:
                expected_shape = (int(current.sizes['y']), int(current.sizes['x']))
                if expected_shape != (int(frame.sizes['y']), int(frame.sizes['x'])):
                    raise RuntimeError(f"Grid shape changed within store: {expected_shape} vs {(int(frame.sizes['y']), int(frame.sizes['x']))}")
            group = zarr.open_group(_ci_resolve(str(store)), mode='a')
            original = {}
            for name, array in group.arrays():
                dimensions = list(array.attrs.get('_ARRAY_DIMENSIONS', []))
                if dimensions and dimensions[0] == 'time':
                    original[name] = array.shape
            try:
                frame.to_zarr(store, mode='a', append_dim='time', consolidated=False)
                result = validate_frame_in_store(store, requested, expected_shape)
            except Exception:
                for name, shape in original.items():
                    if name in group:
                        group[name].resize(shape)
                raise
        else:
            staging = store.with_name(store.name + f'.creating.{os.getpid()}')
            if staging.exists():
                shutil.rmtree(staging)
            try:
                frame.to_zarr(staging, mode='w', encoding=output_encoding(frame), consolidated=False)
                validate_frame_in_store(staging, requested)
                os.replace(staging, store)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
        if not store.exists():
            raise RuntimeError(f'Store creation failed: {store}')
        if 'result' not in locals() or result.get('time_count') != 1:
            result = validate_frame_in_store(store, requested)
        result['already_present'] = False
        return result

def append_batch_verified(batch: xr.Dataset, store: Path) -> dict[pd.Timestamp, float]:
    """Transactionally append and reopen-verify several unique timestamps."""
    store = Path(_ci_resolve(store))
    requested = [utc_naive(value) for value in np.asarray(batch.time.values)]
    if not requested or len(requested) != len(set(requested)):
        raise RuntimeError('Append batch must contain unique timestamps')
    store.parent.mkdir(parents=True, exist_ok=True)
    apply_time_encoding(batch)
    with store_lock(store):
        journal = Path(_ci_resolve(str(store) + '.append_transaction.json'))
        if journal.exists() and store.exists():
            transaction = json.loads(journal.read_text())
            group = zarr.open_group(_ci_resolve(str(store)), mode='a')
            for name, shape in transaction.get('original_shapes', {}).items():
                if name in group:
                    group[name].resize(tuple((int(value) for value in shape)))
            journal.unlink()
        expected_shape = (int(batch.sizes['y']), int(batch.sizes['x']))
        original: dict[str, tuple[int, ...]] = {}
        if store.exists():
            with xr.open_zarr(store, consolidated=False) as current:
                current_times = pd.DatetimeIndex(pd.to_datetime(current.time.values))
                overlap = current_times.intersection(pd.DatetimeIndex(requested))
                if len(overlap):
                    raise RuntimeError(f'Refusing duplicate batch timestamps: {list(overlap)}')
                current_shape = (int(current.sizes['y']), int(current.sizes['x']))
                if current_shape != expected_shape:
                    raise RuntimeError(f'Grid shape changed within store: {current_shape} vs {expected_shape}')
            group = zarr.open_group(_ci_resolve(str(store)), mode='a')
            for name, array in group.arrays():
                dimensions = list(array.attrs.get('_ARRAY_DIMENSIONS', []))
                if dimensions and dimensions[0] == 'time':
                    original[name] = array.shape
            journal_tmp = journal.with_name(journal.name + f'.tmp.{os.getpid()}')
            journal_tmp.write_text(json.dumps({'requested_times': [stamp.isoformat() for stamp in requested], 'original_shapes': {name: list(shape) for name, shape in original.items()}}, indent=2) + '\n')
            os.replace(journal_tmp, journal)
            try:
                batch.to_zarr(store, mode='a', append_dim='time', consolidated=False)
                fractions = validate_batch_in_store(store, requested, expected_shape)
                journal.unlink(missing_ok=True)
                return fractions
            except BaseException:
                for name, shape in original.items():
                    if name in group:
                        group[name].resize(shape)
                journal.unlink(missing_ok=True)
                raise
        staging = store.with_name(store.name + f'.creating.{os.getpid()}')
        if staging.exists():
            shutil.rmtree(staging)
        try:
            batch.to_zarr(staging, mode='w', encoding=output_encoding(batch), consolidated=False)
            fractions = validate_batch_in_store(staging, requested, expected_shape)
            os.replace(staging, store)
            return fractions
        finally:
            if staging.exists():
                shutil.rmtree(staging)

def safe_cleanup_raw(paths: Iterable[Path], *, append_verified: bool) -> None:
    if not append_verified:
        raise RuntimeError('Refusing raw cleanup before verified append')
    for path in paths:
        Path(_ci_resolve(path)).unlink(missing_ok=True)
    parents = sorted({Path(_ci_resolve(path)).parent for path in paths}, key=lambda p: len(p.parts), reverse=True)
    for parent in parents:
        try:
            parent.rmdir()
        except OSError:
            pass

def load_progress(path: Path) -> pd.DataFrame:
    if not Path(_ci_resolve(path)).exists():
        return pd.DataFrame(columns=PROGRESS_COLUMNS)
    result = pd.read_parquet(_ci_resolve(path))
    result['requested_time'] = pd.to_datetime(result['requested_time'])
    return result

def update_progress(path: Path, requested_time: Any, **updates: Any) -> pd.DataFrame:
    requested = utc_naive(requested_time)
    frame = load_progress(path)
    mask = frame.requested_time.eq(requested) if len(frame) else np.asarray([], dtype=bool)
    base = {column: None for column in PROGRESS_COLUMNS}
    base.update({'requested_time': requested, 'year': requested.year, 'month': requested.month, 'status': 'pending', 'attempts': 0, 'updated_at': pd.Timestamp.utcnow().tz_localize(None)})
    if mask.any():
        row = frame.loc[mask].iloc[-1].to_dict()
        frame = frame.loc[~mask].copy()
        base.update(row)
    base.update(updates)
    base['updated_at'] = pd.Timestamp.utcnow().tz_localize(None)
    frame = pd.concat([frame, pd.DataFrame([base])], ignore_index=True)
    frame = frame[PROGRESS_COLUMNS].sort_values('requested_time').reset_index(drop=True)
    atomic_write_parquet(frame, path)
    return frame

def directory_size(path: Path) -> int:
    if not Path(_ci_resolve(path)).exists():
        return 0
    return sum((item.stat().st_size for item in Path(_ci_resolve(path)).rglob('*') if item.is_file()))
