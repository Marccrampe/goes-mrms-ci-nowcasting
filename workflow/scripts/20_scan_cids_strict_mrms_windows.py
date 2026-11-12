"""
Streaming CIDS-style convective-initiation scan for MRMS composite reflectivity.

Purpose
-------
Scan a large Northeast / Mid-Atlantic AOI without retaining the full MRMS
archive locally. For each 10-min analysis time, the script downloads only the
nearest MRMS Composite Reflectivity file, crops it immediately, applies the
CIDS decision sequence, writes daily Parquet checkpoints, and removes the raw
file.

CIDS-style sequence
-------------------
1. Segment convective systems (CS): contiguous CR > 35 dBZ regions with at
   least 16 radar pixels.
2. For each CS at t, compute dense optical flow CR(t) -> CR(t-10).
3. Retro-advect every current-CS pixel. If zero retro-advected pixels overlap
   a valid CS at t-10, mark it as a potential CI.
4. Compute a separate flow CR(t) -> CR(t+10). If at least one advected pixel
   overlaps a valid CS at t+10, retain it as a persistent CI; otherwise label
   it transient.
5. Follow associated CSs through +10, +20 and +30 min. A CI is "developing"
   when total associated area and maximum dBZ are both greater than their t0
   values at all three future times. Otherwise it is "declining".

Important implementation choices
--------------------------------
- Reproduces the scientific ordering of CIDS, not bugs/hard-coded assumptions
  in the released scripts.
- Uses 8-connected filled regions rather than contour-edge pixels.
- Rejects objects close to the cropped AOI boundary to prevent false births
  caused by objects entering the regional domain.
- Writes both persistent CI and potential-CI diagnostic catalogs.
- Checkpoints every UTC day and supports safe resume.

Example
-------
python scripts/20_scan_cids_strict_mrms.py   --year 2021   --start-date 2021-03-01   --end-date 2021-10-31   --output-root data/ci_catalog_cids_strict_2021_2025   --resume
"""
from __future__ import annotations
from ci_nowcasting.paths import resolve_path as _ci_resolve, project_root as _ci_project, source_root as _ci_source
import argparse
import gc
import gzip
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import xml.etree.ElementTree as ET
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import cv2
import numpy as np
import pandas as pd
import requests
import xarray as xr
from scipy import ndimage
ALGORITHM_VERSION = 'cids_strict_mrms_severe_windows_v2'
PRODUCT_DIRS = ('MergedReflectivityQCComposite_00.50', 'MergedReflectivityQCComposite')
FILE_RE = re.compile('^(?:MRMS_)?MergedReflectivityQCComposite(?:_00\\.50)?_(\\d{8}-\\d{6})\\.grib2(?:\\.gz)?$')
CI_COLUMNS = ['algorithm_version', 'year', 'day', 'ci_id', 'ci_time', 'ci_class', 'is_developing', 'centroid_y_px', 'centroid_x_px', 'latitude', 'longitude', 'area_px_t0', 'max_dbz_t0', 'mean_dbz_t0', 'bbox_y0', 'bbox_y1', 'bbox_x0', 'bbox_x1', 'past_overlap_px', 'past_overlap_fraction', 'future_overlap_px', 'future_overlap_fraction', 'area_px_tplus10', 'area_px_tplus20', 'area_px_tplus30', 'max_dbz_tplus10', 'max_dbz_tplus20', 'max_dbz_tplus30', 'n_objects_tplus10', 'n_objects_tplus20', 'n_objects_tplus30', 'labels_tplus10', 'labels_tplus20', 'labels_tplus30', 'flow_back_median_dx_px', 'flow_back_median_dy_px', 'flow_forward_median_dx_px', 'flow_forward_median_dy_px', 'valid_fraction_tminus10', 'valid_fraction_t0', 'valid_fraction_tplus10', 'valid_fraction_tplus20', 'valid_fraction_tplus30', 'source_tminus10', 'source_t0', 'source_tplus10', 'source_tplus20', 'source_tplus30']
POTENTIAL_COLUMNS = CI_COLUMNS + ['future_confirmed', 'rejection_reason']

@dataclass
class ListedFile:
    valid_time: pd.Timestamp
    url: str
    filename: str
    product_dir: str
    source: str

@dataclass
class ConvectiveObject:
    label_id: int
    area_px: int
    max_dbz: float
    mean_dbz: float
    centroid_y: float
    centroid_x: float
    bbox_y0: int
    bbox_y1: int
    bbox_x0: int
    bbox_x1: int
    touches_edge: bool

@dataclass
class FrameData:
    anchor_time: pd.Timestamp
    actual_time: pd.Timestamp
    source: ListedFile
    dbz: np.ndarray
    valid: np.ndarray
    gray: np.ndarray
    labels: np.ndarray
    objects: Dict[int, ConvectiveObject]
    latitude: np.ndarray
    longitude: np.ndarray
    valid_fraction: float

class DailyLogger:

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, message: str) -> None:
        timestamp = pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        line = f'[{timestamp}] {message}'
        print(line, flush=True)
        with self.log_path.open('a') as f:
            f.write(line + '\n')

class S3MRMSArchive:
    """Public NOAA MRMS S3 listing with AWS-CLI and XML fallbacks."""

    def __init__(self, timeout: int=60):
        self.timeout = int(timeout)
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=8, max_retries=0)
        self.session.mount('https://', adapter)
        self._listing_cache: Dict[str, List[ListedFile]] = {}

    @staticmethod
    def _day_key(day: pd.Timestamp) -> str:
        return pd.Timestamp(day).strftime('%Y%m%d')

    def _list_aws_cli(self, product_dir: str, day: pd.Timestamp) -> List[ListedFile]:
        prefix = f's3://noaa-mrms-pds/CONUS/{product_dir}/{day:%Y%m%d}/'
        https_base = f'https://noaa-mrms-pds.s3.amazonaws.com/CONUS/{product_dir}/{day:%Y%m%d}/'
        try:
            proc = subprocess.run(['aws', 's3', 'ls', '--no-sign-request', prefix], check=False, capture_output=True, text=True, timeout=120)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return []
        if proc.returncode != 0:
            return []
        rows: List[ListedFile] = []
        for line in proc.stdout.splitlines():
            parts = line.split()
            if len(parts) < 4:
                continue
            filename = parts[-1]
            match = FILE_RE.match(filename)
            if not match:
                continue
            valid_time = pd.to_datetime(match.group(1), format='%Y%m%d-%H%M%S')
            rows.append(ListedFile(valid_time=valid_time, url=https_base + filename, filename=filename, product_dir=product_dir, source='aws_cli'))
        return rows

    def _list_s3_xml(self, product_dir: str, day: pd.Timestamp) -> List[ListedFile]:
        bucket_url = 'https://noaa-mrms-pds.s3.amazonaws.com/'
        prefix = f'CONUS/{product_dir}/{day:%Y%m%d}/'
        https_base = bucket_url + prefix
        continuation: Optional[str] = None
        rows: List[ListedFile] = []
        while True:
            params = {'list-type': '2', 'prefix': prefix}
            if continuation:
                params['continuation-token'] = continuation
            response = self.session.get(bucket_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            root = ET.fromstring(response.text)
            namespace = ''
            if root.tag.startswith('{'):
                namespace = root.tag.split('}')[0] + '}'
            for item in root.findall(f'.//{namespace}Contents'):
                key_el = item.find(f'{namespace}Key')
                if key_el is None or not key_el.text:
                    continue
                filename = key_el.text.split('/')[-1]
                match = FILE_RE.match(filename)
                if not match:
                    continue
                valid_time = pd.to_datetime(match.group(1), format='%Y%m%d-%H%M%S')
                rows.append(ListedFile(valid_time=valid_time, url=https_base + filename, filename=filename, product_dir=product_dir, source='s3_xml'))
            truncated_el = root.find(f'{namespace}IsTruncated')
            truncated = truncated_el is not None and str(truncated_el.text).lower() == 'true'
            if not truncated:
                break
            token_el = root.find(f'{namespace}NextContinuationToken')
            if token_el is None or not token_el.text:
                break
            continuation = token_el.text
        return rows

    def list_day(self, day: pd.Timestamp) -> List[ListedFile]:
        day = pd.Timestamp(day).normalize()
        key = self._day_key(day)
        if key in self._listing_cache:
            return self._listing_cache[key]
        all_rows: List[ListedFile] = []
        for product_dir in PRODUCT_DIRS:
            rows = self._list_aws_cli(product_dir, day)
            if not rows:
                try:
                    rows = self._list_s3_xml(product_dir, day)
                except Exception:
                    rows = []
            all_rows.extend(rows)
            if rows:
                break
        deduplicated: Dict[str, ListedFile] = {}
        for row in sorted(all_rows, key=lambda x: x.valid_time):
            deduplicated[row.filename] = row
        rows = sorted(deduplicated.values(), key=lambda x: x.valid_time)
        self._listing_cache[key] = rows
        return rows

    def nearest_file(self, anchor: pd.Timestamp, tolerance_seconds: float) -> Optional[ListedFile]:
        anchor = pd.Timestamp(anchor)
        candidates: List[ListedFile] = []
        for day in [anchor.normalize() - pd.Timedelta(days=1), anchor.normalize(), anchor.normalize() + pd.Timedelta(days=1)]:
            candidates.extend(self.list_day(day))
        if not candidates:
            return None
        times_ns = np.array([row.valid_time.value for row in candidates], dtype=np.int64)
        target_ns = anchor.value
        index = int(np.argmin(np.abs(times_ns - target_ns)))
        offset_seconds = abs(int(times_ns[index]) - int(target_ns)) / 1000000000.0
        if offset_seconds > tolerance_seconds:
            return None
        return candidates[index]

    def download(self, listed_file: ListedFile, destination: Path, retries: int=3) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.stat().st_size > 0:
            return destination
        partial = destination.with_suffix(destination.suffix + '.part')
        for attempt in range(retries + 1):
            try:
                with self.session.get(listed_file.url, stream=True, timeout=(30, 180)) as response:
                    response.raise_for_status()
                    with partial.open('wb') as handle:
                        for chunk in response.iter_content(1024 * 1024):
                            if chunk:
                                handle.write(chunk)
                if partial.stat().st_size <= 0:
                    raise IOError('empty download')
                partial.replace(destination)
                return destination
            except Exception:
                partial.unlink(missing_ok=True)
                if attempt >= retries:
                    raise
                time.sleep(2.0 * (attempt + 1))
        raise RuntimeError('unreachable')

def choose_data_array(dataset: xr.Dataset) -> xr.DataArray:
    candidates: List[xr.DataArray] = []
    for name in dataset.data_vars:
        array = dataset[name].squeeze(drop=True)
        if array.ndim >= 2:
            candidates.append(array)
    if not candidates:
        raise ValueError(f'No 2-D data variable found: {list(dataset.data_vars)}')
    candidates.sort(key=lambda arr: int(np.prod(arr.shape)), reverse=True)
    return candidates[0]

def find_coordinate(dataset: xr.Dataset, array: xr.DataArray, names: Sequence[str]) -> Optional[np.ndarray]:
    lower_names = {str(name).lower(): name for name in array.coords}
    lower_names.update({str(name).lower(): name for name in dataset.coords})
    for candidate in names:
        key = lower_names.get(candidate.lower())
        if key is None:
            continue
        source = array.coords[key] if key in array.coords else dataset.coords[key]
        return np.asarray(source.values)
    return None

def crop_to_bbox(data: np.ndarray, latitude: np.ndarray, longitude: np.ndarray, lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.asarray(data)
    latitude = np.asarray(latitude)
    longitude = np.asarray(longitude)
    longitude = np.where(longitude > 180.0, longitude - 360.0, longitude)
    if latitude.ndim == 1 and longitude.ndim == 1:
        if data.shape[-2] != latitude.size or data.shape[-1] != longitude.size:
            if data.shape[-2] == longitude.size and data.shape[-1] == latitude.size:
                data = data.T
            else:
                raise ValueError(f'Coordinate lengths do not match the data shape: data={data.shape}, lat={latitude.shape}, lon={longitude.shape}')
        y_indices = np.where((latitude >= lat_min) & (latitude <= lat_max))[0]
        x_indices = np.where((longitude >= lon_min) & (longitude <= lon_max))[0]
        if len(y_indices) == 0 or len(x_indices) == 0:
            raise ValueError('Requested AOI does not intersect the MRMS grid')
        y0, y1 = (int(y_indices.min()), int(y_indices.max()) + 1)
        x0, x1 = (int(x_indices.min()), int(x_indices.max()) + 1)
        return (data[y0:y1, x0:x1], latitude[y0:y1], longitude[x0:x1])
    if latitude.ndim == 2 and longitude.ndim == 2:
        if latitude.shape != data.shape or longitude.shape != data.shape:
            raise ValueError(f'2-D coordinate mismatch: data={data.shape}, lat={latitude.shape}, lon={longitude.shape}')
        inside = (latitude >= lat_min) & (latitude <= lat_max) & (longitude >= lon_min) & (longitude <= lon_max)
        rows, cols = np.where(inside)
        if rows.size == 0:
            raise ValueError('Requested AOI does not intersect the MRMS grid')
        y0, y1 = (int(rows.min()), int(rows.max()) + 1)
        x0, x1 = (int(cols.min()), int(cols.max()) + 1)
        return (data[y0:y1, x0:x1], latitude[y0:y1, x0:x1], longitude[y0:y1, x0:x1])
    raise ValueError(f'Unsupported coordinate dimensionality: lat={latitude.shape}, lon={longitude.shape}')

def decode_grib_crop(compressed_path: Path, work_dir: Path, bbox: Tuple[float, float, float, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    work_dir.mkdir(parents=True, exist_ok=True)
    grib_path = compressed_path
    temporary_grib: Optional[Path] = None
    if compressed_path.suffix == '.gz':
        temporary_grib = work_dir / compressed_path.name[:-3]
        with gzip.open(compressed_path, 'rb') as src, temporary_grib.open('wb') as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
        grib_path = temporary_grib
    dataset: Optional[xr.Dataset] = None
    try:
        dataset = xr.open_dataset(_ci_resolve(grib_path), engine='cfgrib', backend_kwargs={'indexpath': ''})
        array = choose_data_array(dataset)
        values = np.asarray(array.squeeze(drop=True).values)
        while values.ndim > 2:
            values = values[0]
        latitude = find_coordinate(dataset, array, ['latitude', 'lat', 'y'])
        longitude = find_coordinate(dataset, array, ['longitude', 'lon', 'x'])
        if latitude is None or longitude is None:
            raise ValueError('Could not find latitude/longitude coordinates in GRIB2')
        cropped, lat_crop, lon_crop = crop_to_bbox(values, latitude, longitude, lat_min=bbox[0], lat_max=bbox[1], lon_min=bbox[2], lon_max=bbox[3])
        cropped = np.asarray(cropped, dtype=np.float32)
        finite = np.isfinite(cropped)
        valid = finite & (cropped > -900.0)
        cropped = np.where(finite & (cropped > -90.0), cropped, -10.0)
        cropped = cropped.astype(np.float32)
        return (cropped, valid, np.asarray(lat_crop), np.asarray(lon_crop))
    finally:
        if dataset is not None:
            dataset.close()
        if temporary_grib is not None:
            temporary_grib.unlink(missing_ok=True)

def cids_gray(dbz: np.ndarray) -> np.ndarray:
    scaled = np.zeros(dbz.shape, dtype=np.float32)
    middle = (dbz >= 20.0) & (dbz <= 80.0)
    scaled[middle] = 253.0 * (dbz[middle] - 20.0) / 60.0
    scaled[dbz > 80.0] = 255.0
    return cv2.GaussianBlur(scaled, (5, 5), 0)

def coordinate_at(latitude: np.ndarray, longitude: np.ndarray, y: float, x: float) -> Tuple[float, float]:
    iy = int(np.clip(round(y), 0, latitude.shape[0] - 1)) if latitude.ndim == 1 else int(np.clip(round(y), 0, latitude.shape[0] - 1))
    ix = int(np.clip(round(x), 0, longitude.shape[-1] - 1))
    if latitude.ndim == 1:
        lat = float(latitude[iy])
    else:
        lat = float(latitude[iy, ix])
    if longitude.ndim == 1:
        lon = float(longitude[ix])
    else:
        lon = float(longitude[iy, ix])
    return (lat, lon)

def segment_convective_systems(dbz: np.ndarray, valid: np.ndarray, threshold_dbz: float, min_area_px: int, edge_buffer_px: int) -> Tuple[np.ndarray, Dict[int, ConvectiveObject]]:
    raw_mask = valid & (dbz > threshold_dbz)
    raw_labels, n_raw = ndimage.label(raw_mask, structure=np.ones((3, 3), dtype=np.uint8))
    slices = ndimage.find_objects(raw_labels)
    output_labels = np.zeros(raw_labels.shape, dtype=np.int32)
    objects: Dict[int, ConvectiveObject] = {}
    next_id = 1
    height, width = dbz.shape
    for raw_id in range(1, n_raw + 1):
        object_slice = slices[raw_id - 1]
        if object_slice is None:
            continue
        local = raw_labels[object_slice] == raw_id
        area_px = int(local.sum())
        if area_px < min_area_px:
            continue
        ys_local, xs_local = np.where(local)
        y0, x0 = (object_slice[0].start, object_slice[1].start)
        ys = ys_local + y0
        xs = xs_local + x0
        values = dbz[ys, xs]
        bbox_y0, bbox_y1 = (int(ys.min()), int(ys.max()) + 1)
        bbox_x0, bbox_x1 = (int(xs.min()), int(xs.max()) + 1)
        touches_edge = bbox_y0 < edge_buffer_px or bbox_x0 < edge_buffer_px or bbox_y1 > height - edge_buffer_px or (bbox_x1 > width - edge_buffer_px)
        output_labels[ys, xs] = next_id
        objects[next_id] = ConvectiveObject(label_id=next_id, area_px=area_px, max_dbz=float(np.max(values)), mean_dbz=float(np.mean(values)), centroid_y=float(np.mean(ys)), centroid_x=float(np.mean(xs)), bbox_y0=bbox_y0, bbox_y1=bbox_y1, bbox_x0=bbox_x0, bbox_x1=bbox_x1, touches_edge=bool(touches_edge))
        next_id += 1
    return (output_labels, objects)

def object_pixels(labels: np.ndarray, label_ids: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    label_ids = [int(value) for value in label_ids if int(value) > 0]
    if not label_ids:
        return (np.array([], dtype=np.int32), np.array([], dtype=np.int32))
    mask = np.isin(labels, label_ids)
    ys, xs = np.where(mask)
    return (ys.astype(np.int32), xs.astype(np.int32))

def warp_pixels(ys: np.ndarray, xs: np.ndarray, flow: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if ys.size == 0:
        empty = np.array([], dtype=np.int32)
        return (empty, empty, np.array([], dtype=bool))
    dx = flow[ys, xs, 0]
    dy = flow[ys, xs, 1]
    warped_x = np.rint(xs.astype(np.float32) + dx).astype(np.int32)
    warped_y = np.rint(ys.astype(np.float32) + dy).astype(np.int32)
    valid = (warped_y >= 0) & (warped_y < flow.shape[0]) & (warped_x >= 0) & (warped_x < flow.shape[1])
    return (warped_y, warped_x, valid)

def overlap_labels_after_flow(source_labels: np.ndarray, source_label_ids: Sequence[int], flow: np.ndarray, destination_labels: np.ndarray) -> Tuple[int, float, List[int], float, float]:
    ys, xs = object_pixels(source_labels, source_label_ids)
    total = int(ys.size)
    if total == 0:
        return (0, 0.0, [], float('nan'), float('nan'))
    warped_y, warped_x, valid = warp_pixels(ys, xs, flow)
    hit_values = np.zeros(total, dtype=np.int32)
    hit_values[valid] = destination_labels[warped_y[valid], warped_x[valid]]
    overlap_px = int(np.count_nonzero(hit_values > 0))
    labels_hit = sorted((int(value) for value in np.unique(hit_values) if value > 0))
    median_dx = float(np.median(flow[ys, xs, 0]))
    median_dy = float(np.median(flow[ys, xs, 1]))
    return (overlap_px, float(overlap_px / total), labels_hit, median_dx, median_dy)

def aggregate_objects(frame: FrameData, label_ids: Sequence[int]) -> Tuple[float, float, int]:
    selected = [frame.objects[int(label)] for label in label_ids if int(label) in frame.objects]
    if not selected:
        return (float('nan'), float('nan'), 0)
    return (float(sum((obj.area_px for obj in selected))), float(max((obj.max_dbz for obj in selected))), len(selected))

def farneback(source_gray: np.ndarray, destination_gray: np.ndarray) -> np.ndarray:
    return cv2.calcOpticalFlowFarneback(source_gray, destination_gray, None, 0.5, 5, 25, 5, 7, 1.5, 0)

class FrameCache:

    def __init__(self, archive: S3MRMSArchive, temp_root: Path, bbox: Tuple[float, float, float, float], tolerance_seconds: float, threshold_dbz: float, min_area_px: int, edge_buffer_px: int, max_frames: int=8):
        self.archive = archive
        self.temp_root = temp_root
        self.bbox = bbox
        self.tolerance_seconds = float(tolerance_seconds)
        self.threshold_dbz = float(threshold_dbz)
        self.min_area_px = int(min_area_px)
        self.edge_buffer_px = int(edge_buffer_px)
        self.max_frames = int(max_frames)
        self.frames: OrderedDict[str, FrameData] = OrderedDict()
        self.flows: OrderedDict[Tuple[str, str], np.ndarray] = OrderedDict()

    @staticmethod
    def _key(timestamp: pd.Timestamp) -> str:
        return pd.Timestamp(timestamp).strftime('%Y%m%d%H%M')

    def get(self, anchor: pd.Timestamp) -> Optional[FrameData]:
        anchor = pd.Timestamp(anchor)
        key = self._key(anchor)
        if key in self.frames:
            self.frames.move_to_end(key)
            return self.frames[key]
        listed = self.archive.nearest_file(anchor, self.tolerance_seconds)
        if listed is None:
            return None
        day_dir = self.temp_root / f'{listed.valid_time:%Y%m%d}'
        day_dir.mkdir(parents=True, exist_ok=True)
        raw_path = day_dir / listed.filename
        self.archive.download(listed, raw_path)
        try:
            dbz, valid, latitude, longitude = decode_grib_crop(raw_path, work_dir=day_dir / 'decoded', bbox=self.bbox)
        finally:
            raw_path.unlink(missing_ok=True)
        labels, objects = segment_convective_systems(dbz, valid, threshold_dbz=self.threshold_dbz, min_area_px=self.min_area_px, edge_buffer_px=self.edge_buffer_px)
        frame = FrameData(anchor_time=anchor, actual_time=listed.valid_time, source=listed, dbz=dbz, valid=valid, gray=cids_gray(dbz), labels=labels, objects=objects, latitude=latitude, longitude=longitude, valid_fraction=float(valid.mean()))
        self.frames[key] = frame
        self.frames.move_to_end(key)
        while len(self.frames) > self.max_frames:
            self.frames.popitem(last=False)
        return frame

    def flow(self, source: FrameData, destination: FrameData) -> np.ndarray:
        key = (self._key(source.anchor_time), self._key(destination.anchor_time))
        if key in self.flows:
            self.flows.move_to_end(key)
            return self.flows[key]
        result = farneback(source.gray, destination.gray)
        self.flows[key] = result
        self.flows.move_to_end(key)
        while len(self.flows) > 8:
            self.flows.popitem(last=False)
        return result

    def clear(self) -> None:
        self.frames.clear()
        self.flows.clear()
        gc.collect()

def empty_frame(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype='object') for column in columns})

def write_parquet_atomic(frame: pd.DataFrame, path: Path, columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        frame = empty_frame(columns)
    else:
        for column in columns:
            if column not in frame.columns:
                frame[column] = np.nan
        frame = frame[list(columns)]
    temporary = path.with_suffix(path.suffix + '.tmp')
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)

def build_base_row(year: int, day: pd.Timestamp, ci_index: int, current: FrameData, previous: FrameData, future10: FrameData, future20: FrameData, future30: FrameData, obj: ConvectiveObject, past_overlap_px: int, past_overlap_fraction: float, future_overlap_px: int, future_overlap_fraction: float, back_dx: float, back_dy: float, forward_dx: float, forward_dy: float) -> Dict[str, object]:
    latitude, longitude = coordinate_at(current.latitude, current.longitude, obj.centroid_y, obj.centroid_x)
    ci_time = current.actual_time
    return {'algorithm_version': ALGORITHM_VERSION, 'year': int(year), 'day': day.strftime('%Y-%m-%d'), 'ci_id': f'{ci_time:%Y%m%dT%H%M%S}_{ci_index:04d}', 'ci_time': ci_time, 'centroid_y_px': obj.centroid_y, 'centroid_x_px': obj.centroid_x, 'latitude': latitude, 'longitude': longitude, 'area_px_t0': obj.area_px, 'max_dbz_t0': obj.max_dbz, 'mean_dbz_t0': obj.mean_dbz, 'bbox_y0': obj.bbox_y0, 'bbox_y1': obj.bbox_y1, 'bbox_x0': obj.bbox_x0, 'bbox_x1': obj.bbox_x1, 'past_overlap_px': past_overlap_px, 'past_overlap_fraction': past_overlap_fraction, 'future_overlap_px': future_overlap_px, 'future_overlap_fraction': future_overlap_fraction, 'flow_back_median_dx_px': back_dx, 'flow_back_median_dy_px': back_dy, 'flow_forward_median_dx_px': forward_dx, 'flow_forward_median_dy_px': forward_dy, 'valid_fraction_tminus10': previous.valid_fraction, 'valid_fraction_t0': current.valid_fraction, 'valid_fraction_tplus10': future10.valid_fraction, 'valid_fraction_tplus20': future20.valid_fraction, 'valid_fraction_tplus30': future30.valid_fraction, 'source_tminus10': previous.source.filename, 'source_t0': current.source.filename, 'source_tplus10': future10.source.filename, 'source_tplus20': future20.source.filename, 'source_tplus30': future30.source.filename}

def object_local_valid_fraction(valid: np.ndarray, obj: ConvectiveObject, padding_px: int=30) -> float:
    """Return radar-valid fraction in a local box around one CS.

    Large regional AOIs naturally include areas outside MRMS coverage over
    Canada/ocean. QC must therefore be local to the object rather than based
    on the valid fraction of the complete cropped domain.
    """
    height, width = valid.shape
    y0 = max(0, obj.bbox_y0 - padding_px)
    y1 = min(height, obj.bbox_y1 + padding_px)
    x0 = max(0, obj.bbox_x0 - padding_px)
    x1 = min(width, obj.bbox_x1 + padding_px)
    local = valid[y0:y1, x0:x1]
    if local.size == 0:
        return 0.0
    return float(np.mean(local))

def load_window_anchors(manifest_path: Path, year: int, analysis_step_minutes: int) -> tuple[dict[pd.Timestamp, pd.DatetimeIndex], pd.DataFrame]:
    """Load severe-event windows and return unique anchors grouped by UTC day."""
    if manifest_path.suffix.lower() == '.parquet':
        windows = pd.read_parquet(_ci_resolve(manifest_path))
    else:
        windows = pd.read_csv(_ci_resolve(manifest_path))
    required = {'window_start_utc', 'window_end_utc'}
    missing = required.difference(windows.columns)
    if missing:
        raise ValueError(f'Window manifest missing columns: {sorted(missing)}')
    windows = windows.copy()
    windows['window_start_utc'] = pd.to_datetime(windows['window_start_utc'], errors='coerce')
    windows['window_end_utc'] = pd.to_datetime(windows['window_end_utc'], errors='coerce')
    windows = windows.dropna(subset=['window_start_utc', 'window_end_utc'])
    year_start = pd.Timestamp(year=year, month=1, day=1)
    year_end = pd.Timestamp(year=year + 1, month=1, day=1)
    windows = windows.loc[(windows['window_end_utc'] >= year_start) & (windows['window_start_utc'] < year_end)].copy()
    all_anchors: list[pd.Timestamp] = []
    frequency = f'{analysis_step_minutes}min'
    for row in windows.itertuples(index=False):
        start = max(pd.Timestamp(row.window_start_utc), year_start)
        end = min(pd.Timestamp(row.window_end_utc), year_end - pd.Timedelta(seconds=1))
        start = start.ceil(frequency)
        end = end.floor(frequency)
        if end < start:
            continue
        all_anchors.extend(pd.date_range(start, end, freq=frequency).tolist())
    unique = pd.DatetimeIndex(sorted(set(all_anchors)))
    grouped: dict[pd.Timestamp, pd.DatetimeIndex] = {}
    if len(unique):
        for day, values in pd.Series(unique, index=unique).groupby(unique.normalize()):
            grouped[pd.Timestamp(day)] = pd.DatetimeIndex(values.tolist())
    return (grouped, windows)

def analyze_time(year: int, day: pd.Timestamp, anchor: pd.Timestamp, frames: Sequence[FrameData], cache: FrameCache, min_valid_fraction: float, ci_counter_start: int) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, int]]:
    previous, current, future10, future20, future30 = frames
    stats = {'objects_current': len(current.objects), 'objects_edge_rejected': 0, 'objects_data_rejected': 0, 'potential_ci': 0, 'persistent_ci': 0, 'transient_echo': 0, 'developing_ci': 0, 'declining_ci': 0}
    ci_rows: List[Dict[str, object]] = []
    potential_rows: List[Dict[str, object]] = []
    flow_backward = cache.flow(current, previous)
    flow_forward = cache.flow(current, future10)
    flow_10_20 = cache.flow(future10, future20)
    flow_20_30 = cache.flow(future20, future30)
    ci_counter = ci_counter_start
    for label_id, obj in current.objects.items():
        if obj.touches_edge:
            stats['objects_edge_rejected'] += 1
            continue
        local_valid_fractions = [object_local_valid_fraction(frame.valid, obj, padding_px=30) for frame in frames]
        if min(local_valid_fractions) < min_valid_fraction:
            stats['objects_data_rejected'] += 1
            continue
        past_overlap_px, past_overlap_fraction, _, back_dx, back_dy = overlap_labels_after_flow(current.labels, [label_id], flow_backward, previous.labels)
        if past_overlap_px != 0:
            continue
        stats['potential_ci'] += 1
        future_overlap_px, future_overlap_fraction, labels10, forward_dx, forward_dy = overlap_labels_after_flow(current.labels, [label_id], flow_forward, future10.labels)
        base = build_base_row(year=year, day=day, ci_index=ci_counter, current=current, previous=previous, future10=future10, future20=future20, future30=future30, obj=obj, past_overlap_px=past_overlap_px, past_overlap_fraction=past_overlap_fraction, future_overlap_px=future_overlap_px, future_overlap_fraction=future_overlap_fraction, back_dx=back_dx, back_dy=back_dy, forward_dx=forward_dx, forward_dy=forward_dy)
        ci_counter += 1
        if future_overlap_px <= 0 or not labels10:
            row = dict(base)
            row.update({'ci_class': 'transient_echo', 'is_developing': False, 'future_confirmed': False, 'rejection_reason': 'no_overlap_at_tplus10', 'labels_tplus10': '', 'labels_tplus20': '', 'labels_tplus30': ''})
            potential_rows.append(row)
            stats['transient_echo'] += 1
            continue
        stats['persistent_ci'] += 1
        area10, max10, n10 = aggregate_objects(future10, labels10)
        _, _, labels20, _, _ = overlap_labels_after_flow(future10.labels, labels10, flow_10_20, future20.labels)
        area20, max20, n20 = aggregate_objects(future20, labels20)
        labels30: List[int] = []
        if labels20:
            _, _, labels30, _, _ = overlap_labels_after_flow(future20.labels, labels20, flow_20_30, future30.labels)
        area30, max30, n30 = aggregate_objects(future30, labels30)
        developing = bool(labels10 and labels20 and labels30 and (area10 > obj.area_px) and (area20 > obj.area_px) and (area30 > obj.area_px) and (max10 > obj.max_dbz) and (max20 > obj.max_dbz) and (max30 > obj.max_dbz))
        ci_class = 'developing' if developing else 'declining'
        stats[f'{ci_class}_ci'] += 1
        row = dict(base)
        row.update({'ci_class': ci_class, 'is_developing': developing, 'area_px_tplus10': area10, 'area_px_tplus20': area20, 'area_px_tplus30': area30, 'max_dbz_tplus10': max10, 'max_dbz_tplus20': max20, 'max_dbz_tplus30': max30, 'n_objects_tplus10': n10, 'n_objects_tplus20': n20, 'n_objects_tplus30': n30, 'labels_tplus10': ','.join(map(str, labels10)), 'labels_tplus20': ','.join(map(str, labels20)), 'labels_tplus30': ','.join(map(str, labels30))})
        ci_rows.append(row)
        potential_row = dict(row)
        potential_row.update({'future_confirmed': True, 'rejection_reason': ''})
        potential_rows.append(potential_row)
    return (ci_rows, potential_rows, stats)

def merge_daily_catalogs(year_dir: Path, year: int) -> Tuple[Path, Path]:
    daily_dir = year_dir / 'daily'
    ci_files = sorted(daily_dir.glob('ci_*.parquet'))
    potential_files = sorted(daily_dir.glob('potential_*.parquet'))
    ci_frames = [pd.read_parquet(_ci_resolve(path)) for path in ci_files]
    ci_frames = [frame for frame in ci_frames if len(frame)]
    ci = pd.concat(ci_frames, ignore_index=True) if ci_frames else empty_frame(CI_COLUMNS)
    if len(ci):
        ci['ci_time'] = pd.to_datetime(ci['ci_time'])
        ci = ci.sort_values(['ci_time', 'latitude', 'longitude']).reset_index(drop=True)
    potential_frames = [pd.read_parquet(_ci_resolve(path)) for path in potential_files]
    potential_frames = [frame for frame in potential_frames if len(frame)]
    potential = pd.concat(potential_frames, ignore_index=True) if potential_frames else empty_frame(POTENTIAL_COLUMNS)
    if len(potential):
        potential['ci_time'] = pd.to_datetime(potential['ci_time'])
        potential = potential.sort_values(['ci_time', 'latitude', 'longitude']).reset_index(drop=True)
    ci_path = year_dir / f'cids_strict_ci_{year}.parquet'
    potential_path = year_dir / f'cids_strict_potential_{year}.parquet'
    write_parquet_atomic(ci, ci_path, CI_COLUMNS)
    write_parquet_atomic(potential, potential_path, POTENTIAL_COLUMNS)
    return (ci_path, potential_path)

def process_day(day: pd.Timestamp, year: int, cache: FrameCache, output_dir: Path, analysis_step_minutes: int, min_valid_fraction: float, logger: DailyLogger, anchors: Optional[pd.DatetimeIndex]=None) -> Dict[str, object]:
    daily_dir = output_dir / 'daily'
    date_key = day.strftime('%Y%m%d')
    ci_path = daily_dir / f'ci_{date_key}.parquet'
    potential_path = daily_dir / f'potential_{date_key}.parquet'
    stats_path = daily_dir / f'stats_{date_key}.json'
    if anchors is None:
        anchors = pd.date_range(day.normalize(), day.normalize() + pd.Timedelta(days=1) - pd.Timedelta(minutes=analysis_step_minutes), freq=f'{analysis_step_minutes}min')
    else:
        anchors = pd.DatetimeIndex(sorted(pd.to_datetime(anchors).unique()))
    all_ci: List[Dict[str, object]] = []
    all_potential: List[Dict[str, object]] = []
    totals = {'day': day.strftime('%Y-%m-%d'), 'anchors_total': len(anchors), 'anchors_complete': 0, 'anchors_missing': 0, 'objects_current': 0, 'objects_edge_rejected': 0, 'objects_data_rejected': 0, 'potential_ci': 0, 'persistent_ci': 0, 'transient_echo': 0, 'developing_ci': 0, 'declining_ci': 0, 'errors': 0}
    ci_counter = 1
    for anchor_index, anchor in enumerate(anchors, start=1):
        required_times = [anchor - pd.Timedelta(minutes=10), anchor, anchor + pd.Timedelta(minutes=10), anchor + pd.Timedelta(minutes=20), anchor + pd.Timedelta(minutes=30)]
        try:
            frames = [cache.get(timestamp) for timestamp in required_times]
            if any((frame is None for frame in frames)):
                totals['anchors_missing'] += 1
                continue
            totals['anchors_complete'] += 1
            ci_rows, potential_rows, stats = analyze_time(year=year, day=day, anchor=anchor, frames=frames, cache=cache, min_valid_fraction=min_valid_fraction, ci_counter_start=ci_counter)
            ci_counter += len(potential_rows)
            all_ci.extend(ci_rows)
            all_potential.extend(potential_rows)
            for key, value in stats.items():
                totals[key] += int(value)
        except Exception as exc:
            totals['errors'] += 1
            logger.write(f'ERROR {anchor}: {type(exc).__name__}: {exc}\n' + traceback.format_exc(limit=3))
        if anchor_index % 24 == 0:
            logger.write(f"{day:%Y-%m-%d} {anchor_index:03d}/{len(anchors)} anchors | potential={totals['potential_ci']} persistent={totals['persistent_ci']} developing={totals['developing_ci']}")
    ci_frame = pd.DataFrame(all_ci)
    potential_frame = pd.DataFrame(all_potential)
    write_parquet_atomic(ci_frame, ci_path, CI_COLUMNS)
    write_parquet_atomic(potential_frame, potential_path, POTENTIAL_COLUMNS)
    stats_path.write_text(json.dumps(totals, indent=2, default=str))
    cache.clear()
    logger.write(f"DONE {day:%Y-%m-%d} | persistent={totals['persistent_ci']} developing={totals['developing_ci']} transient={totals['transient_echo']} missing_anchors={totals['anchors_missing']} errors={totals['errors']}")
    return totals

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--start-date', default=None)
    parser.add_argument('--end-date', default=None)
    parser.add_argument('--window-manifest', type=Path, default=None, help='Parquet/CSV with window_start_utc and window_end_utc. When supplied, only anchors inside these windows are scanned.')
    parser.add_argument('--bbox', nargs=4, type=float, metavar=('LAT_MIN', 'LAT_MAX', 'LON_MIN', 'LON_MAX'), default=(36.0, 46.0, -81.0, -68.0))
    parser.add_argument('--threshold-dbz', type=float, default=35.0)
    parser.add_argument('--min-area-px', type=int, default=16)
    parser.add_argument('--analysis-step-min', type=int, default=10)
    parser.add_argument('--time-tolerance-sec', type=float, default=180.0)
    parser.add_argument('--edge-buffer-px', type=int, default=10)
    parser.add_argument('--min-valid-fraction', type=float, default=0.98)
    parser.add_argument('--output-root', type=Path, default=Path(_ci_resolve('data/ci_catalog_cids_strict_2021_2025')))
    parser.add_argument('--temp-root', type=Path, default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--max-days', type=int, default=None)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    project_root = Path.cwd()
    manifest_path: Optional[Path] = None
    day_anchors: Optional[dict[pd.Timestamp, pd.DatetimeIndex]] = None
    selected_windows = pd.DataFrame()
    if args.window_manifest is not None:
        manifest_path = args.window_manifest
        if not manifest_path.is_absolute():
            manifest_path = project_root / manifest_path
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        day_anchors, selected_windows = load_window_anchors(manifest_path=manifest_path, year=args.year, analysis_step_minutes=args.analysis_step_min)
        start_date = min(day_anchors) if day_anchors else pd.Timestamp(args.year, 1, 1)
        end_date = max(day_anchors) if day_anchors else pd.Timestamp(args.year, 12, 31)
    else:
        if args.start_date is None or args.end_date is None:
            raise ValueError('Provide --window-manifest, or both --start-date and --end-date')
        start_date = pd.Timestamp(args.start_date).normalize()
        end_date = pd.Timestamp(args.end_date).normalize()
        if start_date.year != args.year or end_date.year != args.year:
            raise ValueError('start-date and end-date must belong to --year')
        if end_date < start_date:
            raise ValueError('end-date is before start-date')
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = project_root / output_root
    year_dir = output_root / str(args.year)
    daily_dir = year_dir / 'daily'
    daily_dir.mkdir(parents=True, exist_ok=True)
    if args.temp_root is None:
        base_temp = Path(_ci_resolve(os.environ.get('TMPDIR', f"/tmp/{os.environ.get('USER', 'user')}")))
        temp_root = base_temp / f'cids_mrms_{args.year}'
    else:
        temp_root = args.temp_root
    temp_root.mkdir(parents=True, exist_ok=True)
    logger = DailyLogger(year_dir / f'scan_{args.year}.log')
    config = {'algorithm_version': ALGORITHM_VERSION, 'year': args.year, 'start_date': str(start_date.date()), 'end_date': str(end_date.date()), 'window_manifest': str(manifest_path) if manifest_path else None, 'selected_windows': int(len(selected_windows)), 'selected_anchors': int(sum((len(values) for values in day_anchors.values()))) if day_anchors is not None else None, 'bbox': list(args.bbox), 'threshold_dbz': args.threshold_dbz, 'min_area_px': args.min_area_px, 'analysis_step_min': args.analysis_step_min, 'time_tolerance_sec': args.time_tolerance_sec, 'edge_buffer_px': args.edge_buffer_px, 'min_valid_fraction': args.min_valid_fraction, 'output_root': str(output_root), 'temp_root': str(temp_root), 'python': sys.executable}
    (year_dir / 'config.json').write_text(json.dumps(config, indent=2))
    logger.write('CONFIG ' + json.dumps(config, sort_keys=True))
    archive = S3MRMSArchive()
    cache = FrameCache(archive=archive, temp_root=temp_root, bbox=tuple((float(value) for value in args.bbox)), tolerance_seconds=args.time_tolerance_sec, threshold_dbz=args.threshold_dbz, min_area_px=args.min_area_px, edge_buffer_px=args.edge_buffer_px)
    if day_anchors is not None:
        days = pd.DatetimeIndex(sorted(day_anchors))
    else:
        days = pd.date_range(start_date, end_date, freq='D')
    if args.max_days is not None:
        days = days[:args.max_days]
    yearly_stats: List[Dict[str, object]] = []
    for day_index, day in enumerate(days, start=1):
        date_key = day.strftime('%Y%m%d')
        ci_path = daily_dir / f'ci_{date_key}.parquet'
        potential_path = daily_dir / f'potential_{date_key}.parquet'
        stats_path = daily_dir / f'stats_{date_key}.json'
        complete = ci_path.exists() and potential_path.exists() and stats_path.exists()
        if complete and args.resume and (not args.overwrite):
            logger.write(f'SKIP completed {day:%Y-%m-%d} ({day_index}/{len(days)})')
            yearly_stats.append(json.loads(stats_path.read_text()))
            continue
        if args.overwrite:
            ci_path.unlink(missing_ok=True)
            potential_path.unlink(missing_ok=True)
            stats_path.unlink(missing_ok=True)
        logger.write(f'START {day:%Y-%m-%d} ({day_index}/{len(days)})')
        stats = process_day(day=day, year=args.year, cache=cache, output_dir=year_dir, analysis_step_minutes=args.analysis_step_min, min_valid_fraction=args.min_valid_fraction, logger=logger, anchors=day_anchors.get(day) if day_anchors is not None else None)
        yearly_stats.append(stats)
    stats_frame = pd.DataFrame(yearly_stats)
    stats_frame.to_csv(year_dir / f'daily_stats_{args.year}.csv', index=False)
    ci_path, potential_path = merge_daily_catalogs(year_dir, args.year)
    summary = {'year': args.year, 'days_requested': len(days), 'days_completed': int(len(stats_frame)), 'window_manifest': str(manifest_path) if manifest_path else None, 'selected_windows': int(len(selected_windows)), 'selected_anchors': int(stats_frame.get('anchors_total', pd.Series(dtype=int)).sum()), 'persistent_ci': int(stats_frame.get('persistent_ci', pd.Series(dtype=int)).sum()), 'developing_ci': int(stats_frame.get('developing_ci', pd.Series(dtype=int)).sum()), 'declining_ci': int(stats_frame.get('declining_ci', pd.Series(dtype=int)).sum()), 'transient_echo': int(stats_frame.get('transient_echo', pd.Series(dtype=int)).sum()), 'potential_ci': int(stats_frame.get('potential_ci', pd.Series(dtype=int)).sum()), 'ci_catalog': str(ci_path), 'potential_catalog': str(potential_path)}
    (year_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    (year_dir / '_SUCCESS').write_text(pd.Timestamp.utcnow().isoformat())
    logger.write('YEAR COMPLETE ' + json.dumps(summary, sort_keys=True))
    try:
        shutil.rmtree(temp_root)
    except Exception:
        pass
if __name__ == '__main__':
    main()
