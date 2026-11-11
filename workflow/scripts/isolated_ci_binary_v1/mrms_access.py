"""Thin adapter around the tested NOAA MRMS access/GRIB decoder."""
from __future__ import annotations
from ci_nowcasting.paths import resolve_path as _ci_resolve, project_root as _ci_project, source_root as _ci_source
import importlib.util
import sys
from pathlib import Path
import pandas as pd

def load_legacy(project_root):
    path = Path(_ci_resolve(project_root)) / str(_ci_source() / 'scripts/20_scan_cids_strict_mrms_windows.py')
    spec = importlib.util.spec_from_file_location('legacy_cids_mrms_access', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def list_period_files(archive, start, end):
    start, end = (pd.Timestamp(start), pd.Timestamp(end))
    rows = []
    for day in pd.date_range(start.normalize(), end.normalize(), freq='D'):
        rows.extend(archive.list_day(day))
    unique = {row.filename: row for row in rows if start <= row.valid_time <= end}
    return sorted(unique.values(), key=lambda row: (row.valid_time, row.filename))

def decode_downloaded(legacy, archive, listed, temp_dir, bbox):
    raw_path = temp_dir / listed.filename
    archive.download(listed, raw_path)
    try:
        return legacy.decode_grib_crop(raw_path, temp_dir / 'decoded', tuple(bbox))
    finally:
        raw_path.unlink(missing_ok=True)
