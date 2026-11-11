"""Resolve causal GOES-East source files for every unique requested time."""
from __future__ import annotations
from ci_nowcasting.paths import resolve_path as _ci_resolve, project_root as _ci_project, source_root as _ci_source
import argparse
import time
from pathlib import Path
import pandas as pd
import s3fs
from pipeline_common import ACHA_PRODUCT, CHANNELS, CMIPC_PRODUCT, MANIFEST_ROOT, atomic_write_parquet, parse_goes_filename, satellite_for_time, source_rows_for_request
INVENTORY_COLUMNS = ['product', 'channel', 'satellite', 'scan_start', 'scan_end', 'created_time', 'remote_path', 'filename']

def list_hour(filesystem: s3fs.S3FileSystem, satellite: int, product: str, hour: pd.Timestamp, retries: int=4) -> pd.DataFrame:
    bucket = f'noaa-goes{satellite}'
    prefix = f'{bucket}/{product}/{hour:%Y}/{hour:%j}/{hour:%H}/'
    for attempt in range(retries + 1):
        try:
            paths = filesystem.ls(prefix, detail=False)
            records = []
            for path in paths:
                if not str(path).endswith('.nc'):
                    continue
                try:
                    parsed = parse_goes_filename(path)
                except ValueError:
                    continue
                if parsed['product'] != product or parsed['satellite'] != satellite:
                    continue
                if product == CMIPC_PRODUCT and parsed['channel'] not in CHANNELS:
                    continue
                records.append({**parsed, 'remote_path': str(path)})
            return pd.DataFrame(records, columns=INVENTORY_COLUMNS)
        except Exception:
            if attempt >= retries:
                raise
            time.sleep(min(120, 3 * 2 ** attempt))
    raise AssertionError('unreachable')

def cache_path(cache_root: Path, satellite: int, product: str, hour: pd.Timestamp) -> Path:
    return cache_root / f'G{satellite}' / product / f'{hour:%Y}' / f'{hour:%j}' / f'{hour:%H}.parquet'

def required_listing_hours(requests: pd.Series) -> list[tuple[int, pd.Timestamp]]:
    pairs = set()
    for value in pd.to_datetime(requests):
        for offset in (-1, 0):
            hour = pd.Timestamp(value).floor('h') + pd.Timedelta(hours=offset)
            pairs.add((satellite_for_time(value), hour))
    return sorted(pairs, key=lambda item: (item[1], item[0]))

def build(request_path: Path, output_dir: Path, refresh: bool=False, max_requests: int | None=None) -> pd.DataFrame:
    requests = pd.read_parquet(_ci_resolve(request_path))
    if 'requested_time' not in requests:
        raise ValueError(f'{request_path} lacks requested_time')
    requests['requested_time'] = pd.to_datetime(requests['requested_time'], errors='coerce')
    requests = requests.dropna(subset=['requested_time']).drop_duplicates('requested_time').sort_values('requested_time').reset_index(drop=True)
    if max_requests is not None:
        requests = requests.head(int(max_requests))
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = output_dir / 'inventory_cache'
    filesystem = s3fs.S3FileSystem(anon=True)
    inventory_parts = []
    failures = []
    pairs = required_listing_hours(requests.requested_time)
    total = len(pairs) * 2
    count = 0
    for satellite, hour in pairs:
        for product in (CMIPC_PRODUCT, ACHA_PRODUCT):
            count += 1
            path = cache_path(cache_root, satellite, product, hour)
            try:
                if path.exists() and (not refresh):
                    frame = pd.read_parquet(_ci_resolve(path))
                else:
                    frame = list_hour(filesystem, satellite, product, hour)
                    atomic_write_parquet(frame, path)
                inventory_parts.append(frame)
                if count == 1 or count % 100 == 0 or count == total:
                    print(f'Listed/cached {count:,}/{total:,} hourly product prefixes', flush=True)
            except Exception as exc:
                failures.append({'satellite': satellite, 'product': product, 'hour': hour, 'error': repr(exc)})
                print(f'[WARN] G{satellite} {product} {hour}: {exc!r}', flush=True)
    inventory = pd.concat(inventory_parts, ignore_index=True) if inventory_parts else pd.DataFrame(columns=INVENTORY_COLUMNS)
    for name in ('scan_start', 'scan_end', 'created_time'):
        inventory[name] = pd.to_datetime(inventory[name], errors='coerce')
    inventory = inventory.dropna(subset=['scan_start', 'scan_end', 'remote_path']).sort_values(['scan_start', 'product', 'channel', 'created_time']).drop_duplicates('remote_path', keep='last').reset_index(drop=True)
    atomic_write_parquet(inventory, output_dir / 'goes_remote_inventory.parquet')
    atomic_write_parquet(pd.DataFrame(failures, columns=['satellite', 'product', 'hour', 'error']), output_dir / 's3_listing_failures.parquet')
    mapping_records = []
    for index, requested in enumerate(requests.requested_time, start=1):
        mapping_records.extend(source_rows_for_request(inventory, requested))
        if index % 1000 == 0:
            print(f'Matched {index:,}/{len(requests):,} requested times', flush=True)
    mapping = pd.DataFrame(mapping_records).sort_values(['requested_time', 'product', 'channel']).reset_index(drop=True)
    atomic_write_parquet(mapping, output_dir / 'goes_source_manifest.parquet')
    selected = mapping.loc[mapping.matched].copy()
    unique_files = selected.sort_values(['remote_path', 'requested_time']).drop_duplicates('remote_path', keep='first')[['product', 'channel', 'satellite', 'scan_start', 'scan_end', 'remote_path', 'filename']].reset_index(drop=True)
    atomic_write_parquet(unique_files, output_dir / 'goes_unique_remote_files.parquet')
    completeness = mapping.groupby('requested_time').matched.sum()
    print(f'Requested timestamps: {len(requests):,}')
    print(f'Fully matched (8/8):  {int((completeness == 8).sum()):,}')
    print(f'Unmatched timestamps: {int((completeness < 8).sum()):,}')
    print(f'Unique remote files:  {len(unique_files):,}')
    print(f'Listing failures:     {len(failures):,}')
    print(f"Output: {output_dir / 'goes_source_manifest.parquet'}")
    return mapping

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--requests', type=Path, default=MANIFEST_ROOT / 'goes_request_times.parquet')
    parser.add_argument('--output-dir', type=Path, default=MANIFEST_ROOT)
    parser.add_argument('--refresh', action='store_true')
    parser.add_argument('--max-requests', type=int)
    args = parser.parse_args()
    build(args.requests, args.output_dir, args.refresh, args.max_requests)
if __name__ == '__main__':
    main()
