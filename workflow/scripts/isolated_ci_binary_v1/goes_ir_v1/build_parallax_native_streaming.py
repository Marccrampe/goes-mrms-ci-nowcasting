"""Stream causal GOES sources into monthly native-grid parallax Zarr stores."""
from __future__ import annotations
from ci_nowcasting.paths import resolve_path as _ci_resolve, project_root as _ci_project, source_root as _ci_source
import argparse
import json
import os
import resource
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from pipeline_common import ACHA_PRODUCT, CHANNELS, CMIPC_PRODUCT, MANIFEST_ROOT, MIN_FREE_BYTES, OUTPUT_ROOT, PROGRESS_COLUMNS, STATE_ROOT, TEMP_ROOT, ParallaxStaticCache, append_batch_verified, atomic_write_parquet, corrected_native_fields, directory_size, download_file, human_bytes, load_progress, make_output_frame, monthly_store, read_native_crop, safe_cleanup_raw, store_times, utc_naive, validate_batch_in_store, wait_for_disk

def frame_raw_dir(root: Path, requested_time: Any) -> Path:
    return root / utc_naive(requested_time).strftime('%Y%m%dT%H%M%S')

def download_sources(rows: pd.DataFrame, raw_dir: Path, workers: int) -> list[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    with ThreadPoolExecutor(max_workers=min(max(1, workers), len(rows))) as executor:
        future_map = {executor.submit(download_file, row.remote_path, raw_dir / row.filename): row for row in rows.itertuples(index=False)}
        for future in as_completed(future_map):
            paths.append(future.result())
    return sorted(paths)

def prepare_frame(rows: pd.DataFrame, raw_dir: Path, workers: int, nprocs: int, static_cache: ParallaxStaticCache | None=None, timings: dict[str, float] | None=None) -> xr.Dataset:
    requested = utc_naive(rows.requested_time.iloc[0])
    if len(rows) != 8 or not rows.matched.all():
        missing = rows.loc[~rows.matched, ['product', 'channel']].to_dict('records')
        raise RuntimeError(f'Incomplete source set ({len(rows)}/8 rows), missing={missing}')
    if rows.remote_path.duplicated().any():
        raise RuntimeError('One request maps multiple inputs to the same remote file')
    if (pd.to_datetime(rows.scan_end) > requested).any():
        raise RuntimeError('Source manifest violates strict causality')
    stage_started = time.perf_counter()
    paths = download_sources(rows, raw_dir, workers)
    if timings is not None:
        timings['network_seconds'] += time.perf_counter() - stage_started
    path_by_name = {path.name: path for path in paths}
    cmi: dict[int, xr.Dataset] = {}
    acha: xr.Dataset | None = None
    try:
        stage_started = time.perf_counter()
        for row in rows.itertuples(index=False):
            path = path_by_name[str(row.filename)]
            if row.product == CMIPC_PRODUCT:
                cmi[int(row.channel)] = read_native_crop(path, 'CMI', max_dqf=1)
            elif row.product == ACHA_PRODUCT:
                acha = read_native_crop(path, 'HT', max_dqf=1)
        if sorted(cmi) != list(CHANNELS) or acha is None:
            raise RuntimeError('Downloaded inputs do not contain C08-C14 plus ACHA')
        if timings is not None:
            timings['decode_crop_seconds'] += time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        corrected, grid = corrected_native_fields(cmi, acha, requested, nprocs=nprocs, static_cache=static_cache)
        frame = make_output_frame(requested, rows, corrected, cmi[13], grid, static_cache=static_cache)
        if timings is not None:
            timings['parallax_seconds'] += time.perf_counter() - stage_started
        return frame
    finally:
        for dataset in cmi.values():
            dataset.close()
        if acha is not None:
            acha.close()

def initialize_progress(requests: list[pd.Timestamp], path: Path, output_root: Path) -> pd.DataFrame:
    progress = load_progress(path)
    existing = set(pd.to_datetime(progress.requested_time)) if len(progress) else set()
    additions = []
    now = pd.Timestamp.utcnow().tz_localize(None)
    for requested in requests:
        if requested not in existing:
            additions.append({'requested_time': requested, 'year': requested.year, 'month': requested.month, 'status': 'pending', 'attempts': 0, 'updated_at': now, 'satellite': None, 'output_store': str(monthly_store(requested, output_root)), 'finite_fraction': None, 'message': ''})
    if additions:
        progress = pd.concat([progress, pd.DataFrame(additions)], ignore_index=True)
        progress = progress[PROGRESS_COLUMNS].sort_values('requested_time').reset_index(drop=True)
        atomic_write_parquet(progress, path)
    return progress

def record_failure(audit_path: Path, requested: pd.Timestamp, attempts: int, error: Exception, raw_dir: Path) -> None:
    columns = ['requested_time', 'attempts', 'error', 'raw_directory', 'recorded_at']
    audit = pd.read_parquet(_ci_resolve(audit_path)) if audit_path.exists() else pd.DataFrame(columns=columns)
    audit = audit.loc[~pd.to_datetime(audit.requested_time).eq(requested)].copy()
    audit = pd.concat([audit, pd.DataFrame([{'requested_time': requested, 'attempts': attempts, 'error': repr(error), 'raw_directory': str(raw_dir), 'recorded_at': pd.Timestamp.utcnow().tz_localize(None)}])], ignore_index=True)
    atomic_write_parquet(audit[columns].sort_values('requested_time'), audit_path)

def projected_storage(output_root: Path, written_this_run: int, total_requested: int) -> dict[str, Any]:
    used = directory_size(output_root)
    all_written = sum((len(store_times(path)) for path in output_root.glob('year=*/month=*/goes_ir.zarr')))
    bytes_per_frame = used / max(1, all_written)
    return {'measured_output_bytes': used, 'written_frames': all_written, 'bytes_per_frame': bytes_per_frame, 'total_requested_frames': total_requested, 'projected_final_bytes': int(bytes_per_frame * total_requested), 'written_this_run': written_this_run}

def run(args: argparse.Namespace) -> None:
    manifest = pd.read_parquet(_ci_resolve(args.manifest))
    manifest['requested_time'] = pd.to_datetime(manifest.requested_time)
    for column in ('scan_start', 'scan_end'):
        manifest[column] = pd.to_datetime(manifest[column])
    requested = sorted((pd.Timestamp(value) for value in manifest.requested_time.unique()))
    if args.start:
        requested = [value for value in requested if value >= utc_naive(args.start)]
    if args.end:
        requested = [value for value in requested if value <= utc_naive(args.end)]
    if args.year:
        requested = [value for value in requested if value.year == args.year]
    if args.month:
        requested = [value for value in requested if value.month == args.month]
    if args.max_frames:
        requested = requested[:args.max_frames]
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.state_root.mkdir(parents=True, exist_ok=True)
    args.temp_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.state_root / 'progress.parquet'
    audit_path = args.state_root / 'permanent_failures.parquet'
    runtime_path = args.state_root / 'runtime_metrics.csv'
    progress = initialize_progress(requested, progress_path, args.output_root)
    print(f'Selected requested timestamps: {len(requested):,}')
    print(f'Output root: {args.output_root}')
    print(f'Verified append batch size: {args.batch_size}')
    written_this_run = 0
    timings = {'network_seconds': 0.0, 'decode_crop_seconds': 0.0, 'parallax_seconds': 0.0, 'zarr_write_seconds': 0.0}
    static_cache = ParallaxStaticCache()
    run_started = time.perf_counter()
    completed_durations: list[float] = []
    progress = progress.set_index('requested_time', drop=False)

    def save_progress() -> None:
        snapshot = progress[PROGRESS_COLUMNS].reset_index(drop=True)
        atomic_write_parquet(snapshot.sort_values('requested_time').reset_index(drop=True), progress_path)

    def set_state(stamp: pd.Timestamp, **values: Any) -> None:
        for name, value in values.items():
            progress.at[stamp, name] = value
        progress.at[stamp, 'updated_at'] = pd.Timestamp.utcnow().tz_localize(None)

    def log_runtime(completed: int, current: pd.Timestamp) -> None:
        elapsed = max(time.perf_counter() - run_started, 1e-09)
        overall_rate = 3600.0 * completed / elapsed
        recent = completed_durations[-args.rate_window:]
        rolling_rate = 3600.0 * len(recent) / sum(recent) if recent else 0.0
        remaining = max(0, len(requested) - completed)
        eta_hours = remaining / rolling_rate if rolling_rate > 0 else np.nan
        row = pd.DataFrame([{'logged_at': pd.Timestamp.utcnow().tz_localize(None), 'current_time': current, 'completed': completed, 'selected': len(requested), 'rolling_timestamps_per_hour': rolling_rate, 'overall_timestamps_per_hour': overall_rate, 'eta_hours': eta_hours, **timings, 'peak_ram_bytes': int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)}])
        if runtime_path.exists():
            row.to_csv(runtime_path, mode='a', header=False, index=False)
        else:
            row.to_csv(runtime_path, index=False)
        print(f"[{completed}/{len(requested)}] {current} | rolling={rolling_rate:.1f}/h overall={overall_rate:.1f}/h ETA={eta_hours:.2f}h | net={timings['network_seconds']:.1f}s decode={timings['decode_crop_seconds']:.1f}s parallax={timings['parallax_seconds']:.1f}s write={timings['zarr_write_seconds']:.1f}s", flush=True)
    known_by_store: dict[Path, set[pd.Timestamp]] = {}
    for stamp in requested:
        store = monthly_store(stamp, args.output_root)
        if store not in known_by_store:
            known_by_store[store] = set(store_times(store))
    completed = 0
    for chunk_start in range(0, len(requested), args.batch_size):
        chunk = requested[chunk_start:chunk_start + args.batch_size]
        pending: list[pd.Timestamp] = []
        existing_by_store: dict[Path, list[pd.Timestamp]] = {}
        for stamp in chunk:
            state = progress.loc[stamp]
            if state.status == 'failed' and (not args.retry_failed):
                print(f'{stamp} failed previously; use --retry-failed')
                completed += 1
                continue
            store = monthly_store(stamp, args.output_root)
            if stamp in known_by_store[store]:
                existing_by_store.setdefault(store, []).append(stamp)
                continue
            pending.append(stamp)
            rows = manifest.loc[manifest.requested_time.eq(stamp)]
            previous_attempts = pd.to_numeric(progress.at[stamp, 'attempts'], errors='coerce')
            set_state(stamp, status='downloading', attempts=0 if pd.isna(previous_attempts) else int(previous_attempts), satellite=int(rows.satellite.iloc[0]) if len(rows) else None, output_store=str(store), message='')
        for store, stamps in existing_by_store.items():
            fractions = validate_batch_in_store(store, stamps)
            for stamp in stamps:
                raw_dir = frame_raw_dir(args.temp_root, stamp)
                safe_cleanup_raw(list(raw_dir.glob('*.nc')), append_verified=True)
                set_state(stamp, status='written', output_store=str(store), finite_fraction=fractions[stamp], message='verified existing frame')
                completed += 1
        save_progress()
        prepared_by_store: dict[Path, list[tuple[pd.Timestamp, xr.Dataset, Path, float]]] = {}
        for stamp in pending:
            rows = manifest.loc[manifest.requested_time.eq(stamp)].copy()
            raw_dir = frame_raw_dir(args.temp_root, stamp)
            last_error: Exception | None = None
            for attempt in range(1, args.max_attempts + 1):
                frame_started = time.perf_counter()
                try:
                    wait_for_disk(args.output_root, args.min_free_gb * 1024 ** 3, poll_seconds=args.disk_poll_seconds, no_wait=args.no_wait_for_disk)
                    set_state(stamp, status='downloading', attempts=attempt, message='')
                    frame = prepare_frame(rows, raw_dir, args.download_workers, args.nprocs, static_cache=static_cache, timings=timings)
                    set_state(stamp, status='processed', attempts=attempt, message='native parallax complete; awaiting verified batch')
                    prepared_by_store.setdefault(monthly_store(stamp, args.output_root), []).append((stamp, frame, raw_dir, frame_started))
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    set_state(stamp, status='pending' if attempt < args.max_attempts else 'failed', attempts=attempt, message=repr(exc))
                    if attempt < args.max_attempts:
                        delay = min(300, 5 * 2 ** (attempt - 1))
                        print(f'{stamp}: {exc!r}; retrying in {delay}s', flush=True)
                        time.sleep(delay)
            if last_error is not None:
                record_failure(audit_path, stamp, args.max_attempts, last_error, raw_dir)
                print(f'{stamp}: PERMANENT FAILURE {last_error!r}; raw retained', flush=True)
                completed += 1
                if args.fail_fast:
                    save_progress()
                    raise last_error
        save_progress()
        for store, prepared in prepared_by_store.items():
            frames = [item[1] for item in prepared]
            stamps = [item[0] for item in prepared]
            batch = xr.concat(frames, dim='time', data_vars='minimal', coords='minimal', compat='override').sortby('time')
            write_started = time.perf_counter()
            append_error: Exception | None = None
            try:
                for append_attempt in range(1, args.max_attempts + 1):
                    try:
                        fractions = append_batch_verified(batch, store)
                        zarr.consolidate_metadata(str(store))
                        fractions = validate_batch_in_store(store, stamps)
                        append_error = None
                        break
                    except Exception as exc:
                        append_error = exc
                        if append_attempt < args.max_attempts:
                            delay = min(300, 5 * 2 ** (append_attempt - 1))
                            print(f'{store}: append {exc!r}; retrying in {delay}s', flush=True)
                            time.sleep(delay)
            finally:
                timings['zarr_write_seconds'] += time.perf_counter() - write_started
                batch.close()
                for frame in frames:
                    frame.close()
            if append_error is not None:
                for stamp, _, raw_dir, _ in prepared:
                    set_state(stamp, status='failed', message=repr(append_error))
                    record_failure(audit_path, stamp, args.max_attempts, append_error, raw_dir)
                    completed += 1
                save_progress()
                if args.fail_fast:
                    raise append_error
                continue
            for stamp, _, raw_dir, frame_started in prepared:
                safe_cleanup_raw(list(raw_dir.glob('*.nc')), append_verified=True)
                try:
                    raw_dir.rmdir()
                except OSError:
                    pass
                known_by_store[store].add(stamp)
                set_state(stamp, status='written', finite_fraction=fractions[stamp], message='')
                written_this_run += 1
                completed += 1
                completed_durations.append(time.perf_counter() - frame_started)
            save_progress()
            log_runtime(completed, stamps[-1])
    total_requested = pd.read_parquet(_ci_resolve(MANIFEST_ROOT / 'goes_request_times.parquet')).shape[0] if (MANIFEST_ROOT / 'goes_request_times.parquet').exists() else len(requested)
    projection = projected_storage(args.output_root, written_this_run, total_requested)
    projection_path = args.state_root / 'storage_projection.json'
    projection_path.write_text(json.dumps(projection, indent=2) + '\n')
    run_wall = time.perf_counter() - run_started
    runtime_summary = {'selected_timestamps': len(requested), 'completed_timestamps': completed, 'written_this_run': written_this_run, 'wall_seconds': run_wall, 'timestamps_per_hour': 3600.0 * completed / max(run_wall, 1e-09), **timings, 'peak_ram_bytes': int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024), 'batch_size': int(args.batch_size), 'download_workers': int(args.download_workers)}
    summary_path = args.state_root / 'runtime_summary.json'
    summary_tmp = summary_path.with_name(summary_path.name + f'.tmp.{os.getpid()}')
    summary_tmp.write_text(json.dumps(runtime_summary, indent=2) + '\n')
    summary_tmp.replace(summary_path)
    print(f"Projected final storage: {human_bytes(projection['projected_final_bytes'])} ({human_bytes(projection['bytes_per_frame'])}/frame)")
    print(f'Storage projection: {projection_path}')
    print(f'Runtime metrics: {runtime_path}')
    print(f'Runtime summary: {summary_path}')

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, default=MANIFEST_ROOT / 'goes_source_manifest.parquet')
    parser.add_argument('--output-root', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--state-root', type=Path, default=STATE_ROOT)
    parser.add_argument('--temp-root', type=Path, default=TEMP_ROOT)
    parser.add_argument('--year', type=int)
    parser.add_argument('--month', type=int)
    parser.add_argument('--start')
    parser.add_argument('--end')
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--max-attempts', type=int, default=4)
    parser.add_argument('--download-workers', type=int, default=4)
    parser.add_argument('--nprocs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--rate-window', type=int, default=32)
    parser.add_argument('--min-free-gb', type=float, default=20.0)
    parser.add_argument('--disk-poll-seconds', type=int, default=300)
    parser.add_argument('--no-wait-for-disk', action='store_true')
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--fail-fast', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1 or args.rate_window < 1:
        parser.error('--batch-size and --rate-window must be positive')
    run(args)
if __name__ == '__main__':
    main()
