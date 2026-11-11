"""Repair missing causal MRMS inputs without changing V3 stores or labels.
Archive product is fixed to the QC composite used in the existing CI pipeline.
"""
from ci_nowcasting.paths import resolve_path as _ci_resolve, project_root as _ci_project, source_root as _ci_source
import argparse
import gzip
import hashlib
import importlib.util
import json
import re
import shutil
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path
import numpy as np
import pandas as pd
PRODUCT = 'MergedReflectivityQCComposite_00.50'
BASE = 'https://noaa-mrms-pds.s3.amazonaws.com'
MINUTE = 60000000000
OFFSETS = np.array([-30, -20, -10, 0], np.int64)

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj

def write_json(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, default=str))
    temp.replace(path)

def write_frame(frame, path):
    temp = path.with_name(path.name + '.tmp')
    frame.to_parquet(temp, index=False)
    temp.replace(path)

def prior(times, anchor):
    j = int(np.searchsorted(times, anchor, side='right')) - 1
    return j if j >= 0 and 0 <= anchor - int(times[j]) <= 150000000000 else None

def get_url(url, destination=None):
    last = None
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'CCNY-CI-MRMS-input-repair/1.0'})
            with urllib.request.urlopen(request, timeout=60) as response:
                if destination is None:
                    return response.read()
                with destination.open('wb') as handle:
                    shutil.copyfileobj(response, handle)
            if destination.stat().st_size == 0:
                raise OSError('Empty download')
            return
        except Exception as exc:
            last = exc
            if attempt < 3:
                time.sleep(min(20, 3 * 2 ** attempt))
    raise OSError(f'Archive request failed after retries: {url}: {last}') from last

def list_day(day, root):
    cache = root / (day + '.json')
    if cache.exists():
        return json.loads(cache.read_text())
    records = []
    token = None
    while True:
        query = {'list-type': '2', 'prefix': f'CONUS/{PRODUCT}/{day}/', 'max-keys': '1000'}
        if token:
            query['continuation-token'] = token
        xml = ET.fromstring(get_url(BASE + '/?' + urllib.parse.urlencode(query)))
        namespace = {'s': 'http://s3.amazonaws.com/doc/2006-03-01/'}
        for node in xml.findall('s:Contents', namespace):
            key = node.findtext('s:Key', namespaces=namespace)
            match = re.fullmatch('MRMS_' + re.escape(PRODUCT) + '_(\\d{8})-(\\d{6})\\.grib2\\.gz', Path(_ci_resolve(key)).name)
            if match:
                stamp = pd.to_datetime(''.join(match.groups()), format='%Y%m%d%H%M%S').value
                records.append(dict(key=key, time_ns=int(stamp), bytes=int(node.findtext('s:Size', namespaces=namespace))))
        if xml.findtext('s:IsTruncated', namespaces=namespace) != 'true':
            break
        token = xml.findtext('s:NextContinuationToken', namespaces=namespace)
        if not token:
            raise RuntimeError('Missing archive pagination token')
    records.sort(key=lambda x: (x['time_ns'], x['key']))
    if records:
        write_json(cache, records)
    return records

def decoder_backend():
    try:
        import pygrib
        return 'pygrib'
    except ImportError:
        try:
            import eccodes
            return 'eccodes'
        except ImportError:
            raise RuntimeError('A GRIB decoder is required: pygrib or eccodes. Nothing has been downloaded yet.')

def decode(path, backend, source_ns, latitude, longitude, nearest):
    if backend == 'pygrib':
        import pygrib
        with pygrib.open(str(path)) as gribs:
            if gribs.messages != 1:
                raise RuntimeError('Expected one composite field')
            message = gribs.message(1)
            field = np.asarray(np.ma.filled(message.values, np.nan), np.float32)
            lat, lon = message.latlons()
            valid_time = pd.Timestamp(message.validDate).value
    else:
        import eccodes as ec
        with path.open('rb') as handle:
            gid = ec.codes_grib_new_from_file(handle)
            if gid is None:
                raise RuntimeError('Empty GRIB')
            try:
                ni = int(ec.codes_get(gid, 'Ni'))
                nj = int(ec.codes_get(gid, 'Nj'))
                values = ec.codes_get_values(gid).astype(np.float32)
                lat = ec.codes_get_array(gid, 'latitudes')
                lon = ec.codes_get_array(gid, 'longitudes')
                shape = (ni, nj) if int(ec.codes_get(gid, 'jPointsAreConsecutive')) else (nj, ni)
                field = values.reshape(shape)
                lat = lat.reshape(shape)
                lon = lon.reshape(shape)
                if shape == (ni, nj):
                    field = field.T
                    lat = lat.T
                    lon = lon.T
                missing = float(ec.codes_get(gid, 'missingValue'))
                field[field == missing] = np.nan
                valid_time = pd.to_datetime(str(ec.codes_get(gid, 'validityDate')) + str(ec.codes_get(gid, 'validityTime')).zfill(4), format='%Y%m%d%H%M').value
            finally:
                ec.codes_release(gid)
            extra = ec.codes_grib_new_from_file(handle)
            if extra is not None:
                ec.codes_release(extra)
                raise RuntimeError('Multiple GRIB messages')
    if abs(valid_time - source_ns) > MINUTE:
        raise RuntimeError('GRIB date does not match archive scan date')
    lon = (np.asarray(lon) + 180.0) % 360.0 - 180.0
    if field.shape != np.asarray(lat).shape or field.shape != lon.shape:
        raise RuntimeError('GRIB coordinate mismatch')
    la = np.asarray(lat)[:, 0]
    lo = lon[0, :]
    if not np.allclose(lat, la[:, None], rtol=0, atol=1e-05) or not np.allclose(lon, lo[None, :], rtol=0, atol=1e-05):
        raise RuntimeError('Nonrectilinear MRMS grid; do not resample silently')
    yy = nearest(la, latitude)
    xx = nearest(lo, longitude)
    if not np.allclose(la[yy], latitude, rtol=0, atol=2e-05) or not np.allclose(lo[xx], longitude, rtol=0, atol=2e-05):
        raise RuntimeError('Downloaded MRMS grid differs from retained native grid')
    cropped = np.asarray(field[np.ix_(yy, xx)], np.float32)
    if np.any(np.isfinite(cropped) & (cropped > 120)):
        raise RuntimeError('Unexpected reflectivity values')
    return cropped

def project(field, mapping, shape):
    values = field[mapping['source_y'], mapping['source_x']]
    good = np.isfinite(values) & (values >= -50.0)
    flat = mapping['target_y'].astype(np.int64) * shape[1] + mapping['target_x']
    total = np.asarray(mapping['total_count']).ravel()
    count = np.bincount(flat[good], minlength=len(total))
    valid = (total > 0) & (count >= 0.75 * total)
    peak = np.full(len(total), -50.0, np.float32)
    np.maximum.at(peak, flat[good], values[good])
    peak[~valid] = 0.0
    return (peak.reshape(shape), valid.reshape(shape))

def save_scan(path, field, mapping, shape, stamp, origin, source):
    peak, valid = project(field, mapping, shape)
    if not valid.any():
        raise RuntimeError('No observed MRMS pixels on GOES grid')
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as handle:
        np.savez_compressed(handle, reflectivity_dbz=peak.astype(np.float16), valid_bits=np.packbits(valid.ravel()), shape=np.array(shape, np.int32), time_ns=np.int64(stamp))
    temp.replace(path)
    write_json(path.with_suffix('.json'), dict(time_ns=int(stamp), origin=origin, source=source, product=PRODUCT, grid_hash=mapping['grid_hash'], aggregation='native maximum per GOES cell'))

def selftest():
    assert prior(np.array([100, 200, 300]), 250) == 1
    assert prior(np.array([100, 200]), 99) is None
    assert prior(np.array([100]), 100 + 151000000000) is None
    mapping = dict(source_y=np.array([0, 0, 1]), source_x=np.array([0, 1, 0]), target_y=np.array([0, 0, 0]), target_x=np.array([0, 0, 1]), total_count=np.array([[2, 1]]))
    peak, valid = project(np.array([[10.0, 40.0], [20.0, 0.0]]), mapping, (1, 2))
    assert peak.tolist() == [[40.0, 20.0]] and valid.all()
    peak, valid = project(np.array([[np.nan, 40.0], [20.0, 0.0]]), mapping, (1, 2))
    assert peak.tolist() == [[0.0, 20.0]] and valid.tolist() == [[False, True]]
    print('MRMS INPUT REPAIR SELFTEST PASSED: causal selection, peak projection, validity', flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', type=Path)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--selftest', action='store_true')
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    if not args.dataset_root or not args.output_root:
        parser.error('Dataset and output roots required')
    import zarr
    dataset = args.dataset_root
    output = args.output_root
    code = Path(_ci_resolve(__file__)).resolve().parent
    if not (dataset / 'READY.json').exists():
        raise RuntimeError('Two-window dataset queue has not passed its checks')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'scans').mkdir(exist_ok=True)
    (output / 'listings').mkdir(exist_ok=True)
    for stale in ['READY.json', 'PARTIAL.json']:
        (output / stale).unlink(missing_ok=True)
    fingerprint = dict(version=1, product=PRODUCT, history_offsets_minutes=OFFSETS.tolist(), maximum_age_seconds=150, input_timing='latest valid-time scan <= each input frame time; no future fill', dataset_config_sha256=hashlib.sha256((dataset / 'build_config.json').read_bytes()).hexdigest(), script_sha256=hashlib.sha256(Path(_ci_resolve(__file__)).read_bytes()).hexdigest())
    config = output / 'config.json'
    if config.exists() and json.loads(config.read_text()) != fingerprint:
        raise RuntimeError('Repair configuration changed; use another output directory')
    write_json(config, fingerprint)
    tables = []
    for split in ['train', 'validation', 'test']:
        table = pd.read_parquet(_ci_resolve(dataset / f'{split}_issuance_index_natural.parquet'))
        table['split'] = split
        tables.append(table)
    frame = pd.concat(tables, ignore_index=True)
    if frame.duplicated(['period_id', 'issuance_time']).any():
        raise RuntimeError('Duplicate/split-leaking issuance keys')
    missing = frame[~frame.mrms_history_complete.astype(bool)].copy()
    print('Eligible issuances:', len(frame), 'Incomplete MRMS histories:', len(missing), flush=True)
    builder = None
    grid = None
    backend = None
    repaired = {}
    unresolved = []
    manifest = []
    network_downloads = 0
    copied_existing = 0
    period_count = 0
    if len(missing):
        builder = load('mrms_repair_projection', code / 'build_ci_soft_targets_native_goes.py')
        grid = builder.load_and_validate_goes_grid()
        backend = decoder_backend()
    for pid, subset in missing.groupby('period_id', sort=True):
        period_count += 1
        radar_path = Path(_ci_resolve(subset.radar_path.iloc[0]))
        if subset.radar_path.nunique() != 1:
            raise RuntimeError('Several native stores per period')
        radar = zarr.open_group(_ci_resolve(str(radar_path)), mode='r')
        times = np.asarray(radar['time_ns'][:], np.int64)
        if len(times) == 0 or np.any(np.diff(times) <= 0):
            raise RuntimeError('Invalid existing native scan times')
        latitude = np.asarray(radar['latitude'][:])
        longitude = np.asarray(radar['longitude'][:])
        mapping = builder.build_mapping(radar_path.parent, grid, {})
        if tuple(mapping['expected_shape']) != (len(latitude), len(longitude)):
            raise RuntimeError('Native grid orientation changed')
        for row in subset.itertuples(index=False):
            issuance = pd.Timestamp(row.issuance_time).value
            paths = []
            stamps = []
            for offset in OFFSETS:
                anchor = issuance + int(offset) * MINUTE
                index = prior(times, anchor)
                source = None
                origin = 'existing_native'
                stamp = None
                if index is not None:
                    stamp = int(times[index])
                    source = str(radar_path) + '#time_ns=' + str(stamp)
                else:
                    origin = 'downloaded_archive'
                    days = {pd.Timestamp(anchor).strftime('%Y%m%d'), pd.Timestamp(anchor - 150000000000).strftime('%Y%m%d')}
                    entries = []
                    for day in sorted(days):
                        entries.extend(list_day(day, output / 'listings'))
                    candidates = [e for e in entries if 0 <= anchor - e['time_ns'] <= 150000000000]
                    if candidates:
                        entry = max(candidates, key=lambda e: (e['time_ns'], e['key']))
                        stamp = entry['time_ns']
                        source = BASE + '/' + urllib.parse.quote(entry['key'], safe='/')
                if stamp is None:
                    unresolved.append(dict(period_id=pid, split=row.split, issuance_time=row.issuance_time, input_offset_minutes=int(offset), input_time=pd.Timestamp(anchor), reason='no_causal_archive_scan_within_150_seconds', product=PRODUCT))
                    break
                name = hashlib.sha256((mapping['grid_hash'] + '|' + str(stamp) + '|' + PRODUCT).encode()).hexdigest()[:28]
                path = output / 'scans' / (name + '.npz')
                if not path.exists() or not path.with_suffix('.json').exists():
                    try:
                        if index is not None:
                            field = np.asarray(radar['reflectivity_dbz'][index], np.float32)
                            copied_existing += 1
                        else:
                            with tempfile.TemporaryDirectory(prefix='ci_mrms_input_', dir=output) as temporary:
                                raw = Path(_ci_resolve(temporary)) / 'radar.grib2.gz'
                                decoded = Path(_ci_resolve(temporary)) / 'radar.grib2'
                                get_url(source, raw)
                                with gzip.open(raw, 'rb') as inp, decoded.open('wb') as out:
                                    shutil.copyfileobj(inp, out)
                                field = decode(decoded, backend, stamp, latitude, longitude, builder.nearest_axis_indices)
                                network_downloads += 1
                        save_scan(path, field, mapping, tuple(grid['shape']), stamp, origin, source)
                    except Exception as exc:
                        unresolved.append(dict(period_id=pid, split=row.split, issuance_time=row.issuance_time, input_offset_minutes=int(offset), input_time=pd.Timestamp(anchor), reason=str(exc), product=PRODUCT))
                        print('UNRESOLVED', pid, pd.Timestamp(anchor), str(exc), flush=True)
                        break
                with np.load(_ci_resolve(path), allow_pickle=False) as scan:
                    if int(scan['time_ns']) != stamp or tuple(scan['shape']) != tuple(grid['shape']):
                        raise RuntimeError('Repair cache time/grid mismatch')
                    if not np.isfinite(scan['reflectivity_dbz']).all():
                        raise RuntimeError('Nonfinite cached radar input')
                    if not scan['valid_bits'].any():
                        raise RuntimeError('Cached radar scan has no valid coverage')
                paths.append(str(path))
                stamps.append(stamp)
                manifest.append(dict(period_id=pid, issuance_time=row.issuance_time, input_offset_minutes=int(offset), anchor_time=pd.Timestamp(anchor), actual_time=pd.Timestamp(stamp), origin=origin, source=source, cache_path=str(path)))
            if len(paths) == 4:
                repaired[str(pid), issuance] = (paths, stamps)
        print(f'PERIOD {period_count}/{missing.period_id.nunique()} {pid} downloaded={network_downloads} repaired={len(repaired)}', flush=True)
        write_json(output / 'progress.json', dict(periods_completed=period_count, missing_periods=int(missing.period_id.nunique()), repaired=len(repaired), downloads=network_downloads))
    summaries = []
    for split in ['train', 'validation', 'test']:
        table = frame[frame.split == split].copy()
        paths_column = []
        times_column = []
        ready = []
        for row in table.itertuples(index=False):
            entry = repaired.get((str(row.period_id), pd.Timestamp(row.issuance_time).value))
            paths_column.append(json.dumps(entry[0]) if entry else '[]')
            times_column.append(json.dumps(entry[1]) if entry else '[]')
            ready.append(bool(row.mrms_history_complete) or entry is not None)
        table['mrms_repair_history_paths'] = paths_column
        table['mrms_repair_history_times_ns'] = times_column
        table['mrms_history_complete_after_repair'] = ready
        write_frame(table[pd.Series(ready, index=table.index)], output / f'{split}_issuance_index_mrms_ready.parquet')
        summaries.append(dict(split=split, eligible_issuances=len(table), mrms_ready=sum(ready), remaining_missing=len(table) - sum(ready)))
    pd.DataFrame(manifest, columns=['period_id', 'issuance_time', 'input_offset_minutes', 'anchor_time', 'actual_time', 'origin', 'source', 'cache_path']).to_csv(output / 'input_scan_manifest.csv', index=False)
    pd.DataFrame(unresolved, columns=['period_id', 'split', 'issuance_time', 'input_offset_minutes', 'input_time', 'reason', 'product']).to_csv(output / 'unresolved_inputs.csv', index=False)
    pd.DataFrame(summaries).to_csv(output / 'coverage_summary.csv', index=False)
    left = sum((x['remaining_missing'] for x in summaries))
    report = dict(status='complete' if left == 0 else 'partial_archive_or_decode_gaps', summary=summaries, histories_repaired=len(repaired), unique_downloads_this_invocation=network_downloads, existing_scans_projected_this_invocation=copied_existing, remaining_missing_histories=left, native_stores_modified=False, labels_modified=False, raw_grib_files_retained=False)
    write_json(output / ('coverage_complete.json' if left == 0 else 'PARTIAL.json'), report)
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)
    if left:
        raise RuntimeError(f'{left} MRMS histories remain unavailable; see unresolved_inputs.csv; no READY marker')
    print('MRMS COVERAGE COMPLETE; loader smoke still required.', flush=True)
if __name__ == '__main__':
    main()
