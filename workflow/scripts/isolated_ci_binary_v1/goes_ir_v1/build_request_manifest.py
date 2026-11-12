"""Expand unique strict-CI issuances into a deduplicated causal GOES history."""
from __future__ import annotations
from ci_nowcasting.paths import resolve_path as _ci_resolve, project_root as _ci_project, source_root as _ci_source
import argparse
from pathlib import Path
import pandas as pd
from pipeline_common import DATA_ROOT, MANIFEST_ROOT, atomic_write_parquet, year_label_path
HISTORY_OFFSETS = tuple(range(-120, 1, 10))

def collapse_issuances(labels: pd.DataFrame, year: int) -> pd.DataFrame:
    required = {'period_id', 'issuance_time', 'target_temporally_valid'}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f'label_index is missing columns: {sorted(missing)}')
    frame = labels.copy()
    frame['issuance_time'] = pd.to_datetime(frame['issuance_time'], errors='coerce')
    frame = frame.dropna(subset=['issuance_time', 'period_id'])
    collapsed = frame.groupby(['period_id', 'issuance_time'], as_index=False, sort=True).agg(any_temporally_valid=('target_temporally_valid', 'any'))
    collapsed = collapsed.loc[collapsed.any_temporally_valid].copy()
    collapsed.insert(0, 'year', int(year))
    return collapsed[['year', 'period_id', 'issuance_time']].reset_index(drop=True)

def expand_history(issuances: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    for row in issuances.itertuples(index=False):
        issuance = pd.Timestamp(row.issuance_time)
        for history_index, offset in enumerate(HISTORY_OFFSETS):
            records.append({'year': int(row.year), 'period_id': str(row.period_id), 'issuance_time': issuance, 'history_index': int(history_index), 'history_offset_minutes': int(offset), 'requested_time': issuance + pd.Timedelta(minutes=offset)})
    links = pd.DataFrame(records, columns=['year', 'period_id', 'issuance_time', 'history_index', 'history_offset_minutes', 'requested_time']).sort_values(['year', 'period_id', 'issuance_time', 'history_index']).reset_index(drop=True)
    unique = links[['requested_time']].drop_duplicates().sort_values('requested_time').reset_index(drop=True)
    return (links, unique)

def build(years: list[int], output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    issuance_parts = []
    for year in years:
        path = year_label_path(year)
        if not path.exists():
            raise FileNotFoundError(path)
        part = collapse_issuances(pd.read_parquet(_ci_resolve(path)), year)
        issuance_parts.append(part)
        print(f'{year}: {len(part):,} unique temporally valid issuances')
    issuances = pd.concat(issuance_parts, ignore_index=True)
    duplicated = issuances.duplicated(['year', 'period_id', 'issuance_time'])
    if duplicated.any():
        raise RuntimeError('Issuance collapse produced duplicates')
    links, requests = expand_history(issuances)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(links, output_dir / 'goes_history_links.parquet')
    atomic_write_parquet(requests, output_dir / 'goes_request_times.parquet')
    if 2021 in years:
        catalog_path = DATA_ROOT / 'year_2021_strict_v2_001' / 'isolated_ci_catalog.parquet'
        catalog = pd.read_parquet(_ci_resolve(catalog_path)).sort_values(['birth_time', 'object_id'])
        if catalog.empty:
            raise RuntimeError('2021 strict dataset has no accepted CI for the smoke selection')
        event = catalog.iloc[0]
        candidates = issuances.loc[issuances.period_id.eq(event.period_id)].copy()
        candidates['distance'] = (candidates.issuance_time - pd.Timestamp(event.birth_time)).abs()
        issuance = candidates.sort_values(['distance', 'issuance_time']).iloc[0].issuance_time
        smoke_links = links.loc[links.period_id.eq(event.period_id) & links.issuance_time.eq(issuance)].copy()
        smoke_requests = smoke_links[['requested_time']].drop_duplicates().sort_values('requested_time')
        if not 13 <= len(smoke_requests) <= 25:
            raise RuntimeError(f'Smoke selection has {len(smoke_requests)} frames, expected 13-25')
        atomic_write_parquet(smoke_requests, output_dir / 'goes_smoke_request_times.parquet')
        atomic_write_parquet(pd.DataFrame([{'period_id': event.period_id, 'object_id': event.object_id, 'birth_time': event.birth_time, 'ci_latitude': event.latitude, 'ci_longitude': event.longitude, 'issuance_time': issuance, 'requested_count': len(smoke_requests)}]), output_dir / 'goes_smoke_selection.parquet')
        print(f'Smoke period:                 {event.period_id}')
        print(f'Smoke accepted CI:            {event.object_id}')
        print(f'Smoke requested times:        {len(smoke_requests):,}')
    print(f'Total unique issuances:       {len(issuances):,}')
    print(f'History link rows:            {len(links):,}')
    print(f'Total unique requested times: {len(requests):,}')
    print(f'First requested time:         {requests.requested_time.min()}')
    print(f'Last requested time:          {requests.requested_time.max()}')
    print(f'Output: {output_dir}')
    return (links, requests)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--years', nargs='+', type=int, default=list(range(2021, 2026)))
    parser.add_argument('--output-dir', type=Path, default=MANIFEST_ROOT)
    args = parser.parse_args()
    build(args.years, args.output_dir)
if __name__ == '__main__':
    main()
