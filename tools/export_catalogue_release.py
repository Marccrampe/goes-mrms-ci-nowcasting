"""Export an existing accepted catalogue and exact grouped splits; never rebuild events."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import pandas as pd


def read(path):
    return pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--events', type=Path, required=True, help='Final accepted event table, not candidates')
    ap.add_argument('--period-splits', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True, help='New output directory')
    ap.add_argument('--expected-events', type=int)
    a = ap.parse_args()
    if a.output.exists():
        raise RuntimeError('Existing output preserved; choose a new directory')
    events, splits = read(a.events), read(a.period_splits)
    if events.empty:
        raise RuntimeError('The accepted event table is empty')
    for name in ['event_id', 'period_id']:
        if name not in events or events[name].isna().any():
            raise RuntimeError('Missing/null event field: '+name)
    if events.event_id.duplicated().any():
        raise RuntimeError('Duplicate event IDs')
    if a.expected_events is not None and len(events) != a.expected_events:
        raise RuntimeError(f'Event count {len(events)} differs from expected {a.expected_events}; no filtering applied')
    if not {'period_id', 'split'} <= set(splits.columns):
        raise RuntimeError('Split table requires period_id and split')
    if splits[['period_id', 'split']].isna().any().any():
        raise RuntimeError('Null split assignment')
    assignment = splits[['period_id', 'split']].drop_duplicates()
    if assignment.period_id.duplicated().any():
        raise RuntimeError('A period is assigned to multiple splits')
    if not set(assignment.split) <= {'train', 'validation', 'test'}:
        raise RuntimeError('Unexpected split names; normalize explicitly before export')
    merged = events.merge(assignment.rename(columns={'split': '_release_split'}),
                          on='period_id', how='left', validate='many_to_one')
    if merged._release_split.isna().any():
        raise RuntimeError('Some events have no original split assignment')
    if 'split' in merged and not merged['split'].eq(merged._release_split).all():
        raise RuntimeError('Event and period split columns conflict')
    merged['split'] = merged.pop('_release_split')
    a.output.mkdir(parents=True)
    shutil.copy2(a.events, a.output/('accepted_events_original'+''.join(a.events.suffixes)))
    # JSONL preserves nested footprint/list columns; tabular fields are not discarded.
    merged.to_json(a.output/'accepted_events.jsonl.gz', orient='records', lines=True,
                   date_format='iso', double_precision=15, compression='gzip')
    assignment.to_csv(a.output/'period_split_assignment.csv', index=False)
    manifest = dict(n_events=len(merged), n_periods=merged.period_id.nunique(),
        split_counts=merged.split.value_counts().to_dict(), rebuilt_events=False,
        fields={k:str(v) for k,v in merged.dtypes.items()},
        source_files={str(p.resolve()):sha(p) for p in [a.events, a.period_splits]},
        output_files={p.name:sha(p) for p in a.output.iterdir() if p.is_file()})
    (a.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('EXPORTED', len(merged), 'existing events to', a.output)


if __name__ == '__main__':
    main()
