"""Rebuild paired soft/binary windows with common local ignore masks.

Default: ignore the four TRAIN cases the user reviewed as mostly radar noise.
Ignore is conservative: remove their positive references and mask the union
of old soft support and new binary support in affected windows. They are not
silently converted to negative targets. Other warm/geometry flags stay intact.
Original inputs, splits, masks, labels and repair caches are never modified.
"""
from ci_nowcasting.paths import resolve_path as _ci_resolve, project_root as _ci_project, source_root as _ci_source
import argparse
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
import numpy as np
import pandas as pd
PROJECT = Path(_ci_resolve(_ci_resolve('/crest/students/mcrampe00/PDE_CNN_precip')))
DATA = PROJECT / 'data/ci_isolated_mrms_binary_v1/goes_ir_parallax_native_filled_v1'
SOURCE = DATA / 'datasets/ci_v3_birth_2x30min_v1'
MRMS = DATA / 'datasets/ci_v3_mrms_inputs_complete_v1'
V4 = DATA / 'labels/ci_v4_binary20_birth_envelopes_v1'
QC = PROJECT / 'experiments/ci_v4_binary20_quality_review_v1'
OUT = DATA / 'datasets/ci_v4_soft_binary_paired_qc_v1'
SHAPE = (212, 230)

def sha(path):
    h = hashlib.sha256()
    with Path(_ci_resolve(path)).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def canonical_path(value):
    path = Path(_ci_resolve(str(value)))
    return str((path if path.is_absolute() else PROJECT / path).resolve())

def parse(value):
    if isinstance(value, str):
        value = json.loads(value)
    if value is None:
        return []
    return list(value)

def atomic_json(path, value):
    temporary = Path(_ci_resolve(str(path) + '.tmp'))
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)

def atomic_npz(path, value):
    temporary = Path(_ci_resolve(str(path) + '.tmp'))
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **value)
    temporary.replace(path)

def atomic_parquet(path, frame):
    temporary = Path(_ci_resolve(str(path) + '.tmp'))
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)

class Targets:

    def __init__(self, manifest, ignore_events):
        self.lookup = {}
        self.cache = OrderedDict()
        self.ignore = set(ignore_events)
        for row in manifest.itertuples(index=False):
            key = canonical_path(row.old_soft_target_path)
            if key in self.lookup:
                raise RuntimeError('Duplicate old target path: ' + key)
            self.lookup[key] = dict(soft=key, binary=canonical_path(row.v4_target_path), object_id=str(row.object_id), event_id=str(row.canonical_event_id), birth_time=pd.Timestamp(row.birth_time), period_id=str(row.period_id))

    def get(self, path):
        path = canonical_path(path)
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]
        if path not in self.lookup:
            raise RuntimeError('Window target absent from V4 manifest: ' + path)
        info = self.lookup[path].copy()
        arrays = {}
        for kind in ['soft', 'binary']:
            with np.load(_ci_resolve(info[kind]), allow_pickle=False) as stored:
                if tuple(stored['shape']) != SHAPE:
                    raise RuntimeError('Unexpected target shape: ' + info[kind])
                y, x = (stored['soft_y'], stored['soft_x'])
                values = stored['soft_value'].astype(np.float32)
                if not np.isfinite(values).all() or np.any((values <= 0) | (values > 1)):
                    raise RuntimeError('Invalid sparse target values')
                support = np.zeros(SHAPE, bool)
                support[y, x] = values > 0
                core = np.zeros(SHAPE, bool)
                core[stored['core_y'], stored['core_x']] = True
                if kind == 'binary' and (not (values == 1).all()):
                    raise RuntimeError('Nonbinary V4 target')
                if not support[core].all():
                    raise RuntimeError('Core outside positive supervision')
                arrays[kind] = (support, core)
        if not np.array_equal(arrays['soft'][1], arrays['binary'][1]):
            raise RuntimeError('Paired birth core differs')
        info.update(support=arrays['soft'][0] | arrays['binary'][0], core=arrays['soft'][1], ignored=info['event_id'] in self.ignore)
        self.cache[path] = info
        if len(self.cache) > 192:
            self.cache.popitem(last=False)
        return info

def clean_window(mask, items, usable):
    """Only events with a supervised birth core remain; others become local ignores."""
    mask = mask.copy()
    ignored = [item for item in items if item['ignored']]
    retained = [item for item in items if not item['ignored']]
    for item in ignored:
        mask[item['support']] = False
    unobservable = []
    if not usable:
        return (mask, [], ignored, retained)
    while True:
        hidden = [item for item in retained if not np.any(mask & item['core'])]
        if not hidden:
            break
        hidden_ids = {item['object_id'] for item in hidden}
        retained = [item for item in retained if item['object_id'] not in hidden_ids]
        for item in hidden:
            mask[item['support']] = False
        unobservable.extend(hidden)
    return (mask, retained, ignored, unobservable)

def choose_ignores(selected, manifest, numbers):
    selected = selected.reset_index(drop=True)
    chosen = []
    for number in numbers:
        if not 1 <= number <= len(selected):
            raise RuntimeError('Unknown reviewed case number: ' + str(number))
        item = selected.iloc[number - 1]
        if item.reason not in {'known_24July_suspect', 'warm_two_GOES_scans_review'}:
            raise RuntimeError('Default ignore is restricted to reviewed suspect cases, not random controls')
        rows = manifest.loc[manifest.object_id == item.object_id]
        if len(rows) != 1 or rows.iloc[0].split != 'train':
            raise RuntimeError('Reviewed ignore must resolve to a unique TRAIN detection')
        row = rows.iloc[0]
        chosen.append(dict(case_number=int(number), object_id=str(row.object_id), canonical_event_id=str(row.canonical_event_id), period_id=str(row.period_id), action='ignore_local_supervision', reason='User visual review: mostly suspected radar noise; conservative ignore, not negative relabel'))
    return pd.DataFrame(chosen)

def rebuild_split(args, split, targets, masks_dir):
    source = pd.read_parquet(_ci_resolve(args.source_dataset / f'{split}_lead_index_natural.parquet'))
    source.issuance_time = pd.to_datetime(source.issuance_time)
    if source.duplicated(['period_id', 'issuance_time', 'lead_index']).any():
        raise RuntimeError('Duplicate source lead rows')
    soft, binary = (source.copy(), source.copy())
    logs, flags, mask_remap = ([], {}, {})
    positive_windows = 0
    for period_number, (pid, period_rows) in enumerate(source.groupby('period_id', sort=True), 1):
        stores = {}
        changed = set()
        pending = []
        for index, row in period_rows.iterrows():
            paths = parse(row.soft_target_paths)
            items = [targets.get(path) for path in paths]
            if any((item['period_id'] != str(pid) for item in items)):
                raise RuntimeError('Cross-period event reference')
            lead = int(row.lead_index)
            if lead not in [0, 1]:
                raise RuntimeError('Expected two 30-minute windows')
            for item in items:
                offset = (item['birth_time'] - row.issuance_time).total_seconds() / 60.0
                if not np.isfinite(offset):
                    raise RuntimeError('Invalid birth time')
            old_mask_path = canonical_path(row.loss_mask_path)
            if old_mask_path not in stores:
                with np.load(_ci_resolve(old_mask_path), allow_pickle=False) as stored:
                    stores[old_mask_path] = {key: stored[key].copy() for key in stored.files}
            store = stores[old_mask_path]
            mr = int(row.mask_row)
            if tuple(store['shape']) != SHAPE or int(store['issuance_time_ns'][mr]) != row.issuance_time.value:
                raise RuntimeError('Source mask shape/time mismatch')
            packed = store['mask_bits'][mr, lead]
            old_mask = np.unpackbits(packed, count=int(np.prod(SHAPE))).reshape(SHAPE).astype(bool)
            mask, keep, ignored, hidden = clean_window(old_mask, items, bool(row.usable_lead))
            if not np.array_equal(mask, old_mask):
                store['mask_bits'][mr, lead] = np.packbits(mask.ravel())
                changed.add(old_mask_path)
            usable = bool(row.usable_lead) and bool(mask.any())
            if not usable:
                keep = []
            pos = bool(keep) and usable
            key = (str(pid), int(row.issuance_time.value))
            flags[key] = flags.get(key, False) or pos
            positive_windows += int(pos)
            for kind, frame in [('soft', soft), ('binary', binary)]:
                frame.at[index, 'soft_target_paths'] = json.dumps([item[kind] for item in keep])
                if 'positive_event_ids' in frame:
                    frame.at[index, 'positive_event_ids'] = json.dumps(sorted({item['event_id'] for item in keep}))
                frame.at[index, 'usable_lead'] = usable
                for name in ['has_usable_positive', 'is_positive', 'target_present']:
                    if name in frame:
                        frame.at[index, name] = pos
                for name in ['positive_event_count', 'n_positive_events']:
                    if name in frame:
                        frame.at[index, name] = len({item['event_id'] for item in keep})
            if ignored or hidden or (not np.array_equal(mask, old_mask)):
                logs.append(dict(split=split, period_id=str(pid), issuance_time=row.issuance_time, lead_index=lead, ignored_events=json.dumps(sorted({i['event_id'] for i in ignored})), unsupervised_core_events=json.dumps(sorted({i['event_id'] for i in hidden})), ignored_pixels_added=int(old_mask.sum() - mask.sum())))
            pending.append((index, old_mask_path))
        for old_path in stores:
            if old_path in changed:
                destination = masks_dir / (hashlib.sha256(old_path.encode()).hexdigest()[:24] + '.npz')
                atomic_npz(destination, stores[old_path])
                mask_remap[old_path] = str(destination)
            else:
                mask_remap[old_path] = old_path
        for index, old_path in pending:
            soft.at[index, 'loss_mask_path'] = mask_remap[old_path]
            binary.at[index, 'loss_mask_path'] = mask_remap[old_path]
        if period_number == 1 or period_number % 25 == 0:
            print(f'WINDOWS {split} period={period_number}/{source.period_id.nunique()}', flush=True)
    shared = ['period_id', 'issuance_time', 'lead_index', 'usable_lead', 'loss_mask_path', 'mask_row']
    if not soft[shared].equals(binary[shared]):
        raise RuntimeError('Paired window/mask contract mismatch')
    for kind, frame in [('soft', soft), ('binary', binary)]:
        root = args.output_dir / kind
        atomic_parquet(root / f'{split}_lead_index_natural.parquet', frame)
    original_natural = pd.read_parquet(_ci_resolve(args.source_dataset / f'{split}_issuance_index_natural.parquet'))
    original_ready = pd.read_parquet(_ci_resolve(args.mrms_input_root / f'{split}_issuance_index_mrms_ready.parquet'))
    summary = dict(split=split, eligible=len(original_natural), retained=len(original_ready), excluded=len(original_natural) - len(original_ready), positive_windows=positive_windows)
    for suffix, original in [('natural', original_natural), ('common', original_ready)]:
        frame = original.copy()
        frame.issuance_time = pd.to_datetime(frame.issuance_time)
        values = []
        for row in frame.itertuples(index=False):
            key = (str(row.period_id), int(pd.Timestamp(row.issuance_time).value))
            if key not in flags:
                raise RuntimeError('Issuance lacks rebuilt lead rows: ' + str(key))
            values.append(flags[key])
        frame['any_usable_positive'] = np.asarray(values, bool)
        if 'loss_mask_path' in frame:
            frame['loss_mask_path'] = frame.loss_mask_path.map(lambda value: mask_remap.get(canonical_path(value), str(value)))
        if suffix == 'common':
            if not frame.mrms_history_complete_after_repair.astype(bool).all():
                raise RuntimeError('Incomplete radar input in common cohort')
            summary['positive_common_issuances'] = int(frame.any_usable_positive.sum())
            summary['positive_common_fraction'] = float(frame.any_usable_positive.mean())
        for kind in ['soft', 'binary']:
            atomic_parquet(args.output_dir / kind / f'{split}_issuance_index_{suffix}.parquet', frame)
    return (summary, logs)

def selftest():
    shape = (8, 8)
    mask = np.ones(shape, bool)
    a = np.zeros(shape, bool)
    a[2, 2] = True
    support = np.zeros(shape, bool)
    support[1:4, 1:4] = True
    item = dict(object_id='suspect', event_id='e0', ignored=True, core=a, support=support)
    b = np.zeros(shape, bool)
    b[6, 6] = True
    good = dict(object_id='good', event_id='e1', ignored=False, core=b, support=b)
    new, keep, ignored, hidden = clean_window(mask, [item, good], True)
    assert [k['object_id'] for k in keep] == ['good'] and (not new[2, 2]) and new[6, 6]
    assert mask.all() and len(ignored) == 1 and (not hidden)
    ambiguous = dict(object_id='overlap', event_id='e2', ignored=False, core=a, support=support)
    new, keep, ignored, hidden = clean_window(mask, [item, ambiguous, good], True)
    assert len(hidden) == 1 and len(keep) == 1
    print('PAIRED LOCAL IGNORE / COMMON CORE / NO NEGATIVE RELABEL SELFTEST PASSED', flush=True)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selftest', action='store_true')
    p.add_argument('--source-dataset', type=Path, default=SOURCE)
    p.add_argument('--mrms-input-root', type=Path, default=MRMS)
    p.add_argument('--v4-root', type=Path, default=V4)
    p.add_argument('--qc-root', type=Path, default=QC)
    p.add_argument('--output-dir', type=Path, default=OUT)
    p.add_argument('--ignore-reviewed-cases', nargs='+', type=int, default=[1, 2, 3, 4])
    args = p.parse_args()
    if args.selftest:
        selftest()
        return
    if args.output_dir.resolve() in {args.source_dataset.resolve(), args.v4_root.resolve(), args.mrms_input_root.resolve()}:
        raise RuntimeError('Use a separate output directory')
    qc = json.loads((args.qc_root / 'audit_summary.json').read_text())
    if qc['file_errors']:
        raise RuntimeError('V4 integrity errors remain')
    finished = json.loads((args.qc_root / 'completed.json').read_text())
    if finished['status'] != 'quality_diagnostics_generated_requires_human_review':
        raise RuntimeError('QC generation incomplete')
    manifest = pd.read_parquet(_ci_resolve(args.v4_root / 'target_manifest.parquet'))
    if len(manifest) != int(qc['detections']) or manifest.object_id.duplicated().any():
        raise RuntimeError('QC/manifest mismatch')
    if (file_sha := json.loads((args.qc_root / 'qc_config.json').read_text()).get('manifest_sha256')):
        if file_sha != sha(args.v4_root / 'target_manifest.parquet'):
            raise RuntimeError('QC belongs to a different target build')
    selected = pd.read_csv(_ci_resolve(args.qc_root / 'selected_cases.csv'))
    decisions = choose_ignores(selected, manifest, args.ignore_reviewed_cases)
    ignore_events = set(decisions.canonical_event_id)
    source_files = [args.v4_root / 'target_manifest.parquet', args.qc_root / 'selected_cases.csv']
    for split in ['train', 'validation', 'test']:
        source_files.extend([args.source_dataset / f'{split}_lead_index_natural.parquet', args.source_dataset / f'{split}_issuance_index_natural.parquet', args.mrms_input_root / f'{split}_issuance_index_mrms_ready.parquet'])
    config = dict(version=1, source_sha256={str(path): sha(path) for path in source_files}, ignore_decisions=decisions.to_dict('records'), policy='Only user-reviewed TRAIN cases 1-4: local ignore of soft/binary support union. Other 644 warm flags NOT automatically excluded.', temporal_rule='Preserve original V3 canonical-event window assignment, 0-30 / 30-60 min', common_positivity='Retained >=35 birth core must intersect the shared supervised mask', original_cohort_and_input_order_preserved=True)
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / 'preparation_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError('Preparation changed: choose a new output directory')
    atomic_json(config_path, config)
    completed = args.output_dir / 'completed.json'
    if completed.exists():
        for kind in ['soft', 'binary']:
            if not (args.output_dir / kind / 'READY.json').exists():
                raise RuntimeError('Partial READY state')
        print('PAIRED DATASETS ALREADY READY:', args.output_dir, flush=True)
        return
    for kind in ['soft', 'binary']:
        (args.output_dir / kind).mkdir(exist_ok=True)
    masks_dir = args.output_dir / 'shared_loss_masks'
    masks_dir.mkdir(exist_ok=True)
    decisions.to_csv(args.output_dir / 'reviewed_local_ignores.csv', index=False)
    targets = Targets(manifest, ignore_events)
    summaries = []
    logs = []
    for split in ['train', 'validation', 'test']:
        summary, changes = rebuild_split(args, split, targets, masks_dir)
        summaries.append(summary)
        logs.extend(changes)
        print('SPLIT REBUILT', json.dumps(summary), flush=True)
    if sum((s['excluded'] for s in summaries)) != 106:
        raise RuntimeError('Original common-cohort exclusions changed')
    pd.DataFrame(summaries).to_csv(args.output_dir / 'summary.csv', index=False)
    pd.DataFrame(logs, columns=['split', 'period_id', 'issuance_time', 'lead_index', 'ignored_events', 'unsupervised_core_events', 'ignored_pixels_added']).to_csv(args.output_dir / 'window_mask_changes.csv', index=False)
    manifest.assign(review_action=np.where(manifest.canonical_event_id.isin(ignore_events), 'ignore_local_supervision', 'retained_no_noise_verdict')).to_parquet(args.output_dir / 'event_review_actions.parquet', index=False)
    for kind in ['soft', 'binary']:
        root = args.output_dir / kind
        files = {path.name: sha(path) for path in sorted(root.glob('*.parquet'))}
        atomic_json(root / 'READY.json', dict(status='paired_dataset_ready_with_explicit_local_ignores', target_kind=kind, pair_identity=identity, index_sha256=files, summary=summaries, ignore_events=sorted(ignore_events), original_mrms_input_root=str(args.mrms_input_root), geometry='Original V3 buffered soft target' if kind == 'soft' else 'Seeded >=20-dBZ native envelope at birth, binary projection', note='Unreviewed warm/geometry flags retained. No test skill inspected.'))
    atomic_json(completed, dict(status='paired_datasets_ready', pair_identity=identity, ignored_reviewed_train_events=len(ignore_events), warm_category_automatically_removed=0, input_issuances_removed=0, splits=summaries))
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)
    print('PAIRED SOFT / BINARY DATASETS READY:', args.output_dir, flush=True)
if __name__ == '__main__':
    main()
