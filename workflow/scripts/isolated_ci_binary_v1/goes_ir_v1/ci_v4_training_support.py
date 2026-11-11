"""Paired soft/binary datasets, identical cohort, masks and crop anchors."""
from ci_nowcasting.paths import resolve_path as _ci_resolve, project_root as _ci_project, source_root as _ci_source
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ci_goes_mrms_v3_dataset import CIGoesMRMSV3Dataset

def sha(path):
    h = hashlib.sha256()
    with Path(_ci_resolve(path)).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

class CIGoesV4Dataset(CIGoesMRMSV3Dataset):

    def __init__(self, dataset_root, mrms_input_root, split='train', **kwargs):
        root = Path(_ci_resolve(dataset_root))
        ready = json.loads((root / 'READY.json').read_text())
        if ready['status'] != 'paired_dataset_ready_with_explicit_local_ignores':
            raise RuntimeError('Paired dataset not ready')
        self.target_kind = ready['target_kind']
        self.pair_identity = ready['pair_identity']
        for name, expected in ready['index_sha256'].items():
            if name.startswith(split + '_') and sha(root / name) != expected:
                raise RuntimeError('Prepared index changed: ' + name)
        super().__init__(root, mrms_input_root, split=split, **kwargs)
        original_keys = self.issuances[['period_id', 'issuance_time']].copy()
        cohort = pd.read_parquet(_ci_resolve(root / f'{split}_issuance_index_common.parquet'))
        cohort.issuance_time = pd.to_datetime(cohort.issuance_time)
        if not cohort[['period_id', 'issuance_time']].equals(original_keys):
            raise RuntimeError('Original complete input cohort order/keys changed')
        self.issuances = cohort

    def _build_targets(self, rows):
        targets, masks, _ = super()._build_targets(rows)
        candidates = []
        for row in rows.itertuples(index=False):
            lead = int(row.lead_index)
            if not row.usable_lead:
                continue
            for path in json.loads(row.soft_target_paths):
                stored = self._load_soft_target(path)
                ys, xs = (stored['core_y'], stored['core_x'])
                keep = masks[lead, ys, xs] > 0
                candidates.extend(((int(y), int(x), lead) for y, x in zip(ys[keep], xs[keep])))
        if self.target_kind == 'binary' and (not np.isin(targets, [0, 1]).all()):
            raise RuntimeError('Binary target contains soft values')
        return (targets, masks, candidates)

class CommonCoreEvaluationDataset(CIGoesV4Dataset):
    """Score every model against the same masked >=35-dBZ birth cores."""

    def _build_targets(self, rows):
        _, masks, candidates = super()._build_targets(rows)
        cores = np.zeros_like(masks)
        for row in rows.itertuples(index=False):
            lead = int(row.lead_index)
            if not row.usable_lead:
                continue
            for path in json.loads(row.soft_target_paths):
                stored = self._load_soft_target(path)
                cores[lead, stored['core_y'], stored['core_x']] = 1.0
        return (cores * masks, masks, candidates)
