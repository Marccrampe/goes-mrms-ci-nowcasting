"""GOES+MRMS on the common fully observed cohort, using optional repaired inputs."""
from ci_nowcasting.paths import resolve_path as _ci_resolve, project_root as _ci_project, source_root as _ci_source
import argparse
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
HERE = Path(_ci_resolve(__file__)).resolve().parent
spec = importlib.util.spec_from_file_location('v3_two_window_inputs', HERE / 'ci_goes_v3_30min_dataset.py')
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

class CIGoesMRMSV3Dataset(base.CIGoesV3Dataset):

    def __init__(self, dataset_root, mrms_input_root, split='train', include_mrms=True, **kwargs):
        super().__init__(dataset_root, split=split, include_mrms=False, **kwargs)
        self.mrms_input_root = Path(_ci_resolve(mrms_input_root))
        self.issuances = pd.read_parquet(_ci_resolve(self.mrms_input_root / f'{split}_issuance_index_mrms_ready.parquet'))
        self.issuances['issuance_time'] = pd.to_datetime(self.issuances.issuance_time)
        if self.issuances.duplicated(['period_id', 'issuance_time']).any():
            raise RuntimeError('Duplicate MRMS-ready keys')
        if not self.issuances.mrms_history_complete_after_repair.astype(bool).all():
            raise RuntimeError('Incomplete radar cohort')
        self.include_mrms = bool(include_mrms)

    def _mrms_history(self, row):
        paths = json.loads(row.mrms_repair_history_paths)
        if not paths:
            return super()._mrms_history(row)
        stamps = np.array(json.loads(row.mrms_repair_history_times_ns), np.int64)
        expected = pd.Timestamp(row.issuance_time).value + np.array([-30, -20, -10, 0], np.int64) * 60000000000
        if len(paths) != 4 or stamps.shape != (4,) or np.any(stamps > expected) or np.any(expected - stamps > 150000000000):
            raise RuntimeError('Noncausal/incomplete repaired MRMS input')
        fields = []
        masks = []
        for path, stamp in zip(paths, stamps):
            path = Path(_ci_resolve(path))
            if self.mrms_input_root.resolve() not in path.resolve().parents:
                raise RuntimeError('Repair cache path outside root')
            with np.load(_ci_resolve(path), allow_pickle=False) as scan:
                if tuple(scan['shape']) != base.base.GRID_SHAPE or int(scan['time_ns']) != stamp:
                    raise RuntimeError('Repair cache grid/time mismatch')
                field = scan['reflectivity_dbz'].astype(np.float32)
                valid = np.unpackbits(scan['valid_bits'], count=field.size).reshape(field.shape).astype(np.float32)
            if not np.isfinite(field).all():
                raise RuntimeError('Nonfinite MRMS input')
            fields.append(field[None])
            masks.append(valid[None])
        return (np.stack(fields), np.stack(masks), stamps)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['mrms_history_complete'] = torch.tensor(True)
        return item

def smoke(dataset_root, input_root):
    from torch.utils.data import DataLoader, Subset
    reports = []
    for split in ['train', 'validation', 'test']:
        dataset = CIGoesMRMSV3Dataset(dataset_root, input_root, split=split)
        if not len(dataset):
            raise RuntimeError('Empty MRMS-ready split ' + split)
        indices = [0]
        repaired = np.flatnonzero(dataset.issuances.mrms_repair_history_paths.ne('[]').to_numpy())
        if len(repaired):
            indices.append(int(repaired[0]))
        indices = list(dict.fromkeys(indices))
        for index in indices:
            item = dataset[index]
            assert item['inputs'].shape[:2] == (4, 7)
            assert item['mrms_reflectivity_dbz'].shape[:2] == (4, 1)
            assert item['targets'].shape[0] == 2
            assert torch.isfinite(item['inputs']).all() and torch.isfinite(item['mrms_reflectivity_dbz']).all()
            assert item['mrms_valid_masks'].shape == item['mrms_reflectivity_dbz'].shape
            reports.append(dict(split=split, index=index, repaired=bool(json.loads(dataset.issuances.iloc[index].mrms_repair_history_paths)), inputs=list(item['inputs'].shape), radar_inputs=list(item['mrms_reflectivity_dbz'].shape)))
        batch = next(iter(DataLoader(Subset(dataset, indices), batch_size=2, num_workers=2)))
        assert batch['targets'].shape[1] == 2
        goes = CIGoesMRMSV3Dataset(dataset_root, input_root, split=split, include_mrms=False)
        assert dataset.issuances[['period_id', 'issuance_time']].equals(goes.issuances[['period_id', 'issuance_time']])
    root = Path(_ci_resolve(input_root))
    (root / 'loader_smoke.json').write_text(json.dumps(dict(status='passed', cases=reports), indent=2))
    coverage = json.loads((root / 'coverage_complete.json').read_text())
    if coverage['remaining_missing_histories'] != 0:
        raise RuntimeError('Radar coverage incomplete')
    (root / 'READY.json').write_text(json.dumps(dict(status='all_eligible_histories_ready_and_loader_smoke_passed', dataset_root=str(dataset_root), coverage=coverage['summary'], common_cohort=True, product='MergedReflectivityQCComposite_00.50', input_offsets_minutes=[-30, -20, -10, 0]), indent=2))
    print('GOES + MRMS INPUTS READY; common cohort verified.', flush=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--mrms-input-root', type=Path, required=True)
    args = parser.parse_args()
    smoke(args.dataset_root, args.mrms_input_root)
