"""V3 two-window dataset. Train crops; validation/test full domain + fixed tiles."""
from ci_nowcasting.paths import resolve_path as _ci_resolve, project_root as _ci_project, source_root as _ci_source
import argparse
import importlib.util
import json
import sys
from collections import OrderedDict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import zarr
HERE = Path(_ci_resolve(__file__)).resolve().parent
spec = importlib.util.spec_from_file_location('v3_original_goes_loader', HERE / 'ci_goes_patch_dataset.py')
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

class CIGoesV3Dataset(base.CIGoesPatchDataset):

    def __init__(self, dataset_root, split='train', goes_root=base.DEFAULT_ROOT, normalization=None, seed=20260930, flatten_time=False, mode=None, include_mrms=False):
        self.dataset_root = Path(_ci_resolve(dataset_root))
        self.goes_root = Path(_ci_resolve(goes_root))
        self.split = split
        self.mode = mode or ('train' if split == 'train' else 'full')
        if split not in {'train', 'validation', 'test'} or self.mode not in {'train', 'full'}:
            raise ValueError('Invalid split/mode')
        if split != 'train' and self.mode != 'full':
            raise ValueError('Validation/test must use full-domain fixed tiles')
        self.seed = int(seed)
        self.epoch = 0
        self.patch_size = 128
        self.positive_margin = 12
        self.flatten_time = bool(flatten_time)
        self.include_mrms = bool(include_mrms)
        suffix = 'mrms_ready' if include_mrms else 'natural'
        self.issuances = pd.read_parquet(_ci_resolve(self.dataset_root / f'{split}_issuance_index_{suffix}.parquet'))
        self.leads = pd.read_parquet(_ci_resolve(self.dataset_root / f'{split}_lead_index_natural.parquet'))
        for table in [self.issuances, self.leads]:
            table['issuance_time'] = pd.to_datetime(table.issuance_time)
        if self.issuances.duplicated(['period_id', 'issuance_time']).any():
            raise RuntimeError('Duplicate issuances')
        self.leads = self.leads.sort_values(['period_id', 'issuance_time', 'lead_index']).reset_index(drop=True)
        self.lead_lookup = {}
        for (pid, t), rows in self.leads.groupby(['period_id', 'issuance_time']).groups.items():
            rows = np.array(list(rows), np.int64)
            if self.leads.iloc[rows].lead_index.tolist() != [0, 1]:
                raise RuntimeError('Two ordered windows required')
            self.lead_lookup[str(pid), int(pd.Timestamp(t).value)] = rows
        self.month_cache = {}
        self.valid_mask_cache = {}
        self.soft_target_cache = OrderedDict()
        self.period_cache = OrderedDict()
        self.radar_cache = OrderedDict()
        self.normalization = None
        self._mapper = None
        self._radar_grid = None
        if normalization is not None:
            if isinstance(normalization, (str, Path)):
                normalization = json.loads(Path(_ci_resolve(normalization)).read_text())
            mean = np.asarray(normalization.get('channel_mean_k', normalization.get('mean_k')), np.float32)
            std = np.asarray(normalization.get('channel_std_k', normalization.get('std_k')), np.float32)
            if mean.shape != (7,) or std.shape != (7,) or (not np.isfinite(mean).all()) or (not np.isfinite(std).all()) or (std <= 0).any():
                raise RuntimeError('Normalization needs seven finite means/positive stds')
            self.normalization = {'mean': mean.reshape(1, 7, 1, 1), 'std': std.reshape(1, 7, 1, 1)}

    def __getstate__(self):
        state = super().__getstate__()
        state['period_cache'] = OrderedDict()
        state['radar_cache'] = OrderedDict()
        state['_mapper'] = None
        state['_radar_grid'] = None
        return state

    def _period(self, path):
        path = str(path)
        if path not in self.period_cache:
            with np.load(_ci_resolve(path), allow_pickle=False) as z:
                self.period_cache[path] = {k: z[k].copy() for k in z.files}
            if len(self.period_cache) > 4:
                self.period_cache.popitem(last=False)
        self.period_cache.move_to_end(path)
        return self.period_cache[path]

    def _lead_rows(self, row):
        return self.leads.iloc[self.lead_lookup[str(row.period_id), int(pd.Timestamp(row.issuance_time).value)]]

    def _build_targets(self, rows):
        shape = base.GRID_SHAPE
        targets = np.zeros((2, *shape), np.float32)
        masks = np.zeros_like(targets)
        candidates = []
        for row in rows.itertuples(index=False):
            lead = int(row.lead_index)
            if not row.usable_lead:
                continue
            stored = self._period(row.loss_mask_path)
            if tuple(stored['shape']) != shape or stored['issuance_time_ns'][row.mask_row] != pd.Timestamp(row.issuance_time).value:
                raise RuntimeError('Mask grid/time mismatch')
            masks[lead] = np.unpackbits(stored['mask_bits'][row.mask_row, lead], count=int(np.prod(shape))).reshape(shape)
            for path in base.parse_list(row.soft_target_paths):
                soft = self._load_soft_target(path)
                np.maximum.at(targets[lead], (soft['soft_y'], soft['soft_x']), soft['soft_value'])
                if self.mode == 'train':
                    candidates.extend(((int(y), int(x), lead) for y, x in zip(soft['core_y'], soft['core_x'])))
            targets[lead][masks[lead] == 0] = 0
        while len(self.soft_target_cache) > 256:
            self.soft_target_cache.popitem(last=False)
        return (targets, masks, candidates)

    def _load_history(self, issuance_time):
        frames = []
        for offset in (-30, -20, -10, 0):
            t = pd.Timestamp(issuance_time) + pd.Timedelta(minutes=offset)
            month = self._month_data(t)
            if int(t.value) not in month['time_lookup']:
                raise RuntimeError('Missing GOES time ' + str(t))
            frame = np.asarray(month['group']['CMI'][month['time_lookup'][int(t.value)]], np.float32)
            if frame.shape != (7, *base.GRID_SHAPE) or not np.isfinite(frame).all():
                raise RuntimeError('GOES data shape/nonfinite failure ' + str(t))
            frames.append(frame)
        return np.stack(frames)

    def _mrms_history(self, row):
        path = str(row.radar_path)
        if self._mapper is None:
            soft_spec = importlib.util.spec_from_file_location('v3_mrms_mapper', HERE / 'build_ci_soft_targets_native_goes.py')
            mapper = importlib.util.module_from_spec(soft_spec)
            soft_spec.loader.exec_module(mapper)
            self._mapper = mapper
            self._radar_grid = mapper.load_and_validate_goes_grid()
        if path not in self.radar_cache:
            mapping = self._mapper.build_mapping(Path(_ci_resolve(path)).parent, self._radar_grid, {})
            group = zarr.open_group(_ci_resolve(path), mode='r')
            self.radar_cache[path] = (group, np.asarray(group['time_ns'][:], np.int64), mapping)
            if len(self.radar_cache) > 2:
                self.radar_cache.popitem(last=False)
        self.radar_cache.move_to_end(path)
        group, times, mapping = self.radar_cache[path]
        stamp = self._period(row.loss_mask_path)['mrms_input_times_ns'][int(row.mask_row)]
        expected = pd.Timestamp(row.issuance_time).value + np.array([-30, -20, -10, 0], np.int64) * 60000000000
        if np.any(stamp < 0) or np.any(stamp > expected) or np.any(expected - stamp > 150000000000):
            raise RuntimeError('Unavailable/noncausal MRMS input')
        flat = mapping['target_y'].astype(np.int64) * base.GRID_SHAPE[1] + mapping['target_x']
        total = np.asarray(mapping['total_count']).ravel()
        size = len(total)
        fields = []
        valids = []
        for t in stamp:
            index = int(np.searchsorted(times, t))
            if index >= len(times) or times[index] != t:
                raise RuntimeError('MRMS timestamp disappeared')
            f = np.asarray(group['reflectivity_dbz'][index], np.float32)
            values = f[mapping['source_y'], mapping['source_x']]
            good = np.isfinite(values) & (values >= -50.0)
            n = np.bincount(flat[good], minlength=size)
            valid = (total > 0) & (n >= 0.75 * total)
            peak = np.full(size, -50.0, np.float32)
            np.maximum.at(peak, flat[good], values[good])
            peak[~valid] = 0.0
            fields.append(peak.reshape(1, *base.GRID_SHAPE))
            valids.append(valid.reshape(1, *base.GRID_SHAPE))
        return (np.stack(fields), np.stack(valids).astype(np.float32), stamp.copy())

    def __getitem__(self, index):
        row = self.issuances.iloc[int(index)]
        rows = self._lead_rows(row)
        targets, masks, candidates = self._build_targets(rows)
        history = self._load_history(row.issuance_time)
        radar = self._mrms_history(row) if self.include_mrms else None
        if self.mode == 'train':
            crop = self._choose_crop(self._rng(int(index), row), bool(row.any_usable_positive), candidates)
            y0, x0 = (crop['y0'], crop['x0'])
            region = np.s_[y0:y0 + 128, x0:x0 + 128]
        else:
            y0 = x0 = -1
            region = np.s_[:, :]
        history = history[:, :, region[0], region[1]]
        targets = targets[:, region[0], region[1]]
        masks = masks[:, region[0], region[1]]
        if self.normalization is not None:
            history = (history - self.normalization['mean']) / self.normalization['std']
        if self.flatten_time:
            history = history.reshape(28, *history.shape[-2:])
        tensor = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
        item = dict(inputs=tensor(history), targets=tensor(targets), loss_masks=tensor(masks), lead_valid=tensor(rows.usable_lead.to_numpy()), is_positive=torch.tensor(bool(row.any_usable_positive)), period_id=str(row.period_id), issuance_time_ns=torch.tensor(int(pd.Timestamp(row.issuance_time).value)), crop_origin_yx=torch.tensor([y0, x0]), mrms_history_complete=torch.tensor(bool(row.mrms_history_complete)))
        if radar is not None:
            item['mrms_reflectivity_dbz'] = tensor(radar[0][:, :, region[0], region[1]])
            item['mrms_valid_masks'] = tensor(radar[1][:, :, region[0], region[1]])
            item['mrms_input_times_ns'] = torch.from_numpy(radar[2])
        return item

def fixed_tiles(item):
    if tuple(item['inputs'].shape[-2:]) != base.GRID_SHAPE:
        raise ValueError('Full domain required')
    origins = [(0, 0), (0, 102), (84, 0), (84, 102)]
    return ({key: torch.stack([item[key][..., y:y + 128, x:x + 128] for y, x in origins]) for key in ['inputs', 'targets', 'loss_masks']}, origins)

def smoke(root, normalization=None):
    from torch.utils.data import DataLoader
    reports = []
    for split in ['train', 'validation', 'test']:
        dataset = CIGoesV3Dataset(root, split=split, normalization=normalization)
        if not len(dataset):
            raise RuntimeError('Empty split ' + split)
        positives = np.flatnonzero(dataset.issuances.any_usable_positive.to_numpy(bool))
        negatives = np.flatnonzero(~dataset.issuances.any_usable_positive.to_numpy(bool))
        if not len(positives) or not len(negatives):
            raise RuntimeError('Missing positive/negative class ' + split)
        for index in [int(positives[0]), int(negatives[0])]:
            item = dataset[index]
            assert item['targets'].shape[0] == 2 and item['inputs'].shape[:2] == (4, 7)
            assert torch.isfinite(item['inputs']).all() and torch.isfinite(item['targets']).all()
            assert item['targets'].min() >= 0 and item['targets'].max() <= 1
            assert torch.count_nonzero(item['targets'][item['loss_masks'] == 0]) == 0
            if not bool(item['is_positive']):
                assert torch.count_nonzero(item['targets']) == 0
            if split != 'train':
                tiles, origins = fixed_tiles(item)
                assert tiles['inputs'].shape == (4, 4, 7, 128, 128)
            reports.append(dict(split=split, index=index, inputs=list(item['inputs'].shape), positive=bool(item['is_positive']), supervised_pixels=int(item['loss_masks'].sum())))
        batch = next(iter(DataLoader(torch.utils.data.Subset(dataset, [int(positives[0]), int(negatives[0])]), batch_size=2, num_workers=2)))
        assert batch['targets'].shape[1] == 2
        ready = CIGoesV3Dataset(root, split=split, include_mrms=True)
        if len(ready):
            item = ready[0]
            assert item['mrms_reflectivity_dbz'].shape[:2] == (4, 1)
            assert torch.isfinite(item['mrms_reflectivity_dbz']).all()
    path = Path(_ci_resolve(root)) / 'loader_smoke.json'
    path.write_text(json.dumps(dict(status='passed', cases=reports), indent=2))
    print('V3 TWO-WINDOW LOADER SMOKE PASSED', flush=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--normalization-json', type=Path)
    args = parser.parse_args()
    smoke(args.dataset_root, args.normalization_json)
