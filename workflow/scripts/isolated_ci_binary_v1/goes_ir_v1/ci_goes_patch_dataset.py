from ci_nowcasting.paths import resolve_path as _ci_resolve, project_root as _ci_project, source_root as _ci_source
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import zarr
DEFAULT_ROOT = Path(_ci_resolve(_ci_resolve('/crest/students/mcrampe00/PDE_CNN_precip/data/ci_isolated_mrms_binary_v1/goes_ir_parallax_native_filled_v1')))
DEFAULT_SPLIT_ROOT = DEFAULT_ROOT / 'splits/period_grouped_80_10_10_v1'
HISTORY_OFFSETS_MINUTES = tuple(range(-120, 1, 10))
EXPECTED_CHANNELS = tuple(range(8, 15))
GRID_SHAPE = (212, 230)

def parse_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        value = value.strip()
        return [] if not value else list(json.loads(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    if pd.isna(value):
        return []
    raise TypeError(f'Unsupported list value: {type(value)}')

def decode_time_array(values, attrs):
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.datetime64):
        return pd.DatetimeIndex(values.astype('datetime64[ns]'))
    units_text = str(attrs.get('units', '')).lower()
    if 'nanosecond' in units_text:
        unit = 'ns'
    elif 'microsecond' in units_text:
        unit = 'us'
    elif 'millisecond' in units_text:
        unit = 'ms'
    elif 'second' in units_text:
        unit = 's'
    else:
        magnitude = float(np.nanmedian(np.abs(values.astype(np.float64))))
        if magnitude > 1e+17:
            unit = 'ns'
        elif magnitude > 100000000000000.0:
            unit = 'us'
        elif magnitude > 100000000000.0:
            unit = 'ms'
        else:
            unit = 's'
    return pd.DatetimeIndex(pd.to_datetime(values, unit=unit))

class CIGoesPatchDataset(Dataset):

    def __init__(self, issuance_index, lead_index, goes_root=DEFAULT_ROOT, patch_size=128, positive_margin=12, seed=20260930, flatten_time=False, normalization=None):
        self.issuance_path = Path(_ci_resolve(issuance_index))
        self.lead_path = Path(_ci_resolve(lead_index))
        self.goes_root = Path(_ci_resolve(goes_root))
        self.patch_size = int(patch_size)
        self.positive_margin = int(positive_margin)
        self.seed = int(seed)
        self.epoch = 0
        self.flatten_time = bool(flatten_time)
        if self.patch_size > min(GRID_SHAPE):
            raise ValueError(f'patch_size={self.patch_size} exceeds grid={GRID_SHAPE}')
        if not 0 <= self.positive_margin < self.patch_size // 2:
            raise ValueError('positive_margin must be between zero and half the patch size')
        self.issuances = pd.read_parquet(_ci_resolve(self.issuance_path))
        self.leads = pd.read_parquet(_ci_resolve(self.lead_path))
        self.issuances['issuance_time'] = pd.to_datetime(self.issuances['issuance_time'])
        self.leads['issuance_time'] = pd.to_datetime(self.leads['issuance_time'])
        if self.issuances.duplicated(['period_id', 'issuance_time']).any():
            raise RuntimeError('Duplicated issuance keys')
        self.leads = self.leads.sort_values(['period_id', 'issuance_time', 'lead_index']).reset_index(drop=True)
        self.lead_lookup = {}
        for key, indices in self.leads.groupby(['period_id', 'issuance_time'], sort=False).groups.items():
            index_array = np.asarray(list(indices), dtype=np.int64)
            if len(index_array) != 6:
                raise RuntimeError(f'Expected six leads for {key}, found {len(index_array)}')
            self.lead_lookup[str(key[0]), int(pd.Timestamp(key[1]).value)] = index_array
        for row in self.issuances.itertuples(index=False):
            key = (str(row.period_id), int(pd.Timestamp(row.issuance_time).value))
            if key not in self.lead_lookup:
                raise RuntimeError(f'Issuance missing lead rows: {key}')
        self.month_cache = {}
        self.valid_mask_cache = {}
        self.soft_target_cache = {}
        self.normalization = None
        if normalization is not None:
            if isinstance(normalization, (str, Path)):
                with open(_ci_resolve(normalization), 'r', encoding='utf-8') as handle:
                    normalization = json.load(handle)
            mean = np.asarray(normalization['channel_mean_k'], dtype=np.float32)
            std = np.asarray(normalization['channel_std_k'], dtype=np.float32)
            if mean.shape != (7,) or std.shape != (7,):
                raise ValueError('Normalization must contain seven channel means and standard deviations')
            if np.any(std <= 0):
                raise ValueError('Normalization standard deviations must be positive')
            self.normalization = {'mean': mean.reshape(1, 7, 1, 1), 'std': std.reshape(1, 7, 1, 1)}

    def __getstate__(self):
        state = self.__dict__.copy()
        state['month_cache'] = {}
        state['valid_mask_cache'] = {}
        state['soft_target_cache'] = {}
        return state

    def __len__(self):
        return len(self.issuances)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _rng(self, index, row):
        payload = f"{self.seed}|{self.epoch}|{index}|{row['period_id']}|{pd.Timestamp(row['issuance_time']).value}".encode('utf-8')
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        integer_seed = int.from_bytes(digest, byteorder='little', signed=False)
        return np.random.default_rng(integer_seed)

    def _month_data(self, timestamp):
        timestamp = pd.Timestamp(timestamp)
        key = (int(timestamp.year), int(timestamp.month))
        if key in self.month_cache:
            return self.month_cache[key]
        store_path = self.goes_root / f'year={timestamp.year:04d}' / f'month={timestamp.month:02d}' / 'goes_ir.zarr'
        if not store_path.exists():
            raise FileNotFoundError(store_path)
        group = zarr.open_group(_ci_resolve(str(store_path)), mode='r')
        if 'CMI' not in group:
            raise KeyError(f'CMI missing from {store_path}')
        times = decode_time_array(group['time'][:], group['time'].attrs.asdict())
        time_lookup = {int(value.value): index for index, value in enumerate(times)}
        channels = tuple((int(value) for value in np.asarray(group['channel'][:])))
        if channels != EXPECTED_CHANNELS:
            raise RuntimeError(f'Unexpected channels in {store_path}: {channels}')
        if tuple(group['CMI'].shape[-2:]) != GRID_SHAPE:
            raise RuntimeError(f"Unexpected grid in {store_path}: {group['CMI'].shape}")
        result = {'group': group, 'path': store_path, 'time_lookup': time_lookup}
        self.month_cache[key] = result
        return result

    def _load_history(self, issuance_time):
        frames = []
        for offset in HISTORY_OFFSETS_MINUTES:
            timestamp = pd.Timestamp(issuance_time) + pd.Timedelta(minutes=offset)
            month = self._month_data(timestamp)
            timestamp_ns = int(timestamp.value)
            if timestamp_ns not in month['time_lookup']:
                raise RuntimeError(f'Missing GOES history time {timestamp} for issuance {issuance_time}')
            time_index = month['time_lookup'][timestamp_ns]
            frame = np.asarray(month['group']['CMI'][time_index], dtype=np.float32)
            if frame.shape != (7, GRID_SHAPE[0], GRID_SHAPE[1]):
                raise RuntimeError(f'Unexpected CMI frame shape: {frame.shape}')
            if not np.isfinite(frame).all():
                raise RuntimeError(f'Non-finite GOES input at {timestamp}')
            frames.append(frame)
        history = np.stack(frames, axis=0)
        if history.shape != (13, 7, GRID_SHAPE[0], GRID_SHAPE[1]):
            raise RuntimeError(f'Unexpected history shape: {history.shape}')
        return history

    def _load_valid_mask(self, path):
        path = str(path)
        if path not in self.valid_mask_cache:
            with np.load(_ci_resolve(path), allow_pickle=False) as archive:
                mask = np.asarray(archive['valid_mask'], dtype=bool)
            if mask.shape != GRID_SHAPE:
                raise RuntimeError(f'Unexpected valid mask shape: {path} {mask.shape}')
            self.valid_mask_cache[path] = mask
        return self.valid_mask_cache[path]

    def _load_soft_target(self, path):
        path = str(path)
        if path not in self.soft_target_cache:
            with np.load(_ci_resolve(path), allow_pickle=False) as archive:
                target = {'soft_y': np.asarray(archive['soft_y'], dtype=np.int64), 'soft_x': np.asarray(archive['soft_x'], dtype=np.int64), 'soft_value': np.asarray(archive['soft_value'], dtype=np.float32), 'core_y': np.asarray(archive['core_y'], dtype=np.int64), 'core_x': np.asarray(archive['core_x'], dtype=np.int64)}
            self.soft_target_cache[path] = target
        return self.soft_target_cache[path]

    def _lead_rows(self, issuance_row):
        key = (str(issuance_row['period_id']), int(pd.Timestamp(issuance_row['issuance_time']).value))
        rows = self.leads.iloc[self.lead_lookup[key]].sort_values('lead_index')
        if rows['lead_index'].tolist() != list(range(6)):
            raise RuntimeError(f'Invalid lead ordering for {key}')
        return rows

    def _build_targets(self, lead_rows):
        targets = np.zeros((6, *GRID_SHAPE), dtype=np.float32)
        loss_masks = np.zeros((6, *GRID_SHAPE), dtype=bool)
        core_candidates = []
        for _, row in lead_rows.iterrows():
            lead_index = int(row['lead_index'])
            if not bool(row['usable_lead']):
                continue
            mask_paths = parse_list(row['valid_mask_paths'])
            if mask_paths:
                valid = self._load_valid_mask(mask_paths[0]).copy()
                for mask_path in mask_paths[1:]:
                    valid &= self._load_valid_mask(mask_path)
            else:
                valid = np.zeros(GRID_SHAPE, dtype=bool)
            target = np.zeros(GRID_SHAPE, dtype=np.float32)
            loss_mask = valid.copy()
            for soft_path in parse_list(row['soft_target_paths']):
                soft = self._load_soft_target(soft_path)
                np.maximum.at(target, (soft['soft_y'], soft['soft_x']), soft['soft_value'])
                loss_mask[soft['core_y'], soft['core_x']] = True
                for y_value, x_value in zip(soft['core_y'], soft['core_x']):
                    core_candidates.append((int(y_value), int(x_value), lead_index))
            target[~loss_mask] = 0.0
            targets[lead_index] = target
            loss_masks[lead_index] = loss_mask
        return (targets, loss_masks, core_candidates)

    def _choose_crop(self, rng, is_positive, core_candidates):
        patch = self.patch_size
        height, width = GRID_SHAPE
        if is_positive:
            if not core_candidates:
                raise RuntimeError('Positive issuance has no usable positive-core candidate')
            candidate_index = int(rng.integers(0, len(core_candidates)))
            anchor_y, anchor_x, anchor_lead = core_candidates[candidate_index]
            local_low = self.positive_margin
            local_high = patch - self.positive_margin
            desired_local_y = int(rng.integers(local_low, local_high))
            desired_local_x = int(rng.integers(local_low, local_high))
            y0 = int(np.clip(anchor_y - desired_local_y, 0, height - patch))
            x0 = int(np.clip(anchor_x - desired_local_x, 0, width - patch))
            actual_local_y = anchor_y - y0
            actual_local_x = anchor_x - x0
            if not (0 <= actual_local_y < patch and 0 <= actual_local_x < patch):
                raise RuntimeError('Positive anchor disappeared from crop')
            anchor_global = (anchor_y, anchor_x)
            anchor_local = (actual_local_y, actual_local_x)
        else:
            y0 = int(rng.integers(0, height - patch + 1))
            x0 = int(rng.integers(0, width - patch + 1))
            anchor_lead = -1
            anchor_global = (-1, -1)
            anchor_local = (-1, -1)
        return {'y0': y0, 'x0': x0, 'anchor_lead': anchor_lead, 'anchor_global': anchor_global, 'anchor_local': anchor_local}

    def __getitem__(self, index):
        row = self.issuances.iloc[int(index)]
        lead_rows = self._lead_rows(row)
        targets, loss_masks, core_candidates = self._build_targets(lead_rows)
        is_positive = bool(row['any_usable_positive'])
        rng = self._rng(int(index), row)
        crop = self._choose_crop(rng, is_positive, core_candidates)
        y0 = crop['y0']
        x0 = crop['x0']
        y1 = y0 + self.patch_size
        x1 = x0 + self.patch_size
        history = self._load_history(row['issuance_time'])
        history = history[:, :, y0:y1, x0:x1]
        targets = targets[:, y0:y1, x0:x1]
        loss_masks = loss_masks[:, y0:y1, x0:x1]
        if self.normalization is not None:
            history = (history - self.normalization['mean']) / self.normalization['std']
        if self.flatten_time:
            history = history.reshape(13 * 7, self.patch_size, self.patch_size)
        history = np.ascontiguousarray(history, dtype=np.float32)
        targets = np.ascontiguousarray(targets, dtype=np.float32)
        loss_masks = np.ascontiguousarray(loss_masks, dtype=np.float32)
        lead_valid = np.asarray(lead_rows['usable_lead'], dtype=np.float32)
        return {'inputs': torch.from_numpy(history), 'targets': torch.from_numpy(targets), 'loss_masks': torch.from_numpy(loss_masks), 'lead_valid': torch.from_numpy(lead_valid), 'is_positive': torch.tensor(is_positive, dtype=torch.bool), 'period_id': str(row['period_id']), 'issuance_time_ns': torch.tensor(int(pd.Timestamp(row['issuance_time']).value), dtype=torch.int64), 'crop_origin_yx': torch.tensor([y0, x0], dtype=torch.int64), 'anchor_global_yx': torch.tensor(crop['anchor_global'], dtype=torch.int64), 'anchor_local_yx': torch.tensor(crop['anchor_local'], dtype=torch.int64), 'anchor_lead_index': torch.tensor(crop['anchor_lead'], dtype=torch.int64)}

def smoke_test(args):
    dataset = CIGoesPatchDataset(issuance_index=args.issuance_index, lead_index=args.lead_index, goes_root=args.goes_root, patch_size=args.patch_size, positive_margin=args.positive_margin, seed=args.seed, flatten_time=args.flatten_time)
    positive_indices = np.flatnonzero(dataset.issuances['any_usable_positive'].to_numpy(dtype=bool))
    negative_indices = np.flatnonzero(~dataset.issuances['any_usable_positive'].to_numpy(dtype=bool))
    if len(positive_indices) == 0:
        raise RuntimeError('No positive issuance in dataset')
    if len(negative_indices) == 0:
        raise RuntimeError('No negative issuance in dataset')
    selected = [('positive', int(positive_indices[0])), ('negative', int(negative_indices[0]))]
    items = []
    print('\n===== CI GOES PATCH DATASET SMOKE TEST =====')
    print('Dataset rows:', len(dataset))
    print('History offsets:', HISTORY_OFFSETS_MINUTES)
    print('Channels:', EXPECTED_CHANNELS)
    for name, index in selected:
        item = dataset[index]
        items.append(item)
        target_nonzero = (item['targets'] > 0).reshape(6, -1).sum(dim=1)
        supervised_pixels = (item['loss_masks'] > 0).reshape(6, -1).sum(dim=1)
        print(f'\n{name.upper()}')
        print(' index:', index)
        print(' period_id:', item['period_id'])
        print(' inputs:', tuple(item['inputs'].shape))
        print(' targets:', tuple(item['targets'].shape))
        print(' loss_masks:', tuple(item['loss_masks'].shape))
        print(' finite inputs:', bool(torch.isfinite(item['inputs']).all()))
        print(' target nonzero by lead:', target_nonzero.tolist())
        print(' supervised pixels by lead:', supervised_pixels.tolist())
        print(' crop origin:', item['crop_origin_yx'].tolist())
        print(' anchor global:', item['anchor_global_yx'].tolist())
        print(' anchor local:', item['anchor_local_yx'].tolist())
        print(' anchor lead:', int(item['anchor_lead_index']))
        if name == 'positive':
            local_y, local_x = item['anchor_local_yx'].tolist()
            if not (0 <= local_y < args.patch_size and 0 <= local_x < args.patch_size):
                raise RuntimeError('Positive anchor is outside patch')
            if int(target_nonzero.sum()) == 0:
                raise RuntimeError('Positive patch contains no target')
        if name == 'negative':
            if int(target_nonzero.sum()) != 0:
                raise RuntimeError('Negative patch contains a nonzero target')
    input_batch = torch.stack([item['inputs'] for item in items])
    target_batch = torch.stack([item['targets'] for item in items])
    mask_batch = torch.stack([item['loss_masks'] for item in items])
    print('\nBATCH')
    print(' inputs:', tuple(input_batch.shape))
    print(' targets:', tuple(target_batch.shape))
    print(' loss_masks:', tuple(mask_batch.shape))
    print('SMOKE TEST PASSED')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--issuance-index', type=Path, default=DEFAULT_SPLIT_ROOT / 'train_issuance_pool_with_weights.parquet')
    parser.add_argument('--lead-index', type=Path, default=DEFAULT_SPLIT_ROOT / 'train_lead_index.parquet')
    parser.add_argument('--goes-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--patch-size', type=int, default=128)
    parser.add_argument('--positive-margin', type=int, default=12)
    parser.add_argument('--seed', type=int, default=20260930)
    parser.add_argument('--flatten-time', action='store_true')
    args = parser.parse_args()
    smoke_test(args)
if __name__ == '__main__':
    main()
