from ci_nowcasting.paths import resolve_path as _ci_resolve, project_root as _ci_project, source_root as _ci_source
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(_ci_resolve(_ci_resolve('/project/data/ci_isolated_mrms_binary_v1/goes_ir_parallax_native_filled_v1')))
STATE = ROOT / 'state'
LEAD_INPUT = STATE / 'ci_goes_lead_index.parquet'
ISSUANCE_INPUT = STATE / 'ci_goes_issuance_index.parquet'
SOFT_MANIFEST = ROOT / 'labels/ci_soft_targets_v1/state/soft_target_build_manifest.parquet'
VALID_MANIFEST = ROOT / 'labels/ci_valid_masks_native_goes_v1/state/projected_valid_mask_manifest.parquet'
LEAD_OUTPUT = STATE / 'ci_goes_training_lead_index.parquet'
ISSUANCE_OUTPUT = STATE / 'ci_goes_training_issuance_index.parquet'
SUMMARY_OUTPUT = STATE / 'ci_goes_training_index_summary.json'

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
    raise TypeError(f'Unsupported list representation: {type(value)}')
print('Loading indexes...')
lead = pd.read_parquet(_ci_resolve(LEAD_INPUT))
issuance = pd.read_parquet(_ci_resolve(ISSUANCE_INPUT))
soft = pd.read_parquet(_ci_resolve(SOFT_MANIFEST))
valid = pd.read_parquet(_ci_resolve(VALID_MANIFEST))
lead['issuance_time'] = pd.to_datetime(lead['issuance_time'])
issuance['issuance_time'] = pd.to_datetime(issuance['issuance_time'])
if soft['object_id'].duplicated().any():
    raise RuntimeError('Duplicated object_id in soft-target manifest')
if valid.duplicated(['period_id', 'valid_mask_id']).any():
    raise RuntimeError('Duplicated period/valid-mask pair')
if lead.duplicated(['period_id', 'issuance_time', 'lead_index']).any():
    raise RuntimeError('Duplicated lead-index key')
if issuance.duplicated(['period_id', 'issuance_time']).any():
    raise RuntimeError('Duplicated issuance-index key')
print('Checking referenced files...')
missing_soft_files = [path for path in soft['output_path'].astype(str) if not Path(_ci_resolve(path)).exists()]
missing_valid_files = [path for path in valid['projected_valid_mask_path'].astype(str) if not Path(_ci_resolve(path)).exists()]
if missing_soft_files:
    raise RuntimeError(f'Missing soft-target files: {len(missing_soft_files)}')
if missing_valid_files:
    raise RuntimeError(f'Missing projected valid-mask files: {len(missing_valid_files)}')
soft_lookup = {str(row.object_id): {'period_id': str(row.period_id), 'path': str(row.output_path), 'core_pixels': int(row.goes_core_pixels), 'soft_pixels': int(row.soft_target_pixels)} for row in soft.itertuples(index=False)}
valid_lookup = {(str(row.period_id), str(row.valid_mask_id)): {'path': str(row.projected_valid_mask_path), 'valid_pixels': int(row.valid_pixels), 'empty': bool(row.projected_mask_empty)} for row in valid.itertuples(index=False)}
valid_array_cache = {}
valid_combination_pixel_cache = {}

def load_projected_valid_mask(mask_path):
    mask_path = str(mask_path)
    if mask_path not in valid_array_cache:
        with np.load(_ci_resolve(mask_path), allow_pickle=False) as archive:
            mask = np.asarray(archive['valid_mask'], dtype=bool)
        if mask.shape != (212, 230):
            raise RuntimeError(f'Unexpected projected mask shape: {mask_path} {mask.shape}')
        valid_array_cache[mask_path] = mask
    return valid_array_cache[mask_path]

def combined_valid_pixel_count(mask_paths):
    key = tuple(sorted((str(value) for value in mask_paths)))
    if not key:
        return 0
    if key not in valid_combination_pixel_cache:
        combined = load_projected_valid_mask(key[0]).copy()
        for mask_path in key[1:]:
            combined &= load_projected_valid_mask(mask_path)
        valid_combination_pixel_cache[key] = int(np.count_nonzero(combined))
    return valid_combination_pixel_cache[key]
resolved_object_ids = []
soft_target_paths = []
valid_mask_ids = []
valid_mask_paths = []
base_valid_pixels = []
positive_core_pixels_sum = []
positive_soft_pixels_sum = []
base_valid_mask_empty = []
usable_leads = []
label_kinds = []
loss_mask_rules = []
core_only_positive = []
zero_target_generated = []
print('Resolving 294,030 lead rows...')
for row in lead.itertuples(index=False):
    period_id = str(row.period_id)
    object_ids = [str(value) for value in parse_list(row.positive_object_ids)]
    required_mask_ids = [str(value) for value in parse_list(row.required_valid_mask_ids)]
    if int(row.positive_count) != len(object_ids):
        raise RuntimeError(f'positive_count mismatch for {period_id} {row.issuance_time} lead={row.lead_index}')
    if not required_mask_ids and bool(row.target_valid):
        raise RuntimeError(f'A valid lead has no required valid mask: {period_id} {row.issuance_time} lead={row.lead_index}')
    required_mask_ids = sorted(set(required_mask_ids))
    resolved_mask_ids = []
    mask_paths = []
    for mask_id in required_mask_ids:
        mask_key = (period_id, mask_id)
        if mask_key not in valid_lookup:
            if bool(row.target_valid):
                raise RuntimeError(f'Unresolved valid mask for a valid lead: {mask_key}')
            continue
        resolved_mask_ids.append(mask_id)
        mask_paths.append(valid_lookup[mask_key]['path'])
    mask_pixels = combined_valid_pixel_count(mask_paths)
    mask_empty = mask_pixels == 0
    resolved_object_ids_for_paths = []
    object_paths = []
    core_pixel_sum = 0
    soft_pixel_sum = 0
    for object_id in object_ids:
        if object_id not in soft_lookup:
            if bool(row.target_valid):
                raise RuntimeError(f'Unresolved CI object for a valid lead: {object_id}')
            continue
        object_entry = soft_lookup[object_id]
        if object_entry['period_id'] != period_id:
            raise RuntimeError(f'Object/period mismatch: {object_id}')
        resolved_object_ids_for_paths.append(object_id)
        object_paths.append(object_entry['path'])
        core_pixel_sum += object_entry['core_pixels']
        soft_pixel_sum += object_entry['soft_pixels']
    upstream_target_valid = bool(row.target_valid)
    is_positive = len(object_ids) > 0
    usable = upstream_target_valid and (mask_pixels > 0 or is_positive)
    is_core_only = usable and is_positive and mask_empty
    is_generated_zero = usable and (not is_positive)
    if not usable:
        label_kind = 'ignore'
        loss_rule = 'no_supervision'
    elif is_core_only:
        label_kind = 'positive'
        loss_rule = 'positive_core_only'
    elif is_positive:
        label_kind = 'positive'
        loss_rule = 'base_valid_or_positive_core'
    else:
        label_kind = 'negative'
        loss_rule = 'base_valid_zero_target'
    resolved_object_ids.append(json.dumps(resolved_object_ids_for_paths))
    soft_target_paths.append(json.dumps(object_paths))
    valid_mask_ids.append(json.dumps(resolved_mask_ids))
    valid_mask_paths.append(json.dumps(mask_paths))
    base_valid_pixels.append(mask_pixels)
    positive_core_pixels_sum.append(core_pixel_sum)
    positive_soft_pixels_sum.append(soft_pixel_sum)
    base_valid_mask_empty.append(mask_empty)
    usable_leads.append(usable)
    label_kinds.append(label_kind)
    loss_mask_rules.append(loss_rule)
    core_only_positive.append(is_core_only)
    zero_target_generated.append(is_generated_zero)
training_lead = lead.copy()
training_lead['resolved_positive_object_ids'] = resolved_object_ids
training_lead['unresolved_positive_object_count'] = training_lead['positive_count'].astype(int) - training_lead['resolved_positive_object_ids'].apply(lambda value: len(json.loads(value)))
training_lead['soft_target_paths'] = soft_target_paths
training_lead['resolved_valid_mask_ids'] = valid_mask_ids
training_lead['valid_mask_paths'] = valid_mask_paths
training_lead['resolved_valid_mask_count'] = training_lead['resolved_valid_mask_ids'].apply(lambda value: len(json.loads(value)))
training_lead['unresolved_valid_mask_count'] = training_lead.apply(lambda row: len(parse_list(row['required_valid_mask_ids'])) - int(row['resolved_valid_mask_count']), axis=1)
training_lead['base_valid_pixels'] = np.asarray(base_valid_pixels, dtype=np.int32)
training_lead['positive_core_pixels_sum'] = np.asarray(positive_core_pixels_sum, dtype=np.int32)
training_lead['positive_soft_pixels_sum'] = np.asarray(positive_soft_pixels_sum, dtype=np.int32)
training_lead['base_valid_mask_empty'] = np.asarray(base_valid_mask_empty, dtype=bool)
training_lead['usable_lead'] = np.asarray(usable_leads, dtype=bool)
training_lead['label_kind'] = label_kinds
training_lead['loss_mask_rule'] = loss_mask_rules
training_lead['core_only_positive'] = np.asarray(core_only_positive, dtype=bool)
training_lead['zero_target_generated'] = np.asarray(zero_target_generated, dtype=bool)
derived_issuance = training_lead.groupby(['period_id', 'issuance_time', 'year'], as_index=False).agg(usable_lead_count=('usable_lead', 'sum'), usable_positive_lead_count=('core_only_positive', lambda values: int(training_lead.loc[values.index, 'label_kind'].eq('positive').sum())), ignored_lead_count=('usable_lead', lambda values: int((~values).sum())), core_only_positive_lead_count=('core_only_positive', 'sum'), generated_zero_lead_count=('zero_target_generated', 'sum'))
positive_counts = training_lead.assign(usable_positive=training_lead['usable_lead'] & training_lead['label_kind'].eq('positive')).groupby(['period_id', 'issuance_time', 'year'], as_index=False)['usable_positive'].sum().rename(columns={'usable_positive': 'usable_positive_lead_count'})
derived_issuance = derived_issuance.drop(columns=['usable_positive_lead_count']).merge(positive_counts, on=['period_id', 'issuance_time', 'year'], how='left', validate='one_to_one')
derived_issuance['all_six_leads_usable'] = derived_issuance['usable_lead_count'] == 6
derived_issuance['any_lead_usable'] = derived_issuance['usable_lead_count'] > 0
derived_issuance['any_usable_positive'] = derived_issuance['usable_positive_lead_count'] > 0
training_issuance = issuance.merge(derived_issuance, on=['period_id', 'issuance_time', 'year'], how='left', validate='one_to_one')
if training_issuance['usable_lead_count'].isna().any():
    raise RuntimeError('Some issuances were not resolved')
training_lead = training_lead.sort_values(['year', 'period_id', 'issuance_time', 'lead_index']).reset_index(drop=True)
training_issuance = training_issuance.sort_values(['year', 'period_id', 'issuance_time']).reset_index(drop=True)
STATE.mkdir(parents=True, exist_ok=True)
training_lead.to_parquet(LEAD_OUTPUT, index=False)
training_issuance.to_parquet(ISSUANCE_OUTPUT, index=False)
usable = training_lead['usable_lead']
positive = training_lead['label_kind'].eq('positive')
negative = training_lead['label_kind'].eq('negative')
usable_issuance = training_issuance[training_issuance['any_lead_usable']].copy()
summary = {'passed': True, 'lead_rows': int(len(training_lead)), 'issuances': int(len(training_issuance)), 'usable_leads': int(usable.sum()), 'ignored_leads': int((~usable).sum()), 'positive_usable_leads': int((usable & positive).sum()), 'negative_usable_leads': int((usable & negative).sum()), 'core_only_positive_leads': int(training_lead['core_only_positive'].sum()), 'empty_mask_negative_leads_ignored': int((training_lead['base_valid_mask_empty'] & training_lead['positive_count'].eq(0) & training_lead['target_valid']).sum()), 'all_six_leads_usable_issuances': int(training_issuance['all_six_leads_usable'].sum()), 'partially_usable_issuances': int((training_issuance['any_lead_usable'] & ~training_issuance['all_six_leads_usable']).sum()), 'issuances_with_usable_ci': int(usable_issuance['any_usable_positive'].sum()), 'natural_positive_issuance_fraction': float(usable_issuance['any_usable_positive'].mean()), 'natural_positive_lead_fraction': float(positive[usable].mean()), 'zero_target_arrays_materialized': False, 'negative_target_rule': 'zero target generated at load time inside projected valid mask', 'positive_target_rule': 'pixelwise maximum of referenced object soft targets', 'positive_loss_mask_rule': 'projected valid mask OR positive core', 'grid_shape': [212, 230]}
with open(_ci_resolve(SUMMARY_OUTPUT), 'w', encoding='utf-8') as handle:
    json.dump(summary, handle, indent=2)
print('\n===== FINAL TRAINING LABEL INDEX =====')
for key, value in summary.items():
    print(f'{key}: {value}')
print('\nLead index:', LEAD_OUTPUT)
print('Issuance index:', ISSUANCE_OUTPUT)
print('Summary:', SUMMARY_OUTPUT)
