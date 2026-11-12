from ci_nowcasting.paths import resolve_path as _ci_resolve, project_root as _ci_project, source_root as _ci_source
import json
from pathlib import Path
import numpy as np
import pandas as pd
SEED = 20260930
N_RESTARTS = 1500
ROOT = Path(_ci_resolve(_ci_resolve('/project/data/ci_isolated_mrms_binary_v1/goes_ir_parallax_native_filled_v1')))
STATE = ROOT / 'state'
OUTPUT = ROOT / 'splits/period_grouped_80_10_10_v1'
LEAD_PATH = STATE / 'ci_goes_training_lead_index.parquet'
ISSUANCE_PATH = STATE / 'ci_goes_training_issuance_index.parquet'
SOFT_MANIFEST_PATH = ROOT / 'labels/ci_soft_targets_v1/state/soft_target_build_manifest.parquet'
SPLIT_NAMES = np.asarray(['train', 'validation', 'test'])
SPLIT_RATIOS = np.asarray([0.8, 0.1, 0.1], dtype=np.float64)
FEATURES = ['unique_ci_objects', 'positive_issuances', 'issuances', 'period_count']
FEATURE_WEIGHTS = np.asarray([12.0, 5.0, 1.0, 0.5], dtype=np.float64)

def score_assignment(assignment, feature_values, year_indices, n_years):
    total = feature_values.sum(axis=0)
    actual = np.zeros((3, len(FEATURES)), dtype=np.float64)
    for split_index in range(3):
        actual[split_index] = feature_values[assignment == split_index].sum(axis=0)
    expected = SPLIT_RATIOS[:, None] * total[None, :]
    denominator = np.maximum(expected, 1.0)
    score = float(np.sum(FEATURE_WEIGHTS[None, :] * ((actual - expected) / denominator) ** 2))
    for year_index in range(n_years):
        selected = year_indices == year_index
        if not selected.any():
            continue
        year_total = feature_values[selected].sum(axis=0)
        year_actual = np.zeros_like(actual)
        for split_index in range(3):
            year_actual[split_index] = feature_values[selected & (assignment == split_index)].sum(axis=0)
        year_expected = SPLIT_RATIOS[:, None] * year_total[None, :]
        year_denominator = np.maximum(year_expected, 1.0)
        score += 0.5 * float(np.sum(FEATURE_WEIGHTS[None, :] * ((year_actual - year_expected) / year_denominator) ** 2))
    return score

def make_assignment(feature_values, year_indices, n_years, seed):
    rng = np.random.default_rng(seed)
    difficulty = feature_values[:, 0] * 100.0 + feature_values[:, 1] * 5.0 + np.sqrt(feature_values[:, 2])
    order = np.argsort(-(difficulty * rng.uniform(0.85, 1.15, size=len(difficulty))))
    assignment = np.full(len(feature_values), -1, dtype=np.int8)
    global_actual = np.zeros((3, len(FEATURES)), dtype=np.float64)
    global_processed = np.zeros(len(FEATURES), dtype=np.float64)
    year_actual = np.zeros((n_years, 3, len(FEATURES)), dtype=np.float64)
    year_processed = np.zeros((n_years, len(FEATURES)), dtype=np.float64)
    for period_index in order:
        values = feature_values[period_index]
        year_index = year_indices[period_index]
        processed_after = global_processed + values
        year_processed_after = year_processed[year_index] + values
        candidate_scores = []
        for split_index in range(3):
            candidate_global = global_actual.copy()
            candidate_global[split_index] += values
            expected_global = SPLIT_RATIOS[:, None] * processed_after[None, :]
            global_denominator = np.maximum(expected_global, 1.0)
            candidate_score = float(np.sum(FEATURE_WEIGHTS[None, :] * ((candidate_global - expected_global) / global_denominator) ** 2))
            candidate_year = year_actual[year_index].copy()
            candidate_year[split_index] += values
            expected_year = SPLIT_RATIOS[:, None] * year_processed_after[None, :]
            year_denominator = np.maximum(expected_year, 1.0)
            candidate_score += 0.5 * float(np.sum(FEATURE_WEIGHTS[None, :] * ((candidate_year - expected_year) / year_denominator) ** 2))
            candidate_score += float(rng.uniform(0.0, 1e-10))
            candidate_scores.append(candidate_score)
        selected_split = int(np.argmin(candidate_scores))
        assignment[period_index] = selected_split
        global_actual[selected_split] += values
        global_processed += values
        year_actual[year_index, selected_split] += values
        year_processed[year_index] += values
    return assignment
print('Loading final dataset indexes...')
lead = pd.read_parquet(_ci_resolve(LEAD_PATH))
issuance_source = pd.read_parquet(_ci_resolve(ISSUANCE_PATH))
soft = pd.read_parquet(_ci_resolve(SOFT_MANIFEST_PATH))
lead['issuance_time'] = pd.to_datetime(lead['issuance_time'])
issuance_source['issuance_time'] = pd.to_datetime(issuance_source['issuance_time'])
issuance = issuance_source[issuance_source['any_lead_usable']].copy()
excluded_issuances = issuance_source[~issuance_source['any_lead_usable']].copy()
if len(issuance) != 49000:
    raise RuntimeError(f'Expected 49,000 usable issuances, found {len(issuance)}')
period_year_counts = issuance.groupby('period_id')['year'].nunique()
if int(period_year_counts.max()) != 1:
    raise RuntimeError('At least one period spans multiple years')
period_stats = issuance.groupby(['period_id', 'year'], as_index=False).agg(issuances=('issuance_time', 'size'), positive_issuances=('any_usable_positive', 'sum'))
period_stats['negative_issuances'] = period_stats['issuances'] - period_stats['positive_issuances']
object_counts = soft.groupby(['period_id', 'year'], as_index=False)['object_id'].nunique().rename(columns={'object_id': 'unique_ci_objects'})
period_stats = period_stats.merge(object_counts, on=['period_id', 'year'], how='left', validate='one_to_one')
period_stats['unique_ci_objects'] = period_stats['unique_ci_objects'].fillna(0).astype(int)
period_stats['period_count'] = 1
period_stats = period_stats.sort_values(['year', 'period_id']).reset_index(drop=True)
feature_values = period_stats[FEATURES].to_numpy(dtype=np.float64)
years = sorted(period_stats['year'].astype(int).unique())
year_lookup = {year: index for index, year in enumerate(years)}
year_indices = np.asarray([year_lookup[int(year)] for year in period_stats['year']], dtype=np.int64)
print(f'Optimizing grouped split across {len(period_stats)} periods...')
best_assignment = None
best_score = np.inf
for restart in range(N_RESTARTS):
    assignment = make_assignment(feature_values, year_indices, len(years), SEED + restart)
    score = score_assignment(assignment, feature_values, year_indices, len(years))
    if score < best_score:
        best_score = score
        best_assignment = assignment.copy()
if best_assignment is None:
    raise RuntimeError('Split optimization produced no assignment')
period_stats['split'] = SPLIT_NAMES[best_assignment]
period_stats['split_ratio_target'] = period_stats['split'].map({'train': 0.8, 'validation': 0.1, 'test': 0.1})
period_to_split = dict(zip(period_stats['period_id'], period_stats['split']))
issuance['split'] = issuance['period_id'].map(period_to_split)
if issuance['split'].isna().any():
    raise RuntimeError('Some usable issuances lack a split')
usable_keys = issuance[['period_id', 'issuance_time', 'split']]
lead_dataset = lead.merge(usable_keys, on=['period_id', 'issuance_time'], how='inner', validate='many_to_one')
if len(lead_dataset) != len(issuance) * 6:
    raise RuntimeError(f'Lead/issuance cardinality mismatch: {len(lead_dataset)} versus {len(issuance) * 6}')
OUTPUT.mkdir(parents=True, exist_ok=True)
period_stats.to_parquet(OUTPUT / 'period_split_assignment.parquet', index=False)
excluded_issuances.to_parquet(OUTPUT / 'excluded_no_usable_lead_issuances.parquet', index=False)
split_rows = []
for split_name in SPLIT_NAMES:
    split_issuance = issuance[issuance['split'] == split_name].copy()
    split_lead = lead_dataset[lead_dataset['split'] == split_name].copy()
    split_issuance = split_issuance.sort_values(['year', 'period_id', 'issuance_time']).reset_index(drop=True)
    split_lead = split_lead.sort_values(['year', 'period_id', 'issuance_time', 'lead_index']).reset_index(drop=True)
    split_issuance.to_parquet(OUTPUT / f'{split_name}_issuance_index_natural.parquet', index=False)
    split_lead.to_parquet(OUTPUT / f'{split_name}_lead_index.parquet', index=False)
    split_periods = period_stats[period_stats['split'] == split_name]
    split_rows.append({'split': split_name, 'periods': int(split_issuance['period_id'].nunique()), 'issuances': int(len(split_issuance)), 'positive_issuances': int(split_issuance['any_usable_positive'].sum()), 'negative_issuances': int((~split_issuance['any_usable_positive']).sum()), 'positive_issuance_pct': float(100.0 * split_issuance['any_usable_positive'].mean()), 'unique_ci_objects': int(split_periods['unique_ci_objects'].sum()), 'usable_leads': int(split_lead['usable_lead'].sum()), 'positive_usable_leads': int((split_lead['usable_lead'] & split_lead['label_kind'].eq('positive')).sum())})
split_summary = pd.DataFrame(split_rows)
split_summary['ci_object_pct'] = 100.0 * split_summary['unique_ci_objects'] / split_summary['unique_ci_objects'].sum()
split_summary['issuance_pct'] = 100.0 * split_summary['issuances'] / split_summary['issuances'].sum()
train = issuance[issuance['split'] == 'train'].copy()
train['sampling_class'] = np.where(train['any_usable_positive'], 'positive', 'negative')
n_train_positive = int(train['any_usable_positive'].sum())
n_train_negative = int((~train['any_usable_positive']).sum())
train['balanced_sampling_weight'] = np.where(train['any_usable_positive'], 0.5 / n_train_positive, 0.5 / n_train_negative)
train.to_parquet(OUTPUT / 'train_issuance_pool_with_weights.parquet', index=False)
train_positive = train[train['any_usable_positive']].copy()
train_negative = train[~train['any_usable_positive']].copy()
if len(train_negative) < len(train_positive):
    raise RuntimeError('Not enough train negatives for 50/50 sampling')
epoch_negative = train_negative.sample(n=len(train_positive), replace=False, random_state=SEED)
balanced_epoch = pd.concat([train_positive, epoch_negative], ignore_index=True)
balanced_epoch = balanced_epoch.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
balanced_epoch.insert(0, 'epoch_position', np.arange(len(balanced_epoch), dtype=np.int64))
balanced_epoch.to_parquet(OUTPUT / 'train_balanced_50_50_epoch000_issuance_index.parquet', index=False)
balanced_keys = balanced_epoch[['epoch_position', 'period_id', 'issuance_time']]
balanced_leads = lead_dataset.merge(balanced_keys, on=['period_id', 'issuance_time'], how='inner', validate='many_to_one')
balanced_leads = balanced_leads.sort_values(['epoch_position', 'lead_index']).reset_index(drop=True)
balanced_leads.to_parquet(OUTPUT / 'train_balanced_50_50_epoch000_lead_index.parquet', index=False)
period_sets = {split_name: set(period_stats.loc[period_stats['split'] == split_name, 'period_id']) for split_name in SPLIT_NAMES}
if period_sets['train'] & period_sets['validation']:
    raise RuntimeError('Period leakage between train and validation')
if period_sets['train'] & period_sets['test']:
    raise RuntimeError('Period leakage between train and test')
if period_sets['validation'] & period_sets['test']:
    raise RuntimeError('Period leakage between validation and test')
object_split = soft[['object_id', 'period_id']].copy()
object_split['split'] = object_split['period_id'].map(period_to_split)
if object_split['split'].isna().any():
    raise RuntimeError('Some CI objects lack a split')
if object_split['object_id'].duplicated().any():
    raise RuntimeError('A CI object occurs more than once')
year_table = issuance.groupby(['year', 'split'], as_index=False).agg(issuances=('issuance_time', 'size'), positive_issuances=('any_usable_positive', 'sum'))
year_table['positive_pct'] = 100.0 * year_table['positive_issuances'] / year_table['issuances']
year_table.to_csv(OUTPUT / 'split_by_year.csv', index=False)
split_summary.to_csv(OUTPUT / 'split_summary.csv', index=False)
summary = {'passed': True, 'seed': SEED, 'grouping_unit': 'period_id', 'target_ratios': {'train': 0.8, 'validation': 0.1, 'test': 0.1}, 'optimization_priority': 'unique CI objects, positive issuances, total issuances, periods, and year balance', 'source_issuances': int(len(issuance_source)), 'dataset_issuances': int(len(issuance)), 'excluded_no_usable_lead_issuances': int(len(excluded_issuances)), 'source_lead_rows': int(len(lead)), 'dataset_lead_rows': int(len(lead_dataset)), 'unique_ci_objects': int(soft['object_id'].nunique()), 'optimization_score': float(best_score), 'train_natural_pool_retained': True, 'train_positive_issuances': int(len(train_positive)), 'train_negative_issuances': int(len(train_negative)), 'balanced_epoch_size': int(len(balanced_epoch)), 'balanced_epoch_positive': int(balanced_epoch['any_usable_positive'].sum()), 'balanced_epoch_negative': int((~balanced_epoch['any_usable_positive']).sum()), 'validation_sampling': 'natural', 'test_sampling': 'natural', 'period_leakage': False, 'object_leakage': False, 'splits': split_summary.to_dict(orient='records')}
with open(_ci_resolve(OUTPUT / 'split_summary.json'), 'w', encoding='utf-8') as handle:
    json.dump(summary, handle, indent=2)
print('\n===== PERIOD-GROUPED CI SPLIT =====')
print(split_summary.to_string(index=False))
print('\n===== BY YEAR =====')
print(year_table.to_string(index=False))
print('\n===== BALANCED TRAIN EPOCH 000 =====')
print('Positive:', len(train_positive))
print('Negative sampled:', len(epoch_negative))
print('Epoch size:', len(balanced_epoch))
print('Positive fraction:', float(balanced_epoch['any_usable_positive'].mean()))
print('\nExcluded issuances with no usable lead:', len(excluded_issuances))
print('Period leakage: False')
print('Object leakage: False')
print('Saved under:', OUTPUT)
