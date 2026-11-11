"""Consistency validation for the MRMS-only binary isolated-CI product."""
from __future__ import annotations
from ci_nowcasting.paths import resolve_path as _ci_resolve, project_root as _ci_project, source_root as _ci_source
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from core import forecast_intervals, lead_index

def validate(root, cfg=None):
    root = Path(_ci_resolve(root)).resolve()
    if cfg is None:
        cfg = json.loads((root / 'config.json').read_text())
    objects = pd.read_parquet(_ci_resolve(root / 'isolated_ci_catalog.parquet'))
    labels = pd.read_parquet(_ci_resolve(root / 'label_index.parquet'))
    frames = pd.read_parquet(_ci_resolve(root / 'frame_inventory.parquet'))
    checks = {}
    checks['unique_object_id'] = not objects.object_id.duplicated().any()
    checks['unique_physical_birth_signature'] = not objects.duplicated(['period_id', 'birth_time', 'birth_y_px', 'birth_x_px']).any()
    mask_ids, area_ok, masks_nonempty = (set(), True, True)
    spacing = frames.groupby('period_id')[['dy_km', 'dx_km']].first()
    for row in objects.itertuples(index=False):
        data = np.load(_ci_resolve(row.native_birth_mask_path))
        y, x = (data['y'], data['x'])
        masks_nonempty &= len(y) > 0 and len(y) == len(x)
        dy, dx = spacing.loc[row.period_id]
        area_ok &= len(y) == int(row.area_px) and np.isclose(len(y) * dy * dx, row.area_km2)
        mask_ids.add(row.object_id)
    checks['positive_masks_nonempty'] = bool(masks_nonempty)
    checks['mask_area_matches_catalog'] = bool(area_ok)
    checks['all_confirmed'] = bool((pd.to_datetime(objects.confirmation_time) > pd.to_datetime(objects.birth_time)).all())
    checks['all_past_isolation_distances_at_least_threshold'] = bool((objects.past_min_mature_distance_km >= cfg['isolation_radius_km']).all())
    checks['all_same_time_isolation_distances_at_least_threshold'] = bool((objects.same_time_min_mature_distance_km >= cfg['isolation_radius_km']).all())
    linked = []
    unique_interval = True
    for row in labels.itertuples(index=False):
        ids = json.loads(row.positive_object_ids)
        linked.extend(ids)
        unique_interval &= len(ids) == len(set(ids)) and len(ids) == row.positive_count
        for object_id in ids:
            birth = objects.set_index('object_id').loc[object_id, 'birth_time'] if object_id in mask_ids else None
            unique_interval &= birth is not None and lead_index(birth, row.issuance_time, cfg['forecast_bin_minutes'], cfg['max_forecast_minutes']) == row.lead_index
    checks['all_label_links_exist'] = set(linked) <= mask_ids
    exploded = labels.assign(ids=labels.positive_object_ids.map(json.loads)).explode('ids').dropna(subset=['ids'])
    checks['positive_once_per_issuance'] = bool(unique_interval and (not exploded.duplicated(['period_id', 'issuance_time', 'ids']).any()))
    checks['six_or_configured_bins'] = labels.lead_index.nunique() == len(forecast_intervals(cfg['forecast_bin_minutes'], cfg['max_forecast_minutes']))
    checks['temporally_valid_intervals_reference_spatial_masks'] = not (labels.target_temporally_valid & labels.required_valid_mask_ids.eq('[]')).any()
    checks['valid_mask_references_exist'] = frames.valid_mask_path.map(lambda p: Path(_ci_resolve(p)).is_file()).all()
    checks['reflectivity_references_exist'] = frames.reflectivity_path.map(lambda p: Path(_ci_resolve(p)).is_dir()).all()
    checks = {key: bool(value) for key, value in checks.items()}
    result = {'passed': bool(all(checks.values())), 'checks': checks, 'objects': len(objects), 'label_rows': len(labels), 'frame_rows': len(frames)}
    (root / 'validation_report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise SystemExit(1)
    return result
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    validate(args.root)
