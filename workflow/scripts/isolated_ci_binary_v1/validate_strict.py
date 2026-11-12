"""Validate strict-isolation metrics in a finalized catalogue."""
from ci_nowcasting.paths import resolve_path as _ci_resolve, project_root as _ci_project, source_root as _ci_source
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(_ci_resolve(args.root)).resolve()
    cfg = json.loads((root / 'config.json').read_text())
    catalog = pd.read_parquet(_ci_resolve(root / 'isolated_ci_catalog.parquet'))
    checks = {'strict_columns_present': all((column in catalog.columns for column in ['strict_isolated', 'background_max_fraction_ge10', 'background_min_valid_fraction', 'connected20_outer_area_km2']))}
    if checks['strict_columns_present'] and len(catalog):
        checks['all_objects_strict'] = bool(catalog['strict_isolated'].astype(bool).all())
        checks['background_fraction_within_limit'] = bool((catalog['background_max_fraction_ge10'] <= cfg['strict_maximum_background_fraction_ge10'] + 1e-12).all())
        checks['connected_area_within_limit'] = bool((catalog['connected20_outer_area_km2'] <= cfg['strict_maximum_connected20_outer_area_km2'] + 1e-12).all())
        checks['strict_metrics_finite'] = bool(np.isfinite(catalog[['background_max_fraction_ge10', 'background_min_valid_fraction', 'connected20_outer_area_km2']].to_numpy(dtype=float)).all())
    else:
        checks['all_objects_strict'] = True
        checks['background_fraction_within_limit'] = True
        checks['connected_area_within_limit'] = True
        checks['strict_metrics_finite'] = True
    result = {'passed': bool(all(checks.values())), 'checks': checks, 'objects': int(len(catalog))}
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise SystemExit(1)
if __name__ == '__main__':
    main()
