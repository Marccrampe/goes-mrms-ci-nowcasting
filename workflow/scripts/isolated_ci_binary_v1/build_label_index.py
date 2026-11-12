"""Regenerate forecast-bin labels without rerunning physical CI detection."""
from __future__ import annotations
from ci_nowcasting.paths import resolve_path as _ci_resolve, project_root as _ci_project, source_root as _ci_source
import argparse
import json
from pathlib import Path
import pandas as pd
from pipeline import build_label_index

def main(root, max_forecast_minutes, output=None):
    root = Path(_ci_resolve(root)).resolve()
    cfg = json.loads((root / 'config.json').read_text())
    if max_forecast_minutes > cfg['reserved_forecast_minutes']:
        raise ValueError('Requested horizon exceeds reserved MRMS future support')
    cfg['max_forecast_minutes'] = int(max_forecast_minutes)
    frames = pd.read_parquet(_ci_resolve(root / 'frame_inventory.parquet'))
    objects = pd.read_parquet(_ci_resolve(root / 'isolated_ci_catalog.parquet'))
    periods = pd.read_parquet(_ci_resolve(root / 'period_manifest.parquet'))
    labels = build_label_index(cfg, frames, objects, periods)
    destination = Path(_ci_resolve(output)).resolve() if output else root / f'label_index_{max_forecast_minutes}min.parquet'
    temporary = Path(_ci_resolve(str(destination) + '.tmp'))
    labels.to_parquet(temporary, index=False)
    temporary.replace(destination)
    print(f'Wrote {len(labels)} rows to {destination}; physical catalogue unchanged')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--max-forecast-minutes', type=int, required=True)
    parser.add_argument('--output')
    args = parser.parse_args()
    main(args.root, args.max_forecast_minutes, args.output)
