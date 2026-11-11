#!/usr/bin/env python3
"""Run a documented scientific stage with a configurable local workspace."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from ci_nowcasting.paths import REPOSITORY, source_root, project_root

SCI='scripts/isolated_ci_binary_v1/goes_ir_v1/'
TASKS={
    'candidate-catalogue':'scripts/20_scan_cids_strict_mrms_windows.py',
    'native-periods':'scripts/isolated_ci_binary_v1/pipeline.py',
    'native-label-index':'scripts/isolated_ci_binary_v1/build_label_index.py',
    'goes-requests':SCI+'build_request_manifest.py',
    'goes-inventory':SCI+'build_s3_manifest.py',
    'goes-parallax':SCI+'build_parallax_native_streaming.py',
    'goes-fill':SCI+'fill_native_goes_parallax_nans.py',
    'ci35-catalogue':SCI+'build_ci_labels_v3_contour20.py',
    'training-index':SCI+'build_ci_training_label_index.py',
    'period-splits':SCI+'build_ci_period_splits.py',
    'radar-aux':SCI+'build_ci_radar_aux_v1.py',
    'valid-masks':SCI+'build_ci_valid_masks_native_goes.py',
    'dataset-two-windows':SCI+'prepare_ci_v3_30min_dataset.py',
    'mrms-complete':SCI+'complete_ci_v3_mrms_inputs.py',
    'binary20-build':'scripts/build_ci_v4_binary20.py',
    'binary20-qc':'scripts/qc_ci_v4_binary20.py',
    'dataset-paired':SCI+'prepare_ci_v4_paired_datasets.py',
    'catalogue-check':'scripts/isolated_ci_binary_v1/validate_strict.py',
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,default=project_root(),help='Data/output tree; defaults to repository root')
    p.add_argument('task',choices=['configure','list']+list(TASKS))
    p.add_argument('arguments',nargs=argparse.REMAINDER)
    a=p.parse_args();workspace=a.workspace.expanduser().resolve()
    if a.task=='list':
        for name,script in TASKS.items():print(name,script)
        return
    if a.task=='configure':
        workspace.mkdir(parents=True,exist_ok=True)
        target=workspace/'configs';target.mkdir(exist_ok=True)
        old='/crest/students/mcrampe00/PDE_CNN_precip'
        def rebase(v):
            if isinstance(v,dict):return {k:rebase(x) for k,x in v.items()}
            if isinstance(v,list):return [rebase(x) for x in v]
            if isinstance(v,str) and '${CI_WORKSPACE}' in v:return v.replace('${CI_WORKSPACE}',str(workspace))
            if isinstance(v,str) and (v==old or v.startswith(old+'/')):return str(workspace/v[len(old):].lstrip('/'))
            return v
        for src in (REPOSITORY/'configs/catalogue_templates').glob('*.json'):
            dest=target/src.name
            if dest.exists():raise RuntimeError('Existing config preserved: '+str(dest))
            cfg=rebase(json.loads(src.read_text()))
            if 'project_root' in cfg:cfg['project_root']=str(workspace)
            if 'temporary_root' in cfg:cfg['temporary_root']=str(workspace/'temporary')
            dest.write_text(json.dumps(cfg,indent=2)+'\n');print('CONFIG',dest)
        return
    script=source_root()/TASKS[a.task]
    env=os.environ.copy();env['CI_WORKSPACE']=str(workspace)
    env['PYTHONPATH']=os.pathsep.join([str(REPOSITORY),str(script.parent),str(source_root()/SCI),env.get('PYTHONPATH','')])
    tail=a.arguments[1:] if a.arguments[:1]==['--'] else a.arguments
    # The original splitting program has no argparse and otherwise starts work.
    if a.task in ['period-splits','training-index'] and '--help' in tail:
        print('Uses CI_WORKSPACE data/ci_isolated_mrms_binary_v1; see docs/pipeline.md. Run with no extra arguments.')
        return
    workspace.mkdir(parents=True,exist_ok=True)
    raise SystemExit(subprocess.call([sys.executable,str(script),*tail],cwd=workspace,env=env))

if __name__=='__main__':main()
