#!/usr/bin/env python3
"""Run release checks and save a concise, honest local validation report."""
from pathlib import Path
import json
import platform
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pipeline import TASKS

def main():
    logs=[f'Python: {sys.version}',f'Platform: {platform.platform()}']
    commands=[['-m','unittest','discover','-s','tests','-v'],['notebooks/run_quickstart.py'],
        ['models/train.py','--help'],['verification/strict_ci35/score.py','--help'],
        ['pipeline.py','binary20-build','--','--selftest'],
        ['pipeline.py','binary20-qc','--','--selftest']]
    commands += [['pipeline.py',task,'--','--help'] for task in TASKS]
    for command in commands:
        result=subprocess.run([sys.executable,*command],cwd=ROOT,text=True,capture_output=True,timeout=120)
        logs += ['COMMAND: '+' '.join(command),result.stdout,result.stderr]
        print('PASS' if result.returncode==0 else 'FAIL',' '.join(command),flush=True)
        if result.returncode:
            (ROOT/'docs/local_validation.txt').write_text('\n'.join(logs))
            raise SystemExit(result.returncode)
    with tempfile.TemporaryDirectory(prefix='CI external workspace ') as work:
        result=subprocess.run([sys.executable,str(ROOT/'pipeline.py'),'--workspace',work,'configure'],
            cwd=work,text=True,capture_output=True,check=True)
        for file in (Path(work)/'configs').glob('*.json'):
            cfg=json.loads(file.read_text())
            assert Path(cfg['project_root']).resolve()==Path(work).resolve()
            assert '/crest/' not in file.read_text()
        logs += ['EXTERNAL WORKSPACE CONFIGURATION PASSED',result.stdout]
    logs += ['RESULT: all requested local checks passed',
             'NOT EXECUTED: full native data retrieval, catalogue rebuild, CUDA training, full TEST score reproduction.']
    (ROOT/'docs/local_validation.txt').write_text('\n'.join(logs)+'\n')
    freeze=subprocess.run([sys.executable,'-m','pip','freeze'],text=True,capture_output=True,check=True)
    (ROOT/'docs/tested_environment.txt').write_text(f'# {platform.platform()} / Python {platform.python_version()}\n'+freeze.stdout)

if __name__=='__main__':main()
