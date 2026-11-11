#!/usr/bin/env python3
"""Package source, weights and example; exclude local outputs and environments."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]
EXCLUDE={'__pycache__','.git','.venv','outputs','data','runs','temporary','.ipynb_checkpoints'}

def files():
    return sorted(p for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink()
        and not set(p.relative_to(ROOT).parts)&EXCLUDE
        and p.suffix not in {'.pyc','.pyo'} and p.name!='.DS_Store')

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    output=a.output.resolve()
    if output==ROOT or ROOT in output.parents:raise RuntimeError('Archive must be outside the repository')
    if output.exists():raise RuntimeError('Existing archive preserved')
    hashes={}
    for f in files():
        if f.name=='release_checksums.json':continue
        with f.open('rb') as handle:hashes[str(f.relative_to(ROOT))]=hashlib.file_digest(handle,'sha256').hexdigest()
    (ROOT/'release_checksums.json').write_text(json.dumps(hashes,indent=2)+'\n')
    with tarfile.open(output,'w:gz',compresslevel=1) as tf:
        for f in files():tf.add(f,arcname=str(Path(ROOT.name)/f.relative_to(ROOT)),recursive=False)
    print('ARCHIVE',output)
    print('FILES',len(files()))
    print('MiB',round(output.stat().st_size/1024**2,1))

if __name__=='__main__':main()
