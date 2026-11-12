"""Read-only scan of source, archives and table cells for personal filesystem paths."""
from pathlib import Path
import argparse
import io
import json
import re
import tarfile
import zipfile

PATTERNS = [rb'/Users/[A-Za-z0-9_.-]+/', rb'/home/[A-Za-z0-9_.-]+/',
            rb'/crest/students/[A-Za-z0-9_.-]+/', rb'crest-cluster-node-[0-9]+']


def scan(data, name, findings, tally, depth=0):
    if depth > 6:
        raise RuntimeError('Archive nesting exceeds scan limit: '+name)
    tally[0] += 1
    if any(re.search(p, data) for p in PATTERNS):
        findings.append(name)
    if name.endswith('.tar.gz'):
        with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as tf:
            for m in tf.getmembers():
                header = (m.name+' '+m.uname+' '+m.gname).encode()
                if any(re.search(p, header) for p in PATTERNS): findings.append(name+'::header')
                if m.isfile(): scan(tf.extractfile(m).read(), name+'::'+m.name, findings, tally, depth+1)
    elif zipfile.is_zipfile(io.BytesIO(data)):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for m in z.infolist():
                if not m.is_dir(): scan(z.read(m), name+'::'+m.filename, findings, tally, depth+1)
    elif name.endswith('.parquet'):
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError('Install pyarrow to scan Parquet contents; audit is incomplete') from exc
        table = pq.read_table(io.BytesIO(data))
        for column in table.column_names:
            text = '\n'.join(str(v) for v in table[column].to_pylist()).encode()
            if any(re.search(p, text) for p in PATTERNS): findings.append(name+'::'+column)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('path', nargs='?', type=Path, default=Path(__file__).resolve().parents[1])
    a = ap.parse_args(); findings = []; tally = [0]
    paths = [a.path] if a.path.is_file() else sorted(a.path.rglob('*'))
    for p in paths:
        if not p.is_file() or p.is_symlink() or any(x in {'.git','__pycache__','.venv','outputs'}
                or x.endswith('.egg-info') for x in p.parts): continue
        scan(p.read_bytes(), str(p.relative_to(a.path)) if a.path.is_dir() else p.name, findings, tally)
    print(json.dumps(dict(scanned_files_and_members=tally[0], findings=sorted(set(findings))), indent=2))
    if findings: raise SystemExit(1)


if __name__ == '__main__':
    main()
