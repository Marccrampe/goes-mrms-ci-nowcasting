#!/usr/bin/env python3
"""Execute the real miniature inference from any current directory."""
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parents[1]
subprocess.run([sys.executable,str(root/'models/inference.py'),
    '--checkpoint',str(root/'weights/goes_mrms_ci35_hard_r2.pt'),
    '--input',str(root/'examples/mini_case.npz'),
    '--normalization',str(root/'preprocessing/dataset/normalization.json'),
    '--output',str(root/'outputs/mini_case_output.npz'),
    '--expected-checkpoint-sha256','9231c9bb9aa540717986c1ae729cdbbf56629ed74b072de10a200be4f9ae507e'],check=True)
print('REAL QUICKSTART PASSED')
