#!/usr/bin/env python3
"""Execute the real miniature inference from any current directory."""
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parents[1]
subprocess.run([sys.executable,str(root/'models/inference.py'),
    '--checkpoint',str(root/'weights/goes_mrms_ci35_hard_r2_public.pt'),
    '--input',str(root/'examples/mini_case.npz'),
    '--normalization',str(root/'preprocessing/dataset/normalization.json'),
    '--output',str(root/'outputs/mini_case_output.npz'),
    '--expected-checkpoint-sha256','9bc69822fb18d9b5036bc4e90fa7487aa08aafdaeda364ebb176fc55b01d5b0b'],check=True)
print('REAL QUICKSTART PASSED')
