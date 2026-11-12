from pathlib import Path
import ast
import numpy as np
from snapshots import choose_scans
root=Path(__file__).resolve().parent
for p in root.glob('*.py'):ast.parse(p.read_text(),filename=str(p),feature_version=(3,9))
minute=60_000_000_000
t=np.asarray([-12,-2,8,18,28,38],np.int64)*minute
assert choose_scans(t,0)==[2,3,4]
t=np.asarray([7,9,18,28],np.int64)*minute
assert choose_scans(t,0)==[0,2,3] # equal-distance tie selects earlier scan
for t in [np.array([-2,0,38])*minute,np.array([8,8,18,28])*minute,np.array([0,2,4])*minute]:
    try:choose_scans(t,0)
    except RuntimeError:pass
    else:raise AssertionError('Invalid/incomplete radar timing accepted')
draw=(root/'draw.py').read_text();tree=ast.parse(draw)
orange=[]
for n in ast.walk(tree):
    if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='scatter':
        kw={k.arg:k.value for k in n.keywords}
        if isinstance(kw.get('c'),ast.Name) and kw['c'].id=='ORANGE':
            orange.append(ast.literal_eval(kw['marker']))
assert orange==['o']
assert 'label="CI25 growth"' not in draw and 'label="CI35 birth"' not in draw
assert 'r["counts"]' in draw and 'counts["accepted_growth"]' in draw
# No matching call is introduced by the visual-only loader.
for n in ast.walk(ast.parse((root/'run.py').read_text())):
    if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute):assert n.func.attr not in ['matching','classify','predict','fit']
print('PASS: Python3.9 syntax, actual-window scan selection, timing tolerance/ties, missing scans rejection, orange dots, no matching/inference/refit calls.')
