"""Small provenance/cache helpers; no original source or data is modified."""
from pathlib import Path
import hashlib,importlib.util,json,sys
import numpy as np

MINUTE=60_000_000_000
EVENTS=[('20240523','2024-05-23 08:00','2024-05-23 18:00'),
        ('20250619','2025-06-19 16:00','2025-06-19 23:00'),
        ('20250725','2025-07-25 16:00','2025-07-25 23:00')]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    def clean(v):
        if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [clean(x) for x in v]
        if isinstance(v,np.ndarray):return clean(v.tolist())
        if isinstance(v,np.generic):return clean(v.item())
        if isinstance(v,float) and not np.isfinite(v):return None
        if isinstance(v,Path):return str(v)
        return v
    tmp.write_text(json.dumps(clean(value),indent=2,allow_nan=False));tmp.replace(p)
def freeze(p,value):
    value=json.loads(json.dumps(value,default=str));p=Path(p)
    if p.exists() and read(p)!=value:raise RuntimeError('Existing contract differs; preserved: '+str(p))
    if not p.exists():write(p,value)
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def require(ok,message):
    if not ok:raise RuntimeError(message)
def truth(v):return str(v).lower() in ['true','1','1.0']
def array_sha(*arrays):
    h=hashlib.sha256()
    for a in arrays:
        a=np.ascontiguousarray(a);h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()
def save_npz(p,contract,**arrays):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
    tmp.replace(p);write(p.with_suffix('.json'),dict(contract=contract,sha256=sha(p)))
def cached(p,contract):
    p=Path(p);marker=p.with_suffix('.json')
    if not p.exists() or not marker.exists():return False
    d=read(marker);require(d['contract']==contract,'Cached inference inputs/settings changed: '+str(p));require(d['sha256']==sha(p),'Cache changed: '+str(p));return True

def classify(p,reference,events,coverage,threshold,strict_km,growth_km,dy,dx,C,N):
    """Strict counts fixed; growth only reinterprets nonduplicate strict FP."""
    p=np.asarray(p);reference=np.asarray(reference,bool)
    require(p.shape==reference.shape and np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'Invalid probability/reference map')
    observed,no=C.components(reference,1);predicted,npred=C.components(p>=threshold,2)
    distance=C.matrix(C.truth_fields(observed,dy,dx),predicted,npred);pairs=C.matching(distance,strict_km)
    hit_pred={int(j)+1 for i,j in pairs};hit_obs={int(i)+1 for i,j in pairs}
    covered=set(np.flatnonzero(np.any(distance<=strict_km+1e-8,axis=1))+1) if npred else set()
    supported=set(np.flatnonzero(np.any(distance<=strict_km+1e-8,axis=0))+1) if no else set()
    duplicates=supported-hit_pred;eligible=[i for i in range(npred) if i+1 not in hit_pred|duplicates]
    fields=[]
    from scipy.ndimage import distance_transform_edt
    for event in events:
        mask=np.zeros(p.shape,bool);mask.ravel()[np.asarray(event['flat'],int)]=True;fields.append(distance_transform_edt(~mask,sampling=(dy,dx)))
    ed=C.matrix(np.asarray(fields),predicted,npred);safe=N.coverage_checks(predicted,npred,coverage,[growth_km],dy,dx)[growth_km]
    eligible=[i for i in eligible if safe[i]];accepted={}
    if events and eligible:
        for i,j in C.matching(ed[:,eligible],growth_km):accepted[eligible[int(j)]+1]=events[int(i)]
    strict=np.zeros(p.shape,np.uint8);secondary=np.zeros_like(strict)
    for k in range(1,npred+1):
        strict[predicted==k]=1 if k in hit_pred else 3
        secondary[predicted==k]=1 if k in hit_pred else 2 if k in accepted else 3
    counts=dict(tp=len(pairs),fp=npred-len(pairs),fn=no-len(pairs),observed_objects=no,predicted_objects=npred,
                accepted_growth=len(accepted),remaining_FP=npred-len(pairs)-len(accepted),duplicate_FP_not_promoted=len(duplicates),FN_covered=len(covered-hit_obs),FN_uncovered=no-len(covered))
    require(counts['tp']+counts['fp']==npred and counts['tp']+counts['fn']==no,'Strict contingency error')
    require(counts['accepted_growth']+counts['remaining_FP']==counts['fp'],'FP partition error')
    require(not (hit_pred&set(accepted)) and not (duplicates&set(accepted)),'Double-counted accepted forecast')
    return dict(probability=p,observed=observed,predicted=predicted,pairs=[(int(i),int(j)) for i,j in pairs],hit_pred=hit_pred,hit_obs=hit_obs,covered_obs=covered,
                accepted=accepted,duplicates=duplicates,strict=strict,secondary=secondary,counts=counts,threshold=threshold,distance=distance)
