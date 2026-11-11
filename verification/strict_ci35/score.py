#!/usr/bin/env python3
"""Full-domain strict CI35 scoring with optional separate association diagnostics."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from ci_nowcasting.paths import activate_sources
activate_sources()
import ci_common_exploratory_core as C
from verification.strict_ci35 import competition as K


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,help='Full-domain NPZ; documented in docs/verification.md')
    p.add_argument('--thresholds',required=True,help='VALIDATION-selected threshold JSON')
    p.add_argument('--model',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--association',action='store_true',help='Write separately named association diagnostics')
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise RuntimeError('Choose a new output directory')
    contract=json.loads(Path(a.thresholds).read_text())
    if contract['selection_split']!='validation':raise RuntimeError('Threshold selection must use VALIDATION')
    qs=contract['thresholds'][a.model]
    with np.load(a.input,allow_pickle=False) as z:
        prob=z['probability'];core=z['core'];keys=z['issuance_ns'];period=z['period_id']
        dy=float(z['dy_km']);dx=float(z['dx_km'])
        split=str(z['split'].item())
    if split!='test':raise RuntimeError('This entry point reports held-out TEST only')
    if len(qs)!=2 or any(not 0<float(q)<=1 for q in qs):raise RuntimeError('Two valid frozen thresholds required')
    if prob.ndim!=4 or prob.shape[1]!=2 or core.shape!=prob.shape:raise RuntimeError('Expected N,2,H,W full-domain arrays')
    if len(keys)!=len(prob) or len(period)!=len(prob) or len(set(zip(period.tolist(),keys.tolist())))!=len(prob):
        raise RuntimeError('Invalid/duplicate issuance keys')
    if not np.isfinite(prob).all() or (prob<0).any() or (prob>1).any():raise RuntimeError('Invalid probabilities')
    if not np.isin(core,[0,1]).all() or min(dx,dy)<=0:raise RuntimeError('Binary uninflated observed cores/positive spacing required')
    out.mkdir(parents=True);rows=[];recovered=[]
    for i in range(len(keys)):
        for lead,q in enumerate(qs):
            r,_,_,_=K.score(C,prob[i,lead],core[i,lead],q,dy,dx)
            c=r['counts'];rows.append(dict(period_id=str(period[i]),issuance_ns=int(keys[i]),lead_index=lead,
                threshold=q,TP=c['TP_old'],FP=c['FP_old'],FN=c['FN_old'],FN_recovered=c['FN_recovered']))
            if a.association:
                for ob,fc,distance in r['recovered']:
                    recovered.append(dict(period_id=str(period[i]),issuance_ns=int(keys[i]),lead_index=lead,
                        observed_id=ob+1,forecast_id=fc+1,distance_km=distance))
    def write(name,values,columns):
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(values)
    write('strict_per_issuance.csv',[{k:r[k] for k in r if k!='FN_recovered'} for r in rows],
          ['period_id','issuance_ns','lead_index','threshold','TP','FP','FN'])
    strict=[];assoc=[]
    div=lambda n,d: n/d if d else None
    for lead in range(2):
        total={k:sum(r[k] for r in rows if r['lead_index']==lead) for k in ['TP','FP','FN','FN_recovered']}
        tp,fp,fn,rec=[total[k] for k in ['TP','FP','FN','FN_recovered']]
        strict.append(dict(model=a.model,lead_index=lead,threshold=qs[lead],N_issuances=len(keys),TP=tp,FP=fp,FN=fn,
            CSI=div(tp,tp+fp+fn),POD=div(tp,tp+fn),FAR=div(fp,tp+fp)))
        assoc.append(dict(model=a.model,lead_index=lead,threshold=qs[lead],TP_old=tp,FP_old=fp,FN_old=fn,
            FN_recovered=rec,H_new=tp+rec,FN_new=fn-rec,FP_new=fp,
            CSI_association=div(tp+rec,tp+fn+fp),POD_event=div(tp+rec,tp+fn),FAR_forecast=div(fp,tp+fp)))
    write('strict_scores.csv',strict,list(strict[0]))
    if a.association:
        write('association_scores.csv',assoc,list(assoc[0]))
        write('competition_recoveries.csv',recovered,['period_id','issuance_ns','lead_index','observed_id','forecast_id','distance_km'])
    sha=lambda path:hashlib.file_digest(Path(path).open('rb'),'sha256').hexdigest()
    (out/'contract.json').write_text(json.dumps(dict(input_sha256=sha(a.input),thresholds_sha256=sha(a.thresholds),
        model=a.model,split=split,full_domain_required=True,radius_km=5,min_forecast_pixels=2,
        association_definition=K.DEFINITION if a.association else None),indent=2)+'\n')
    print(json.dumps(strict,indent=2));print('OUTPUT',out.resolve())

if __name__=='__main__':main()
