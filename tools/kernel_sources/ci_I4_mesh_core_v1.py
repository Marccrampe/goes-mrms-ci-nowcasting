"""CPU-only distance/native-area/probability mesh on unchanged cached forecasts."""
import numpy as np
from scipy.ndimage import distance_transform_edt
import ci_common_exploratory_core as C
import ci_I4_growth_sweep_core_v1 as S

RADII=(1.,3.,5.,10.,15.,20.,30.)
AREAS=(1,2,3,5)

def coverage_checks(labels,n,coverage,radii,dy,dx):
 if coverage is None:return {r:np.zeros(n,bool) for r in radii}
 if coverage.all():return {r:np.ones(n,bool) for r in radii}
 bad_distance=distance_transform_edt(coverage,sampling=(dy,dx));out={r:np.ones(n,bool) for r in radii}
 for k in range(1,n+1):
  foot=labels==k;safe=float(bad_distance[foot].min());needs=[r for r in radii if safe<=r+1e-8]
  if not needs:continue
  field=distance_transform_edt(~foot,sampling=(dy,dx))
  for r in needs:out[r][k-1]=bool(coverage[field<=r+1e-8].mean()>=.95)
 return out

def evaluate(p,core,coverage,events_by_level,dy,dx,thresholds=S.THRESHOLDS,radii=RADII,areas=AREAS):
 p=np.asarray(p);core=np.asarray(core,bool)
 if p.ndim!=2 or p.shape!=core.shape or not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise RuntimeError('Invalid unmasked forecast/core schema')
 obs,nobs=C.components(core);truth_fields=C.truth_fields(obs,dy,dx)
 event_data={}
 for level,events in events_by_level.items():
  fields=[]
  for e in events:
   mask=np.zeros(p.shape,bool);mask.ravel()[e['flat']]=True
   fields.append(distance_transform_edt(~mask,sampling=(dy,dx)))
  event_data[level]=(events,np.asarray(fields))
 output=[]
 for threshold in thresholds:
  labels,n=C.components(p>=threshold,2);dist=C.matrix(truth_fields,labels,n)
  support_counts={r:np.any(dist<=r+1e-8,axis=0) if nobs else np.zeros(n,bool) for r in radii}
  strict={};cov=coverage_checks(labels,n,coverage,radii,dy,dx)
  for radius in radii:
   pairs=C.matching(dist,radius);tp_ids={int(pi)+1 for oi,pi in pairs};tp=len(pairs)
   dup={pi+1 for pi in range(n) if support_counts[radius][pi] and pi+1 not in tp_ids}
   covered=int(np.any(dist<=radius+1e-8,axis=1).sum()) if n else 0
   strict[radius]=dict(tp_ids=tp_ids,dup_ids=dup,tp=tp,fn=nobs-tp,fp=n-tp,covered=covered)
  for level,(events,fields) in event_data.items():
   event_dist=C.matrix(fields,labels,n)
   for area in areas:
    eligible=np.asarray([i for i,e in enumerate(events) if e['birth_native_pixels']>=area],dtype=int)
    ed=event_dist[eligible] if len(eligible) else np.empty((0,n))
    for radius in radii:
     base=strict[radius];assigned={}
     if coverage is not None and n and len(eligible):
      for ei,pi in C.matching(ed,radius):
       oid=int(pi)+1
       if oid in base['tp_ids'] or oid in base['dup_ids'] or not cov[radius][pi]:continue
       assigned[oid]=events[eligible[ei]]
     g=len(assigned);dup=len(base['dup_ids'])
     if g+dup>base['fp']:raise RuntimeError('FP categories overlap')
     output.append(dict(radar_level_dbz=int(level),min_native_birth_pixels=int(area),tolerance_km=float(radius),threshold=float(threshold),
      tp=base['tp'],fn=base['fn'],fp=base['fp'],observed_objects=nobs,predicted_objects=n,
      valid_cases=1,negative_cases=int(nobs==0),negative_alarm_cases=int(nobs==0 and n>0),
      observed_covered=base['covered'],predicted_supported=int(support_counts[radius].sum()),competition_FN_covered=base['covered']-base['tp'],
      radar_growth_promotions=g,duplicate_promotions=dup,
      promoted_growth_below35=sum(e['maximum_native_dbz']<35 for e in assigned.values()),
      promoted_growth_birth_one_pixel=sum(e['birth_native_pixels']==1 for e in assigned.values()),
      FP_with_incomplete_radar_window=base['fp'] if coverage is None else 0,
      predicted_pixels=int((labels>0).sum()),domain_pixels=int(p.size),reference_pixels=int(core.sum())))
 return output

def selftest():
 S.selftest()
 p=np.zeros((25,30),np.float32);core=np.zeros_like(p,bool);core[5,4]=True
 p[5,4:6]=.8;p[5,8:10]=.4;p[17,22:24]=.2
 coverage=np.ones_like(core);event=dict(flat=np.array([17*30+22]),birth_native_pixels=1,maximum_native_dbz=29)
 rows=evaluate(p,core,coverage,{25:[event]},1,1,thresholds=[.1,.3,.5],radii=[1.,5.,20.],areas=[1,2,3,5])
 assert len(rows)==36
 # Exact reproduction of the previous final-rule numeric engine for every threshold.
 old=S.evaluate(p,S.preparation(core,coverage,[event],1,1),[.1,.3,.5])
 for a in old:
  b=next(r for r in rows if r['threshold']==a['threshold'] and r['tolerance_km']==5 and r['min_native_birth_pixels']==1)
  for key,value in a.items():assert b[key]==value,(key,b[key],value)
 assert next(r for r in rows if r['threshold']==.1 and r['tolerance_km']==5 and r['min_native_birth_pixels']==2)['radar_growth_promotions']==0
 # Fast shared coverage calculations must reproduce original coverage_ok, including masks/edges.
 coverage[17,22]=False;coverage[0,:]=False
 labels,n=C.components(p>=.1,2);fast=coverage_checks(labels,n,coverage,[1.,3.,5.,20.],1.,1.)
 ctx=dict(coverage=coverage,to_bad=distance_transform_edt(coverage),dy=1.,dx=1.)
 for radius,flags in fast.items():
  for oid in range(1,n+1):assert flags[oid-1]==C.coverage_ok(ctx,labels,oid,radius)
 # Missing radar cannot promote a growth FP; area changes cannot alter the strict reference/counts.
 rows=evaluate(p,core,None,{25:[event]},1,1,[.1],[5.],[1,2])
 assert all(r['radar_growth_promotions']==0 for r in rows)
 assert len({(r['tp'],r['fn'],r['fp']) for r in rows})==1
 print('MESH SELFTEST PASSED: exact previous 5km rule, distance/native-area separation, shared physical coverage, missing evidence, unchanged strict truth',flush=True)
