"""Existing growing-echo rule plus measured, contiguous post-birth persistence."""
import numpy as np
from scipy.ndimage import distance_transform_edt

MINUTE=60_000_000_000
PERSISTENCES=(0,5,10,15,20)

def persistence_flags(event,start,end,persistence,growth_flags):
    # Birth, first 3 scans and growth retain the EXACT previous within-window rule.
    base=growth_flags(event,start,end,1)
    if base is None:return None
    if persistence==0:return dict(base,persistence_confirmation_ns=int(base['qualifying_time_ns']))
    rows=event['profile'];birth=int(event['birth_ns']);initial=[]
    for row in rows:
        if row['time_ns']<birth:continue
        if initial and row['time_ns']-initial[-1]['time_ns']>180e9:break
        initial.append(row)
    if not initial or initial[0]['time_ns']!=birth:return None
    confirmation=next((r for r in initial if r['time_ns']-birth>=persistence*MINUTE),None)
    if confirmation is None:return None
    decision=max(int(base['qualifying_time_ns']),int(confirmation['time_ns']))
    if any(birth<=t<=decision for t in event.get('topology_times',[])):return None
    return dict(base,persistence_confirmation_ns=decision,
                confirmed_duration_minutes=(confirmation['time_ns']-birth)/MINUTE)

def candidates(reference,level,start,end,valid,persistences,growth_flags):
    selected=[]
    for event in reference.events:
        if event['level_dbz']!=level or event.get('birth_touches_domain_boundary'):continue
        flags={p:persistence_flags(event,start,end,p,growth_flags) for p in persistences}
        if flags[0] is None:continue
        points=np.asarray(event['birth_flat'],int);points=points[valid.ravel()[points]]
        if not len(points):continue
        birth=int(event['birth_ns']);initial=[]
        for row in event['profile']:
            if row['time_ns']<birth:continue
            if initial and row['time_ns']-initial[-1]['time_ns']>180e9:break
            initial.append(row)
        peak=max(float(r['native_max_dbz']) for r in initial if r['time_ns']<=birth+20*MINUTE)
        selected.append(dict(id=f'Z{level}_{event["track_id"]}',flat=points,birth_ns=birth,
                             **flags[0],persistence_flags=flags,peak_fixed20_dbz=peak,
                             peak_ambiguous20=any(birth<=t<=birth+20*MINUTE for t in event.get('topology_times',[])),
                             continuous_observed_duration_capped20=min(20.,(initial[-1]['time_ns']-birth)/MINUTE)))
    return selected

def signature(events):
    keys=['birth_native_pixels','p90_gain_dbz','native_area_gain','maximum_native_dbz','growth_time_ns','qualifying_time_ns','observed_scans']
    return sorted((int(e['birth_ns']),tuple(sorted(map(int,e['flat']))),tuple(float(e[k]) for k in keys)) for e in events)

def evaluate(probability,core,coverage,events_by_level,dy,dx,thresholds,radii,persistences,C,N,peak_thresholds=(-1,)):
    p=np.asarray(probability);core=np.asarray(core,bool)
    if p.ndim!=2 or p.shape!=core.shape or not np.isfinite(p).all() or (p<0).any() or (p>1).any():raise RuntimeError('Invalid forecast/core')
    observed,nobs=C.components(core);truth=C.truth_fields(observed,dy,dx);event_data={}
    for level,events in events_by_level.items():
        fields=[]
        for event in events:
            mask=np.zeros(p.shape,bool);mask.ravel()[event['flat']]=True
            fields.append(distance_transform_edt(~mask,sampling=(dy,dx)))
        eligible={(duration,peak):np.array([i for i,e in enumerate(events) if e['persistence_flags'][duration] is not None
                        and (peak<0 or not e.get('peak_ambiguous20',False) and e['peak_fixed20_dbz']>=peak)],int) for duration in persistences for peak in peak_thresholds}
        event_data[level]=(events,np.asarray(fields),eligible)
    output=[]
    for threshold in thresholds:
        labels,n=C.components(p>=threshold,2);distance=C.matrix(truth,labels,n)
        supported={r:np.any(distance<=r+1e-8,axis=0) if nobs else np.zeros(n,bool) for r in radii}
        cov=N.coverage_checks(labels,n,coverage,radii,dy,dx);strict={}
        for radius in radii:
            pairs=C.matching(distance,radius);tp_ids={int(pi)+1 for oi,pi in pairs}
            dup={pi+1 for pi in range(n) if supported[radius][pi] and pi+1 not in tp_ids}
            covered=int(np.any(distance<=radius+1e-8,axis=1).sum()) if n else 0
            strict[radius]=dict(tp_ids=tp_ids,dup_ids=dup,tp=len(pairs),fn=nobs-len(pairs),fp=n-len(pairs),covered=covered)
        for level,(events,fields,eligible) in event_data.items():
            event_distance=C.matrix(fields,labels,n)
            for (duration,peak),ids in eligible.items():
                ed=event_distance[ids] if len(ids) else np.empty((0,n))
                for radius in radii:
                    base=strict[radius];assigned={}
                    if coverage is not None and n and len(ids):
                        # Same all-prediction assignment as the frozen engine. Strict TP/duplicates consume matches.
                        for ei,pi in C.matching(ed,radius):
                            oid=int(pi)+1
                            if oid in base['tp_ids'] or oid in base['dup_ids'] or not cov[radius][pi]:continue
                            assigned[oid]=events[ids[ei]]
                    growth=len(assigned);dup=len(base['dup_ids'])
                    if growth+dup>base['fp']:raise RuntimeError('Overlapping FP categories')
                    output.append(dict(radar_level_dbz=int(level),min_native_birth_pixels=1,persistence_minutes=int(duration),peak_min_dbz=int(peak),
                        tolerance_km=float(radius),threshold=float(threshold),tp=base['tp'],fn=base['fn'],fp=base['fp'],
                        observed_objects=nobs,predicted_objects=n,valid_cases=1,negative_cases=int(nobs==0),negative_alarm_cases=int(nobs==0 and n>0),
                        observed_covered=base['covered'],predicted_supported=int(supported[radius].sum()),competition_FN_covered=base['covered']-base['tp'],
                        radar_growth_promotions=growth,duplicate_promotions=dup,candidate_occurrences=len(ids),
                        promoted_growth_below35=sum(e['maximum_native_dbz']<35 for e in assigned.values()),
                        promoted_growth_birth_one_pixel=sum(e['birth_native_pixels']==1 for e in assigned.values()),
                        FP_with_incomplete_radar_window=base['fp'] if coverage is None else 0,
                        predicted_pixels=int((labels>0).sum()),domain_pixels=int(p.size),reference_pixels=int(core.sum())))
    return output
