"""Original Fig11 numerical functions, extracted unchanged; see kernel_sources.json."""

import numpy as np

from scipy.ndimage import label, distance_transform_edt

from scipy.optimize import linear_sum_assignment

def components(binary, min_area=1):
    labels, n = label(binary, np.ones((3, 3), np.uint8))
    if n and min_area > 1:
        keep = np.bincount(labels.ravel()) >= min_area
        keep[0] = False
        labels, n = label(keep[labels], np.ones((3, 3), np.uint8))
    return labels, n

def prediction_segments(predicted_labels):
    points=np.flatnonzero(predicted_labels).astype(np.int32)
    values=predicted_labels.ravel()[points]
    order=np.argsort(values,kind='stable')
    ordered=points[order]
    starts=np.r_[0,np.flatnonzero(np.diff(values[order]))+1] if len(points) else np.array([],np.int32)
    return ordered,starts

def matching(distance, tolerance):
    if not distance.size:
        return []
    allowed = distance <= tolerance+1e-8
    penalty = (min(distance.shape)+1)*(tolerance+1)
    rows, cols = linear_sum_assignment(np.where(allowed, distance, penalty))
    return [(int(r), int(c)) for r, c in zip(rows, cols) if allowed[r, c]]

def truth_fields(labels, dy, dx):
    return np.asarray([distance_transform_edt(labels != k, sampling=(dy, dx))
                       for k in range(1, int(labels.max()) + 1)])

def matrix(fields, labels, count):
    if not len(fields) or not count:
        return np.empty((len(fields), count), np.float64)
    points, starts = prediction_segments(labels)
    return np.asarray([np.minimum.reduceat(field.ravel()[points], starts) for field in fields])

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

def require(ok,message):
    if not ok:raise RuntimeError(message)

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

from types import SimpleNamespace

C = SimpleNamespace(components=components, truth_fields=truth_fields, matrix=matrix, matching=matching)

N = SimpleNamespace(coverage_checks=coverage_checks)


def growth_flags(event,start,end,area,min_gain=5.,area_ratio=1.5,area_gain=1):
    if not start<int(event['birth_ns'])<=end:return None
    if not event.get('eligible_new_echo_proxy'):return None
    rows=[p for p in event['profile'] if event['birth_ns']<=p['time_ns']<=min(end,event['birth_ns']+30*MINUTE)]
    if len(rows)<3 or rows[0]['time_ns']!=event['birth_ns']:return None
    if rows[0]['native_pixels']<area:return None
    # Require at least three contiguous scans; no promotion from a lone spike.
    initial=[]
    for row in rows:
        if initial and row['time_ns']-initial[-1]['time_ns']>180e9:break
        initial.append(row)
    if len(initial)<3 or initial[2]['time_ns']-initial[0]['time_ns']<180e9:return None
    p0=initial[0];growth=None
    for row in initial[1:]:
        if row['native_p90_dbz']-p0['native_p90_dbz']>=min_gain or (
            row['native_pixels']>=p0['native_pixels']*area_ratio and row['native_pixels']-p0['native_pixels']>=area_gain):
            growth=row;break
    if growth is None:return None
    decision=max(initial[2]['time_ns'],growth['time_ns'])
    if any(event['birth_ns']<=t<=decision for t in event.get('topology_times',[])):return None
    return dict(birth_native_pixels=p0['native_pixels'],p90_gain_dbz=max(r['native_p90_dbz'] for r in initial)-p0['native_p90_dbz'],
        native_area_gain=max(r['native_pixels'] for r in initial)-p0['native_pixels'],
        maximum_native_dbz=max(r['native_max_dbz'] for r in initial),growth_time_ns=growth['time_ns'],
        qualifying_time_ns=decision,observed_scans=len(initial))


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

MINUTE = 60_000_000_000
