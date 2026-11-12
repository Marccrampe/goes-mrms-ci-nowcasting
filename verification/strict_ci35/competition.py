"""Recover only unmatched observations adjacent to ORIGINAL matched forecasts."""
import numpy as np

COUNT = ['TP_old','TP_pred_old','FP_old','FN_old','FN_recovered','H_new',
         'FN_new','FP_new','n_observed_objects','n_forecast_objects']
METRICS = ['CSI_association','POD_event','FAR_forecast']
DEFINITION = {
    'CSI_association':'H_new/(H_new+FN_new+FP_old); hybrid event/object association score',
    'CSI_one_to_one_old':'TP_old/(TP_old+FN_old+FP_old)',
    'POD_event':'H_new/(H_new+FN_new)',
    'FAR_forecast':'FP_old/(TP_pred_old+FP_old); exactly unchanged',
    'recovery':'Only original one-to-one FN adjacent to an originally matched forecast',
    'ties':'minimum original edge distance, then lowest zero-based forecast ID',
    'distance_tolerance_km':5.0,'original_numeric_epsilon_km':1e-8,
    'min_forecast_pixels':2,'connectivity':8,
    'threshold_selection':'NONE: original archived operational thresholds only',
    'promotions':'NO radar-growth or duplicate-forecast promotions',
    'undefined_ratios':'NaN when denominator is zero',
}

def ensure(ok, message):
    if not ok:
        raise RuntimeError(message)

def rates(c):
    c={k:int(c[k]) for k in COUNT}
    tp,k,fp,fn,rec,h,miss,newfp,no,nf=[c[n] for n in COUNT]
    ensure(tp==k and fp==newfp and h==tp+rec and miss==fn-rec,
           'Changed forecast FP, original pairing, or recovery arithmetic.')
    ensure(min(c.values())>=0 and h+miss==no and tp+fn==no and k+fp==nf,
           'Observed/forecast population conservation failed.')
    div=lambda a,b:float(a/b) if b else float('nan')
    c.update(CSI_one_to_one_old=div(tp,tp+fp+fn),POD_old=div(tp,no),FAR_old=div(fp,nf),
             CSI_association=div(h,h+miss+fp),POD_event=div(h,no),FAR_forecast=div(fp,nf),
             FN_recovered_percent=100*div(rec,fn))
    return c

def recover(d, original_pairs, tolerance=5.0):
    d=np.asarray(d,dtype=np.float64)
    ensure(d.ndim==2 and not np.isnan(d).any() and (d>=0).all(),'Invalid distance matrix.')
    no,nf=d.shape
    pairs=[(int(o),int(f)) for o,f in original_pairs]
    old_obs={o for o,f in pairs};old_forecasts={f for o,f in pairs}
    ensure(len(old_obs)==len(pairs)==len(old_forecasts),'Original matches are not one-to-one.')
    ensure(all(0<=o<no and 0<=f<nf and d[o,f]<=tolerance+1e-8 for o,f in pairs),
           'Invalid original pair.')
    assignments=np.full(no,-1,dtype=np.int64)
    original_assignments=assignments.copy()
    for o,f in pairs:
        assignments[o]=original_assignments[o]=f
    original_fn=sorted(set(range(no))-old_obs)
    original_fp=sorted(set(range(nf))-old_forecasts)
    # A maximum-cardinality match cannot leave an unmatched observation next to
    # an unmatched forecast: that edge would increase its cardinality by one.
    ensure(not original_fn or not original_fp or
           not np.any(d[np.ix_(original_fn,original_fp)]<=tolerance+1e-8),
           'Original match is not maximum cardinality: eligible unmatched O/F edge.')
    fixed_eligible=np.asarray(sorted(old_forecasts),dtype=np.int64)
    recovered=[]
    for o in original_fn:
        eligible=fixed_eligible[d[o,fixed_eligible]<=tolerance+1e-8]
        if eligible.size:
            f=int(eligible[np.argmin(d[o,eligible])])
            assignments[o]=f
            recovered.append((o,f,float(d[o,f])))
    recovered_ids={o for o,f,dist in recovered}
    ensure(len(recovered_ids)==len(recovered) and not recovered_ids&old_obs,
           'Observation counted twice.')
    ensure(all(f in old_forecasts for o,f,dist in recovered),
           'Recovery attempted to promote a forecast FP.')
    ensure(all(assignments[o]==f for o,f in pairs),'An original pair changed.')
    ensure(set(assignments[assignments>=0])==old_forecasts,
           'Matched forecast-object population changed.')
    # Equivalence in observed hits to eligibility coverage is checked, never used
    # instead of the actual two-stage algorithm or its explicit association audit.
    covered=(d<=tolerance+1e-8).any(axis=1) if nf else np.zeros(no,bool)
    ensure(np.array_equal(assignments>=0,covered),'Maximum-cardinality coverage invariant failed.')
    tp=len(pairs);rec=len(recovered)
    counts=rates(dict(TP_old=tp,TP_pred_old=tp,FP_old=nf-tp,FN_old=no-tp,
        FN_recovered=rec,H_new=tp+rec,FN_new=no-tp-rec,FP_new=nf-tp,
        n_observed_objects=no,n_forecast_objects=nf))
    return dict(counts=counts,pairs=pairs,recovered=recovered,assignments=assignments,
        original_assignments=original_assignments,matched_forecast_ids=np.asarray(sorted(old_forecasts),int),
        unmatched_forecast_ids=np.asarray(original_fp,int),
        missed_observed_ids=np.flatnonzero(assignments<0))

def score(C,field,core,q,dy,dx):
    field=np.asarray(field);core=np.asarray(core,bool)
    ensure(field.ndim==2 and field.shape==core.shape and np.isfinite(field).all(),
           'Invalid native forecast/reference grid.')
    labels,nf=C.components(field>=float(q),2)
    truth,no=C.components(core)
    d=C.matrix(C.truth_fields(truth,dy,dx),labels,nf)
    ensure(d.shape==(no,nf),'Distance matrix shape differs.')
    result=recover(d,C.matching(d,5.0))
    return result,labels,truth,d
