import numpy as np
def choose_scans(times,initial,offsets=(8,18,28),maximum_error_seconds=150):
    times=np.asarray(times,np.int64)
    if times.ndim!=1 or len(times)==0 or not np.all(np.diff(times)>0):raise RuntimeError('Radar times missing, duplicated or unsorted')
    allowed=np.flatnonzero((times>initial)&(times<=initial+30*60_000_000_000))
    if len(allowed)<len(offsets):raise RuntimeError('Insufficient radar scans inside forecast window')
    ids=[]
    for offset in offsets:
        target=initial+int(offset)*60_000_000_000
        i=int(allowed[np.argmin(np.abs(times[allowed]-target))])
        if abs(int(times[i])-target)>maximum_error_seconds*1e9:raise RuntimeError('No radar scan near requested T0+'+str(offset)+'min')
        ids.append(i)
    if len(set(ids))!=len(ids):raise RuntimeError('Requested contextual frames map to the same scan')
    return ids
