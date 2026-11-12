"""Tight-distance sensitivity to small, newly growing radar echoes.

Candidate lineages are independent of forecast locations. Native threshold=1
pixel for tracking; minimum NATIVE area at birth is varied only at scoring.
These are radar-growth proxies, not certified physical CI or new training labels.
"""
from collections import deque,defaultdict
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from ci_v4_relaxed_core import EpisodeTracker,components,matching
from ci_common_exploratory_core import MINUTE,matrix,coverage_ok

LEVELS=(20,25,30,35)
AREAS=(1,2,3)
RADII=(1.,3.,5.)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jw(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def cw(path,frame):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');frame.to_csv(tmp,index=False);tmp.replace(path)


def build_reference(job):
    import zarr
    from build_ci_soft_targets_native_goes import build_mapping,load_and_validate_goes_grid
    root=Path(job['directory']);root.mkdir(parents=True,exist_ok=True)
    g=zarr.open_group(job['radar_path'],mode='r');times=np.asarray(g['time_ns'][:],np.int64)
    if not len(times) or np.any(np.diff(times)<=0):raise RuntimeError('Invalid native radar time axis')
    contract=dict(job=job,source_time_sha256=hashlib.sha256(times.tobytes()).hexdigest(),
        native_shape=list(g['reflectivity_dbz'].shape),script_sha256=sha(__file__))
    done=root/'completed.json'
    if done.exists():
        meta=json.loads(done.read_text())
        if meta['contract']!=contract:raise RuntimeError('Tiny-echo source changed')
        for name,digest in meta['outputs'].items():
            if sha(root/name)!=digest:raise RuntimeError('Tiny-echo cache changed: '+name)
        print('REUSE TINY RADAR',job['period_id'],flush=True);return
    grid=load_and_validate_goes_grid();shape=tuple(job['shape']);size=int(np.prod(shape));w=shape[1]
    m=build_mapping(Path(job['radar_path']).parent,grid,{})
    sy,sx=m['source_y'],m['source_x'];flat=m['target_y'].astype(np.int64)*w+m['target_x']
    native_key=sy.astype(np.int64)*g['reflectivity_dbz'].shape[2]+sx
    if len(np.unique(native_key))!=len(native_key):raise RuntimeError('Duplicate native cells in mapping')
    total=np.asarray(m['total_count']).ravel()
    chosen=np.flatnonzero((times>=job['start_ns']-35*MINUTE)&(times<=job['end_ns']))
    if not len(chosen):raise RuntimeError('Empty native radar scope')
    trackers={z:EpisodeTracker(shape,job['dy'],job['dx'],scans=3,speed_km_min=2.) for z in LEVELS}
    prior={z:np.zeros(shape,np.int16) for z in LEVELS};bad=np.zeros(shape,np.int16);history=deque()
    profiles={z:defaultdict(list) for z in LEVELS};topology={z:defaultdict(list) for z in LEVELS}
    validity=[]
    for frame,scan in enumerate(chosen):
        stamp=int(times[scan]);field=np.asarray(g['reflectivity_dbz'][int(scan)],np.float32)
        values=field[sy,sx];good=np.isfinite(values)&(values>=-50)
        counts=np.bincount(flat[good],minlength=size)
        valid=((total>0)&(counts>=.75*total)).reshape(shape);validity.append(np.packbits(valid.ravel()))
        peak=np.full(size,-50,np.float32);np.maximum.at(peak,flat[good],values[good]);peak=peak.reshape(shape)
        while history and history[0][0]<stamp-30*MINUTE:
            _,v,echoes=history.popleft();bad-=~v
            for z in LEVELS:prior[z]-=echoes[z]
        history_ok=bool(history and stamp-history[0][0]>=28*MINUTE and np.diff([h[0] for h in history]+[stamp]).max()<=180e9)
        hv=(bad==0)&valid if history_ok else np.zeros(shape,bool)
        echoes={}
        for z,tracker in trackers.items():
            # Native cutoff=1: no tiny echo removed before projection/tracking.
            selected=good&(values>=z)
            projected=np.zeros(size,bool);projected[flat[selected]]=True;projected=projected.reshape(shape)&valid
            old_labels=tracker.previous.copy();old_roots=dict(tracker.previous_roots)
            tracker.update(projected,valid,stamp,frame,hv,prior[z]>0,peak)
            echoes[z]=projected;labels=tracker.previous;roots=tracker.previous_roots
            parents=defaultdict(set);children=defaultdict(set)
            if tracker.previous_time is not None:
                both=(old_labels>0)&(labels>0)
                for before,after in np.unique(np.c_[old_labels[both],labels[both]],axis=0):
                    if int(before) not in old_roots:continue
                    parent=old_roots[int(before)];parents[int(after)].add(parent);children[parent].add(int(after))
                flagged=set()
                for child,ps in parents.items():
                    if len(ps)>1:flagged|=ps|{roots[child]}
                for parent,parts in children.items():
                    if len(parts)>1:flagged.add(parent)
                for r in flagged:topology[z][r].append(stamp)
            lut=np.zeros(int(labels.max())+1,np.int32)
            for label,r in roots.items():lut[label]=r
            native_roots=lut[labels].ravel()[flat]
            eligible=selected&(native_roots>0)
            ordered=np.flatnonzero(eligible);ordered=ordered[np.argsort(native_roots[ordered],kind='stable')]
            if len(ordered):
                ids,starts=np.unique(native_roots[ordered],return_index=True);ends=np.r_[starts[1:],len(ordered)]
                for r,a,b in zip(ids,starts,ends):
                    event=tracker.events[int(r)];birth=event['birth_ns']
                    if not job['start_ns']<birth<=job['end_ns'] or stamp>birth+30*MINUTE:continue
                    if event['censored_history'] or event['prior_echo_at_birth_location']:continue
                    v=values[ordered[a:b]]
                    profiles[z][int(r)].append(dict(time_ns=stamp,native_pixels=int(len(v)),
                        native_p90_dbz=float(np.quantile(v,.9)),native_max_dbz=float(v.max())))
        history.append((stamp,valid.copy(),{z:echoes[z].copy() for z in LEVELS}));bad+=~valid
        for z in LEVELS:prior[z]+=echoes[z]
        if frame%100==0 or frame==len(chosen)-1:print('TINY RADAR',job['period_id'],f'{frame+1}/{len(chosen)}',flush=True)
    events=[]
    for z,tracker in trackers.items():
        for e in tracker.export():
            if not e['eligible_new_echo_proxy'] or e['track_id'] not in profiles[z]:continue
            yy,xx=np.divmod(np.asarray(e['birth_flat'],int),shape[1])
            e['birth_touches_domain_boundary']=bool((yy==0).any() or (yy==shape[0]-1).any() or (xx==0).any() or (xx==shape[1]-1).any())
            e.update(level_dbz=z,profile=profiles[z][e['track_id']],topology_times=topology[z][e['track_id']])
            events.append(e)
    tmp=root/'coverage.npz.tmp'
    with tmp.open('wb') as stream:np.savez_compressed(stream,time_ns=times[chosen],shape=shape,valid_bits=np.asarray(validity))
    tmp.replace(root/'coverage.npz');jw(root/'events.json',events)
    jw(done,dict(contract=contract,events=len(events),outputs={n:sha(root/n) for n in ['coverage.npz','events.json']}))
    print('TINY RADAR DONE',job['period_id'],len(events),'new persistent candidates; no prediction-based selection',flush=True)


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


class GrowthReference:
    def __init__(self,root):
        self.root=Path(root);done=json.loads((self.root/'completed.json').read_text())
        for name,digest in done['outputs'].items():
            if sha(self.root/name)!=digest:raise RuntimeError('Tiny reference changed: '+name)
        with np.load(self.root/'coverage.npz',allow_pickle=False) as z:
            self.times=z['time_ns'];self.shape=tuple(z['shape']);self.valid=z['valid_bits']
        self.events=json.loads((self.root/'events.json').read_text());self.size=int(np.prod(self.shape))
    def window(self,start,end):
        anchor=int(np.searchsorted(self.times,start,side='right'))-1
        after=np.flatnonzero((self.times>start)&(self.times<=end))
        if anchor<0 or not len(after):return None
        if start-self.times[anchor]>150e9 or end-self.times[after[-1]]>150e9 or np.diff(np.r_[self.times[anchor],self.times[after],end]).max()>180e9:return None
        return np.unpackbits(np.bitwise_and.reduce(self.valid[np.r_[anchor,after]],axis=0),count=self.size).reshape(self.shape).astype(bool)
    def candidates(self,level,area,start,end,valid):
        result=[]
        for e in self.events:
            if e['level_dbz']!=level:continue
            if e.get('birth_touches_domain_boundary'):continue
            flag=growth_flags(e,start,end,area)
            if flag is None:continue
            pts=np.asarray(e['birth_flat'],int);pts=pts[valid.ravel()[pts]]
            if len(pts):result.append(dict(id=f'Z{level}_{e["track_id"]}',flat=pts,birth_ns=e['birth_ns'],**flag))
        return result


def assign_growth(events,labels,npred,strict_tp,coverage,dy,dx,radius):
    if not events or not npred or coverage is None:return {}
    fields=[]
    for event in events:
        mask=np.zeros(labels.shape,bool);mask.ravel()[event['flat']]=True
        fields.append(distance_transform_edt(~mask,sampling=(dy,dx)))
    d=matrix(np.asarray(fields),labels,npred)
    to_bad=None if coverage.all() else distance_transform_edt(coverage,sampling=(dy,dx))
    ctx=dict(coverage=coverage,to_bad=to_bad,dy=dy,dx=dx)
    result={}
    # Match ALL predictions first. A candidate consumed by a strict TP cannot
    # also promote another FP. Each candidate promotes at most one FP.
    for ei,pi in matching(d,radius):
        oid=pi+1
        if oid in strict_tp or not coverage_ok(ctx,labels,oid,radius):continue
        result[oid]=dict(events[ei],edge_distance_km=float(d[ei,pi]))
    return result


def selftest():
    birth=1700000000000000000;start=birth-MINUTE;end=birth+30*MINUTE
    def event(values=(22,24,28),areas=(1,1,2)):
        return dict(birth_ns=birth,eligible_new_echo_proxy=True,topology_times=[],profile=[dict(time_ns=birth+i*2*MINUTE,
            native_pixels=a,native_p90_dbz=z,native_max_dbz=z) for i,(z,a) in enumerate(zip(values,areas))])
    e=event();assert growth_flags(e,start,end,1) and not growth_flags(e,start,end,2)
    assert not growth_flags(event((22,22,22),(1,1,1)),start,end,1)
    assert growth_flags(event((22,22,22),(1,2,2)),start,end,1)
    assert not growth_flags(e,start,birth+MINUTE,1)
    e['topology_times']=[birth+2*MINUTE];assert not growth_flags(e,start,end,1)
    lab=np.zeros((16,20),int);lab[4:6,4:6]=1;lab[4:6,9:11]=2
    ev=[dict(id='e',flat=np.array([4*20+4]))]
    cov=np.ones(lab.shape,bool)
    assert set(assign_growth(ev,lab,2,set(),cov,2.,2.,1.))=={1}
    assert not assign_growth(ev,lab,2,{1},cov,2.,2.,5.)
    assert not assign_growth(ev,lab,2,set(),None,2.,2.,5.)
    print('TINY GROWTH SELFTEST PASSED: native birth area, measured growth, three scans, same-bin timing, topology rejection, one-to-one, coverage,1-5km',flush=True)
