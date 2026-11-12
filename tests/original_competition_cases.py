"""Synthetic tests only; no real-case analysis is executed on import."""
import numpy as np
from competition import recover,score

def run_tests(C):
    tested=[]
    def check(name,d,pairs,expected):
        r=recover(np.asarray(d,float),pairs)
        for k,v in expected.items():
            assert r['counts'][k]==v,(name,k,r['counts'][k],v)
        tested.append(name)
        return r
    check('one forecast / multiple observations',[[1],[2],[8]],[(0,0)],
          dict(TP_old=1,FN_recovered=1,H_new=2,FN_new=1,FP_new=0))
    r=check('two eligible matched forecasts / no double counting',
            [[0,8],[8,0],[2,1]],[(0,0),(1,1)],dict(FN_recovered=1,H_new=3))
    assert r['assignments'].tolist()==[0,1,1]
    r=check('exact tie / lowest originally matched forecast ID',
            [[0,8],[8,0],[2,2]],[(0,0),(1,1)],dict(FN_recovered=1))
    assert r['assignments'].tolist()==[0,1,0]
    r=check('duplicate forecast remains FP',[[1,2],[3,9]],[(0,0)],
            dict(TP_old=1,FP_old=1,FP_new=1,FN_recovered=1))
    assert r['unmatched_forecast_ids'].tolist()==[1]
    assert r['counts']['FAR_forecast']==r['counts']['FAR_old']==.5
    check('isolated FP remains FP',[[1,9],[2,10]],[(0,0)],dict(FP_new=1,FN_recovered=1))
    check('outside tolerance remains FN',[[0],[5.0001]],[(0,0)],dict(FN_recovered=0,FN_new=1))
    check('original epsilon preserved',[[0],[5+5e-9]],[(0,0)],dict(FN_recovered=1))
    check('no forecasts',np.empty((3,0)),[],dict(H_new=0,FN_new=3,FP_new=0))
    check('no observations',np.empty((0,3)),[],dict(H_new=0,FN_new=0,FP_new=3))
    r=check('empty populations',np.empty((0,0)),[],dict(H_new=0,FN_new=0,FP_new=0))
    assert np.isnan(r['counts']['CSI_association'])
    for name,d,pairs in [
        ('reject nonmaximum original match',[[1]],[]),
        ('reject duplicate original observation',[[1,1]],[(0,0),(0,1)]),
        ('reject invalid distance',[[np.nan]],[])]:
        try:
            recover(d,pairs)
        except RuntimeError:
            tested.append(name)
        else:
            raise AssertionError(name)
    # Compare the untouched matcher with an independent cardinality-only matcher.
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching
    rng=np.random.default_rng(20261107)
    for _ in range(250):
        no,nf=map(int,rng.integers(0,10,size=2))
        d=rng.uniform(0,12,size=(no,nf))
        pairs=C.matching(d,5.)
        if no and nf:
            maximum=maximum_bipartite_matching(csr_matrix(d<=5+1e-8),perm_type='column')
            assert len(pairs)==int((maximum>=0).sum())
        else:
            assert len(pairs)==0
        r=recover(d,pairs)
        assert r['counts']['FP_old']==r['counts']['FP_new']
    tested.append('250 original-matcher / independent maximum-cardinality graph comparisons')
    core=np.zeros((28,32),bool);core[5,5]=True;core[5,15]=True
    field=np.zeros(core.shape,np.float32)
    field[5,5:16]=.8;field[22,23:26]=.8;field[18,2]=.8
    old_core=core.copy();old_field=field.copy()
    r,labels,truth,d=score(C,field,core,.4,2.,2.)
    assert (r['counts']['TP_old'],r['counts']['FP_old'],r['counts']['FN_recovered'])==(1,1,1)
    assert labels[18,2]==0  # Original min-area=2 filter, unchanged.
    assert np.array_equal(core,old_core) and np.array_equal(field,old_field)
    tested.append('original extraction, min-area filter and immutable input arrays')
    return tested
