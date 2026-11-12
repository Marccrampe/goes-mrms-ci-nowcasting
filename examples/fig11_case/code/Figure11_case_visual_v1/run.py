#!/usr/bin/env python3
"""Figure11 visual changes only; reconstruct original labels from saved decisions, never rematch."""
import os
os.environ.setdefault('MPLBACKEND','Agg');os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse,datetime as dt,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import common as H
from snapshots import choose_scans

def project_radar(field,m,shape):
    flat=m['target_y'].astype(np.int64)*shape[1]+m['target_x'];total=m['total_count'].ravel()
    v=field[m['source_y'],m['source_x']];good=np.isfinite(v)&(v>=-50)
    count=np.bincount(flat[good],minlength=len(total));valid=(total>0)&(count>=.75*total)
    peak=np.full(len(total),-50,np.float32);np.maximum.at(peak,flat[good],v[good]);peak[~valid]=np.nan
    return peak.reshape(shape),valid.reshape(shape)
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',type=Path,default=Path('/project'))
    ap.add_argument('--case-root',type=Path);ap.add_argument('--output',type=Path)
    a=ap.parse_args();root=a.project.resolve();source=a.case_root or root/'experiments/ci_case_study_eval_v1'
    out=a.output or root/'figure_finale/Main/Fig11'/('visual_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    H.require(not out.exists(),'Use a NEW output directory');out.mkdir(parents=True)
    evidence={}
    def record(p):p=Path(p);evidence[str(p)]=H.sha(p);return p
    def read(p):return H.read(record(p))
    try:
        t=pd.Timestamp('2025-07-25 19:00:00');date='20250725';name='T0_'+t.strftime('%Y%m%dT%H%M%S')
        cfg=read(source/'config.json');decision_path=source/date/(name+'_decisions.json');decision=read(decision_path)
        score=decision['score'];births=decision['births'];accepted={int(k):v for k,v in decision['accepted_candidates'].items()}
        H.require(pd.Timestamp(score['issuance_time']).value==t.value and int(score.get('growth_dbz',25))==25,'Wrong saved case or diagnostic')
        q=float(score['threshold']);pid=str(score['period_id']);split=str(score['split'])
        selection=pd.read_csv(record(source/'selected_issuances.csv'))
        selected=selection[pd.to_datetime(selection.issuance_time,utc=True).astype('int64').eq(t.value)&selection.period_id.astype(str).eq(pid)]
        H.require(len(selected)==1,'Case selection missing or duplicated');selected=selected.iloc[0]
        cache=source/date/'cache'/(name+'.npz');marker=read(cache.with_suffix('.json'))
        H.require(marker['sha256']==H.sha(record(cache)),'Probability cache hash changed')
        contract=marker['contract'];H.require(str(contract['period_id'])==pid and pd.Timestamp(contract['issuance_time']).value==t.value,'Wrong probability case')
        H.require(contract['checkpoint_sha256']==cfg['checkpoint_sha256'],'Wrong checkpoint in saved case cache')
        with np.load(cache,allow_pickle=False) as z:p=z['probability'].copy()
        H.require(p.ndim==2 and np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'Invalid saved probability map')
        code=root/'scripts/isolated_ci_binary_v1/goes_ir_v1';sys.path.insert(0,str(code))
        # Existing scientific component labelling only; there is NO matching or classification call.
        for name0 in ['ci_common_exploratory_core.py','ci_v4_relaxed_core.py','build_ci_soft_targets_native_goes.py']:
            f=code/name0;h=cfg.get('readers_sha256',{}).get(str(f))
            H.require(h is not None and H.sha(record(f))==h,'Original source kernel differs: '+str(f))
        import ci_common_exploratory_core as C
        core=np.zeros(p.shape,bool)
        for birth in births:
            yy=np.asarray(birth['core_y'],int);xx=np.asarray(birth['core_x'],int)
            H.require(len(yy)==len(xx) and len(yy)>0,'Saved birth footprint empty')
            H.require(((yy>=0)&(yy<p.shape[0])&(xx>=0)&(xx<p.shape[1])).all(),'Saved birth footprint outside grid')
            core[yy,xx]=True
        observed,no=C.components(core,1);predicted,nf=C.components(p>=q,2)
        ledger=pd.read_csv(record(source/'object_ledger.csv'))
        ledger=ledger[ledger.period_id.astype(str).eq(pid)&pd.to_datetime(ledger.issuance_time,utc=True).astype('int64').eq(t.value)]
        H.require(not ledger.duplicated(['object_kind','object_id']).any(),'Duplicate saved object IDs')
        forecasts=ledger[ledger.object_kind.eq('forecast')];observations=ledger[ledger.object_kind.eq('observation')]
        H.require(set(forecasts.object_id.astype(int))==set(range(1,nf+1)) and set(observations.object_id.astype(int))==set(range(1,no+1)),'Original object populations differ')
        for r in ledger.itertuples():
            labels=predicted if r.object_kind=='forecast' else observed
            yy,xx=np.where(labels==int(r.object_id))
            H.require(np.isclose(xx.mean(),r.grid_x,atol=1e-8) and np.isclose(yy.mean(),r.grid_y,atol=1e-8),'Saved object geometry/order differs')
        hit_pred=set(forecasts[forecasts.strict_status.eq('TP')].object_id.astype(int))
        hit_obs=set(observations[observations.strict_status.eq('TP_observation')].object_id.astype(int))
        covered_obs=hit_obs|set(observations[observations.covered_competition.map(H.truth)].object_id.astype(int))
        H.require(len(hit_pred)==len(hit_obs)==int(score['tp']) and nf-len(hit_pred)==int(score['fp']) and no-len(hit_obs)==int(score['fn']),'Saved strict counts disagree')
        H.require(set(accepted)==set(forecasts[forecasts.diagnostic_status.eq('accepted_growth')].object_id.astype(int)),'Growth classifications differ')
        H.require(not hit_pred&set(accepted) and len(accepted)==int(score['accepted_growth']),'Original growth population differs')
        H.require(int(score['remaining_FP'])==int(score['fp'])-len(accepted),'Original FP partition differs')
        for k,event in accepted.items():
            r=forecasts[forecasts.object_id.eq(k)].iloc[0]
            H.require(str(r.candidate_id)==str(event['id']),'Saved candidate association differs')
        strict=np.zeros(p.shape,np.uint8);secondary=np.zeros_like(strict)
        for k in range(1,nf+1):
            strict[predicted==k]=1 if k in hit_pred else 3
            secondary[predicted==k]=1 if k in hit_pred else 2 if k in accepted else 3
        counts={k:int(score[k]) for k in ['tp','fp','fn','accepted_growth','remaining_FP']}
        counts.update(FN_covered=len(covered_obs-hit_obs),FN_uncovered=no-len(covered_obs))
        for key in ['FN_covered','FN_uncovered']:
            if key in score:H.require(counts[key]==int(score[key]),'Saved uncovered/competition labels differ')
        result=dict(probability=p,observed=observed,predicted=predicted,strict=strict,secondary=secondary,
                    covered_obs=covered_obs,hit_obs=hit_obs,hit_pred=hit_pred,accepted=accepted,counts=counts,threshold=q)
        mapper=H.load('_figure11_original_mapper',code/'build_ci_soft_targets_native_goes.py')
        grid=mapper.load_and_validate_goes_grid();shape=tuple(grid['shape']);H.require(shape==p.shape,'Grid shape differs')
        from pyproj import Transformer
        inverse=Transformer.from_crs(grid['transformer'].target_crs,'EPSG:4326',always_xy=True)
        xx,yy=np.meshgrid(grid['x_m'],grid['y_m']);lon,lat=inverse.transform(xx,yy)
        H.require(H.array_sha(grid['x_m'],grid['y_m'],lon,lat)==cfg['grid_sha256'],'Original geographic grid identity differs')
        import zarr
        radar_path=Path(selected.radar_path);group=zarr.open_group(str(radar_path),mode='r')
        times=np.asarray(group['time_ns'][:],np.int64)
        chosen=choose_scans(times,int(t.value),[8,18,28],150)
        mapping=mapper.build_mapping(radar_path.parent,grid,{})
        fields=[];stamps=[];audit=[]
        for offset,index in zip([8,18,28],chosen):
            field,valid=project_radar(np.asarray(group['reflectivity_dbz'][index],np.float32),mapping,shape)
            fields.append(np.where(valid,field,np.nan));stamps.append(int(times[index]))
            audit.append(dict(requested_offset_min=offset,requested_time=str(t+pd.Timedelta(minutes=offset)),
                actual_time=str(pd.Timestamp(times[index],tz='UTC')),actual_offset_min=float((times[index]-t.value)/60e9),
                source_index=int(index),decoded_array_sha256=H.array_sha(field,valid)))
        # Same settings/domain rule/threshold and saved assignments; only three display scans and point glyphs change.
        payload=dict(result=result,settings=cfg,lon=lon,lat=lat,t=t,split=split,births=births,
            display_radar_fields=fields,display_radar_times=stamps)
        font=root/'font_nature/Arial.ttf';H.require(font.exists(),'Missing manuscript Arial.ttf');record(font)
        from matplotlib import font_manager as fm,ft2font
        for f in font.parent.glob('*.ttf'):fm.fontManager.addfont(str(f))
        family=fm.FontProperties(fname=str(font)).get_name()
        H.require(family.startswith('Arial'),'Expected manuscript Arial family')
        bold=Path(fm.findfont(fm.FontProperties(family=family,weight='bold'),fallback_to_default=False))
        H.require(ft2font.FT2Font(str(bold)).style_flags&2,'True Arial Bold font missing')
        record(bold)
        os.environ['CI_PAPER_FONT']=str(font)
        import draw
        relative=draw.page(out,name,payload)
        persistence=int(score['persistence_minutes'])
        caption=(f'Case study initialized 25 July 2025 at 19:00 UTC, original split {split}. '
            'Panels a–c show native-source MRMS observations projected with the original radar mapper during the 0–30min forecast window. '
            'Requested times are 19:08, 19:18 and 19:28 UTC; exact selected scan times are printed in the panel titles and recorded in scan_times.csv. '
            'The nearest available scan within 150 s is used; frames outside the forecast window are forbidden. '
            f'Panel d retains the original saved probabilities and raw threshold {q:g}. Panels e/f preserve the original full-domain strict/diagnostic classifications and numerical scores, before the unchanged display zoom. '
            'Green: Accepted; red: Unmatched; blue dots: CI35 observations spatially covered by a forecast; blue crosses: Uncovered miss. Orange dots: CI25. '
            f'Here CI25 denotes eligible ≥25dBZ onsets used by the broader prediction-conditioned growth diagnostic (persistence {persistence}min, minimum one native MRMS pixel, original association radius {score["growth_km"]:g}km); it does not necessarily denote the independent CI25 catalogue. '
            'A covered observation is not necessarily a conventional one-to-one TP. The diagnostic is not the competition-only association metric. '
            'White top-row birth contours follow the original display rule: a birth is shown only in the first displayed scan after its recorded birth and within 6 min, with the original display-only two-cell dilation. They are reference birth footprints, not radar echo tracking contours. '
            'Probability fields, thresholds, classifications and counts are unchanged. Updated top-row scans are verification context, not model inputs. '
            'This case retains its actual split designation; no held-out TEST membership is inferred from the date. Only glyphs, labels, formatting and the three contextual radar frames have changed.')
        (out/'caption.txt').write_text(caption+'\n');pd.DataFrame(audit).to_csv(out/'scan_times.csv',index=False)
        H.write(out/'SAVED_CLASSIFICATIONS_PRESERVED.json',dict(score=score,probability_array_sha256=H.array_sha(p),
            observed_labels_sha256=H.array_sha(observed),forecast_labels_sha256=H.array_sha(predicted),
            strict_display_sha256=H.array_sha(strict),diagnostic_display_sha256=H.array_sha(secondary),
            accepted_candidates=accepted,classification_source=str(decision_path),matching_rerun=False,threshold_changed=False))
        for f,h in evidence.items():H.require(H.sha(f)==h,'Read-only source changed during render: '+f)
        H.write(out/'PROVENANCE.json',dict(input_sha256=evidence,radar_store=str(radar_path),scan_audit=audit,
            split=split,threshold=q,diagnostic_persistence_minutes=persistence,
            code_sha256={p.name:H.sha(p) for p in Path(__file__).parent.glob('*.py')},
            archived_decisions_reused=True,inference=False,verification_recomputed=False,
            previous_config_diagnostic_disagrees_with_saved_score=cfg.get('diagnostic',{}).get('persistence_minutes')!=persistence,
            output=relative,manual_visual_review_required=True))
        H.write(out/'completed.json',dict(status='VISUAL_CHANGES_EXPORTED; REVIEW_REQUIRED',output=str(out)))
        print('FIGURE:',out,flush=True)
        print('SCANS:',audit,flush=True)
        print('Original raw threshold:',q,'; diagnostic persistence:',persistence,'min; saved assignments preserved.',flush=True)
    except Exception as e:
        import traceback
        H.write(out/'FAILED.json',dict(error=str(e),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
