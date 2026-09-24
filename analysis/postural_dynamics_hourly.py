"""Cluster ants from concatenated clock-aligned hourly posture/velocity motifs.

The input dictionary is frozen from day 1. Hourly counts, missingness and all
114 tracked identities are audited before spatial outcomes are opened.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

from analysis.postural_dynamics import (CURRENT, SEED, PRIMARY, MAX_VELOCITY_MM_S,
    causal_pose, clip_quality, history_features, joint_state)
from analysis.postural_dynamics_extract import DAY1_FRAME, DAY_FRAMES, FPS, stamp

MIN_CLIPS = 5
MIN_HOURS = 16
VERSION = 'hourly_concatenated_v1'


def hourly_profiles(labels, ants, minutes, n_ants, n_motifs, min_clips=MIN_CLIPS):
    """[day,ant,clock-hour,motif], with NaN for insufficiently observed hours."""
    hour = np.asarray(minutes) // 60
    if np.any((hour < 0) | (hour >= 48)):
        raise ValueError('Hours must be within the two matched days')
    counts = np.bincount(((hour//24*n_ants + ants)*24 + hour%24)*n_motifs + labels,
                         minlength=2*n_ants*24*n_motifs).reshape(2,n_ants,24,n_motifs)
    n = counts.sum(axis=-1)
    prob = np.divide(counts,n[...,None],out=np.full(counts.shape,np.nan),where=n[...,None]>=min_clips)
    return prob, counts


def mask_and_values(x):
    x = np.asarray(x,float)
    if x.ndim != 3:
        raise ValueError('Expected [ant,hour,motif] features')
    mask = np.isfinite(x).all(axis=-1)
    if np.any(np.isfinite(x).any(axis=-1) != mask):
        raise ValueError('An hour must have all features or be entirely missing')
    return mask, np.nan_to_num(x)


def distances(x, centers):
    """Mean squared feature distance across an ant's observed clock hours."""
    mask, values = mask_and_values(x)
    diff = values[:,None] - centers[None]
    loss = (np.square(diff).sum(axis=-1)*mask[:,None]).sum(axis=-1)
    return np.divide(loss,mask.sum(axis=-1)[:,None],out=np.full(loss.shape,np.inf),where=mask.sum(axis=-1)[:,None]>0)


def pairwise_distances(x):
    """Hour-matched RMS distance; only jointly measured hours contribute."""
    mask, values = mask_and_values(x)
    overlap = mask[:,None] & mask[None,:]
    n = overlap.sum(axis=-1)
    if np.any(n == 0):
        raise ValueError('Some ants share no measured clock hours')
    d = np.sqrt((np.square(values[:,None]-values[None,:]).sum(axis=-1)*overlap).sum(axis=-1)/n)
    np.fill_diagonal(d,0)
    return d


def fit_masked(x, k, *, seed=SEED, n_init=12, max_iter=100, weights=None):
    """K-means objective on concatenated hours, masking missing hour blocks.

    Each ant has total weight one regardless of its number of observed hours.
    Missing center hours use the training colony mean for that clock hour.
    """
    mask, values = mask_and_values(x)
    if np.any(mask.sum(axis=1)==0):
        raise ValueError('An ant must have at least one observed hour')
    weights = np.ones(len(x)) if weights is None else np.asarray(weights,float)
    w = mask * (weights / mask.sum(axis=1))[:,None]
    den = w.sum(axis=0)
    if np.any(den==0):
        raise ValueError('A training hour has no observations')
    global_mean = np.einsum('nh,nhm->hm',w,values)/den[:,None]
    filled = np.where(mask[...,None],values,global_mean)
    rng = np.random.default_rng(seed); best = None
    for _ in range(n_init):
        centers = [filled[rng.choice(len(x),p=weights/weights.sum())]]
        for j in range(1,k):
            scores = distances(x,np.asarray(centers)).min(axis=1)*weights
            centers.append(filled[rng.choice(len(x),p=scores/scores.sum() if scores.sum()>0 else weights/weights.sum())])
        centers = np.asarray(centers).copy()
        for iteration in range(max_iter):
            d = distances(x,centers); labels = d.argmin(axis=1)
            updated=[]
            for j in range(k):
                member = labels==j; ww=w*member[:,None]; dd=ww.sum(axis=0)
                center=np.divide(np.einsum('nh,nhm->hm',ww,values),dd[:,None],out=global_mean.copy(),where=dd[:,None]>0)
                if not np.any(weights[member]>0):
                    center=filled[np.argmax(d.min(axis=1)*weights)].copy()
                updated.append(center)
            updated=np.asarray(updated)
            if np.max(abs(updated-centers))<1e-7:
                centers=updated;break
            centers=updated
        d=distances(x,centers); labels=d.argmin(axis=1)
        objective=float(np.average(d.min(axis=1),weights=weights))
        if best is None or objective<best['objective']:
            best=dict(centers=centers,labels=labels,objective=objective)
    return best


def select_partition(candidates):
    eligible=candidates[(candidates.k>=2)&(candidates.smallest_group>=4)&(candidates.bootstrap_ari_median>=.8)&candidates.silhouette.notna()]
    return 1 if eligible.empty else int(eligible.loc[eligible.silhouette>=eligible.silhouette.max()-.02,'k'].min())


def cluster_hourly(x, activity, *, bootstrap=100, seed=SEED):
    pair=pairwise_distances(x);rng=np.random.default_rng(seed);fits={};rows=[];agreements={}
    for k in range(1,min(6,len(x)-1)+1):
        fit=fit_masked(x,k,seed=seed,n_init=20);fits[k]=fit;labels=fit['labels']
        if k==1:
            rows.append(dict(k=1,silhouette=np.nan,bootstrap_ari_median=np.nan,bootstrap_ari_p10=np.nan,bootstrap_ari_p90=np.nan,smallest_group=len(x),sizes=json.dumps([len(x)])))
            agreements[k]=(np.full(len(x),np.nan),np.zeros(len(x),int));continue
        ari=[]; numerator=np.zeros(len(x));denominator=np.zeros(len(x))
        for b in range(bootstrap):
            weights=np.bincount(rng.integers(len(x),size=len(x)),minlength=len(x))
            # With >=16/24 hours per ant, a missing bootstrap hour is rare;
            # redraw instead of inventing a clock-hour center.
            while np.any((np.isfinite(x[weights>0]).all(axis=-1)).sum(axis=0)==0):
                weights=np.bincount(rng.integers(len(x),size=len(x)),minlength=len(x))
            pred=fit_masked(x,k,seed=seed+b+1,n_init=5,weights=weights)['labels']
            ari.append(adjusted_rand_score(labels,pred))
            tab=np.zeros((k,k),int)
            np.add.at(tab,(pred[weights>0],labels[weights>0]),1)
            a,z=linear_sum_assignment(-tab);lookup=np.zeros(k,int);lookup[a]=z
            absent=weights==0;numerator+=absent&(lookup[pred]==labels);denominator+=absent
        sizes=np.bincount(labels,minlength=k)
        silhouette=silhouette_score(pair,labels,metric='precomputed') if 1<len(np.unique(labels))<len(x) else np.nan
        rows.append(dict(k=k,silhouette=silhouette,bootstrap_ari_median=np.median(ari),bootstrap_ari_p10=np.quantile(ari,.1),bootstrap_ari_p90=np.quantile(ari,.9),smallest_group=int(sizes.min()),sizes=json.dumps(sizes.tolist())))
        agreements[k]=(np.divide(numerator,denominator,out=np.full(len(x),np.nan),where=denominator>0),denominator)
    candidates=pd.DataFrame(rows);chosen=select_partition(candidates);selected=fits[chosen]
    order=sorted(range(chosen),key=lambda g:np.median(activity[selected['labels']==g]))
    lookup=np.argsort(order);labels=lookup[selected['labels']]
    return dict(candidates=candidates,selected_k=chosen,labels=labels,centers=selected['centers'][order],
                fits=fits,oob_agreement=agreements[chosen][0],oob_trials=agreements[chosen][1],pairwise=pair)


def load_all(tasks_path,cache,output):
    tasks=json.loads(tasks_path.read_text());qc=[];pieces=[];provenance=[stamp(tasks_path)]
    if len(tasks)!=114:raise ValueError('Prepare all 114 identities with --all-tracks')
    for i,task in enumerate(tasks):
        path=cache/(task['ant'].replace(':','_')+'.npz');meta_path=path.with_suffix('.json')
        saved=json.loads(meta_path.read_text());actual=stamp(path)
        if saved['signature']['task']!=task or stamp(task['source']['path'])!=task['source']:
            raise ValueError(f'Stale cache or changed source: {path}')
        if saved['signature'].get('feature_version')!='posture_velocity_v2':raise ValueError('Velocity cache required')
        if (actual['size'],actual['mtime_ns'])!=(saved['output']['size'],saved['output']['mtime_ns']):raise ValueError('Cache changed')
        with np.load(path) as z:
            raw=z['shape'];lengths=z['lengths_mm'];cameras=z['cameras'];starts=z['frames'];v=z['velocity_mm_s'];pc=z['position_cameras']
            keep,flags=clip_quality(raw,lengths,cameras,z['duplicate_frame']);pose=keep.copy()
            valid_v=np.isfinite(v).all(axis=(1,2))&(np.linalg.norm(v,axis=-1)<=MAX_VELOCITY_MM_S).all(axis=1)
            valid_cam=np.isfinite(pc).all(axis=1)&(pc>=0).all(axis=1)&(np.ptp(np.nan_to_num(pc,nan=-1),axis=1)==0)
            keep&=valid_v&valid_cam;indices=np.flatnonzero(keep);shape=causal_pose(raw[keep])
            after=np.isfinite(shape).all(axis=(1,2));shape=shape[after];indices=indices[after]
            minute=((starts[indices]-DAY1_FRAME)//(60*FPS)).astype(int);day=minute//1440
            n=np.bincount(minute//60,minlength=48).reshape(2,24)
            row={k:task[k] for k in ('ant','side','track_id','track_name','detection_fraction')}
            row.update(requested=len(raw),complete_clips=int(flags['complete'].sum()),geometry_pass=int(flags['geometry'].sum()),camera_pass=int(flags['same_camera'].sum()),duplicate_clips=int(flags['duplicate'].sum()),posture_quality_clips=int(pose.sum()),velocity_pass=int(valid_v.sum()),position_camera_pass=int(valid_cam.sum()),accepted=len(indices),old_detection_pass=task['detection_fraction']>.4)
            for d in (0,1):
                row[f'day{d+1}_clips']=int(n[d].sum());row[f'day{d+1}_hours']=int((n[d]>=MIN_CLIPS).sum())
                row[f'day{d+1}_eligible']=row[f'day{d+1}_hours']>=MIN_HOURS
            row['old_day1_eligible']=row['old_detection_pass'] and row['day1_clips']>=240 and len(np.unique(minute[day==0]//30))>=24
            row['exclusion_reason']='included' if row['day1_eligible'] else ('no accepted day-1 clips' if not row['day1_clips'] else f'fewer than {MIN_HOURS} hours with >= {MIN_CLIPS} clips')
            qc.append(row)
            pieces.append(dict(shape=shape,velocity=v[indices],ant_index=np.full(len(indices),i,int),minute=minute,day=day,frames=starts[indices],camera=cameras[indices,0].astype(int)))
        provenance.extend([actual,stamp(meta_path),task['source']]);print('HOURLY_QC',task['ant'],row['day1_clips'],row['day1_hours'],row['day1_eligible'],flush=True)
    qc=pd.DataFrame(qc);qc.to_csv(output/'all_ant_quality.csv',index=False)
    bundle={k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    counts=np.bincount(bundle['ant_index']*48+bundle['minute']//60,minlength=len(qc)*48).reshape(len(qc),48)
    pd.DataFrame(counts,index=qc.ant,columns=[f'hour_{h:02d}' for h in range(48)]).to_csv(output/'all_ant_hourly_coverage.csv')
    return qc,bundle,provenance


def project_observed(x,mean,components):
    """Display/continuous-model coordinates, solved using measured hours only."""
    flat=x.reshape(len(x),-1);result=[]
    for row in flat:
        valid=np.isfinite(row);c=components[:,valid]
        result.append(np.linalg.solve(c@c.T+np.eye(len(c))*1e-6,c@(row[valid]-mean[valid])))
    return np.asarray(result)


def observed_loss(target,prediction):
    mask,values=mask_and_values(target)
    loss=(np.square(values-prediction).sum(axis=-1)*mask).sum(axis=-1)
    return np.divide(loss,mask.sum(axis=-1),out=np.full(len(target),np.nan),where=mask.sum(axis=-1)>0)


def fit_hourly(qc,b,dictionary,output,bootstrap):
    frozen=json.loads((dictionary/'SPACE_BLIND_FROZEN.json').read_text())
    if hashlib.sha256((dictionary/'space_blind_models.npz').read_bytes()).hexdigest()!=frozen['model_sha256']:raise ValueError('Dictionary hash changed')
    if frozen.get('feature_version')!='posture_velocity_v2':raise ValueError('Expected frozen posture+velocity dictionary')
    with np.load(dictionary/'space_blind_models.npz') as z:m={k:z[k] for k in z.files}
    rank=int(m['rank']);k=int(m['n_motifs']);shape=b['shape'];ids=b['ant_index'];day=b['day']
    scores=((shape-m['posture_mean'])@m['posture_components'].T)[:,:,:rank].astype(np.float32)
    norm={name:m[name] for name in ('posture_feature_mean','posture_feature_scale','velocity_feature_mean','velocity_feature_std')}
    state,_=joint_state(scores,b['velocity'],[],normalizer=norm)
    primary=history_features(state,1.)
    variants={PRIMARY:primary,'instantaneous':history_features(state,0.),'posture_history':history_features(state[:,:,:rank],1.),'velocity_history':history_features(state[:,:,rank:],1.),'dynamics_only':history_features(state,1.,dynamics_only=True)}
    for weight,name in ((.5,'velocity_half'),(2.,'velocity_double')):
        weighted=state.copy();weighted[:,:,rank:]*=weight;variants[name]=history_features(weighted,1.)
    # Restore the exact camera correction used to train the frozen ablation.
    # Its global mean is recoverable from the equal-ant training cache, stored
    # separately by the original run only through its centered centers. Do not
    # approximate it here: omit this ablation from the hourly revision.
    angles=np.arctan2(shape.reshape(len(shape),28,8,2)[...,1],shape.reshape(len(shape),28,8,2)[...,0])
    motion=np.sqrt(np.square(np.angle(np.exp(1j*np.diff(angles[:,:CURRENT+1],axis=1)))).mean(axis=(1,2)))*12
    speed=np.linalg.norm(b['velocity'][:,:CURRENT+1],axis=-1).mean(axis=1)
    qc=qc.copy()
    for name,values in [('angular_motion_rad_s',motion),('sampled_speed_mm_s',speed)]:
        qc[name]=[float(values[(ids==i)&(day==0)].mean()) if np.any((ids==i)&(day==0)) else np.nan for i in range(len(qc))]
    all_profiles={};all_counts={};assigned=None
    for name,x in variants.items():
        labels=cdist(x,m[name+'_centers'],metric='sqeuclidean').argmin(axis=1)
        prob,count=hourly_profiles(labels,ids,b['minute'],len(qc),k);all_profiles[name]=prob;all_counts[name]=count
        if name==PRIMARY:assigned=labels
    cameras=np.unique(b['camera']);labels=np.searchsorted(cameras,b['camera'])
    all_profiles['camera_only'],all_counts['camera_only']=hourly_profiles(labels,ids,b['minute'],len(qc),len(cameras))
    qc.to_csv(output/'all_ant_quality.csv',index=False)
    np.savez_compressed(output/'hourly_motif_profiles.npz',**all_profiles,counts=all_counts[PRIMARY],ants=qc.ant.to_numpy(str),camera_ids=cameras)
    n=all_counts[PRIMARY].sum(axis=-1);rows=[]
    for d in (0,1):
        for i,a in enumerate(qc.ant):
            for h in range(24):
                row=dict(ant=a,side=qc.iloc[i].side,day=d+1,hour=h,clock_hour=(10+h)%24,n_clips=int(n[d,i,h]),measured=bool(n[d,i,h]>=MIN_CLIPS))
                row.update({f'motif_{j:02d}':all_profiles[PRIMARY][d,i,h,j] for j in range(k)})
                rows.append(row)
    pd.DataFrame(rows).to_csv(output/'hourly_motif_frequencies.csv',index=False)
    results={};arrays={};controls=[]
    for side in ('left','right'):
        ix=np.flatnonzero(qc.side.eq(side)&qc.day1_eligible);t=qc.iloc[ix].copy();x=np.sqrt(all_profiles[PRIMARY][:,ix]);activity=t.angular_motion_rad_s.to_numpy()
        if len(ix)<12:raise ValueError(f'Only {len(ix)} eligible ants in {side}')
        r=cluster_hourly(x[0],activity,bootstrap=bootstrap);r['candidates'].to_csv(output/f'{side}_ant_model_selection.csv',index=False)
        print('SELECTED_HOURLY_K',side,len(ix),r['selected_k'],flush=True)
        mean=r['fits'][1]['centers'][0];filled=np.where(np.isfinite(x[0]),x[0],mean).reshape(len(ix),-1)
        pca=PCA(svd_solver='full').fit(filled);coord=project_observed(x[0],pca.mean_,pca.components_[:2]);coord2=project_observed(x[1],pca.mean_,pca.components_[:2])
        t['group']=r['labels'];t['pc1']=coord[:,0];t['pc2']=coord[:,1];t['day2_pc1']=coord2[:,0];t['day2_pc2']=coord2[:,1]
        t['oob_agreement']=r['oob_agreement'];t['oob_trials']=r['oob_trials']
        t['day2_group']=distances(x[1],r['centers']).argmin(axis=1);t.loc[~t.day2_eligible,'day2_group']=-1
        paired=t.day2_eligible.to_numpy();repeat=float((t.loc[paired,'group']==t.loc[paired,'day2_group']).mean()) if paired.any() and r['selected_k']>1 else None
        baseline=np.broadcast_to(mean,x[1].shape);base=np.nanmean(observed_loss(x[1][paired],baseline[paired]))
        validation=[]
        for count,fit in r['fits'].items():
            pred=fit['centers'][fit['labels']];loss=np.nanmean(observed_loss(x[1][paired],pred[paired]))
            validation.append(dict(model=f'K={count}',complexity=count,mean_mse=loss,explained_vs_mean=1-loss/base,n_ants=int(paired.sum())))
        for count in (1,2):
            c=project_observed(x[0],pca.mean_,pca.components_[:count]);pred=(c@pca.components_[:count]+pca.mean_).reshape(x[0].shape)
            loss=np.nanmean(observed_loss(x[1][paired],pred[paired]));validation.append(dict(model=f'PC{count}',complexity=count,mean_mse=loss,explained_vs_mean=1-loss/base,n_ants=int(paired.sum())))
        validation=pd.DataFrame(validation);validation.to_csv(output/f'{side}_heldout_profile_prediction.csv',index=False)
        tri=np.triu_indices(len(ix),1);d=r['pairwise'][tri]
        for name,profile in all_profiles.items():
            other=pairwise_distances(np.sqrt(profile[0,ix]))[tri]
            rho=float(spearmanr(d,other).statistic) if np.ptp(other)>0 and np.ptp(d)>0 else np.nan
            controls.append(dict(side=side,representation=name,distance_spearman_with_primary=rho))
        observed=np.isfinite(x[0]).all(axis=-1).astype(float);missing_distance=cdist(observed,observed,metric='hamming')[tri]
        controls.append(dict(side=side,representation='availability_only',distance_spearman_with_primary=float(spearmanr(d,missing_distance).statistic)))
        # Same counts but each ant's hours independently permuted: timing control.
        rng=np.random.default_rng(SEED);shuffled=np.stack([p[rng.permutation(24)] for p in x[0]])
        shuffled_fit=cluster_hourly(shuffled,activity,bootstrap=bootstrap,seed=SEED+1)
        shuffled_fit['candidates'].to_csv(output/f'{side}_shuffled_hour_model_selection.csv',index=False)
        hour_control=dict(selected_k=shuffled_fit['selected_k'],ari_with_ordered=float(adjusted_rand_score(r['labels'],shuffled_fit['labels'])))
        r.update(rows=ix,table=t,profile=all_profiles[PRIMARY][:,ix],counts=all_counts[PRIMARY][:,ix],profile_pca=pca,repeat_fraction=repeat,repeat_n=int(paired.sum()),profile_validation=validation,hour_control=hour_control)
        results[side]=r;t.to_csv(output/f'{side}_hourly_groups.csv',index=False)
        arrays.update({side+'_centers':r['centers'],side+'_rows':ix,side+'_profile_pca_mean':pca.mean_,side+'_profile_pca_components':pca.components_,side+'_pairwise':r['pairwise']})
    controls=pd.DataFrame(controls);controls.to_csv(output/'hourly_representation_controls.csv',index=False)
    np.savez_compressed(output/'hourly_ant_models.npz',**arrays)
    frozen_new=dict(feature_version=VERSION,frozen_utc=datetime.now(timezone.utc).isoformat(),occupancy_inputs_loaded=False,
      min_clips_per_hour=MIN_CLIPS,min_hours=MIN_HOURS,n_hours=24,n_motifs=k,concatenated_dimensions=24*k,
      dictionary_sha256=frozen['model_sha256'],dictionary_training_ants=frozen['n_ants'],dictionary_path=str(dictionary),
      selected_ant_k={s:r['selected_k'] for s,r in results.items()},model_sha256=hashlib.sha256((output/'hourly_ant_models.npz').read_bytes()).hexdigest(),
      groups_sha256={s:hashlib.sha256((output/f'{s}_hourly_groups.csv').read_bytes()).hexdigest() for s in results},
      hour_shuffle={s:r['hour_control'] for s,r in results.items()})
    (output/'HOURLY_FROZEN.json').write_text(json.dumps(frozen_new,indent=2)+'\n')
    return results,qc,all_profiles,controls,m


def spatial_validation(block,atoms,output,results):
    """Withheld outcomes; never used to fit motifs, select hours or choose K."""
    from analysis.colony_behavioral_landscape import features
    frozen=json.loads((output/'HOURLY_FROZEN.json').read_text());assert not frozen['occupancy_inputs_loaded']
    assert hashlib.sha256((output/'hourly_ant_models.npz').read_bytes()).hexdigest()==frozen['model_sha256']
    start=pd.Timestamp('2026-07-24 10:00').value//(1800*10**9);provenance=[];summary={};forecasts=[]
    label_path=block/'stitched/grid_occupancy_histograms_arena_0p25mm/track_cluster_ids.csv';labels=pd.read_csv(label_path);labels['ant']=labels.side+':'+labels.TrackID.astype(int).astype(str).str.zfill(3)
    behavior_path=block/'analysis_outputs/long_timescale_0723_0724_0729_20260915/task_bins.parquet'
    behavior=pd.read_parquet(behavior_path);behavior=behavior[behavior.source_block.eq(str(block))&behavior.in_recording_bin].copy()
    behavior['day']=((behavior.timestamp-pd.Timestamp('2026-07-24 10:00')).dt.total_seconds()//86400).astype(int)
    values={}
    for ant,part in behavior[behavior.day.eq(1)].groupby('ant'):
        w=part.n_expected_frames*part.position_coverage;ok=part.colony_percent.notna()&w.gt(0)
        values[ant]=float(np.average(part.loc[ok,'colony_percent'],weights=w[ok])) if ok.any() else np.nan
    provenance.extend([stamp(label_path),stamp(behavior_path),stamp(block/'panorama_regions.csv')]);rng=np.random.default_rng(SEED)
    for side,r in results.items():
        assert hashlib.sha256((output/f'{side}_hourly_groups.csv').read_bytes()).hexdigest()==frozen['groups_sha256'][side]
        t=r['table'].copy();maps=[];coverage=[]
        for ant in t.ant:
            path=atoms/(ant.replace(':','_')+'.npz');meta=json.loads(path.with_suffix('.json').read_text())
            if meta['signature']['entry']['block']!=str(block) or not meta['original_histogram_exact']:raise ValueError('Wrong spatial cache')
            for item in meta['signature']['inputs']:
                actual=stamp(item['path'])
                if (actual['size'],actual['mtime_ns'])!=(item['size'],item['mtime_ns']):raise ValueError('Stale spatial cache')
            mm=[];cc=[]
            with np.load(path) as z:
                for d in (0,1):
                    mask=(z['calendar_bin']>=start+d*48)&(z['calendar_bin']<start+(d+1)*48)
                    if mask.sum()!=48:raise ValueError('Incomplete spatial calendar')
                    n=int(z['detected'][mask].sum());mm.append(z['counts'][mask].sum(axis=0)/max(1,n));cc.append(n/DAY_FRAMES)
                edges=(z['x_edges'],z['y_edges'])
            maps.append(mm);coverage.append(cc);provenance.extend([stamp(path),stamp(path.with_suffix('.json'))])
        maps=np.asarray(maps);coverage=np.asarray(coverage);t['position_coverage_day1']=coverage[:,0];t['position_coverage_day2']=coverage[:,1]
        t['spatial_label']=t.ant.map(labels.set_index('ant').cluster_id);t['day2_colony_percent']=t.ant.map(values)
        comparable=t.spatial_label.notna();ari=adjusted_rand_score(t.loc[comparable,'group'],t.loc[comparable,'spatial_label']) if comparable.any() and r['selected_k']>1 else None
        null=[adjusted_rand_score(rng.permutation(t.loc[comparable,'group']),t.loc[comparable,'spatial_label']) for _ in range(1000)] if ari is not None else []
        ix=np.flatnonzero((coverage>=.4).all(axis=1));a1=maps[ix,0];a2=maps[ix,1];g=t.group.to_numpy()[ix]
        def loss(groups):
            pred=[]
            for j in range(len(ix)):
                other=(np.arange(len(ix))!=j)&(groups==groups[j])
                if not other.any():other=np.arange(len(ix))!=j
                pred.append(a1[other].mean(axis=0))
            return np.square(features(np.asarray(pred),np.ones(len(ix)))-features(a2,np.ones(len(ix)))).sum(axis=1)/2
        base=loss(np.zeros(len(ix),int));cluster=loss(g);gain=base-cluster
        own=np.square(features(a1,np.ones(len(ix)))-features(a2,np.ones(len(ix)))).sum(axis=1)/2
        boot=[rng.choice(gain,len(gain),replace=True).mean() for _ in range(2000)]
        perm=[(base-loss(rng.permutation(g))).mean() for _ in range(1000)] if r['selected_k']>1 else []
        for j,i in enumerate(ix):forecasts.append(dict(side=side,ant=t.iloc[i].ant,baseline_loss=base[j],hourly_group_loss=cluster[j],own_previous_day_loss=own[j],gain=gain[j]))
        r.update(table=t,spatial_maps=maps,spatial_edges=edges,forecast_gain=gain)
        t.to_csv(output/f'{side}_groups_with_spatial_outcomes.csv',index=False)
        summary[side]=dict(n_ants=len(t),tracked_identities=57,selected_k=r['selected_k'],groups=t.group.value_counts().sort_index().to_dict(),spatial_ari=ari,spatial_comparison_n=int(comparable.sum()),spatial_permutation_p=float((1+(np.asarray(null)>=ari).sum())/1001) if ari is not None else None,day2_assignment_retention=r['repeat_fraction'],day2_assignment_n=r['repeat_n'],forecast_n=len(ix),mean_spatial_gain=float(gain.mean()),spatial_gain_ci=np.quantile(boot,[.025,.975]).tolist(),forecast_permutation_p=float((1+(np.asarray(perm)>=gain.mean()).sum())/1001) if perm else None,baseline_spatial_loss=float(base.mean()),group_spatial_loss=float(cluster.mean()),own_previous_day_loss=float(own.mean()),hour_shuffle=r['hour_control'])
        print('HOURLY_SPATIAL_RESULT',side,json.dumps(summary[side]),flush=True)
    pd.DataFrame(forecasts).to_csv(output/'heldout_spatial_forecasts.csv',index=False)
    np.savez_compressed(output/'heldout_spatial_maps.npz',**{s:r['spatial_maps'] for s,r in results.items()})
    (output/'results_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary,provenance


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('tasks','pose-cache','dictionary','block','spatial-atoms','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--bootstrap',type=int,default=100);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.block.parent.name!='20260724' or a.block.name!='block01':raise ValueError('Wrong recording')
    (a.output/'COMPLETE.json').unlink(missing_ok=True)
    qc,b,provenance=load_all(a.tasks,a.pose_cache,a.output)
    results,qc,profiles,controls,models=fit_hourly(qc,b,a.dictionary,a.output,a.bootstrap)
    summary,sources=spatial_validation(a.block,a.spatial_atoms,a.output,results);provenance+=sources
    provenance += [stamp(a.dictionary/n) for n in ('SPACE_BLIND_FROZEN.json','space_blind_models.npz','run_manifest.json','history_prediction_validation.csv','motif_resolution_validation.csv','posture_reconstruction.csv','posture_display_samples.npz')]
    from analysis.postural_dynamics_hourly_plots import render
    render(a.output,results,qc,profiles,controls,models,a.dictionary,a.block,summary)
    if any(stamp(item['path'])!=item for item in provenance):raise ValueError('Input changed during analysis')
    manifest=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},summary=summary,source_fingerprints=provenance,feature_version=VERSION,
      software={k:importlib.metadata.version(k) for k in ('numpy','pandas','scipy','scikit-learn','matplotlib','pyarrow')})
    (a.output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (a.output/'COMPLETE.json').write_text(json.dumps(dict(complete=True,source=str(a.block)))+'\n')


if __name__=='__main__':main()
