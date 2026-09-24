"""Eigenpostures -> short-time patterns -> ant groups -> withheld spatial tests.

All representation and group fitting is completed before spatial inputs load.
Only 20260724/block01 is used. See postural_dynamics.md for the protocol.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_rand_score, silhouette_score

from analysis.postural_dynamics_extract import DAY1_FRAME, DAY_FRAMES, FPS, EDGES, EDGE_NAMES, stamp

CURRENT=24  # raw frame 52, after causal filtering and 12 Hz sampling
FUTURE=27   # raw frame 58: 0.25 seconds later
SEED=724
PRIMARY='posture_velocity_history'
MAX_VELOCITY_MM_S=20.


def joint_state(scores,velocity,training_indices,normalizer=None):
    """Join posture and signed velocity with equal total training variance.

    Preserve the relative variances of posture modes. Each velocity component
    contributes half of the velocity block's variance. All fitting uses only
    the supplied training clips and times through CURRENT.
    """
    scores=np.asarray(scores);velocity=np.asarray(velocity)
    if scores.shape[:2]!=velocity.shape[:2] or velocity.shape[-1]!=2:
        raise ValueError('Velocity and posture must have aligned clip times')
    if normalizer is None:
        p=scores[training_indices,:CURRENT+1];v=velocity[training_indices,:CURRENT+1]
        normalizer=dict(posture_feature_mean=p.mean(axis=(0,1)),
          posture_feature_scale=np.array(max(np.sqrt(p.var(axis=(0,1)).sum()),1e-6)),
          velocity_feature_mean=v.mean(axis=(0,1)),velocity_feature_std=np.maximum(v.std(axis=(0,1)),1e-4))
    posture=(scores-normalizer['posture_feature_mean'])/normalizer['posture_feature_scale']
    motion=(velocity-normalizer['velocity_feature_mean'])/(normalizer['velocity_feature_std']*np.sqrt(2))
    return np.concatenate([posture,motion],axis=-1).astype(np.float32),normalizer


def causal_pose(shape, width=4):
    """Causal moving average of direction cosines, then 12 Hz sampling.

    Output j uses only raw frames <= 4+2*j. No future sample enters the history.
    Renormalization returns each averaged segment to unit length.
    """
    shape=np.asarray(shape,dtype=np.float32)
    if shape.ndim!=3 or shape.shape[1:]!=(60,16) or width>5 or width<1:
        raise ValueError('Expected [clips,60,16], with a causal width from 1 to 5')
    ends=np.arange(4,59,2)+1
    prefix=np.concatenate([np.zeros((len(shape),1,16),np.float32),np.cumsum(shape,axis=1,dtype=np.float32)],axis=1)
    out=(prefix[:,ends]-prefix[:,ends-width])/width
    pairs=out.reshape(len(out),len(ends),8,2)
    norm=np.linalg.norm(pairs,axis=-1,keepdims=True)
    return np.divide(pairs,norm,out=np.full_like(pairs,np.nan),where=norm>.1).reshape(len(out),len(ends),16)


def clip_quality(shape,lengths,cameras,duplicates):
    """Conservative, predeclared geometry bounds; no interpolation or gap fill."""
    complete=np.isfinite(shape).all(axis=(1,2))
    geometry=((lengths[:,:,0]>=.1)&(lengths[:,:,0]<=1.25)).all(axis=1)
    geometry&=((lengths[:,:,1:]>=.05)&(lengths[:,:,1:]<=2.)).all(axis=(1,2))
    known=np.isfinite(cameras).all(axis=1)
    same=np.ptp(np.nan_to_num(cameras,nan=-1),axis=1)==0
    duplicate=duplicates.any(axis=1)
    return complete & geometry & known & same & ~duplicate, dict(complete=complete,geometry=geometry,same_camera=known&same,duplicate=duplicate)


def history_features(scores, seconds=1., *, dynamics_only=False, shuffle=False, seed=SEED):
    """A fixed-endpoint history of posture and/or body-relative velocities."""
    steps=int(round(seconds*12))
    if not np.isclose(steps,seconds*12) or not 0<=steps<=CURRENT:
        raise ValueError('History must lie on the 12 Hz grid within the observed clip')
    history=np.array(scores[:,CURRENT-steps:CURRENT+1],copy=True)
    if shuffle and steps>1:
        rng=np.random.default_rng(seed)
        order=np.argsort(rng.random((len(history),steps)),axis=1)
        history[:,:-1]=np.take_along_axis(history[:,:-1],order[:,:,None],axis=1)
    if dynamics_only:
        history-=history.mean(axis=1,keepdims=True)
    return history.reshape(len(history),-1)/np.sqrt(steps+1)


def balanced_indices(mask,ant_index,cap,seed=SEED):
    rng=np.random.default_rng(seed)
    ids=np.unique(ant_index[mask])
    pools=[np.flatnonzero(mask & (ant_index==a)) for a in ids]
    count=min(cap,min(map(len,pools)))
    if count<10:raise ValueError('Too few training clips to balance all eligible ants')
    return np.concatenate([rng.choice(pool,count,replace=False) for pool in pools])


def ant_mean(values,ids):
    return np.array([np.mean(values[ids==a],axis=0) for a in np.unique(ids)])


def motif_model(x,k,seed=SEED):
    return MiniBatchKMeans(k,n_init=5,batch_size=2048,max_iter=150,random_state=seed,
                          reassignment_ratio=.005,tol=1e-5).fit(x)


def distributions(labels,ants,days,n_ants,k):
    count=np.bincount((days*n_ants+ants)*k+labels,minlength=2*n_ants*k).reshape(2,n_ants,k)
    total=count.sum(axis=-1,keepdims=True)
    return np.divide(count,total,out=np.zeros_like(count,dtype=float),where=total>0),count


def order_labels(labels,activity):
    order=sorted(np.unique(labels),key=lambda k:np.median(activity[labels==k]))
    lookup={old:new for new,old in enumerate(order)}
    return np.array([lookup[y] for y in labels]),order


def cluster_ant_profiles(probabilities,activity,*,bootstrap=100,seed=SEED):
    """Compare K=2–6 and resample whole ants, never frames or snippets."""
    from analysis.colony_behavioral_landscape import bootstrap_candidates
    x=np.sqrt(probabilities)
    candidates,fits,agreement=bootstrap_candidates(x,bootstrap,seed)
    eligible=candidates[(candidates.smallest_group>=4)&(candidates.bootstrap_ari_median>=.8)]
    chosen=1 if eligible.empty else int(eligible.loc[eligible.silhouette>=eligible.silhouette.max()-.02,'k'].min())
    labels,order=order_labels(fits[2].labels_,activity)
    centers=fits[2].cluster_centers_[order]
    return dict(candidates=candidates,selected_k=chosen,labels=labels,centers=centers,
                oob_agreement=agreement[2][0],oob_trials=agreement[2][1])


def load_pose_cache(tasks_path,cache,output):
    tasks=json.loads(tasks_path.read_text());qc=[];pieces=[];meta=[];provenance=[stamp(tasks_path)]
    for task in tasks:
        path=cache/(task['ant'].replace(':','_')+'.npz')
        saved=json.loads(path.with_suffix('.json').read_text())
        if saved['signature'].get('feature_version')!='posture_velocity_v2':
            raise ValueError(f'Velocity extraction is required: {path}')
        if saved['signature']['task']!=task or stamp(task['source']['path'])!=task['source']:
            raise ValueError(f'Stale or mismatched pose cache: {path}')
        actual=stamp(path)
        if (actual['size'],actual['mtime_ns'])!=(saved['output']['size'],saved['output']['mtime_ns']):
            raise ValueError(f'Changed pose cache: {path}')
        with np.load(path) as z:
            raw=z['shape'];lengths=z['lengths_mm'];cameras=z['cameras'];starts=z['frames']
            keep,flags=clip_quality(raw,lengths,cameras,z['duplicate_frame'])
            velocity=z['velocity_mm_s'];position_camera=z['position_cameras']
            velocity_valid=np.isfinite(velocity).all(axis=(1,2))&(np.linalg.norm(velocity,axis=-1)<=MAX_VELOCITY_MM_S).all(axis=1)
            camera_valid=np.isfinite(position_camera).all(axis=1)&(position_camera>=0).all(axis=1)&(np.ptp(np.nan_to_num(position_camera,nan=-1),axis=1)==0)
            original_keep=keep.copy();keep&=velocity_valid&camera_valid
            accepted=causal_pose(raw[keep])
            indices=np.flatnonzero(keep)
            after=np.isfinite(accepted).all(axis=(1,2));accepted=accepted[after];indices=indices[after]
            day=((starts-DAY1_FRAME)//DAY_FRAMES).astype(int)
            minute=((starts-DAY1_FRAME)//(60*FPS)).astype(int)
            row=dict(ant=task['ant'],side=task['side'],track_id=task['track_id'],track_name=task['track_name'],
                     detection_fraction=task['detection_fraction'],requested=len(raw),
                     complete_clips=int(flags['complete'].sum()),geometry_pass=int(flags['geometry'].sum()),
                     camera_pass=int(flags['same_camera'].sum()),duplicate_clips=int(flags['duplicate'].sum()),accepted=len(indices))
            row.update(posture_quality_clips=int(original_keep.sum()),velocity_pass=int(velocity_valid.sum()),position_camera_pass=int(camera_valid.sum()))
            for d in (0,1):
                valid=indices[day[indices]==d]
                row[f'day{d+1}_clips']=len(valid)
                row[f'day{d+1}_halfhours']=len(np.unique(minute[valid]//30))
                row[f'day{d+1}_eligible']=len(valid)>=240 and row[f'day{d+1}_halfhours']>=24
            qc.append(row)
            if row['day1_eligible']:
                index=len(meta);meta.append(row)
                pieces.append(dict(shape=accepted,ant_index=np.full(len(indices),index,np.int32),
                                   velocity=velocity[indices],
                                   day=day[indices],minute=minute[indices],frames=starts[indices],
                                   camera=cameras[indices,0].astype(int),
                                   lengths=np.nanmedian(lengths[indices[day[indices]==0]],axis=(0,1))))
        provenance.extend([actual,stamp(path.with_suffix('.json')),task['source']])
        print('POSE_QC',task['ant'],row['day1_clips'],row['day2_clips'],row['day1_eligible'],flush=True)
    pd.DataFrame(qc).to_csv(output/'pose_quality.csv',index=False)
    ants=pd.DataFrame(meta)
    if any((ants.side==s).sum()<12 for s in ('left','right')):raise ValueError('Fewer than 12 eligible ants in a colony')
    bundle={key:np.concatenate([p[key] for p in pieces],axis=0) for key in ('shape','velocity','ant_index','day','minute','frames','camera')}
    bundle['lengths']=np.median(np.stack([p['lengths'] for p in pieces]),axis=0)
    return ants,bundle,pd.DataFrame(qc),provenance


def fit_space_blind(ants,b,output,bootstrap=100):
    """Fit posture and velocity without occupancy maps, location or labels."""
    shape,ids,day=b['shape'],b['ant_index'],b['day']
    train=(day==0)&((b['minute']//60)%4!=3)
    valid=(day==0)&~train
    balanced=balanced_indices(train,ids,180)
    pca_validation=PCA(svd_solver='full').fit(shape[balanced,::4].reshape(-1,16))
    rank=int(np.searchsorted(np.cumsum(pca_validation.explained_variance_ratio_),.95)+1)
    scores=pca_validation.transform(shape.reshape(-1,16)).reshape(len(shape),28,16)[:,:,:rank].astype(np.float32)
    state,validation_normalizer=joint_state(scores,b['velocity'],balanced)
    target=state[:,FUTURE];current=state[:,CURRENT]
    target_mean=target[balanced].mean(axis=0)
    base=ant_mean(np.square(target[valid]-target_mean).sum(axis=1),ids[valid])
    persistence=ant_mean(np.square(target[valid]-current[valid]).sum(axis=1),ids[valid])
    history_rows=[];errors={}
    for h in (0.,.25,.5,1.,2.):
        for shuffled in ((False,True) if h>0 else (False,)):
            x=history_features(state,h,shuffle=shuffled)
            ridge=Ridge(alpha=1.).fit(x[balanced],target[balanced])
            squared=np.square(ridge.predict(x[valid])-target[valid])
            loss=ant_mean(squared.sum(axis=1),ids[valid])
            errors[f'{h:g}_{shuffled}']=loss
            history_rows.append(dict(history_seconds=h,shuffled=shuffled,mean_mse=loss.mean(),
                                     standard_error=loss.std(ddof=1)/np.sqrt(len(loss)),
                                     explained_vs_mean=1-loss.mean()/base.mean(),
                                     gain_vs_persistence=1-loss.mean()/persistence.mean(),n_ants=len(loss),
                                     posture_block_mse=ant_mean(squared[:,:rank].sum(axis=1),ids[valid]).mean(),
                                     velocity_block_mse=ant_mean(squared[:,rank:].sum(axis=1),ids[valid]).mean()))
    history_table=pd.DataFrame(history_rows)
    history_table.to_csv(output/'history_prediction_validation.csv',index=False)
    x=history_features(state,1.)
    motif_rows=[]
    for k in (12,24,48):
        model=motif_model(x[balanced],k)
        future=np.stack([target[balanced][model.labels_==j].mean(axis=0) if (model.labels_==j).any() else target_mean for j in range(k)])
        loss=ant_mean(np.square(future[model.predict(x[valid])]-target[valid]).sum(axis=1),ids[valid])
        motif_rows.append(dict(motifs=k,mean_mse=loss.mean(),standard_error=loss.std(ddof=1)/np.sqrt(len(loss))))
    resolution=pd.DataFrame(motif_rows);best=resolution.loc[resolution.mean_mse.idxmin()]
    n_motifs=int(resolution.loc[resolution.mean_mse<=best.mean_mse+best.standard_error,'motifs'].min())
    resolution.to_csv(output/'motif_resolution_validation.csv',index=False)
    # Refit on all of day 1 only, after internal validation fixes the resolution.
    balanced=balanced_indices(day==0,ids,240)
    pca=PCA(svd_solver='full').fit(shape[balanced,::4].reshape(-1,16))
    rank=int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_),.95)+1)
    scores=pca.transform(shape.reshape(-1,16)).reshape(len(shape),28,16)[:,:,:rank].astype(np.float32)
    state,normalizer=joint_state(scores,b['velocity'],balanced)
    primary=history_features(state,1.)
    variants={PRIMARY:primary,'instantaneous':history_features(state,0.),
              'posture_history':history_features(state[:,:,:rank],1.),
              'velocity_history':history_features(state[:,:,rank:],1.),
              'dynamics_only':history_features(state,1.,dynamics_only=True)}
    for weight,name in ((.5,'velocity_half'),(2.,'velocity_double')):
        weighted=state.copy();weighted[:,:,rank:]*=weight
        variants[name]=history_features(weighted,1.)
    centered=primary.copy();global_mean=primary[balanced].mean(axis=0)
    camera_means={}
    for camera in np.unique(b['camera']):
        cm=balanced[b['camera'][balanced]==camera]
        mean=primary[cm].mean(axis=0) if len(cm)>=10 else global_mean
        camera_means[int(camera)]=mean
        centered[b['camera']==camera]+=global_mean-mean
    variants['camera_centered']=centered
    angles=np.arctan2(shape.reshape(len(shape),28,8,2)[...,1],shape.reshape(len(shape),28,8,2)[...,0])
    differences=np.angle(np.exp(1j*np.diff(angles[:,:CURRENT+1],axis=1)))
    clip_motion=np.sqrt(np.square(differences).mean(axis=(1,2)))*12
    activity=np.array([clip_motion[(ids==i)&(day==0)].mean() for i in range(len(ants))])
    ants=ants.copy();ants['angular_motion_rad_s']=activity
    for j,name in enumerate(('forward_velocity_mm_s','lateral_velocity_mm_s')):
        ants[name]=[b['velocity'][(ids==i)&(day==0),:CURRENT+1,j].mean() for i in range(len(ants))]
    ants['sampled_speed_mm_s']=[np.linalg.norm(b['velocity'][(ids==i)&(day==0),:CURRENT+1],axis=-1).mean() for i in range(len(ants))]
    results={};model_arrays={};profile_arrays={};motif_labels={};secondary_rows=[]
    for name,x in variants.items():
        model=motif_model(x[balanced],n_motifs)
        assigned=model.predict(x)
        prob,count=distributions(assigned,ids,day,len(ants),n_motifs)
        profile_arrays[name]=prob;motif_labels[name]=assigned
        model_arrays[name+'_centers']=model.cluster_centers_
        if name==PRIMARY:
            examples=[]
            training=x[balanced]
            for motif in range(n_motifs):
                candidates=np.flatnonzero(model.labels_==motif)
                if not len(candidates):candidates=np.arange(len(balanced))
                distance=np.square(training[candidates]-model.cluster_centers_[motif]).sum(axis=1)
                examples.append(balanced[candidates[distance.argmin()]])
            model_arrays['motif_example_shape']=shape[examples]
            model_arrays['motif_example_velocity_mm_s']=b['velocity'][examples]
            centers=model.cluster_centers_.reshape(n_motifs,13,rank+2)*np.sqrt(13)
            model_arrays['motif_center_velocity_mm_s']=centers[:,:,rank:]*(normalizer['velocity_feature_std']*np.sqrt(2))+normalizer['velocity_feature_mean']
            model_arrays['motif_example_ant']=ids[examples]
            model_arrays['motif_example_frames']=b['frames'][examples]
            for side in ('left','right'):
                rows=np.flatnonzero(ants.side.eq(side).to_numpy())
                r=cluster_ant_profiles(prob[0,rows],activity[rows],bootstrap=bootstrap)
                r.update(rows=rows,profile=prob[:,rows],counts=count[:,rows])
                r['candidates'].assign(side=side).to_csv(output/f'{side}_ant_model_selection.csv',index=False)
                table=ants.iloc[rows].copy();table['group']=r['labels'];table['oob_agreement']=r['oob_agreement'];table['oob_trials']=r['oob_trials']
                z=PCA(svd_solver='full').fit(np.sqrt(prob[0,rows]))
                table['pc1'],table['pc2']=z.transform(np.sqrt(prob[0,rows]))[:,:2].T
                table['day2_group']=cdist(np.sqrt(prob[1,rows]),r['centers']).argmin(axis=1)
                table.loc[~table.day2_eligible,'day2_group']=-1
                paired=table.day2_eligible.to_numpy()
                r['table']=table;r['profile_pca']=z
                r['repeat_fraction']=float((table.loc[paired,'group']==table.loc[paired,'day2_group']).mean())
                r['repeat_n']=int(paired.sum())
                # Compare discrete prototypes with continuous posture-profile modes.
                train_features=np.sqrt(prob[0,rows]);test_features=np.sqrt(prob[1,rows])[paired]
                mean=train_features.mean(axis=0)
                base_loss=np.square(test_features-mean).sum(axis=1).mean()
                prediction_rows=[]
                for k in (1,2,3,4,5,6):
                    km=KMeans(k,n_init=30,random_state=SEED).fit(train_features)
                    pred=km.cluster_centers_[km.labels_][paired]
                    loss=np.square(test_features-pred).sum(axis=1).mean()
                    prediction_rows.append(dict(model=f'K={k}',complexity=k,mean_mse=loss,explained_vs_mean=1-loss/base_loss,n_ants=int(paired.sum())))
                for k in (1,2):
                    projection=z.transform(train_features)[:,:k]@z.components_[:k]+z.mean_
                    loss=np.square(test_features-projection[paired]).sum(axis=1).mean()
                    prediction_rows.append(dict(model=f'PC{k}',complexity=k,mean_mse=loss,explained_vs_mean=1-loss/base_loss,n_ants=int(paired.sum())))
                r['profile_validation']=pd.DataFrame(prediction_rows)
                r['profile_validation'].to_csv(output/f'{side}_heldout_profile_prediction.csv',index=False)
                results[side]=r
                model_arrays[side+'_ant_centers']=r['centers']
                model_arrays[side+'_profile_pca_mean']=z.mean_;model_arrays[side+'_profile_pca_components']=z.components_
                model_arrays[side+'_rows']=rows
                table.to_csv(output/f'{side}_posture_groups.csv',index=False)
        for side in ('left','right'):
            rows=np.flatnonzero(ants.side.eq(side).to_numpy());features=np.sqrt(prob[0,rows])
            if name==PRIMARY:labels=results[side]['labels']
            else:
                fit=KMeans(2,n_init=40,random_state=SEED).fit(features)
                labels,order=order_labels(fit.labels_,activity[rows])
            secondary_rows.extend(dict(side=side,ant=ants.iloc[i].ant,representation=name,group=int(labels[j])) for j,i in enumerate(rows))
    # Camera-only profiles are a diagnostic comparator, never a posture input.
    cameras=np.unique(b['camera']);camera_index=np.searchsorted(cameras,b['camera'])
    camera_profiles,_=distributions(camera_index,ids,day,len(ants),len(cameras))
    profile_arrays['camera_only']=camera_profiles
    for side in ('left','right'):
        rows=results[side]['rows'];fit=KMeans(2,n_init=40,random_state=SEED).fit(np.sqrt(camera_profiles[0,rows]))
        labels,_=order_labels(fit.labels_,activity[rows])
        secondary_rows.extend(dict(side=side,ant=ants.iloc[i].ant,representation='camera_only',group=int(labels[j])) for j,i in enumerate(rows))
    secondary=pd.DataFrame(secondary_rows);secondary.to_csv(output/'representation_groups.csv',index=False)
    model_arrays.update(posture_mean=pca.mean_,posture_components=pca.components_,posture_variance=pca.explained_variance_,
                        posture_variance_ratio=pca.explained_variance_ratio_,rank=np.array(rank),n_motifs=np.array(n_motifs),
                        standard_lengths_mm=b['lengths'],ants=ants.ant.to_numpy(str),camera_ids=np.array(list(camera_means)),camera_feature_means=np.stack(list(camera_means.values())))
    model_arrays.update(normalizer,feature_version=np.array('posture_velocity_v2'))
    pd.DataFrame([dict(feature=f'posture_PC{j+1}',center=float(normalizer['posture_feature_mean'][j]),divisor=float(normalizer['posture_feature_scale']),units='direction-cosine coefficient') for j in range(rank)]+
      [dict(feature=n,center=float(normalizer['velocity_feature_mean'][j]),divisor=float(normalizer['velocity_feature_std'][j]*np.sqrt(2)),units='mm/s') for j,n in enumerate(('forward_velocity','lateral_velocity'))]).to_csv(output/'feature_scaling.csv',index=False)
    np.savez_compressed(output/'space_blind_models.npz',**model_arrays)
    np.savez_compressed(output/'ant_motif_profiles.npz',**profile_arrays)
    # Half-hour profiles keep actual elapsed time and sample counts, never fill gaps.
    halfhour=b['minute']//30;temporal=[]
    assigned=motif_labels[PRIMARY]
    for side,r in results.items():
        for row_index,i in enumerate(r['rows']):
            for slot in range(96):
                mask=(ids==i)&(halfhour==slot);n=int(mask.sum())
                if not n:continue
                counts=np.bincount(assigned[mask],minlength=n_motifs);profile=counts/n
                coord=r['profile_pca'].transform(np.sqrt(profile)[None])[0]
                group=int(cdist(np.sqrt(profile)[None],r['centers']).argmin())
                temporal.append(dict(ant=ants.iloc[i].ant,side=side,halfhour=slot,day=slot//48,n_clips=n,
                                     pc1=coord[0],pc2=coord[1],group=group,angular_motion_rad_s=float(clip_motion[mask].mean())))
    pd.DataFrame(temporal).to_csv(output/'temporal_posture_profiles.csv',index=False)
    # Equal-ant reconstruction error on held-out day 2, no basis refitting.
    recon=[]
    for d in (0,1):
        idx=balanced_indices(day==d,ids,150,SEED+d)
        observations=shape[idx,::4].reshape(-1,16);zs=pca.transform(observations)
        baseline=np.square(observations-pca.mean_).sum(axis=1).mean()
        for k in range(1,17):
            residual=observations-pca.mean_-zs[:,:k]@pca.components_[:k]
            recon.append(dict(day=d+1,modes=k,explained_variance=1-np.square(residual).sum(axis=1).mean()/baseline))
    pd.DataFrame(recon).to_csv(output/'posture_reconstruction.csv',index=False)
    # Small, balanced posture sample for density plots and interactive examples.
    sample=balanced_indices(day==0,ids,80)
    np.savez_compressed(output/'posture_display_samples.npz',shape=shape[sample],scores=scores[sample],ant_index=ids[sample],
                        frames=b['frames'][sample],motion=clip_motion[sample],motif=assigned[sample],camera=b['camera'][sample],velocity_mm_s=b['velocity'][sample])
    frozen=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),occupancy_inputs_loaded=False,history_seconds=1.,
                prediction_horizon_seconds=.25,rank95=rank,n_motifs=n_motifs,n_ants=len(ants),n_clips=len(shape),
                feature_version='posture_velocity_v2',input_features='Posture PCA coefficients plus signed anterior/lateral velocity histories; location and global heading excluded',
                metric='Posture block variance 1; two standardized velocity channels with combined variance 1; statistics fit on balanced day-1 data',
                history_samples=13,feature_channels=rank+2,motif_vector_dimensions=13*(rank+2),
                model_sha256=hashlib.sha256((output/'space_blind_models.npz').read_bytes()).hexdigest(),
                groups_sha256={s:hashlib.sha256((output/f'{s}_posture_groups.csv').read_bytes()).hexdigest() for s in results},
                selected_ant_k={s:r['selected_k'] for s,r in results.items()},
                code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name('postural_dynamics_extract.py'))})
    (output/'SPACE_BLIND_FROZEN.json').write_text(json.dumps(frozen,indent=2)+'\n')
    print('SPACE_BLIND_MODELS_FROZEN',json.dumps(frozen),flush=True)
    return results,ants,dict(pca=pca,rank=rank,n_motifs=n_motifs,history_table=history_table,resolution=resolution,
                             model_arrays=model_arrays,profiles=profile_arrays,secondary=secondary,baseline_mse=base.mean(),persistence_mse=persistence.mean())


def spatial_validation(block,atom_root,output,results,ants,fit):
    """First access to spatial outcomes, after all posture fitting is frozen."""
    frozen=json.loads((output/'SPACE_BLIND_FROZEN.json').read_text())
    assert frozen['occupancy_inputs_loaded'] is False
    assert hashlib.sha256((output/'space_blind_models.npz').read_bytes()).hexdigest()==frozen['model_sha256']
    start=pd.Timestamp('2026-07-24 10:00').value//(1800*10**9)
    spatial={};provenance=[]
    for row in ants.itertuples():
        path=atom_root/(row.ant.replace(':','_')+'.npz');metadata=json.loads(path.with_suffix('.json').read_text())
        entry=metadata['signature']['entry']
        if entry['block']!=str(block) or not metadata['original_histogram_exact']:raise ValueError('Wrong spatial source')
        for item in metadata['signature']['inputs']:
            actual=stamp(item['path'])
            if (actual['size'],actual['mtime_ns'])!=(item['size'],item['mtime_ns']):raise ValueError('Stale spatial cache')
        maps=[];coverage=[]
        with np.load(path) as z:
            for d in (0,1):
                mask=(z['calendar_bin']>=start+d*48)&(z['calendar_bin']<start+(d+1)*48)
                if mask.sum()!=48:raise ValueError('Spatial time windows incomplete')
                detected=int(z['detected'][mask].sum());maps.append(z['counts'][mask].sum(axis=0)/max(1,detected));coverage.append(detected/DAY_FRAMES)
            spatial[row.ant]=dict(maps=np.stack(maps),coverage=coverage,edges=(z['x_edges'],z['y_edges']))
        provenance.extend([stamp(path),stamp(path.with_suffix('.json'))])
    label_path=block/'stitched/grid_occupancy_histograms_arena_0p25mm/track_cluster_ids.csv'
    labels=pd.read_csv(label_path);labels['ant']=labels.side+':'+labels.TrackID.astype(int).astype(str).str.zfill(3)
    label_lookup=labels.set_index('ant').cluster_id
    behavior_path=block/'analysis_outputs/long_timescale_0723_0724_0729_20260915/task_bins.parquet'
    behavior=pd.read_parquet(behavior_path);behavior=behavior[behavior.source_block.eq(str(block))&behavior.in_recording_bin].copy()
    behavior['day']=((behavior.timestamp-pd.Timestamp('2026-07-24 10:00')).dt.total_seconds()//86400).astype(int)
    behavior=behavior[behavior.day.isin([0,1])]
    provenance.extend([stamp(label_path),stamp(behavior_path),stamp(block/'panorama_regions.csv')])
    rng=np.random.default_rng(SEED);controls=[];forecast_rows=[];summary={}
    from analysis.colony_behavioral_landscape import features
    for side,r in results.items():
        t=r['table'].copy();t['spatial_label']=t.ant.map(label_lookup)
        t['position_coverage_day1']=[spatial[a]['coverage'][0] for a in t.ant]
        t['position_coverage_day2']=[spatial[a]['coverage'][1] for a in t.ant]
        for day in (0,1):
            for metric,coverage in [('colony_percent','position_coverage'),('mean_speed_mm_s','speed_coverage'),('sleep_percent','sleep_coverage')]:
                values={}
                for ant,part in behavior[behavior.day.eq(day)].groupby('ant'):
                    w=part.n_expected_frames*part[coverage];valid=part[metric].notna()&w.gt(0)
                    values[ant]=float(np.average(part.loc[valid,metric],weights=w[valid])) if valid.any() else np.nan
                t[f'day{day+1}_{metric}']=t.ant.map(values)
        primary_ari=adjusted_rand_score(t.group,t.spatial_label)
        null=np.array([adjusted_rand_score(rng.permutation(t.group),t.spatial_label) for _ in range(1000)])
        for name,part in fit['secondary'][fit['secondary'].side.eq(side)].groupby('representation'):
            joined=t[['ant','group','spatial_label']].merge(part[['ant','group']],on='ant',suffixes=('_primary','_variant'),validate='one_to_one')
            controls.append(dict(side=side,representation=name,agreement_with_spatial=adjusted_rand_score(joined.group_variant,joined.spatial_label),
                                 agreement_with_primary=adjusted_rand_score(joined.group_variant,joined.group_primary)))
        maps=np.stack([spatial[a]['maps'] for a in t.ant])
        good=t.position_coverage_day1.ge(.4)&t.position_coverage_day2.ge(.4)
        ix=np.flatnonzero(good);a1=maps[ix,0];a2=maps[ix,1];g=t.group.to_numpy()[ix]
        def loss_for_groups(groups):
            predicted=[]
            for i in range(len(ix)):
                other=(np.arange(len(ix))!=i)&(groups==groups[i])
                if not other.any():other=np.arange(len(ix))!=i
                predicted.append(a1[other].mean(axis=0))
            return np.square(features(np.stack(predicted))-features(a2)).sum(axis=1)/2
        baseline=loss_for_groups(np.zeros(len(ix),int));cluster=loss_for_groups(g)
        persistence=np.square(features(a1)-features(a2)).sum(axis=1)/2
        gain=baseline-cluster
        boot=np.array([rng.choice(gain,len(gain),replace=True).mean() for _ in range(2000)])
        perm=np.array([(baseline-loss_for_groups(rng.permutation(g))).mean() for _ in range(1000)])
        for j,i in enumerate(ix):forecast_rows.append(dict(side=side,ant=t.iloc[i].ant,baseline_loss=baseline[j],posture_group_loss=cluster[j],own_previous_day_loss=persistence[j],gain=gain[j]))
        r.update(table=t,spatial_maps=maps,spatial_edges=spatial[t.ant.iloc[0]]['edges'],forecast_gain=gain)
        t.to_csv(output/f'{side}_groups_with_spatial_outcomes.csv',index=False)
        summary[side]=dict(n_ants=len(t),groups=t.group.value_counts().sort_index().to_dict(),selected_k=r['selected_k'],
                           spatial_ari=primary_ari,spatial_permutation_p=float((1+(null>=primary_ari).sum())/1001),
                           day2_assignment_retention=r['repeat_fraction'],day2_assignment_n=r['repeat_n'],
                           forecast_n=len(ix),mean_spatial_gain=float(gain.mean()),spatial_gain_ci=np.quantile(boot,[.025,.975]).tolist(),
                           forecast_permutation_p=float((1+(perm>=gain.mean()).sum())/1001),
                           baseline_spatial_loss=float(baseline.mean()),group_spatial_loss=float(cluster.mean()),own_previous_day_loss=float(persistence.mean()))
        print('POSTHOC_SPATIAL_RESULT',side,json.dumps(summary[side]),flush=True)
    pd.DataFrame(controls).to_csv(output/'representation_spatial_comparison.csv',index=False)
    pd.DataFrame(forecast_rows).to_csv(output/'heldout_spatial_forecasts.csv',index=False)
    np.savez_compressed(output/'heldout_spatial_maps.npz',**{s:results[s]['spatial_maps'] for s in results})
    (output/'results_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary,provenance,pd.DataFrame(controls)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks',type=Path,required=True);p.add_argument('--pose-cache',type=Path,required=True)
    p.add_argument('--block',type=Path,required=True);p.add_argument('--spatial-atoms',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--bootstrap',type=int,default=100)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.block.parent.name!='20260724' or args.block.name!='block01':raise ValueError('Wrong recording')
    (args.output/'COMPLETE.json').unlink(missing_ok=True)
    ants,b,qc,provenance=load_pose_cache(args.tasks,args.pose_cache,args.output)
    results,ants,fit=fit_space_blind(ants,b,args.output,args.bootstrap)
    summary,spatial_sources,controls=spatial_validation(args.block,args.spatial_atoms,args.output,results,ants,fit)
    from analysis.postural_dynamics_plots import render
    render(args.output,results,ants,b,qc,fit,summary,controls,args.block)
    provenance.extend(spatial_sources)
    if any(stamp(item['path'])!=item for item in provenance):raise ValueError('Input changed during analysis')
    manifest=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  summary=summary,source_fingerprints=provenance,
                  software={k:importlib.metadata.version(k) for k in ('numpy','pandas','scipy','scikit-learn','matplotlib','pyarrow')},
                  code_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('postural_dynamics*') if f.is_file()})
    (args.output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (args.output/'COMPLETE.json').write_text(json.dumps(dict(complete=True,source=str(args.block)))+'\n')


if __name__=='__main__':main()
