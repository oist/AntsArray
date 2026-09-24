"""Exploratory, activity-only representation search with post hoc spatial tests."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist,pdist
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score,silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.manifold import trustworthiness

from analysis.postural_dynamics import CURRENT,PRIMARY,SEED,joint_state,history_features,balanced_indices
from analysis.postural_dynamics_extract import stamp
from analysis.postural_dynamics_hourly import load_all
from analysis.activity_discovery_features import SPEED_NAMES,VIEWS,aggregate_clips,hourly_quantiles

FAMILY_LABELS={
 'motif_clock':'Denoised hourly motifs',
 'motif_repertoire':'Distribution of hourly motif frequencies',
 'posture_shape':'Hourly posture means and amplitudes',
 'short_dynamics':'Short-time postural and velocity dynamics',
 'locomotion_levels':'Full-trajectory locomotor levels',
 'locomotion_bouts':'Rest-like and movement run structure',
 'locomotion_clock':'Hourly locomotor timing',
 'posture_locomotion':'Posture + dynamics + locomotion',
 'motif_locomotion':'Motif repertoire + locomotion',
 'locomotion_all':'Locomotor levels + runs + timing'}
SPEED_FAMILIES=('locomotion_levels','locomotion_bouts','locomotion_clock','locomotion_all')


class ActivityTransform:
    """Training-only median imputation, robust block scaling, then <=3 PCs."""
    def fit(self,x,blocks):
        x=np.asarray(x,float)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            self.center=np.nanmedian(x,axis=0);q=np.nanquantile(x,[.25,.75],axis=0)
            sd=np.nanstd(x,axis=0)
        self.keep=np.isfinite(self.center)&(sd>1e-7)
        if not self.keep.any():raise ValueError('No varying activity features')
        self.center=self.center[self.keep];scale=(q[1]-q[0])[self.keep]
        self.scale=np.where(scale>1e-6,scale,sd[self.keep])
        block=np.asarray(blocks)[self.keep]
        self.block_scale=np.sqrt(np.array([(block==g).sum() for g in block]))
        z=self._scaled(x)
        self.pca=PCA(n_components=min(3,len(x)-1,z.shape[1]),svd_solver='full').fit(z)
        return self
    def _scaled(self,x):
        x=np.asarray(x,float)[:,self.keep];x=np.where(np.isfinite(x),x,self.center)
        return np.clip((x-self.center)/self.scale,-6,6)/self.block_scale
    def transform(self,x):return self.pca.transform(self._scaled(x))


def model_fit(x,k,algorithm,covariance='full',seed=SEED,n_init=12):
    if algorithm=='gmm':
        fits=[GaussianMixture(k,covariance_type=covariance,reg_covar=1e-4,n_init=n_init,max_iter=500,random_state=seed).fit(x)]
        # A low-variance bimodal axis can be missed by KMeans initialization.
        # Add label-free quantile starts along every retained activity PC.
        if k>1:
            for axis in range(x.shape[1]):
                groups=np.array_split(np.argsort(x[:,axis]),k)
                weights=np.array([len(g)/len(x) for g in groups]);mu=np.stack([x[g].mean(axis=0) for g in groups])
                covariance_full=np.stack([(x[g]-mu[j]).T@(x[g]-mu[j])/len(g)+np.eye(x.shape[1])*1e-4 for j,g in enumerate(groups)])
                if covariance=='full':precision=np.linalg.inv(covariance_full)
                elif covariance=='diag':precision=1/np.diagonal(covariance_full,axis1=1,axis2=2)
                else:precision=np.linalg.inv(np.sum(covariance_full*weights[:,None,None],axis=0))
                fits.append(GaussianMixture(k,covariance_type=covariance,reg_covar=1e-4,n_init=1,max_iter=500,random_state=seed,init_params='random',means_init=mu,weights_init=weights,precisions_init=precision).fit(x))
        converged=[f for f in fits if f.converged_]
        if not converged:raise RuntimeError('No converged Gaussian-mixture start')
        return max(converged,key=lambda f:f.lower_bound_)
    return KMeans(k,n_init=n_init,max_iter=300,random_state=seed).fit(x)


def centers(model):return model.means_ if hasattr(model,'means_') else model.cluster_centers_


def measurement_replication(raw,blocks,k,algorithm,covariance):
    """Fit A and predict B, then reverse. Neither fold fits the other fold."""
    gain=[];retention=[]
    for a,b in ((1,2),(2,1)):
        tr=ActivityTransform().fit(raw[a],blocks);x=tr.transform(raw[a]);y=tr.transform(raw[b])
        fit=model_fit(x,k,algorithm,covariance,n_init=15)
        labels=fit.predict(x);pred=centers(fit)[labels]
        base=np.square(y-x.mean(axis=0)).sum(axis=1).mean();loss=np.square(y-pred).sum(axis=1).mean()
        gain.append(1-loss/base if base>0 else 0.);retention.append(float(np.mean(labels==fit.predict(y))))
    return float(np.mean(gain)),float(np.mean(retention))


def bootstrap_stability(raw,blocks,k,algorithm,covariance,reference,repeats,seed=SEED):
    rng=np.random.default_rng(seed);ari=[]
    for b in range(repeats):
        index=rng.integers(len(raw),size=len(raw));tr=ActivityTransform().fit(raw[index],blocks)
        fit=model_fit(tr.transform(raw[index]),k,algorithm,covariance,seed+b+1,n_init=3)
        ari.append(adjusted_rand_score(reference,fit.predict(tr.transform(raw))))
    return np.asarray(ari)


def evaluate(raw,blocks,algorithm,bootstrap=100):
    tr=ActivityTransform().fit(raw[0],blocks);x=tr.transform(raw[0]);candidate_rows=[]
    if algorithm=='gmm':
        fits=[]
        for cov in ('tied','diag','full'):
            for k in range(1,5):
                fit=model_fit(x,k,algorithm,cov);fits.append((fit.bic(x),k,cov,fit))
                candidate_rows.append(dict(k=k,covariance=cov,bic=float(fit.bic(x))))
        fits.sort(key=lambda a:a[0]);best=fits[0];bic1=min(t[0] for t in fits if t[1]==1)
        proposed=best if best[1]>1 and bic1-best[0]>=10 else min((t for t in fits if t[1]==1),key=lambda t:t[0])
        _,k,cov,fit=proposed;labels=fit.predict(x);sizes=np.bincount(labels,minlength=k)
        if k>1:
            ari=bootstrap_stability(raw[0],blocks,k,algorithm,cov,labels,bootstrap)
            gain,retention=measurement_replication(raw,blocks,k,algorithm,cov)
            sil=silhouette_score(x,labels) if len(np.unique(labels))>1 else np.nan
            supported=sizes.min()>=4 and np.median(ari)>=.8 and retention>=.8
        else:ari=np.array([np.nan]);gain=0.;retention=np.nan;sil=np.nan;supported=False
        detail=dict(candidate_k=k,covariance=cov,bic_gain=float(bic1-best[0]),candidate_gain=gain,replicate_retention=retention,bootstrap_ari_median=float(np.nanmedian(ari)) if k>1 else np.nan,bootstrap_ari_p10=float(np.nanquantile(ari,.1)) if k>1 else np.nan,silhouette=sil,smallest_group=int(sizes.min()))
        selected=k if supported else 1
    else:
        options=[]
        for k in range(2,5):
            fit=model_fit(x,k,'kmeans',n_init=30);labels=fit.predict(x);sizes=np.bincount(labels,minlength=k)
            sil=silhouette_score(x,labels) if len(np.unique(labels))>1 else np.nan
            if sizes.min()>=4 and sil>=.25:
                ari=bootstrap_stability(raw[0],blocks,k,'kmeans','full',labels,bootstrap)
                gain,retention=measurement_replication(raw,blocks,k,'kmeans','full')
            else:ari=np.array([np.nan]);gain=np.nan;retention=np.nan
            med=float(np.nanmedian(ari)) if np.isfinite(ari).any() else np.nan
            detail=dict(candidate_k=k,covariance='none',bic_gain=np.nan,candidate_gain=gain,replicate_retention=retention,bootstrap_ari_median=med,bootstrap_ari_p10=float(np.nanquantile(ari,.1)) if np.isfinite(ari).any() else np.nan,silhouette=sil,smallest_group=int(sizes.min()))
            candidate_rows.append(dict(k=k,**detail))
            if sizes.min()>=4 and sil>=.25 and med>=.8 and retention>=.8:options.append((k,fit,detail))
        if options:
            highest=max(o[2]['silhouette'] for o in options);k,fit,detail=min((o for o in options if o[2]['silhouette']>=highest-.02),key=lambda a:a[0]);selected=k;cov='none'
        else:
            detail=max(candidate_rows,key=lambda d:d['silhouette'] if np.isfinite(d['silhouette']) else -np.inf).copy();detail.pop('k',None);selected=1;cov='none'
    if selected==1:fit=model_fit(x,1,algorithm,'full',n_init=1)
    labels=fit.predict(x);test=tr.transform(raw[3]);test_labels=fit.predict(test)
    means=centers(fit);base=np.square(test-x.mean(axis=0)).sum(axis=1);loss=np.square(test-means[labels]).sum(axis=1)
    model=dict(transform=tr,model=fit,labels=labels,day2_labels=test_labels,coordinates=x,day2_coordinates=test,day2_loss=loss,day2_baseline_loss=base)
    detail.update(selected_k=selected,activity_gain=float(detail['candidate_gain']) if selected>1 else 0.,dimensions=x.shape[1],pca_variance=float(tr.pca.explained_variance_ratio_.sum()))
    return model,detail,candidate_rows


def assemble_bank(tasks,pose_cache,speed_cache,dictionary,output):
    qc,b,provenance=load_all(tasks,pose_cache,output);n=len(qc)
    with np.load(dictionary/'space_blind_models.npz') as z:m={k:z[k] for k in z.files}
    freeze=json.loads((dictionary/'SPACE_BLIND_FROZEN.json').read_text())
    if hashlib.sha256((dictionary/'space_blind_models.npz').read_bytes()).hexdigest()!=freeze['model_sha256']:raise ValueError('Dictionary changed')
    scores=((b['shape']-m['posture_mean'])@m['posture_components'].T)[:,:,:int(m['rank'])]
    state,_=joint_state(scores,b['velocity'],[],normalizer={k:m[k] for k in ('posture_feature_mean','posture_feature_scale','velocity_feature_mean','velocity_feature_std')})
    histories=history_features(state,1.);assigned=cdist(histories,m[PRIMARY+'_centers'],metric='sqeuclidean').argmin(axis=1)
    motif,nclips=aggregate_clips(np.eye(int(m['n_motifs']))[assigned],b['ant_index'],b['minute'],n)
    posture=np.concatenate([scores[:,12:25].mean(axis=1),scores[:,12:25].std(axis=1)],axis=1)
    angles=np.arctan2(b['shape'].reshape(-1,28,8,2)[...,1],b['shape'].reshape(-1,28,8,2)[...,0])
    omega=np.sqrt(np.square(np.angle(np.exp(1j*np.diff(angles[:,12:25],axis=1)))).mean(axis=1))*12
    velocity=b['velocity'][:,12:25];speed=np.linalg.norm(velocity,axis=-1)
    dynamics=np.column_stack([np.log1p(omega),np.arcsinh(velocity[:,:,0].mean(axis=1)/.1),np.log1p(abs(velocity[:,:,1]).mean(axis=1)/.1),np.log1p(speed.mean(axis=1)/.1),np.log1p(speed.std(axis=1)/.1),(speed>.05).mean(axis=1)])
    pose_hour,_=aggregate_clips(posture,b['ant_index'],b['minute'],n);dyn_hour,_=aggregate_clips(dynamics,b['ant_index'],b['minute'],n)
    speed_hour=[];coverage=[]
    for row in qc.itertuples():
        p=speed_cache/(row.ant.replace(':','_')+'.npz');metadata=json.loads(p.with_suffix('.json').read_text())
        if any(stamp(item['path'])!=item for item in metadata['signature']['sources']):raise ValueError('Speed source changed')
        with np.load(p) as z:speed_hour.append(z['hourly']);coverage.append(z['coverage'])
        provenance += [stamp(p),stamp(p.with_suffix('.json')),*metadata['signature']['sources']]
    speed_hour=np.stack(speed_hour,axis=1);coverage=np.stack(coverage,axis=1)
    qc['speed_day1_hours']=np.isfinite(speed_hour[0]).all(axis=-1).sum(axis=-1)
    qc['speed_day2_hours']=np.isfinite(speed_hour[3]).all(axis=-1).sum(axis=-1)
    qc['speed_eligible']=qc.speed_day1_hours>=16;qc['common_eligible']=qc.day1_eligible&qc.speed_eligible
    # Reserve day-2 quality flags for validation; never select training ants by it.
    qc['common_day2_eligible']=qc.day2_eligible&(qc.speed_day2_hours>=16)
    qc['speed_day1_coverage']=coverage[0].mean(axis=-1);qc['speed_day2_coverage']=coverage[3].mean(axis=-1)
    qc.to_csv(output/'activity_cohort_audit.csv',index=False)
    def names(base):return [f'{stat}_{name}' for stat in ('q25','median','q75') for name in base]
    blocks={};definitions={};bank={}
    def add(name,array,feature_names,block=None):
        bank[name]=array;definitions[name]=feature_names;blocks[name]=np.zeros(array.shape[-1],int) if block is None else np.asarray(block)
    add('motif_clock',np.sqrt(motif).reshape(4,n,-1),[f'hour{h:02d}_motif{j:02d}' for h in range(24) for j in range(int(m['n_motifs']))])
    add('motif_repertoire',hourly_quantiles(np.sqrt(motif)),names([f'sqrt_motif{j:02d}' for j in range(int(m['n_motifs']))]))
    add('posture_shape',hourly_quantiles(pose_hour),names([f'pc{j+1}_{s}' for s in ('mean','sd') for j in range(int(m['rank']))]))
    add('short_dynamics',hourly_quantiles(dyn_hour),names([f'angular_motion_{j}' for j in range(8)]+['signed_forward','abs_lateral','speed','speed_sd','moving_fraction']))
    add('locomotion_levels',hourly_quantiles(speed_hour[...,:7]),names(SPEED_NAMES[:7]))
    add('locomotion_bouts',hourly_quantiles(speed_hour[...,7:]),names(SPEED_NAMES[7:]))
    add('locomotion_clock',speed_hour[...,[0,5]].reshape(4,n,-1),[f'hour{h:02d}_{s}' for h in range(24) for s in ('log_mean_speed','fraction_above_02')])
    for name,parts in [('posture_locomotion',['posture_shape','short_dynamics','locomotion_levels','locomotion_bouts']),('motif_locomotion',['motif_repertoire','locomotion_levels','locomotion_bouts']),('locomotion_all',['locomotion_levels','locomotion_bouts','locomotion_clock'])]:
        add(name,np.concatenate([bank[p] for p in parts],axis=-1),[p+':'+s for p in parts for s in definitions[p]],np.concatenate([np.full(bank[p].shape[-1],j) for j,p in enumerate(parts)]))
    np.savez_compressed(output/'activity_feature_bank.npz',**bank,ants=qc.ant.to_numpy(str),pose_hourly=pose_hour,motif_hourly=motif,speed_hourly=speed_hour,speed_coverage=coverage)
    (output/'feature_definitions.json').write_text(json.dumps(dict(features=definitions,blocks={k:v.tolist() for k,v in blocks.items()},views=VIEWS),indent=2)+'\n')
    # Equal numbers of day-1 histories per common-cohort ant for UMAP.
    eligible=qc.common_eligible.to_numpy();mask=(b['day']==0)&eligible[b['ant_index']]
    selected=balanced_indices(mask,b['ant_index'],100)
    np.savez_compressed(output/'motif_umap_input.npz',histories=histories[selected],motif=assigned[selected],ant=qc.ant.to_numpy(str)[b['ant_index'][selected]],speed_mm_s=speed[selected].mean(axis=1),frames=b['frames'][selected])
    provenance += [stamp(dictionary/'space_blind_models.npz'),stamp(dictionary/'SPACE_BLIND_FROZEN.json')]
    return qc,bank,blocks,m,provenance


def choose_activity_method(table):
    """Selection deliberately has no spatial or day-2 quantities in its keys."""
    ranking=table[table.scope.eq('common')].groupby(['representation','algorithm']).agg(supported_colonies=('selected_k',lambda x:int((x>1).sum())),mean_activity_gain=('activity_gain','mean')).reset_index()
    ranking=ranking.sort_values(['supported_colonies','mean_activity_gain','representation','algorithm'],ascending=[False,False,True,True])
    best=ranking.iloc[0]
    selection=dict(representation=best.representation,algorithm=best.algorithm,supported_colonies=int(best.supported_colonies),mean_activity_gain=float(best.mean_activity_gain),criterion='First supported colonies, then mean A/B centroid prediction gain; spatial labels and day 2 excluded')
    return ranking,selection


def fit_all(qc,bank,blocks,output,bootstrap):
    rows=[];models={};candidate_rows=[];assignments=[]
    for scope in ('common','speed_all'):
        families=list(FAMILY_LABELS) if scope=='common' else list(SPEED_FAMILIES)
        eligible=qc.common_eligible if scope=='common' else qc.speed_eligible
        for family in families:
            for side in ('left','right'):
                ix=np.flatnonzero(eligible&qc.side.eq(side));raw=bank[family][:,ix]
                for algorithm in ('gmm','kmeans'):
                    key=f'{scope}__{family}__{side}__{algorithm}'
                    model,detail,candidates=evaluate(raw,blocks[family],algorithm,bootstrap)
                    model['rows']=ix;models[key]=model
                    row=dict(key=key,scope=scope,representation=family,side=side,algorithm=algorithm,n_ants=len(ix),**detail)
                    day2=(qc.common_day2_eligible if scope=='common' else qc.speed_day2_hours>=16).to_numpy()[ix]
                    if day2.any():
                        row['day2_n']=int(day2.sum());row['day2_profile_gain']=float(1-model['day2_loss'][day2].mean()/model['day2_baseline_loss'][day2].mean())
                        row['day2_retention']=float(np.mean(model['labels'][day2]==model['day2_labels'][day2])) if detail['selected_k']>1 else np.nan
                    else:row.update(day2_n=0,day2_profile_gain=np.nan,day2_retention=np.nan)
                    rows.append(row)
                    candidate_rows.extend(dict(key=key,**c) for c in candidates)
                    for j,index in enumerate(ix):assignments.append(dict(key=key,ant=qc.iloc[index].ant,side=side,group=int(model['labels'][j]),day2_group=int(model['day2_labels'][j]) if day2[j] else -1,pc1=model['coordinates'][j,0],pc2=model['coordinates'][j,1] if model['coordinates'].shape[1]>1 else 0.))
                    print('ACTIVITY_FIT',key,json.dumps({k:row[k] for k in ('n_ants','candidate_k','selected_k','bootstrap_ari_median','replicate_retention','activity_gain')}),flush=True)
    table=pd.DataFrame(rows);table.to_csv(output/'activity_model_comparison.csv',index=False)
    pd.DataFrame(candidate_rows).to_csv(output/'all_candidate_tests.csv',index=False)
    pd.DataFrame(assignments).to_csv(output/'all_activity_assignments.csv',index=False)
    ranking,selection=choose_activity_method(table)
    ranking.to_csv(output/'activity_only_method_ranking.csv',index=False)
    joblib.dump(models,output/'activity_models.joblib',compress=3)
    frozen=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),spatial_inputs_loaded=False,selection=selection,source_cohorts=qc.groupby('side')[['day1_eligible','speed_eligible','common_eligible']].sum().to_dict(),model_sha256=hashlib.sha256((output/'activity_models.joblib').read_bytes()).hexdigest(),assignments_sha256=hashlib.sha256((output/'all_activity_assignments.csv').read_bytes()).hexdigest(),ranking_sha256=hashlib.sha256((output/'activity_only_method_ranking.csv').read_bytes()).hexdigest(),feature_sha256=hashlib.sha256((output/'activity_feature_bank.npz').read_bytes()).hexdigest())
    (output/'ACTIVITY_ONLY_FROZEN.json').write_text(json.dumps(frozen,indent=2)+'\n')
    return table,models,selection


def fit_umap(output):
    import umap
    with np.load(output/'motif_umap_input.npz') as z:data={k:z[k] for k in z.files}
    model=umap.UMAP(n_neighbors=30,min_dist=.1,n_components=2,metric='euclidean',random_state=SEED,transform_seed=SEED,n_jobs=1)
    embedding=model.fit_transform(data['histories']);np.savez_compressed(output/'motif_umap.npz',**data,embedding=embedding)
    rng=np.random.default_rng(SEED);ix=rng.choice(len(embedding),min(1500,len(embedding)),replace=False)
    quality=trustworthiness(data['histories'][ix],embedding[ix],n_neighbors=15)
    info=dict(n_histories=len(embedding),n_ants=len(np.unique(data['ant'])),n_features=data['histories'].shape[1],n_neighbors=30,min_dist=.1,seed=SEED,trustworthiness_15nn_on1500=float(quality),clustering_space='Original normalized history features; UMAP is display only',density_note='UMAP density is not the probability density of the original behavioral space')
    (output/'umap_metadata.json').write_text(json.dumps(info,indent=2)+'\n');joblib.dump(model,output/'umap_model.joblib',compress=3)
    print('UMAP_COMPLETE',json.dumps(info),flush=True)


def spatial_comparison(block,output,qc,table,models,selection):
    frozen=json.loads((output/'ACTIVITY_ONLY_FROZEN.json').read_text());assert not frozen['spatial_inputs_loaded']
    for filename,key in [('activity_models.joblib','model_sha256'),('all_activity_assignments.csv','assignments_sha256'),('activity_only_method_ranking.csv','ranking_sha256')]:
        assert hashlib.sha256((output/filename).read_bytes()).hexdigest()==frozen[key]
    path=block/'stitched/grid_occupancy_histograms_arena_0p25mm/track_cluster_ids.csv';labels=pd.read_csv(path)
    labels['ant']=labels.side+':'+labels.TrackID.astype(int).astype(str).str.zfill(3);lookup=labels.set_index('ant').cluster_id
    outcome=[];all_assignments=pd.read_csv(output/'all_activity_assignments.csv');all_assignments['spatial_label']=all_assignments.ant.map(lookup)
    rng=np.random.default_rng(SEED)
    for row in table.itertuples():
        model=models[row.key];ants=qc.iloc[model['rows']];spatial=ants.ant.map(lookup);ok=spatial.notna().to_numpy();g=model['labels'][ok]
        ari=float(adjusted_rand_score(g,spatial[ok])) if row.selected_k>1 and ok.any() else np.nan
        null=np.array([adjusted_rand_score(rng.permutation(g),spatial[ok]) for _ in range(1000)]) if np.isfinite(ari) else np.array([])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pose_cov=ants.day1_hours.to_numpy()/24;speed_cov=ants.speed_day1_coverage.to_numpy()
            pc=model['coordinates'][:,0]
            nuisance=float(spearmanr(pc,pose_cov if row.scope=='common' else speed_cov).statistic)
        outcome.append(dict(key=row.key,spatial_n=int(ok.sum()),spatial_ari=ari,spatial_permutation_p=float((1+(null>=ari).sum())/1001) if len(null) else np.nan,pc1_coverage_spearman=nuisance))
    results=table.merge(pd.DataFrame(outcome),on='key',validate='one_to_one');results.to_csv(output/'activity_spatial_comparison.csv',index=False)
    all_assignments.to_csv(output/'activity_assignments_with_spatial_labels.csv',index=False)
    selected=results[results.scope.eq('common')&results.representation.eq(selection['representation'])&results.algorithm.eq(selection['algorithm'])]
    summary=dict(selection=selection,primary=selected.replace({np.nan:None}).to_dict('records'),best_spatial_exploratory=results.sort_values('spatial_ari',ascending=False).head(6).replace({np.nan:None}).to_dict('records'))
    (output/'results_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('ACTIVITY_SELECTED_POSTHOC',json.dumps(summary),flush=True)
    return results,all_assignments,[stamp(path)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('tasks','pose-cache','speed-cache','dictionary','block','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--bootstrap',type=int,default=100);p.add_argument('--stage',choices=['features','fit','umap','spatial','all'],default='all')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.stage in ('features','all'):
        qc,bank,blocks,models,provenance=assemble_bank(a.tasks,a.pose_cache,a.speed_cache,a.dictionary,a.output)
        (a.output/'feature_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    else:
        qc=pd.read_csv(a.output/'activity_cohort_audit.csv');definitions=json.loads((a.output/'feature_definitions.json').read_text());blocks={k:np.asarray(v) for k,v in definitions['blocks'].items()}
        with np.load(a.output/'activity_feature_bank.npz') as z:bank={k:z[k] for k in FAMILY_LABELS}
    if a.stage in ('fit','all'):table,models,selection=fit_all(qc,bank,blocks,a.output,a.bootstrap)
    if a.stage in ('umap','all'):fit_umap(a.output)
    if a.stage in ('spatial','all'):
        if a.stage=='spatial':
            table=pd.read_csv(a.output/'activity_model_comparison.csv');models=joblib.load(a.output/'activity_models.joblib');selection=json.loads((a.output/'ACTIVITY_ONLY_FROZEN.json').read_text())['selection']
        comparison,assignments,sources=spatial_comparison(a.block,a.output,qc,table,models,selection)
        provenance=json.loads((a.output/'feature_provenance.json').read_text())+sources
        if any(stamp(item['path'])!=item for item in provenance):raise ValueError('Analysis inputs changed')
        manifest=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},selection=selection,sources=provenance,software={k:importlib.metadata.version(k) for k in ('numpy','pandas','scipy','scikit-learn','matplotlib','pyarrow','umap-learn')})
        (a.output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        (a.output/'NUMERICAL_COMPLETE.json').write_text(json.dumps(dict(complete=True,source=str(a.block)))+'\n')

if __name__=='__main__':main()
