"""Static atlas, gap-aware ethograms, and post hoc sleep/role interpretation."""
import argparse
import hashlib
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from sklearn.metrics import adjusted_rand_score
from sklearn.manifold import trustworthiness
from umap import UMAP
from threadpoolctl import threadpool_limits

from analysis.fine_behavior_features import BASE_NAMES, HZ
from analysis.fine_behavior_states import SEED
from analysis.eigenposture_leiden import fit_role_model


def sleep_comparison(block,ants,data,output):
    labels=np.full(len(data['state']),-1,np.int8);provenance=[]
    for ai,ant in enumerate(ants.itertuples()):
        rows=np.flatnonzero(data['ant']==ai)
        if not len(rows):continue
        root=block/'stitched/sleep_motion_labels/per_track'/Path(ant.track_name).stem
        meta_path=root/'sleep_motion_label_metadata.json';meta=json.loads(meta_path.read_text())
        motion_path=block/'stitched/sleep_motion/per_track'/Path(ant.track_name).stem/'sleep_motion_metadata.json'
        motion=json.loads(motion_path.read_text());raw=block/'stitched/per_track'/ant.track_name
        if (raw.stat().st_size,raw.stat().st_mtime_ns)!=(motion['source_size_bytes'],motion['source_mtime_ns']):
            raise ValueError('Sleep cache raw source changed')
        if (motion_path.stat().st_size,motion_path.stat().st_mtime_ns)!=(meta['source_metadata_size_bytes'],meta['source_metadata_mtime_ns']):
            raise ValueError('Sleep-label motion source changed')
        if meta['fps']!=24 or meta['summary']['track_name']!=ant.track_name:raise ValueError('Sleep clock/identity mismatch')
        values=np.load(root/meta['files']['sleep_state'],mmap_mode='r')
        index=data['frame'][rows]-meta['frame_min']
        valid=(index>=0)&(index<len(values));labels[rows[valid]]=values[index[valid]]
        provenance.append(dict(ant=ant.ant,metadata_sha256=hashlib.sha256(meta_path.read_bytes()).hexdigest(),parameters=meta['classifier_parameters']))
    rows=[];rng=np.random.default_rng(SEED)
    for state in range(data['state'].max()+1):
        fractions=[]
        for ai in range(len(ants)):
            ix=(data['ant']==ai)&(data['state']==state)&(labels>=0)
            if ix.sum()>=20:fractions.append(np.mean(labels[ix]==1))
        f=np.asarray(fractions)
        draws=f[rng.integers(len(f),size=(1000,len(f)))].mean(axis=1) if len(f) else np.array([np.nan])
        rows.append(dict(state=state,n_ants=len(f),fraction=float(np.mean(f)) if len(f) else np.nan,
            lower=float(np.quantile(draws,.025)),upper=float(np.quantile(draws,.975))))
    table=pd.DataFrame(rows);table.to_csv(output/'sleep_by_state.csv',index=False)
    np.save(output/'sleep_at_centers.npy',labels)
    (output/'sleep_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return table


def role_analysis(data,ants,output):
    k=int(data['state'].max()+1)
    counts=np.zeros((len(ants),48,k),dtype=int)
    np.add.at(counts,(data['ant'],data['hour'],data['state']),1)
    total=counts.sum(axis=2)
    hourly=np.divide(counts,total[:,:,None],out=np.full(counts.shape,np.nan),where=total[:,:,None]>=20)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning);proportions=np.nanmean(hourly,axis=1)
    eligible=(np.isfinite(hourly).all(axis=2).sum(axis=1)>=24)
    roles=np.full(len(ants),-1,int);summaries={}
    for side in ('left','right'):
        ids=np.flatnonzero(eligible&ants.side.eq(side).to_numpy())
        if len(ids)<10:
            summaries[side]=dict(n=len(ids),k=None,reason='Fewer than 10 eligible ants');continue
        x=np.sqrt(proportions[ids]);fits={j:fit_role_model(x,j,SEED) for j in range(1,5)};rows=[]
        for j,fit in fits.items():
            labels=fit.predict(x);rng=np.random.default_rng(SEED);bootstrap=[];temporal=[]
            if j>1:
                for repeat in range(50):
                    draws=rng.integers(len(x),size=len(x));m=fit_role_model(x[draws],j,SEED+repeat)
                    bootstrap.append(adjusted_rand_score(labels,m.predict(x)))
                for block_hours in (1,2,4):
                    halves=[]
                    for half in (0,1):
                        mask=(np.arange(48)//block_hours)%2==half
                        xx=np.sqrt(np.nanmean(hourly[ids][:,mask],axis=1))
                        halves.append(fit_role_model(xx,j,SEED).predict(xx))
                    temporal.append(adjusted_rand_score(*halves))
            rows.append(dict(k=j,bic=fit.bic(x),minimum_ants=int(np.bincount(labels,minlength=j).min()),
                bootstrap_median=float(np.median(bootstrap)) if bootstrap else np.nan,
                temporal_median=float(np.median(temporal)) if temporal else np.nan))
        table=pd.DataFrame(rows).set_index('k')
        table['qualifies']=(table.bic<table.loc[1,'bic'])&(table.minimum_ants>=5)&(table.bootstrap_median>=.8)&(table.temporal_median>=.6)
        passing=table[table.qualifies]
        chosen=1 if passing.empty else int(passing[passing.bic<=passing.bic.min()+2].index.min())
        roles[ids]=fits[chosen].predict(x)
        table.to_csv(output/(side+'_role_k.csv'))
        summaries[side]=dict(n=len(ids),k=chosen,sizes=np.bincount(roles[ids]).tolist())
    ants=ants.assign(eligible_role=eligible,role=roles,valid_role_hours=(total>=20).sum(axis=1))
    ants.to_csv(output/'ant_roles.csv',index=False)
    pd.DataFrame(proportions,index=ants.ant,columns=[f'S{s}' for s in range(k)]).to_csv(output/'ant_state_proportions.csv')
    np.savez_compressed(output/'hourly_state_proportions.npz',counts=counts,proportions=hourly,ants=ants.ant.to_numpy(str))
    (output/'role_summary.json').write_text(json.dumps(dict(colonies=summaries,
        rule='Equal-weight observed-hour proportions; >=20 windows/hour in >=24 hours. Tied-covariance GMM K1..4, BIC, >=5 ants/group, 50 bootstrap ARI median>=.8, alternating 1/2/4-hour ARI median>=.6. No final PCA.',
        interpretation='Conditional on observed short sequences; missing tracking and state uncertainty limit role inference.'),indent=2)+'\n')
    return ants,proportions,counts,summaries


def make_report(block,source,sequences,features,models,output,comparison=None):
    output.mkdir(parents=True,exist_ok=True)
    selection=json.loads((models/'selection_frozen.json').read_text())
    test=json.loads((models/'test_evaluation.json').read_text())
    ants=pd.read_csv(models/'ant_split.csv')
    with np.load(models/'states.npz') as z:data=dict(z)
    with np.load(features/'features.npz') as z:
        wi=list(z['widths']).index(selection['width'])
        temporal=z['trajectory'][wi].reshape(-1,5,8)
        spectral=z['spectrum'][wi]
    state=data['state'];raw=data['raw'];k=int(state.max()+1)
    matches=np.asarray(test['matched_jaccard'])[data['order']]
    state_names=[f'S{s}'+('*' if matches[s]<.5 else '') for s in range(k)]
    colors=plt.get_cmap('tab20')(np.arange(k)%20)
    if k>20:colors=plt.get_cmap('turbo')(np.linspace(.03,.97,k))
    cmap=ListedColormap(colors);cmap.set_bad('#dddddd')
    norm=BoundaryNorm(np.arange(k+1)-.5,k)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':120})
    profiles=[]
    for s in range(k):
        ix=state==s
        profiles.append(dict(state=s,test_matched_jaccard=float(matches[s]),n_windows=int(ix.sum()),n_ants=int(len(np.unique(data['ant'][ix]))),
            **{name:float(np.median(raw[ix,j])) for j,name in enumerate(BASE_NAMES)},
            residual_mm=float(np.median(data['residual'][ix])),interpolated_fraction=float(np.mean(data['interpolation'][ix]))))
    profiles=pd.DataFrame(profiles);profiles.to_csv(output/'state_profiles.csv',index=False)
    score=profiles[['forward_peak','lateral_peak','antenna_A_mean','antenna_B_mean']].rank().sum(axis=1)
    candidate=int(score.idxmin())
    (output/'interpretation_frozen.json').write_text(json.dumps(dict(low_motion_candidate=candidate,
        rule='Minimum sum of ranks of median unsigned forward, lateral and both antennal speeds; frozen before reading sleep labels',
        selection_sha256=hashlib.sha256((models/'selection_frozen.json').read_bytes()).hexdigest()),indent=2)+'\n')
    sleep=sleep_comparison(block,ants,data,output)
    ants,proportions,counts,roles=role_analysis(data,ants,output)
    reference=pd.read_csv(source/'spatial_reference.csv')[['side','track_id','spatial_cluster']]
    external=ants.merge(reference,on=['side','track_id'],how='left',validate='one_to_one')
    external.to_csv(output/'roles_with_spatial_reference.csv',index=False)
    spatial_scores=[]
    for side in ('left','right'):
        subset=external[external.side.eq(side)&external.eligible_role&external.spatial_cluster.notna()&(external.role>=0)]
        spatial_scores.append(dict(side=side,n=len(subset),ari=float(adjusted_rand_score(subset.spatial_cluster,subset.role)) if len(subset) else None))
    (output/'spatial_role_comparison.json').write_text(json.dumps(spatial_scores,indent=2)+'\n')
    search=pd.read_csv(models/'search.csv')
    with PdfPages(output/'fine_behavior_report.pdf') as pdf:
        def save(fig,name):
            fig.savefig(output/(name+'.png'),dpi=180,bbox_inches='tight');pdf.savefig(fig,bbox_inches='tight');plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
        markers={'kinematics':'o','trajectory':'s','spectrum':'^'}
        for representation,group in search.groupby('representation'):
            for width,g in group.groupby('width'):
                label=f'{representation}, {width:g} s'
                axes[0].plot(g.n_states,g.validation_ari,marker=markers[representation],label=label)
                axes[1].scatter(g.seed_ari,g.validation_ari,s=30+g.n_states*5,marker=markers[representation],label=label)
                for row in g.itertuples():axes[1].annotate(str(row.n_states),(row.seed_ari,row.validation_ari),fontsize=8)
        axes[0].set(xlabel='Number of states',ylabel='Agreement after omitting training ants (ARI)',title='1. Finer states versus reproducibility')
        axes[1].axvline(.75,color='.6',ls='--');axes[1].axhline(.65,color='.6',ls='--')
        axes[1].set(xlabel='Worst agreement between random starts (ARI)',ylabel='Validation-ant agreement (ARI)',title='Numbers show state counts')
        axes[0].legend(fontsize=8);fig.suptitle(f"{selection['model']}: {k} states | {'passes' if selection['supported'] else 'provisional; fails'} stability rule")
        save(fig,'01_state_resolution')

        rng=np.random.default_rng(SEED);rows=np.sort(rng.choice(len(state),min(10000,len(state)),replace=False))
        layouts=[];metrics=[]
        layout_file=output/'umap_layouts.npz'
        if layout_file.exists():
            with np.load(layout_file) as z:
                np.testing.assert_array_equal(z['rows'],rows);layouts=list(z['layouts'])
        else:
            for neighbors,distance in ((15,.03),(50,.1),(100,.3)):
                layouts.append(UMAP(n_neighbors=neighbors,min_dist=distance,random_state=SEED,n_jobs=1,n_epochs=400).fit_transform(data['x'][rows]))
            np.savez_compressed(layout_file,rows=rows,layouts=np.array(layouts))
        fig,axes=plt.subplots(1,3,figsize=(14,4.7),layout='constrained')
        for ax,xy,(nn,dist) in zip(axes,layouts,((15,.03),(50,.1),(100,.3))):
            sample=rng.choice(len(rows),min(1500,len(rows)),replace=False)
            trust=trustworthiness(data['x'][rows[sample]],xy[sample],n_neighbors=15)
            metrics.append(dict(neighbors=nn,min_dist=dist,trustworthiness=trust))
            ax.scatter(xy[:,0],xy[:,1],c=state[rows],cmap=cmap,norm=norm,s=2,alpha=.6,rasterized=True)
            for s in range(k):
                center=np.median(xy[state[rows]==s],axis=0)
                ax.text(*center,state_names[s],fontsize=9,bbox=dict(facecolor='white',alpha=.75,edgecolor='none'))
            ax.set(title=f'neighbors={nn}, min_dist={dist}\nlocal trustworthiness={trust:.3f}',xticks=[],yticks=[])
        fig.suptitle('2. Identical state assignments in three UMAP views | * weak test-ant match (Jaccard < 0.5)')
        save(fig,'02_umap_states');pd.DataFrame(metrics).to_csv(output/'umap_quality.csv',index=False)

        fig,axes=plt.subplots(2,1,figsize=(13,8),layout='constrained',gridspec_kw={'height_ratios':[2,1]})
        values=raw.copy();values[:,8:13]=np.log1p(values[:,8:13]/.1);values[:,13:]=np.log1p(values[:,13:])
        means=np.stack([np.mean(values[state==s],axis=0) for s in range(k)])
        z=(means-values.mean(axis=0))/np.maximum(values.std(axis=0),1e-8)
        im=axes[0].imshow(z,aspect='auto',cmap='RdBu_r',vmin=-2,vmax=2)
        axes[0].set(yticks=range(k),yticklabels=[f'S{s}' for s in range(k)],xticks=range(16),xticklabels=BASE_NAMES,title='3. What distinguishes the states?')
        axes[0].tick_params(axis='x',rotation=40);fig.colorbar(im,ax=axes[0],label='Mean feature deviation (SD)')
        for name,label in [('forward_peak','forward peak'),('lateral_peak','lateral peak'),('antenna_A_mean','antenna A'),('antenna_B_mean','antenna B')]:
            axes[1].plot(range(k),profiles[name],marker='o',label=label)
        axes[1].set(xticks=range(k),xticklabels=[f'S{s}' for s in range(k)],ylabel='Median speed (mm/s)');axes[1].legend(ncol=4)
        save(fig,'03_state_features')

        fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
        rates=np.abs(np.diff(temporal[:,:,:4],axis=1)).mean(axis=1)/(selection['width']/4)
        rates=np.stack([np.median(rates[state==s],axis=0) for s in range(k)])
        im=axes[0].imshow(np.log10(rates+1e-5),aspect='auto',cmap='magma')
        fig.colorbar(im,ax=axes[0],label='log10 median absolute PC change / s')
        axes[0].set(xticks=range(4),xticklabels=['PC1','PC2','PC3','PC4'],yticks=range(k),yticklabels=[f'S{s}' for s in range(k)],title='Posture variation, derived from the five ordered samples')
        if np.any(spectral>0):
            power=spectral.reshape(-1,4,6).sum(axis=1)
            relative=power/np.maximum(power.sum(axis=1,keepdims=True),1e-12)
            for s in range(k):axes[1].plot([.5,.8,1.2,1.8,2.7,4],np.mean(relative[state==s],axis=0),'-o',color=colors[s],label=f'S{s}')
            axes[1].set(xlabel='Frequency (Hz)',ylabel='Mean relative power',title='Frequency structure, not total wavelet energy')
        else:
            for s in range(k):
                value=temporal[state==s,:,:4]
                # RMS change retains variation even when signed phases cancel.
                rms=np.sqrt(np.mean(value**2,axis=(0,2)))
                axes[1].plot(np.linspace(-selection['width']/2,selection['width']/2,5),rms,'-o',color=colors[s],label=f'S{s}')
            axes[1].set(xlabel='Time relative to window center (s)',ylabel='RMS centered posture-PC displacement',title='Movement amplitude through the window')
        axes[1].legend(ncol=2,fontsize=8)
        fig.suptitle('Temporal features distinguish motion from static posture')
        save(fig,'03b_temporal_features')

        # Actual snippets closest to state centers, from distinct ants; no synthetic
        # average pose is presented as an observed action.
        exemplar_rows=[];cache={}
        for s in range(k):
            available=np.flatnonzero((state==s)&(data['confidence']>=.8))
            center=data['x'][available].mean(axis=0)
            order=available[np.argsort(((data['x'][available]-center)**2).sum(axis=1))]
            chosen=[];used=set()
            for row in order:
                ai=int(data['ant'][row])
                if ai in used:continue
                chosen.append(row);used.add(ai)
                if len(chosen)==2:break
            for row in chosen:exemplar_rows.append(dict(state=s,row=int(row),ant=ants.ant.iloc[data['ant'][row]],hour=int(data['hour'][row]),global_frame=int(data['frame'][row])))
        pd.DataFrame(exemplar_rows).to_csv(output/'exemplars.csv',index=False)
        for page,first in enumerate(range(0,k,5)):
            shown=list(range(first,min(first+5,k)));fig,axes=plt.subplots(len(shown),2,figsize=(13,2.05*len(shown)),squeeze=False,layout='constrained')
            for rr,s in enumerate(shown):
                for cc,record in enumerate([v for v in exemplar_rows if v['state']==s]):
                    row=record['row'];ai=int(data['ant'][row]);ant=ants.ant.iloc[ai]
                    if ai not in cache:
                        with np.load(sequences/(ant.replace(':','_')+'.npz')) as z:cache[ai]=z['pose']
                    poses=cache[ai][data['hour'][row],data['sample'][row]+np.array([-12,-6,0,6,12])].reshape(5,2,3,2)
                    ax=axes[rr,cc]
                    for t,pose in enumerate(poses):
                        for antenna,color in zip(pose,('#ce3d37','#167b9b')):
                            chain=np.vstack([np.zeros(2),antenna]);ax.plot(chain[:,0]+t*6,chain[:,1],'-o',color=color,lw=1,ms=2)
                        ax.plot(t*6,0,'k.',ms=4)
                    ax.set_aspect('equal');ax.set(xticks=np.arange(5)*6,xticklabels=['−1 s','−0.5','0','+0.5','+1 s'],yticks=[],title=f'S{s} | {ant} | hour {record["hour"]}')
                    ax.set_ylim(-4,4)
            fig.suptitle('4. Observed antennal sequences, aligned to the body; red/blue identify antennae')
            save(fig,f'04_pose_examples_{page+1}')

        fig,axes=plt.subplots(4,1,figsize=(13,8),layout='constrained')
        examples=[]
        for side in ('left','right'):
            valid=np.flatnonzero(ants.side.eq(side).to_numpy())
            rankings=np.argsort(counts[valid].sum(axis=2).max(axis=1))[::-1]
            for ai in valid[rankings[:2]]:
                hour=int(counts[ai].sum(axis=1).argmax());examples.append((ai,hour))
        for ax,(ai,hour) in zip(axes,examples):
            ix=np.flatnonzero((data['ant']==ai)&(data['hour']==hour))
            line=np.full(240,np.nan);bins=(data['sample'][ix]/HZ/.25).astype(int);line[bins]=state[ix]
            ax.imshow(np.ma.masked_invalid(line[None,:]),extent=(0,60,0,1),aspect='auto',cmap=cmap,norm=norm,interpolation='nearest')
            ax.set(xlabel='Seconds within the sampled minute',yticks=[],title=f'{ants.ant.iloc[ai]} | hour {hour} | gray = no supported assignment')
        fig.suptitle('5. Fine-scale state sequences; gaps are retained')
        save(fig,'05_ethograms')

        fig,axes=plt.subplots(1,2,figsize=(13,8),layout='constrained')
        for ax,side in zip(axes,('left','right')):
            ids=np.flatnonzero(ants.side.eq(side).to_numpy()&np.isfinite(proportions).all(axis=1))
            if len(ids)>1:ids=ids[leaves_list(linkage(np.sqrt(proportions[ids]),method='average'))]
            bottom=np.zeros(len(ids))
            for s in range(k):ax.barh(np.arange(len(ids)),proportions[ids,s],left=bottom,color=colors[s],height=.95);bottom+=proportions[ids,s]
            for y,ai in enumerate(ids):
                spatial=external.spatial_cluster.iloc[ai]
                if np.isfinite(spatial):ax.scatter(1.025,y,c=['#333333' if spatial==0 else '#c5c5c5'],s=9,marker='s',clip_on=False)
            ax.set(yticks=np.arange(len(ids)),yticklabels=[ants.ant.iloc[i] for i in ids],xlabel='Mean proportion across observed hours',title=f'{side}: {len(ids)} ants with hourly estimates; role K={roles[side]["k"]}')
            ax.tick_params(axis='y',labelsize=6)
        fig.suptitle('6. Proportion-based repertoires | right-edge squares: spatial class, post hoc only')
        save(fig,'06_ant_repertoires')

        fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
        cams=np.unique(data['camera']);table=np.array([[np.mean(data['camera'][state==s]==c) for c in cams] for s in range(k)])
        im=axes[0].imshow(table,aspect='auto',vmin=0,vmax=1,cmap='viridis');fig.colorbar(im,ax=axes[0],label='Fraction')
        axes[0].set(xticks=range(len(cams)),xticklabels=cams,yticks=range(k),yticklabels=[f'S{s}' for s in range(k)],xlabel='Camera',title='State camera composition')
        axes[1].bar(range(k),profiles.residual_mm,color=colors);axes[1].set(xlabel='State',ylabel='Median smoothing residual (mm)',title='Tracking-noise diagnostic')
        order=data['order'];j=np.array(test['matched_jaccard'])
        # Matching entries use original dictionary IDs, before speed ordering.
        axes[2].bar(range(k),j[order],color=colors);axes[2].set(xlabel='State',ylim=(0,1),ylabel='Matched Jaccard',title=f'Test ants: partition ARI={test["ant_removal_ari"]:.2f}')
        fig.suptitle('7. Quality and held-out-ant agreement')
        save(fig,'07_quality_stability')

        fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
        axes[0].bar(range(k),sleep.fraction,color=colors,yerr=[sleep.fraction-sleep.lower,sleep.upper-sleep.fraction],capsize=3)
        axes[0].set(xticks=range(k),xticklabels=[f'S{s}' for s in range(k)],ylim=(0,1),ylabel='Existing motion-sleep label fraction',title=f'8. Sleep comparison | low-motion candidate S{candidate}')
        for s,n in enumerate(sleep.n_ants):axes[0].text(s,.02,str(n),ha='center',fontsize=8)
        axes[1].scatter(raw[::5,8],raw[::5,10:12].mean(axis=1),c=state[::5],cmap=cmap,norm=norm,s=2,alpha=.3,rasterized=True)
        axes[1].set(xscale='symlog',yscale='symlog',xlabel='Forward peak speed (mm/s)',ylabel='Mean relative antennal speed (mm/s)',title='Body and antenna motion are retained separately')
        fig.suptitle('Movement-defined sleep is a post hoc comparator, not independent arousal validation')
        save(fig,'08_sleep_comparison')
    summary=dict(selection=selection,test=test,n_windows=len(state),n_ants=int(len(np.unique(data['ant']))),
        nominal_observed_center_hours=len(state)*.25/3600,roles=roles,spatial_role_comparison=spatial_scores,
        weak_test_match_states=np.flatnonzero(matches<.5).tolist(),low_motion_candidate=candidate,
        sleep_candidate_fraction=float(sleep.fraction.iloc[candidate]),
        caution='State durations reflect overlapping windows. Short randomly sampled clips do not establish all-day prevalence, rare-state absence, or validated sleep.')
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('block','source','sequences','features','models','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    with threadpool_limits(limits=4):make_report(a.block,a.source,a.sequences,a.features,a.models,a.output)
