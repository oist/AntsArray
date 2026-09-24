"""Presentation of the activity-only method comparison and motif UMAP."""
from __future__ import annotations
import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.backends.backend_pdf import PdfPages

from analysis.postural_dynamics_plots import setup,draw_pose,map_plot,INK
from analysis.colony_behavioral_landscape import features
from analysis.postural_dynamics_extract import DAY_FRAMES,stamp
from analysis.activity_discovery import ActivityTransform,FAMILY_LABELS,SPEED_FAMILIES

COLORS=['#128a8b','#d7754e','#6671b7','#b7912d']
PDF='0724_activity_discovery.pdf'
FIGURES=[('02_eigenpostures','Discover a posture basis','The fixed posture basis is retained from the preceding analysis.'),
 ('03_short_dynamics','Map posture–velocity motifs with UMAP','UMAP shows the measured history distribution and original-space motif assignments.'),
 ('04_activity_selection','Select groups using activity alone','Compare ten activity representations and two clustering methods before opening spatial outcomes.'),
 ('05_spatial_outcomes','Reveal the spatial outcomes','Inspect spatial correspondence of the activity-selected method on held-out day-2 maps.'),
 ('06_method_comparison','Compare every attempted approach','Report successful and unsuccessful candidates, cohort sizes and nuisance associations.'),
 ('07_activity_validation','Check persistence and interpretation','Validate the following day and inspect the physical activity differences between groups.')]


def title(fig,n,subtitle):
    fig.suptitle(f'{n:02d}  {FIGURES[n-2][1]}',x=.035,ha='left',fontsize=19,fontweight='bold')
    fig.text(.035,.925,subtitle,fontsize=10,color='#596877',va='top')
    fig.subplots_adjust(top=.82,bottom=.12,left=.075,right=.96,hspace=.60,wspace=.48)


def save(fig,n,out,pdf):
    stem=FIGURES[n-2][0];fig.savefig(out/(stem+'.png'),dpi=180);fig.savefig(out/(stem+'.pdf'));pdf.savefig(fig);plt.close(fig)


def label(g,k):return 'Unsplit cohort' if k==1 else f'A{g+1}'


def dots(ax,x,y,g,k):
    for j in range(k):
        keep=np.asarray(g)==j;ax.scatter(np.asarray(x)[keep],np.asarray(y)[keep],s=30,color=COLORS[j],edgecolor='white',lw=.4,label=label(j,k),alpha=.9)


def read_primary(out,block,qc,comparison,all_models,selection):
    frozen=json.loads((out/'ACTIVITY_ONLY_FROZEN.json').read_text())
    assert hashlib.sha256((out/'activity_models.joblib').read_bytes()).hexdigest()==frozen['model_sha256']
    atoms=block/'analysis_outputs/long_timescale_0723_0724_0729_20260915/temporal_clusters_4h/atoms/2'
    behavior_path=block/'analysis_outputs/long_timescale_0723_0724_0729_20260915/task_bins.parquet'
    behavior=pd.read_parquet(behavior_path);part=behavior[behavior.source_block.eq(str(block))&behavior.in_recording_bin&(behavior.timestamp>=pd.Timestamp('2026-07-25 10:00'))&(behavior.timestamp<pd.Timestamp('2026-07-26 10:00'))]
    nest={}
    for ant,p in part.groupby('ant'):
        w=p.n_expected_frames*p.position_coverage;good=w.gt(0)&p.colony_percent.notna();nest[ant]=float(np.average(p.loc[good,'colony_percent'],weights=w[good])) if good.any() else np.nan
    start=pd.Timestamp('2026-07-24 10:00').value//(1800*10**9);results={};provenance=[stamp(behavior_path),stamp(block/'panorama_regions.csv')];forecasts=[];rng=np.random.default_rng(724)
    labels=pd.read_csv(block/'stitched/grid_occupancy_histograms_arena_0p25mm/track_cluster_ids.csv');labels['ant']=labels.side+':'+labels.TrackID.astype(int).astype(str).str.zfill(3)
    for side in ('left','right'):
        row=comparison[(comparison.scope=='common')&(comparison.representation==selection['representation'])&(comparison.algorithm==selection['algorithm'])&(comparison.side==side)].iloc[0]
        model=all_models[row.key];t=qc.iloc[model['rows']].copy();t['group']=model['labels'];t['pc1']=model['coordinates'][:,0];t['pc2']=model['coordinates'][:,1]
        t['day2_pc1']=model['day2_coordinates'][:,0];t['day2_group']=np.where(t.common_day2_eligible,model['day2_labels'],-1);t['spatial_label']=t.ant.map(labels.set_index('ant').cluster_id);t['day2_colony_percent']=t.ant.map(nest)
        maps=[];coverage=[]
        for ant in t.ant:
            path=atoms/(ant.replace(':','_')+'.npz');metadata=json.loads(path.with_suffix('.json').read_text())
            if metadata['signature']['entry']['block']!=str(block) or not metadata['original_histogram_exact']:raise ValueError('Wrong spatial atoms')
            for item in metadata['signature']['inputs']:
                actual=stamp(item['path'])
                if (actual['size'],actual['mtime_ns'])!=(item['size'],item['mtime_ns']):raise ValueError('Stale spatial atoms')
            with np.load(path) as z:
                mm=[];cc=[]
                for d in (0,1):
                    keep=(z['calendar_bin']>=start+d*48)&(z['calendar_bin']<start+(d+1)*48)
                    if keep.sum()!=48:raise ValueError('Wrong spatial clock window')
                    n=int(z['detected'][keep].sum());mm.append(z['counts'][keep].sum(axis=0)/max(n,1));cc.append(n/DAY_FRAMES)
                edges=(z['x_edges'],z['y_edges'])
            maps.append(mm);coverage.append(cc);provenance += [stamp(path),stamp(path.with_suffix('.json'))]
        maps=np.asarray(maps);coverage=np.asarray(coverage);t['position_coverage_day1']=coverage[:,0];t['position_coverage_day2']=coverage[:,1]
        ok=(coverage>=.4).all(axis=1);ix=np.flatnonzero(ok);train=maps[ok,0];test=maps[ok,1];group=t.group.to_numpy()[ok]
        def spatial_loss(groups):
            pred=[]
            for i in range(len(ix)):
                others=(np.arange(len(ix))!=i)&(groups==groups[i])
                if not others.any():others=np.arange(len(ix))!=i
                pred.append(train[others].mean(axis=0))
            return np.square(features(np.stack(pred),np.ones(len(ix)))-features(test,np.ones(len(ix)))).sum(axis=1)/2
        baseline=spatial_loss(np.zeros(len(ix),int));group_loss=spatial_loss(group);gain=baseline-group_loss;boot=np.array([rng.choice(gain,len(gain),replace=True).mean() for _ in range(2000)])
        for i,j in enumerate(ix):forecasts.append(dict(side=side,ant=t.iloc[j].ant,gain=gain[i],baseline_loss=baseline[i],activity_group_loss=group_loss[i]))
        results[side]=dict(table=t,rows=model['rows'],model=model,selection_row=row.to_dict(),selected_k=int(row.selected_k),spatial_maps=maps,spatial_edges=edges,forecast_gain=gain,forecast_ci=np.quantile(boot,[.025,.975]),forecast_mean=float(gain.mean()))
        t.to_csv(out/f'{side}_primary_activity_groups.csv',index=False)
    pd.DataFrame(forecasts).to_csv(out/'primary_spatial_forecasts.csv',index=False)
    np.savez_compressed(out/'primary_spatial_maps.npz',**{s:r['spatial_maps'] for s,r in results.items()})
    return results,provenance


def render(out,block,dictionary,hourly_reference):
    setup();summary=json.loads((out/'results_summary.json').read_text());selection=summary['selection'];comparison=pd.read_csv(out/'activity_spatial_comparison.csv');all_models=joblib.load(out/'activity_models.joblib');qc=pd.read_csv(out/'activity_cohort_audit.csv');candidates=pd.read_csv(out/'all_candidate_tests.csv')
    results,spatial_sources=read_primary(out,block,qc,comparison,all_models,selection)
    spatial_sources += [stamp(hourly_reference/'hourly_motif_profiles.npz')]
    spatial_sources += [stamp(dictionary/name) for name in ('space_blind_models.npz','posture_display_samples.npz','posture_reconstruction.csv','history_prediction_validation.csv','motif_resolution_validation.csv')]
    with np.load(dictionary/'space_blind_models.npz') as z:models={k:z[k] for k in z.files}
    with np.load(out/'activity_feature_bank.npz') as z:bank={k:z[k] for k in z.files}
    with np.load(hourly_reference/'hourly_motif_profiles.npz') as z:camera=z['camera_only'][0]
    nuisance=[]
    for row in comparison.itertuples():
        m=all_models[row.key];cam=camera[m['rows']]
        with np.errstate(invalid='ignore'):
            import warnings
            with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);means=np.nanmean(cam,axis=1)
        valid=np.isfinite(means).all(axis=1)
        correlation=float(spearmanr(pdist(m['coordinates'][valid]),pdist(np.sqrt(means[valid]))).statistic) if valid.sum()>3 else np.nan
        nuisance.append(dict(key=row.key,camera_distance_spearman=correlation,camera_comparison_n=int(valid.sum())))
    comparison=comparison.merge(pd.DataFrame(nuisance),on='key');comparison.to_csv(out/'method_comparison_with_controls.csv',index=False)
    pca=PCA();pca.mean_=models['posture_mean'];pca.components_=models['posture_components'];pca.explained_variance_=models['posture_variance'];pca.explained_variance_ratio_=models['posture_variance_ratio']
    lengths=models['standard_lengths_mm'];examples=models['motif_example_shape'];sample=np.load(dictionary/'posture_display_samples.npz');recon=pd.read_csv(dictionary/'posture_reconstruction.csv');history=pd.read_csv(dictionary/'history_prediction_validation.csv');first=history.iloc[0]
    fit=dict(rank=int(models['rank']),n_motifs=int(models['n_motifs']),history_table=history,resolution=pd.read_csv(dictionary/'motif_resolution_validation.csv'),persistence_mse=first.mean_mse/(1-first.gain_vs_persistence))
    regions=pd.read_csv(block/'panorama_regions.csv').to_dict('records')
    for reg in regions:
        name=reg['semantic_label'];reg['side']='left' if name.endswith('L') or name.endswith('_left') else 'right';reg['region_type']='arena' if name.startswith('arena') else name[:-1]
    metadata={}
    for side,r in results.items():
        track=Path(r['table'].track_name.iloc[0]).stem;metadata[side]=json.loads((block/'stitched/grid_occupancy_histograms_arena_0p25mm/per_track'/track/'grid_occupancy_metadata.json').read_text())
        spatial_sources.append(stamp(block/'stitched/grid_occupancy_histograms_arena_0p25mm/per_track'/track/'grid_occupancy_metadata.json'))
    with PdfPages(out/PDF) as pdf:
        fig,ax=plt.subplots(2,4,figsize=(16,9))
        title(fig,2,f'PCA of 16 direction cosines · equal ant weights · {fit["rank"]} modes retain 95% of day-1 variance · basis frozen from 52 day-1 ants')
        ax[0,0].bar(np.arange(1,17),100*pca.explained_variance_ratio_,color=INK)
        ax[0,0].set(xlabel='Mode',ylabel='Variance explained (%)',title='Eigenposture spectrum',xticks=[1,4,8,12,16])
        for d,color in [(1,INK),(2,COLORS[1])]:
            part=recon[recon.day.eq(d)];ax[0,1].plot(part.modes,part.explained_variance*100,'o-',ms=3,color=color,label=f'Day {d}')
        ax[0,1].axhline(95,color='#a8b1ba',ls=':');ax[0,1].set(xlabel='Retained modes',ylabel='Reconstructed variance (%)',title='Fixed day-1 basis');ax[0,1].legend(frameon=False)
        z=sample['scores'][:,24];counts,xedges,yedges=np.histogram2d(z[:,0],z[:,1],bins=36)
        im=ax[0,2].pcolormesh(xedges,yedges,counts.T/counts.sum(),cmap='magma',norm=PowerNorm(.5),rasterized=True)
        fig.colorbar(im,ax=ax[0,2],shrink=.8,label='Probability per bin')
        ax[0,2].set(xlabel='Mode 1 coefficient',ylabel='Mode 2 coefficient',title='Sampled postures · density heat map')
        for j in range(3):
            ax[0,3].plot(np.arange(8),np.linalg.norm(pca.components_[j].reshape(8,2),axis=1),'o-',ms=3,label=f'Mode {j+1}')
        ax[0,3].set(xticks=range(8),xticklabels=['Head','Gaster','A1','A2','A3','B1','B2','B3'],ylabel='Loading magnitude',title='Which joints contribute?');ax[0,3].tick_params(axis='x',rotation=45);ax[0,3].legend(frameon=False,fontsize=8)
        for j in range(4):
            for sd,color in [(-1.5,COLORS[0]),(0,'#a8b1ba'),(1.5,COLORS[1])]:draw_pose(ax[1,j],pca.mean_+sd*np.sqrt(pca.explained_variance_[j])*pca.components_[j],lengths,color,alpha=.85)
            ax[1,j].set_title(f'Mode {j+1} · {pca.explained_variance_ratio_[j]:.1%}')
        fig.text(.075,.015,'Bottom: mean (gray) ±1.5 SD (teal/orange), normalized to fixed segment lengths for display. These are schematic modes, not video frames.',fontsize=9)
        save(fig,2,out,pdf)

        fig=plt.figure(figsize=(17,17));gs=fig.add_gridspec(4,4,height_ratios=[.55,1.25,.85,1.35])
        title(fig,3,f'{fit["rank"]} posture coefficients + signed forward/lateral velocity → 13 samples across 1 s → {fit["n_motifs"]} shared motifs')
        fig.subplots_adjust(top=.87,bottom=.065,hspace=.65,wspace=.48)
        descriptions=[('A  Measure and align','Project tracked anchor motion onto\nthe anterior and lateral body axes.\nAppend both signed velocities (mm/s).'),
          ('B  Scale and stack',f'Equal total weight: posture and velocity.\n13 time samples × {fit["rank"]+2} channels\n= {13*(fit["rank"]+2)} values per one-second history.'),
          ('C  Learn shared centers','KMeans groups similar histories.\nEach center defines a motif; a history\ngets its nearest-center motif label.'),
          ('D  Describe individual activity','Measure hourly motif use, posture,\nvelocity and locomotor bouts.\nCompare activity-only ant clustering.')]
        for j,(heading,description) in enumerate(descriptions):
            ax=fig.add_subplot(gs[0,j]);ax.axis('off')
            ax.text(0,.95,heading,weight='bold',fontsize=11,va='top')
            ax.text(0,.68,description,fontsize=10,va='top',linespacing=1.6)
            if j<3:ax.text(1.08,.50,'→',fontsize=20,color=COLORS[0])
        with np.load(out/'motif_umap.npz') as z:embedding=z['embedding'];labels=z['motif']
        ax=fig.add_subplot(gs[1,:2]);im=ax.hexbin(embedding[:,0],embedding[:,1],gridsize=65,bins='log',mincnt=1,cmap='magma');fig.colorbar(im,ax=ax,shrink=.75,label='Measured histories / bin (log scale)');ax.set(xlabel='UMAP 1',ylabel='UMAP 2',title=f'UMAP distribution · {len(embedding):,} histories · equal ant sampling')
        ax=fig.add_subplot(gs[1,2:]);palette=plt.cm.tab20(np.linspace(0,1,fit['n_motifs']))
        for motif in range(fit['n_motifs']):
            select=labels==motif;ax.scatter(embedding[select,0],embedding[select,1],s=3,alpha=.6,c=[palette[motif]],label=f'M{motif:02d}',rasterized=True)
        ax.set(xlabel='UMAP 1',ylabel='UMAP 2',title='Same embedding · colored by original-space motif labels');ax.legend(frameon=False,ncol=6,fontsize=7,loc='upper center',bbox_to_anchor=(.5,-.17),markerscale=2)
        ax0=fig.add_subplot(gs[2,:2]);ax1=fig.add_subplot(gs[2,2:]);h=fit['history_table']
        for shuffled,color,legend_text in [(False,INK,'Time-ordered history'),(True,COLORS[1],'Past times shuffled')]:
            part=h[h.shuffled.eq(shuffled)];ax0.errorbar(part.history_seconds,part.mean_mse,yerr=part.standard_error,color=color,marker='o',capsize=3,label=legend_text)
        ax0.axhline(fit['persistence_mse'],color='#8f9dac',ls=':',label='Hold current posture + velocity')
        ax0.set(xlabel='Past history (seconds)',ylabel='Joint prediction error (scaled units)',title='Predict 0.25 s ahead · held-out hours · mean ± SE');ax0.legend(frameon=False,fontsize=8)
        res=fit['resolution'];ax1.errorbar(res.motifs,res.mean_mse,yerr=res.standard_error,color=INK,marker='o',capsize=4)
        best=res.loc[res.mean_mse.idxmin()]
        ax1.axhline(best.mean_mse+best.standard_error,color='#8f9dac',ls=':',label='Best mean + 1 SE')
        ax1.axvline(fit['n_motifs'],color=COLORS[0],ls='--',label='Selected count');ax1.set(xlabel='Number of shared motifs',ylabel='Joint prediction error (scaled units)',xticks=res.motifs,title='Resolution · smallest within 1 SE of best');ax1.legend(frameon=False,fontsize=8)
        motion=np.sqrt(np.square(models['motif_center_velocity_mm_s']).sum(axis=-1).mean(axis=1));order=np.argsort(motion)
        for j,index in enumerate(np.linspace(0,len(examples)-1,4).round().astype(int)):
            motif=int(order[index]);sub=gs[3,j].subgridspec(2,1,height_ratios=[1.2,1],hspace=.55)
            ax=fig.add_subplot(sub[0,0])
            for k,t in enumerate([12,16,20,24]):draw_pose(ax,examples[motif,t],lengths,plt.cm.viridis(k/3),alpha=.8,lw=1.3)
            ax.set_title(f'Motif {motif:02d} · measured example');ax.set_xlabel('Anterior axis (mm)')
            ax=fig.add_subplot(sub[1,0]);v=models['motif_example_velocity_mm_s'][motif,12:25]
            for component,name in enumerate(('Forward','Lateral')):ax.plot(np.arange(13)/12,v[:,component],color=COLORS[component],label=name)
            ax.axhline(0,color='#a8b2bd',lw=.7);ax.set(xlabel='Time within history (s)',ylabel='Velocity (mm/s)',xticks=[0,.5,1])
            if j==0:ax.legend(frameon=False,fontsize=8)
        fig.text(.075,.022,'UMAP: 30 neighbors, min_dist=0.1, fixed seed. Clusters are assigned in the original 143-dimensional history space. UMAP does not preserve original-space density.',fontsize=9)
        fig.text(.075,.009,'Motifs classify short histories; ant groups classify individuals. Examples below the embedding are measured histories, not reconstructed video.',fontsize=9)
        save(fig,3,out,pdf)
        ranking=pd.read_csv(out/'activity_only_method_ranking.csv')
        fig=plt.figure(figsize=(16,12));gs=fig.add_gridspec(2,2,height_ratios=[1,1.4])
        title(fig,4,f"Activity-selected: {FAMILY_LABELS[selection['representation']]} · {selection['algorithm'].upper()} · no spatial labels or day-2 measurements used for selection")
        for j,(side,r) in enumerate(results.items()):
            ax=fig.add_subplot(gs[0,j]);t=r['table'];row=r['selection_row'];k=r['selected_k']
            dots(ax,t.pc1,t.pc2,t.group,k);ax.set(xlabel='Activity PC1',ylabel='Activity PC2',title=f'{side.title()} · n={len(t)} · selected K={k}');ax.legend(frameon=False,fontsize=8)
            ax.text(.02,.98,f"Bootstrap ARI: median {row['bootstrap_ari_median']:.2f}, 10th percentile {row['bootstrap_ari_p10']:.2f}\nA/B retention {row['replicate_retention']:.0%}",transform=ax.transAxes,va='top',fontsize=8)
        ax=fig.add_subplot(gs[1,0]);yy=np.arange(len(ranking));names=[FAMILY_LABELS[r.representation]+' / '+r.algorithm for r in ranking.itertuples()]
        ax.barh(yy,ranking.mean_activity_gain*100,color=[COLORS[0] if x==2 else '#a5afba' for x in ranking.supported_colonies]);ax.set(yticks=yy,yticklabels=names,xlabel='A/B activity prediction gain (%)',title='All methods · ranked using activity only');ax.invert_yaxis();ax.tick_params(axis='y',labelsize=8)
        ax=fig.add_subplot(gs[1,1]);ax.axis('off');ax.text(0,1,'How K and the approach are chosen',va='top',weight='bold',fontsize=13)
        ax.text(0,.91,'GMM: compare K=1–4 and covariance families by BIC.\nRequire ≥10 improvement over a single component.\n\nKMeans: K=2–4, silhouette ≥0.25; choose the\nsmallest K within 0.02 of the best eligible silhouette.\n\nBoth: ≥4 ants/group, bootstrap median ARI ≥0.8,\nand A/B assignment retention ≥80%. Otherwise K=1.\n\nAcross approaches: first favor supported splits in\nboth colonies, then mean A/B prediction gain.\nTeal bars: supported splits in both colonies.\n\nA/B = disjoint alternating five-minute blocks.\nA/B gains compare group-center predictions against\na colony-mean predictor. Day 2 remains held out.',va='top',fontsize=11,linespacing=1.55)
        fig.text(.075,.025,'This exploratory selection favors reproducible partitions. It does not establish that behavior consists of discrete biological types.',fontsize=10)
        fig.subplots_adjust(left=.24,wspace=.55)
        save(fig,4,out,pdf)

        cols=max(r['selected_k'] for r in results.values())+1
        fig,axes=plt.subplots(2,cols,figsize=(max(15,cols*4),10),squeeze=False)
        title(fig,5,'Day-1 activity labels → day-2 occupancy · spatial outcomes opened only after every model and the activity-only choice were frozen')
        for i,(side,r) in enumerate(results.items()):
            t=r['table'];k=r['selected_k'];ok=t.position_coverage_day2.ge(.4).to_numpy();x,y=r['spatial_edges'];area=np.diff(y)[:,None]*np.diff(x)[None,:]
            vmax=float(np.quantile(np.sqrt(r['spatial_maps'][ok,1]/area).ravel(),.997));vmax=max(vmax,1e-6)
            for g in range(cols-1):
                ax=axes[i,g]
                if g>=k:ax.axis('off');continue
                keep=ok&t.group.eq(g).to_numpy();mean=r['spatial_maps'][keep,1].mean(axis=0)
                im=map_plot(ax,mean,r,regions,metadata[side],vmax,f'{side.title()} · {label(g,k)} · n={keep.sum()}');fig.colorbar(im,ax=ax,shrink=.65,label='√ occupancy density (mm⁻¹)')
            ax=axes[i,-1]
            for g in range(k):
                vals=t.loc[ok&t.group.eq(g),'day2_colony_percent'].dropna().to_numpy();jitter=np.linspace(-.12,.12,len(vals));ax.scatter(g+jitter,vals,color=COLORS[g],s=28,alpha=.8)
                if len(vals):ax.plot([g-.22,g+.22],[np.median(vals)]*2,color=INK,lw=2)
            row=r['selection_row'];ari=row['spatial_ari'];ari_text=f'{ari:.2f}' if np.isfinite(ari) else 'N/A (K=1)'
            ax.set(xticks=range(k),xticklabels=[label(g,k) for g in range(k)],ylabel='Day-2 time in colony region (%)',title=f'Existing spatial labels: ARI {ari_text}')
        fig.text(.075,.04,'Maps average ants equally, requiring ≥40% day-2 position coverage. Outlines: colony (teal), food/water (yellow). Group names are local to each colony.',fontsize=10)
        fig.text(.075,.022,'Spatial-label ARI is exploratory: the existing occupancy labels use the recording, including day 2. Day-2 maps are held out from activity fitting, not a new dataset.',fontsize=10)
        save(fig,5,out,pdf)

        fig,axes=plt.subplots(1,3,figsize=(18,12),gridspec_kw={'width_ratios':[1.5,1,1]})
        title(fig,6,'Every common-cohort method is shown · K=1 means no supported split under the stated rule · gray cells are not evidence of disagreement')
        keys=[(r.representation,r.algorithm) for r in ranking.itertuples()];matrix=np.full((len(keys),2),np.nan);ks=np.zeros_like(matrix)
        for i,(family,algorithm) in enumerate(keys):
            for j,side in enumerate(('left','right')):
                row=comparison[(comparison.scope=='common')&(comparison.representation==family)&(comparison.algorithm==algorithm)&(comparison.side==side)].iloc[0];matrix[i,j]=row.spatial_ari;ks[i,j]=row.selected_k
        ax=axes[0];cmap=plt.cm.viridis.copy();cmap.set_bad('#e6e9ed');im=ax.imshow(matrix,aspect='auto',vmin=0,vmax=1,cmap=cmap)
        for i in range(len(keys)):
            for j in range(2):ax.text(j,i,f'K={int(ks[i,j])}'+(f' · {matrix[i,j]:.2f}' if np.isfinite(matrix[i,j]) else ''),ha='center',va='center',fontsize=9,color='white' if np.isfinite(matrix[i,j]) and matrix[i,j]<.55 else INK)
        ax.set(xticks=[0,1],xticklabels=['Left','Right'],yticks=range(len(keys)),yticklabels=names,title='Posthoc spatial ARI');ax.tick_params(axis='y',labelsize=8);fig.colorbar(im,ax=ax,shrink=.45,label='Adjusted Rand index')
        ax=axes[1];counts=qc.groupby('side')[['day1_eligible','speed_eligible','common_eligible']].sum();xx=np.arange(2)
        for j,(col,name) in enumerate([('day1_eligible','Posture'),('speed_eligible','Full speed'),('common_eligible','Common cohort')]):ax.bar(xx+(j-1)*.22,counts.loc[['left','right'],col],.22,label=name,color=[COLORS[2],COLORS[1],COLORS[0]][j])
        ax.axhline(57,color=INK,ls=':',label='57 identities / colony');ax.set(xticks=xx,xticklabels=['Left','Right'],ylabel='Ants',ylim=(0,64),title='All 114 identities audited');ax.legend(frameon=False,fontsize=8)
        ax=axes[2]
        for j,side in enumerate(('left','right')):
            part=comparison[(comparison.scope=='common')&(comparison.side==side)];ax.scatter(part.camera_distance_spearman,part.spatial_ari,c=COLORS[j],s=35,alpha=.7,label=side.title())
            primary=part[(part.representation==selection['representation'])&(part.algorithm==selection['algorithm'])];ax.scatter(primary.camera_distance_spearman,primary.spatial_ari,marker='*',s=180,c=COLORS[j],edgecolors=INK)
        ax.set(xlabel='Activity vs camera-occupancy distance\nSpearman correlation',ylabel='Spatial-label ARI',title='Observation-pattern control');ax.legend(frameon=False,fontsize=9)
        fig.text(.075,.035,'Star = activity-selected approach. Camera occupancy was excluded from fitting, but correlation can still reveal measurement or spatial confounding.',fontsize=10)
        fig.text(.075,.018,'Broader speed-only cohort results, all candidate tests and permutation comparisons are in the accompanying CSVs and report. No posthoc retuning was performed.',fontsize=10)
        fig.subplots_adjust(left=.22,wspace=.55)
        save(fig,6,out,pdf)

        fig,axes=plt.subplots(2,3,figsize=(17,10))
        title(fig,7,'Day 2 was excluded from choosing the approach · physical measurements describe the activity-selected groups afterward')
        for i,(side,r) in enumerate(results.items()):
            t=r['table'];row=r['selection_row'];k=r['selected_k'];ax=axes[i,0]
            vals=[row['replicate_retention'],row['day2_retention']];ax.bar([0,1],np.asarray(vals)*100,color=[COLORS[2],COLORS[0]])
            ax.set(xticks=[0,1],xticklabels=['Day-1 A/B','Day-1 → day-2'],ylabel='Same group assignment (%)',ylim=(0,105),title=f'{side.title()} · day-2 n={int(row["day2_n"])}')
            ax.text(.03,.06,f'Day-2 activity prediction gain: {row["day2_profile_gain"]:.1%}',transform=ax.transAxes,fontsize=9)
            if k==1:ax.text(.5,.5,'Retention is undefined for K=1',transform=ax.transAxes,ha='center',fontsize=9)
            speed=bank['speed_hourly'][0,r['rows']]
            with np.errstate(invalid='ignore'):
                with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);mean_speed=np.nanmedian(np.expm1(speed[:,:,0])*.1,axis=1);active=np.nanmedian(speed[:,:,5],axis=1)*100
            dots(axes[i,1],mean_speed,active,t.group,k);axes[i,1].set(xlabel='Typical block-mean speed (mm/s)',ylabel='Median hourly fraction >0.2 mm/s (%)',title='Physical locomotor interpretation');axes[i,1].legend(frameon=False,fontsize=8)
            ax=axes[i,2]
            for g in range(k):
                values=speed[t.group.eq(g),:,5]*100
                with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);med=np.nanmedian(values,axis=0);lo,hi=np.nanquantile(values,[.25,.75],axis=0)
                ax.plot(range(24),med,color=COLORS[g],label=label(g,k));ax.fill_between(range(24),lo,hi,color=COLORS[g],alpha=.18)
            ax.set(xlabel='Clock hour (day 1)',ylabel='Fraction >0.2 mm/s (%)',xticks=[0,6,12,18,23],xticklabels=['10','16','22','04','09'],title='Hourly activity · median and ant IQR');ax.legend(frameon=False,fontsize=8)
        fig.text(.075,.035,'Speed is smoothed anchor motion with existing >5 mm/s filtering. Missingness and the imaging geometry remain possible confounds.',fontsize=10)
        fig.text(.075,.018,'Activity groups may describe a reproducible division of a continuous activity gradient. A successful spatial match does not by itself prove two discrete behavioral types.',fontsize=10)
        save(fig,7,out,pdf)
    write_outputs(out,results,qc,comparison,ranking,selection,spatial_sources,all_models)


def number(value,fmt='.2f'):
    return format(value,fmt) if pd.notna(value) else 'N/A'


def write_outputs(out,results,qc,comparison,ranking,selection,spatial_sources,all_models):
    chosen=FAMILY_LABELS[selection['representation']]+' / '+selection['algorithm'].upper()
    lines=['# 0724: discovering activity groups before examining space','',
      f'Activity-only selection: **{chosen}**. Ten predefined representations and two clustering families were compared on the same eligible ants. Spatial labels were opened after fitting, assignment and method-selection hashes were saved. This remains exploratory because the spatial segregation was already known from earlier work.','',
      '| Colony | Ants | Selected K | Bootstrap median ARI | A/B retention | Day-2 retention (n) | Day-2 activity prediction gain | Spatial ARI |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for side,r in results.items():
        row=r['selection_row'];lines.append(f'| {side.title()} | {len(r["table"])} | {r["selected_k"]} | {number(row["bootstrap_ari_median"])} | {number(row["replicate_retention"],".0%")} | {number(row["day2_retention"],".0%")} ({int(row["day2_n"])}) | {number(row["day2_profile_gain"],".1%")} | {number(row["spatial_ari"])} |')
    pose_only=comparison[(comparison.scope=='common')&comparison.representation.isin(['motif_clock','motif_repertoire','posture_shape','short_dynamics'])]
    if pose_only.selected_k.eq(1).all():
        lines += ['', '**The improvement comes from full-trajectory locomotor activity.** Neither hourly motif representation, posture shape/amplitude nor short-time postural/velocity dynamics alone produced a supported split under these rules. The primary speed representation comprises seven block-level measurements (mean, median, 90th/99th percentile speed and fractions above 0.05, 0.2 and 1 mm/s), aggregated by hour and summarized by their 25th/50th/75th percentiles across hours. These 21 features describe activity without clock alignment or any location input.']
    lines += ['', '**Sensitivity matters.** '+ ' '.join(f'{side.title()} bootstrap ARI has a 10th percentile of {r["selection_row"]["bootstrap_ari_p10"]:.2f}, despite a median of {r["selection_row"]["bootstrap_ari_median"]:.2f}.' for side,r in results.items())]
    broader=comparison[(comparison.scope=='speed_all')&(comparison.representation==selection['representation'])&(comparison.algorithm==selection['algorithm'])]
    if len(broader):
        lines += ['', 'Using the same representation/model on the broader speed-qualified cohort gives '+ '; '.join(f'{r.side}: n={r.n_ants}, K={r.selected_k}, spatial ARI {number(r.spatial_ari)}' for r in broader.itertuples())+'. Cohort sensitivity limits the claim of a robust two-class division. See all Gaussian-mixture alternatives below; a stable KMeans partition is not evidence by itself for two distributional modes.']
    lines += ['', '## What changed', '',
      'Figure 3 now includes a UMAP distribution panel and the same embedding colored by the 12 posture–velocity motifs. Each motif is still a nearest-center assignment in the original 143-dimensional one-second history space (9 posture PCs + signed forward and lateral velocities, sampled 13 times). KMeans centers and motif count retain the preceding, frozen day-1 fit. The embedding is a display, not a new clustering algorithm. UMAP does not reliably preserve original-space density ([official UMAP documentation](https://umap-learn.readthedocs.io/en/latest/faq.html)).', '',
      'The individual-level comparison tests clock-aligned motif vectors, quantiles of hourly motif frequencies, posture shape/amplitude, short-time postural dynamics, locomotion levels, movement/rest-like bout structure, clock-aligned locomotion, and three combined representations. Motif repertoire features summarize **hourly** measurements; no pooled whole-day motif frequencies are fitted. The low-dimensional posture approach is inspired by [Stephens et al. (2008)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000028), without assuming that ant posture has the same mode structure as worms.', '',
      '## Selection and validation', '',
      'Within each representation, training-only median imputation, robust scaling, feature-block balancing and up to three PCs precede clustering. Gaussian mixtures compare K=1–4 with tied/diagonal/full covariance and require a BIC improvement of at least 10 over the best K=1 fit. KMeans requires silhouette ≥0.25 and chooses the smallest K within 0.02 of the best eligible silhouette. Both require at least four ants per group, median whole-ant bootstrap ARI ≥0.8 (100 resamples including refitted preprocessing), and ≥80% agreement between disjoint alternating five-minute measurement blocks. Otherwise the result is K=1; rejected candidates are retained in `all_candidate_tests.csv`.', '',
      'A shared method is selected first by the number of colonies with supported nontrivial partitions, then by mean A/B activity prediction gain. This explicit preference for stable partitions does **not** establish intrinsic discrete classes. Prediction gain compares a group centroid measured in A with the same ant’s B activity, relative to the colony mean measured in A; A/B are then reversed. Different representations define different prediction targets, so these gains measure internal reproducibility rather than one universally comparable biological loss.', '',
      'Day 1 is July 24 10:00–July 25 10:00; day 2 is the next 24 hours. Day 2 does not select features, K, normalization or the primary method. Day-2 retention uses the fixed day-1 model, and prediction uses day-1 group membership/centroids. Ants are resampling units; this dataset has only two colonies, so the analysis does not establish generality across colonies.', '',
      '## Every attempted method: common cohort', '',
      '| Representation | Algorithm | Left K / spatial ARI | Right K / spatial ARI | Mean A/B gain |', '|---|---|---:|---:|---:|']
    for rr in ranking.itertuples():
        cells=[]
        for side in ('left','right'):
            row=comparison[(comparison.scope=='common')&(comparison.representation==rr.representation)&(comparison.algorithm==rr.algorithm)&(comparison.side==side)].iloc[0];cells.append(f'{int(row.selected_k)} / {number(row.spatial_ari)}')
        lines.append(f'| {FAMILY_LABELS[rr.representation]} | {rr.algorithm} | {cells[0]} | {cells[1]} | {rr.mean_activity_gain:.1%} |')
    eligible=comparison[(comparison.scope=='common')&comparison.spatial_ari.notna()].groupby(['representation','algorithm']).agg(colonies=('side','nunique'),mean_ari=('spatial_ari','mean')).query('colonies == 2').sort_values('mean_ari',ascending=False)
    if len(eligible):
        family,algorithm=eligible.index[0];lines += ['',f'The strongest **posthoc** mean spatial correspondence among methods with supported splits in both colonies was {FAMILY_LABELS[family]} / {algorithm}, mean ARI {eligible.iloc[0].mean_ari:.2f}. This is reported as an exploratory comparison, not substituted for the activity-selected primary method. No model was retuned after spatial comparison.']
    lines += ['', '## Cohort and missingness', '', '| Colony | Identities | Posture qualified | Speed qualified | Common cohort |', '|---|---:|---:|---:|---:|']
    for side,t in qc.groupby('side'):lines.append(f'| {side.title()} | {len(t)} | {t.day1_eligible.sum()} | {t.speed_eligible.sum()} | {t.common_eligible.sum()} |')
    lines += ['', 'Every identity is audited in `activity_cohort_audit.csv`. Posture requires ≥16 hours with ≥5 accepted clips; full speed requires ≥16 hours with ≥50% observed seconds and ≥3 usable five-minute blocks per hour. The common cohort is their intersection. Four locomotor representations are additionally evaluated on all speed-qualified ants, independently of posture eligibility.', '', '## Broader speed-qualified cohort', '', '| Representation | Model | Colony | n | K | Spatial ARI | Day-2 retention |', '|---|---|---|---:|---:|---:|---:|']
    for r in comparison[comparison.scope.eq('speed_all')].itertuples():lines.append(f'| {FAMILY_LABELS[r.representation]} | {r.algorithm} | {r.side} | {r.n_ants} | {r.selected_k} | {number(r.spatial_ari)} | {number(r.day2_retention,".0%")} |')
    lines += ['', '## Spatial prediction and controls', '']
    for side,r in results.items():
        row=comparison[comparison.key.eq(r['selection_row']['key'])].iloc[0];lo,hi=r['forecast_ci'];lines.append(f'- {side.title()}: leave-one-ant-out day-2 spatial prediction gain {r["forecast_mean"]:.4f} squared Hellinger units (ant-bootstrap 95% interval {lo:.4f}–{hi:.4f}; n={len(r["forecast_gain"])}). Activity-PC1/pose-hour-coverage Spearman r={number(row.pc1_coverage_spearman)}; activity-distance/camera-occupancy-distance Spearman r={number(row.camera_distance_spearman)}.')
    lines += ['', 'Spatial prediction compares other ants’ day-1 group-average occupancy with the target ant’s day-2 occupancy, against the other ants’ colony-average map. Positions require ≥40% coverage on both days. The interval treats ants as independent resampling units within a colony and is descriptive. Existing spatial-label agreement is a separate posthoc endpoint; those labels use the recording including day 2 and are not an independent external validation.', '',
      'Camera identity, location and detection coverage were excluded as fitted features. They may still affect measurement availability or apparent movement. Median imputation can turn observation structure into apparent similarity. The distance correlations above are a confounding diagnostic, not an independent-sample statistical test or a correction. Full speed uses smoothed TrackX/Y motion with the existing 5 mm/s cap, up to five-frame gap interpolation and two-frame Gaussian smoothing. Movement/run thresholds (0.05 and 0.2 mm/s) are operational measurements; runs end at missing seconds or five-minute boundaries and are censored lower bounds, not sleep labels.', '',
      'Even a stable K=2 partition can divide a continuous activity gradient. Agreement with spatial groups supports a relationship between activity and occupancy; it does not prove two discrete innate types or eliminate camera/coverage confounding. Confirmation requires a new colony or recording with this feature definition and selection rule frozen.', '',
      '## Files and reproduction', '',
      '[Combined PDF](0724_activity_discovery.pdf) · [Figure gallery](index.html) · [Interactive UMAP and method explorer](explorer.html). `method_comparison_with_controls.csv` contains every fit and validation metric; `all_candidate_tests.csv` records K/covariance tests. Assignments, feature arrays, UMAP coordinates, frozen model hashes, software versions, source stamps and all 114 inclusion decisions are saved. See `PROTOCOL.md` and `REPRODUCE.md` in the published directory. Permutation p-values are uncorrected exploratory diagnostics across a method search; they are not confirmatory significance tests.','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    findings=' '.join(f'{side.title()}: K={r["selected_k"]}, spatial ARI {number(r["selection_row"]["spatial_ari"])}.' for side,r in results.items())
    sections=''.join(f'<section><h2>{i:02d} · {html.escape(name)}</h2><p>{html.escape(desc)}</p><a href="{stem}.pdf"><img src="{stem}.png" alt="{html.escape(name)}" loading="lazy"></a></section>' for i,(stem,name,desc) in enumerate(FIGURES,2))
    (out/'index.html').write_text(f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>0724 activity discovery</title><style>body{{font:17px/1.55 system-ui;background:#f3f5f6;color:#263448;margin:0}}main{{max-width:1300px;margin:auto;padding:35px 24px}}h1{{font-size:38px}}a{{color:#087e80}}section{{background:white;padding:22px;border-radius:12px;margin:24px 0}}img{{width:100%;height:auto}}nav a{{margin-right:22px}}</style><main><p>20260724 / BLOCK01</p><h1>Activity first, spatial outcomes afterward</h1><p>Selected using activity alone: <b>{html.escape(chosen)}</b>. {findings}</p><nav><a href="{PDF}">Six-figure PDF</a><a href="explorer.html">Interactive explorer</a><a href="REPORT.md">Full report</a></nav><p>Ten activity representations × two clustering methods. Every attempted method is reported; no binary partition is forced.</p>{sections}</main></html>')
    with np.load(out/'motif_umap.npz') as z:umap_data=dict(x=z['embedding'][:,0].round(4).tolist(),y=z['embedding'][:,1].round(4).tolist(),motif=z['motif'].astype(int).tolist(),ant=z['ant'].tolist())
    assignments=pd.read_csv(out/'activity_assignments_with_spatial_labels.csv');method_data=[]
    for row in comparison.itertuples():
        part=assignments[assignments.key.eq(row.key)];method_data.append(dict(key=row.key,name=FAMILY_LABELS[row.representation]+' / '+row.algorithm+' / '+row.side+' / '+row.scope,k=int(row.selected_k),ari=number(row.spatial_ari),retention=number(row.day2_retention,'.0%'),points=part[['ant','pc1','pc2','group','spatial_label']].replace({np.nan:None}).to_dict('records')))
    data=dict(umap=umap_data,methods=method_data,primary=next(r['selection_row']['key'] for r in results.values()))
    template=Path(__file__).with_name('activity_discovery_explorer.html').read_text();(out/'explorer.html').write_text(template.replace('__DATA__',json.dumps(data,allow_nan=False).replace('</','<\\/')))
    if any(stamp(item['path'])!=item for item in spatial_sources):raise ValueError('Plot inputs changed during rendering')
    (out/'plot_provenance.json').write_text(json.dumps(spatial_sources,indent=2)+'\n')
    (out/'COMPLETE.json').write_text(json.dumps(dict(complete=True,figures=len(FIGURES),selection=selection))+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','block','dictionary','hourly-reference'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();render(a.output,a.block,a.dictionary,a.hourly_reference)

if __name__=='__main__':main()
