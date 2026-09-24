"""Focused figures for one activity model, K selection and spatial correspondence."""
from __future__ import annotations
import hashlib
import html
import json
from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.backends.backend_pdf import PdfPages
from analysis.activity_discovery import ActivityTransform
from analysis.activity_discovery_plots import read_primary,number
from analysis.postural_dynamics_plots import setup,draw_pose,map_plot,INK
from analysis.postural_dynamics_extract import stamp

COLORS=['#128a8b','#d7754e','#6671b7','#b7912d']
MOTIF_COLORS=['#128a8b','#d7754e','#6671b7','#b7912d','#44966c','#b15c9b','#56687d','#d6a3a1','#7c7843','#549cb3','#a47651','#9579be']
PDF='0724_activity_landscape.pdf'
FIGURES=[('01_eigenpostures','Discover a posture basis','Body-aligned postures and a density heat map.'),
 ('02_motifs','Map posture–velocity motifs','Density-preserving UMAP and the original-space motif assignments.'),
 ('03_choose_k','Choose the number of activity groups','One speed-based representation and KMeans; assess K=2–4 using activity alone.'),
 ('04_spatial_agreement','Compare activity and spatial classes','Identical activity coordinates, independent labels and every disagreement shown.'),
 ('05_spatial_maps','Reveal spatial segregation','Day-2 occupancy of the groups learned from day-1 activity.'),
 ('06_persistence','Validate and interpret the activity groups','Following-day persistence and the physical activity differences.')]


def title(fig,n,subtitle):
    fig.suptitle(f'{n:02d}  {FIGURES[n-1][1]}',x=.035,ha='left',fontsize=19,fontweight='bold')
    fig.text(.035,.925,subtitle,fontsize=10,color='#596877',va='top')
    fig.subplots_adjust(top=.82,bottom=.12,left=.075,right=.96,hspace=.60,wspace=.48)


def save(fig,n,out,pdf):
    stem=FIGURES[n-1][0];fig.savefig(out/(stem+'.png'),dpi=180);fig.savefig(out/(stem+'.pdf'));pdf.savefig(fig);plt.close(fig)


def label(g,k):return 'Unsplit cohort' if k==1 else f'A{g+1}'


def dots(ax,x,y,g,k):
    for j in range(k):
        keep=np.asarray(g)==j;ax.scatter(np.asarray(x)[keep],np.asarray(y)[keep],s=30,color=COLORS[j],edgecolor='white',lw=.4,label=label(j,k),alpha=.9)


def spatial_alignment(table):
    observed=table.dropna(subset=['spatial_label'])
    c=pd.crosstab(observed.group,observed.spatial_label).reindex(index=[0,1],fill_value=0)
    rows,cols=linear_sum_assignment(-c.to_numpy())
    mapping={str(c.columns[j]):int(i) for i,j in zip(rows,cols)}
    matched=int(c.to_numpy()[rows,cols].sum())
    return dict(spatial_to_activity_color=mapping,spatial_labels=c.columns.tolist(),counts=c.to_numpy().tolist(),matched=matched,n=len(observed))


def render(out,block,dictionary):
    setup();frozen=json.loads((out/'ACTIVITY_ONLY_FROZEN.json').read_text());selection=frozen['selection']
    comparison=pd.read_csv(out/'primary_activity_metrics.csv');all_models=joblib.load(out/'activity_models.joblib');qc=pd.read_csv(out/'activity_cohort_audit.csv');candidates=pd.read_csv(out/'primary_k_selection.csv')
    results,sources=read_primary(out,block,qc,comparison,all_models,selection)
    sources += [stamp(dictionary/name) for name in ('space_blind_models.npz','posture_display_samples.npz','posture_reconstruction.csv','history_prediction_validation.csv','motif_resolution_validation.csv')]
    with np.load(dictionary/'space_blind_models.npz') as z:models={k:z[k] for k in z.files}
    with np.load(out/'activity_feature_bank.npz') as z:bank={k:z[k] for k in z.files}
    umap_info=json.loads((out/'umap_metadata.json').read_text())
    manifest=json.loads((out/'run_manifest.json').read_text());source=Path(manifest['source'])
    old_umap=json.loads((source/'umap_metadata.json').read_text())
    umap_info['previous_trustworthiness']=old_umap['trustworthiness_15nn_on1500']
    manifest['reference_input_sha256']={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in ('motif_umap.npz','umap_metadata.json')}
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    sources.append(stamp(source/'umap_metadata.json'))
    pca=PCA();pca.mean_=models['posture_mean'];pca.components_=models['posture_components'];pca.explained_variance_=models['posture_variance'];pca.explained_variance_ratio_=models['posture_variance_ratio']
    lengths=models['standard_lengths_mm'];examples=models['motif_example_shape'];sample=np.load(dictionary/'posture_display_samples.npz');recon=pd.read_csv(dictionary/'posture_reconstruction.csv');history=pd.read_csv(dictionary/'history_prediction_validation.csv');first=history.iloc[0]
    fit=dict(rank=int(models['rank']),n_motifs=int(models['n_motifs']),history_table=history,resolution=pd.read_csv(dictionary/'motif_resolution_validation.csv'),persistence_mse=first.mean_mse/(1-first.gain_vs_persistence))
    regions=pd.read_csv(block/'panorama_regions.csv').to_dict('records')
    for reg in regions:
        name=reg['semantic_label'];reg['side']='left' if name.endswith('L') or name.endswith('_left') else 'right';reg['region_type']='arena' if name.startswith('arena') else name[:-1]
    metadata={}
    for side,r in results.items():
        track=Path(r['table'].track_name.iloc[0]).stem;metadata[side]=json.loads((block/'stitched/grid_occupancy_histograms_arena_0p25mm/per_track'/track/'grid_occupancy_metadata.json').read_text())
        sources.append(stamp(block/'stitched/grid_occupancy_histograms_arena_0p25mm/per_track'/track/'grid_occupancy_metadata.json'))
    with PdfPages(out/PDF) as pdf:
        fig,ax=plt.subplots(2,4,figsize=(16,9))
        title(fig,1,f'PCA of 16 direction cosines · equal ant weights · {fit["rank"]} modes retain 95% of day-1 variance · basis frozen from 52 day-1 ants')
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
        save(fig,1,out,pdf)

        fig=plt.figure(figsize=(17,17));gs=fig.add_gridspec(4,4,height_ratios=[.55,1.25,.85,1.35])
        title(fig,2,f'{fit["rank"]} posture coefficients + signed forward/lateral velocity → 13 samples across 1 s → {fit["n_motifs"]} shared motifs')
        fig.subplots_adjust(top=.87,bottom=.065,hspace=.65,wspace=.48)
        descriptions=[('A  Measure and align','Project tracked anchor motion onto\nthe anterior and lateral body axes.\nAppend both signed velocities (mm/s).'),
          ('B  Scale and stack',f'Equal total weight: posture and velocity.\n13 time samples × {fit["rank"]+2} channels\n= {13*(fit["rank"]+2)} values per one-second history.'),
          ('C  Learn shared centers','KMeans groups similar histories.\nEach center defines a motif; a history\ngets its nearest-center motif label.'),
          ('D  Group individual activity','Use full-trajectory speed distributions.\n21 hourly-summary features → 3 PCs.\nFit KMeans; choose K using activity.')]
        for j,(heading,description) in enumerate(descriptions):
            ax=fig.add_subplot(gs[0,j]);ax.axis('off')
            ax.text(0,.95,heading,weight='bold',fontsize=11,va='top')
            ax.text(0,.68,description,fontsize=10,va='top',linespacing=1.6)
            if j<3:ax.text(1.08,.50,'→',fontsize=20,color=COLORS[0])
        with np.load(out/'motif_densmap.npz') as z:embedding=z['embedding'];labels=z['motif']
        ax=fig.add_subplot(gs[1,:2]);im=ax.hexbin(embedding[:,0],embedding[:,1],gridsize=65,bins='log',mincnt=1,cmap='magma');fig.colorbar(im,ax=ax,shrink=.75,label='Measured histories / bin (log scale)');ax.set(xlabel='densMAP 1',ylabel='densMAP 2',title=f'Density-preserving UMAP · {len(embedding):,} histories · equal ant sampling')
        ax=fig.add_subplot(gs[1,2:]);palette=MOTIF_COLORS
        for motif in range(fit['n_motifs']):
            select=labels==motif;ax.scatter(embedding[select,0],embedding[select,1],s=3,alpha=.6,c=[palette[motif]],label=f'M{motif:02d}',rasterized=True)
        ax.set(xlabel='densMAP 1',ylabel='densMAP 2',title='Same embedding · colored by original-space motif labels');ax.legend(frameon=False,ncol=6,fontsize=7,loc='upper center',bbox_to_anchor=(.5,-.17),markerscale=2)
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
        fig.text(.075,.022,f'densMAP: 15 neighbors, min_dist=0.01, density weight=2. Original/embedded local-radius correlation: {umap_info["original_radius_spearman_new"]:.2f} (previously {umap_info["original_radius_spearman_old"]:.2f}).',fontsize=9)
        fig.text(.075,.009,'Motifs describe short histories; individual groups use full-trajectory speed summaries. The embedding supplies neither motif labels nor ant groups.',fontsize=9)
        save(fig,2,out,pdf)
        fig,axes=plt.subplots(2,3,figsize=(16,11))
        title(fig,3,'7 locomotor measurements × 3 quantiles across hourly estimates → 21 features → robust scaling → 3 activity PCs → KMeans')
        for i,(side,r) in enumerate(results.items()):
            part=candidates[candidates.key.eq(r['selection_row']['key'])].sort_values('k');ks=part.k.to_numpy()
            ax=axes[i,0];ax.plot(ks,part.silhouette,'o-',color=INK);ax.axhline(.25,color='#8f9dac',ls=':',label='Minimum 0.25');ax.scatter([2],[part.loc[part.k.eq(2),'silhouette'].iloc[0]],s=95,c=COLORS[0],zorder=5)
            ax.set(xticks=ks,xlabel='Number of activity groups K',ylabel='Silhouette score',ylim=(0,.82),title=f'{side.title()} · separation and cohesion');ax.legend(frameon=False,fontsize=8)
            ax=axes[i,1];ax.plot(ks,part.bootstrap_ari_median,'o-',color=INK,label='Median');ax.plot(ks,part.bootstrap_ari_p10,'o--',color='#8f9dac',label='10th percentile');ax.axhline(.8,color=COLORS[0],ls=':',label='Required median ≥0.8');ax.set(xticks=ks,xlabel='Number of activity groups K',ylabel='Bootstrap adjusted Rand index',ylim=(0,1.08),title='Reproducibility across ant resamples');ax.legend(frameon=False,fontsize=8)
            ax=axes[i,2];ax.plot(ks,part.replicate_retention*100,'o-',color=INK);ax.axhline(80,color=COLORS[0],ls=':',label='Required ≥80%');ax.set(xticks=ks,xlabel='Number of activity groups K',ylabel='Same group in A/B measurements (%)',ylim=(65,103),title='Disjoint measurement replication');ax.legend(frameon=False,fontsize=8)
        fig.text(.075,.078,'Choose the smallest eligible K within 0.02 of the best silhouette. Eligibility: ≥4 ants/group, bootstrap median ARI ≥0.8, A/B retention ≥80%.',fontsize=10)
        fig.text(.075,.055,'K=2 has the highest silhouette in both colonies and passes every criterion. Left K=3 and both K=4 fits fail the bootstrap criterion; right K=3 has lower silhouette.',fontsize=10)
        fig.text(.075,.032,'K=1 is the fallback if no split qualifies. This selects a useful partition; it does not establish two intrinsic modes. Spatial labels and day 2 do not choose K.',fontsize=10)
        fig.subplots_adjust(bottom=.19,hspace=.65)
        save(fig,3,out,pdf)

        alignment={side:spatial_alignment(r['table']) for side,r in results.items()}
        (out/'spatial_agreement.json').write_text(json.dumps(alignment,indent=2)+'\n')
        fig,axes=plt.subplots(2,3,figsize=(17,10),gridspec_kw={'width_ratios':[1,1,.9]})
        title(fig,4,'The same ants in the same activity coordinates · spatial labels are revealed afterward · circles mark disagreements')
        for i,(side,r) in enumerate(results.items()):
            t=r['table'];a=alignment[side];mapping=a['spatial_to_activity_color'];matched=t.spatial_label.map(mapping);different=matched.notna()&matched.ne(t.group)
            ax=axes[i,0];dots(ax,t.pc1,t.pc2,t.group,2);ax.set(xlabel='Activity PC1',ylabel='Activity PC2',title=f'{side.title()} · activity groups (n={len(t)})');ax.legend(frameon=False,fontsize=8)
            ax=axes[i,1]
            for name,g in sorted(mapping.items(),key=lambda item:item[1]):
                keep=t.spatial_label.eq(name);ax.scatter(t.loc[keep,'pc1'],t.loc[keep,'pc2'],s=30,c=COLORS[g],edgecolors='white',linewidth=.4,label=f'S{g+1} ({name})')
            ax.scatter(t.loc[different,'pc1'],t.loc[different,'pc2'],s=95,facecolors='none',edgecolors=INK,lw=1)
            ax.set(xlabel='Activity PC1',ylabel='Activity PC2',title='Colored by independently fitted spatial classes');ax.legend(frameon=False,fontsize=8)
            columns=[next(n for n,g in mapping.items() if g==j) for j in range(2)];cross=pd.crosstab(t.group,t.spatial_label).reindex(index=[0,1],columns=columns,fill_value=0).to_numpy();proportions=cross/np.maximum(cross.sum(axis=1,keepdims=True),1)
            ax=axes[i,2];im=ax.imshow(proportions,cmap='Blues',vmin=0,vmax=1)
            for row in range(2):
                for col in range(2):ax.text(col,row,f'{cross[row,col]} '+('ant' if cross[row,col]==1 else 'ants')+f'\n{proportions[row,col]:.0%}',ha='center',va='center',color='white' if proportions[row,col]>.6 else INK,fontsize=11)
            ax.set(xticks=[0,1],xticklabels=['S1','S2'],yticks=[0,1],yticklabels=['A1','A2'],xlabel='Spatial class',ylabel='Activity group',title=f'{a["matched"]}/{a["n"]} agree · ARI {r["selection_row"]["spatial_ari"]:.2f}')
        fig.text(.075,.04,'Spatial-class colors are permuted only to align names for display; no ant is reassigned and no activity coordinates change. Percentages are within activity groups.',fontsize=10)
        fig.text(.075,.021,'Agreement is strong but incomplete: the circled ants and off-diagonal cells remain visible. Existing spatial labels use the full recording.',fontsize=10)
        save(fig,4,out,pdf)
        cols=max(r['selected_k'] for r in results.values())+1
        fig,axes=plt.subplots(2,cols,figsize=(max(15,cols*4),10),squeeze=False)
        title(fig,5,'Day-1 activity labels → day-2 occupancy · spatial outcomes opened only after the activity model was frozen')
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

        fig,axes=plt.subplots(2,3,figsize=(17,10))
        title(fig,6,'Day 2 was excluded from choosing the approach · physical measurements describe the activity-selected groups afterward')
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
        save(fig,6,out,pdf)
    write_outputs(out,results,qc,candidates,alignment,umap_info,sources,bank)


def write_outputs(out,results,qc,candidates,alignment,umap_info,sources,bank):
    lines=['# 0724: activity groups and emergent spatial segregation','',
      'One method is used throughout: **KMeans on three activity PCs derived from 21 hourly locomotor-summary features**. The fitted models and individual assignments are retained exactly from the preceding activity-selected analysis. The figures now run from **1 to 6**, with K justification and direct activity–spatial agreement replacing the method exploration.','',
      '## The activity representation','',
      'For each ant, five-minute blocks measure mean, median, 90th- and 99th-percentile speed and fractions moving above 0.05, 0.2 and 1 mm/s. Valid blocks are averaged within each hour. The 25th, 50th and 75th percentiles across hourly measurements give 7 × 3 = 21 features. Training-only median imputation and robust scaling precede three activity PCs and KMeans. Location, camera identity and spatial class are absent from these features. The posture basis and short-history motifs in Figures 1–2 characterize behavior at a different timescale; they are not inputs to this individual clustering.','',
      '## Why K=2','',
      'Compare K=2, 3 and 4 within this single method. Require silhouette ≥0.25, at least four ants per group, median whole-ant bootstrap ARI ≥0.8 across 100 resamples including refitted preprocessing, and ≥80% assignment retention between disjoint alternating five-minute measurement blocks A/B. Among eligible solutions choose the smallest K within 0.02 of the highest silhouette. K=1 is the fallback if no candidate passes. Neither spatial labels nor day 2 choose K.','',
      '| Colony | K | Silhouette | Bootstrap median ARI | A/B retention | Minimum group size |', '|---|---:|---:|---:|---:|---:|']
    for side,r in results.items():
        for row in candidates[candidates.key.eq(r['selection_row']['key'])].sort_values('k').itertuples():lines.append(f'| {side.title()} | {row.k} | {row.silhouette:.3f} | {row.bootstrap_ari_median:.3f} | {row.replicate_retention:.1%} | {int(row.smallest_group)} |')
    lines += ['', '**K=2 has the highest silhouette in both colonies and passes every threshold.** Left K=3 and both K=4 fits fail the bootstrap criterion. Right K=3 passes the thresholds but has substantially lower silhouette. This justifies the two-group partition for this representation and cohort; it does not prove two intrinsic distributional modes.','',
      '## Agreement with spatial clustering','',
      'Figure 4 shows identical activity coordinates colored first by activity group and then by the independently fitted spatial class. Circles and contingency tables display every disagreement. Spatial labels are only permuted for color/name alignment; activity membership and coordinates remain fixed.','',
      '| Colony | Activity ants | Matching spatial class | Spatial ARI | Next-day retention |', '|---|---:|---:|---:|---:|']
    colonies={}
    for side,r in results.items():
        a=alignment[side];row=r['selection_row'];t=r['table'];mapping=a['spatial_to_activity_color']
        lines.append(f'| {side.title()} | {len(t)} | {a["matched"]}/{a["n"]} ({a["matched"]/a["n"]:.1%}) | {row["spatial_ari"]:.3f} | {row["day2_retention"]:.1%} (n={int(row["day2_n"])}) |')
        points=t[['ant','pc1','pc2','group','spatial_label']].copy();points['spatial_color']=points.spatial_label.map(mapping)
        colonies[side]=dict(n=len(t),matched=a['matched'],ari=row['spatial_ari'],retention=row['day2_retention'],points=points.replace({np.nan:None}).to_dict('records'),counts=a['counts'],spatial_labels=a['spatial_labels'])
    lines += ['', 'Figure 5 then reveals the spatial occupancy associated with those activity labels on day 2.']
    for side,r in results.items():
        t=r['table'];t=t[t.position_coverage_day2.ge(.4)];med=t.groupby('group').day2_colony_percent.median()
        lines.append(f'- {side.title()}: median day-2 colony-region occupancy is {med.loc[0]:.1f}% in A1 and {med.loc[1]:.1f}% in A2. Spatial forecast gain over a colony-average map is {r["forecast_mean"]:.4f} squared Hellinger units (ant-bootstrap 95% interval {r["forecast_ci"][0]:.4f}–{r["forecast_ci"][1]:.4f}).')
    lines += ['', 'The occupancy maps average ants equally and require ≥40% day-2 position coverage. Spatial forecasts predict each ant’s day-2 occupancy from other ants’ day-1 group-average maps, with a leave-one-ant-out colony-average baseline. Existing spatial classes use the full recording, so their agreement is a posthoc correspondence endpoint rather than external validation.','',
      '## Density-preserving UMAP','',
      'The previous 30-neighbor, min_dist=0.1 UMAP is replaced by **densMAP with 15 neighbors, min_dist=0.01, density weight 2, density fraction 0.3 and 700 epochs**, with seed 724. All 6,600 histories and their original 143 features are unchanged. Each of the 66 eligible ants contributes 100 histories. Motif labels remain the frozen nearest-center assignments in the original history space; neither motif nor spatial labels supervise the embedding. densMAP explicitly adds local-density preservation to UMAP ([official documentation](https://umap-learn.readthedocs.io/en/latest/densmap_demo.html)).','',
      f'The Spearman correlation of original versus embedded local log-neighbor radii increases from **{umap_info["original_radius_spearman_old"]:.3f} to {umap_info["original_radius_spearman_new"]:.3f}** (15 neighbors, identical samples). Neighborhood trustworthiness changes from {umap_info["previous_trustworthiness"]:.3f} to {umap_info["trustworthiness_15nn_on1500"]:.3f} on the fixed 1,500-history subsample: improved density preservation trades off some neighborhood fidelity. The embedding can preserve density differences more faithfully without implying that its visual regions are distinct behavioral types. Figure 2 displays both history counts per embedding bin and motif colors.','',
      '## Cohort and interpretation limits','',
      'All 114 identities (57 per colony) are audited. The retained common cohort has 30 left and 36 right ants. Posture eligibility requires ≥16 hours with ≥5 accepted clips; speed eligibility requires ≥16 hours with ≥50% observed seconds and ≥3 usable five-minute blocks per hour. Day-2 retention is restricted to independently eligible day-2 measurements (20 left, 31 right). Day 1 starts July 24 at 10:00 and day 2 starts July 25 at 10:00.','',
      'The presentation is streamlined, but the result remains exploratory. The left partition is sensitive to cohort composition: the same rule on all 31 speed-qualified left ants falls back to K=1. Primary bootstrap ARI 10th percentiles are 0.43 and 0.78, despite medians of 1.00. Activity-distance/camera-occupancy-distance correlations are 0.65 and 0.76, so measurement and spatial confounding remain possible. Full speed uses the existing two-frame smoothing, interpolation across ≤5-frame gaps and >5 mm/s filtering. A stable partition can divide a continuous activity gradient; another recording with the procedure fixed is needed for confirmation.','',
      '[Six-figure PDF](0724_activity_landscape.pdf) · [Interactive explorer](explorer.html) · [Figure gallery](index.html). `primary_k_selection.csv`, `spatial_agreement.json`, primary per-ant tables, source hashes and UMAP metadata provide the numerical record. The preceding exploration remains archived at the source path in `run_manifest.json`.','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    sections=''.join(f'<section><h2>{i:02d} · {html.escape(name)}</h2><p>{html.escape(desc)}</p><a href="{stem}.pdf"><img src="{stem}.png" alt="{html.escape(name)}" loading="lazy"></a></section>' for i,(stem,name,desc) in enumerate(FIGURES,1))
    matches=' · '.join(f'{s.title()}: {alignment[s]["matched"]}/{alignment[s]["n"]} spatial-class agreement' for s in results)
    (out/'index.html').write_text(f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>0724 · Activity landscape</title><style>body{{font:17px/1.55 system-ui;background:#f3f5f6;color:#263448;margin:0}}main{{max-width:1300px;margin:auto;padding:35px 24px}}h1{{font-size:38px}}a{{color:#087e80}}section{{background:white;padding:22px;border-radius:12px;margin:24px 0}}img{{width:100%;height:auto}}nav a{{margin-right:22px}}</style><main><p>20260724 / BLOCK01</p><h1>Activity groups, spatial segregation</h1><p>Full-trajectory locomotor summaries → three activity PCs → KMeans. K=2 is selected using activity measurements; spatial correspondence is revealed afterward.</p><p><b>{matches}</b></p><nav><a href="{PDF}">Six-figure PDF</a><a href="explorer.html">Interactive explorer</a><a href="REPORT.md">Methods and results</a></nav>{sections}</main></html>')
    with np.load(out/'motif_densmap.npz') as z:
        density=-z['original_log_radius'];q0,q1=np.quantile(density,[.02,.98]);density=np.clip((density-q0)/(q1-q0),0,1)
        umap_data=dict(x=z['embedding'][:,0].round(5).tolist(),y=z['embedding'][:,1].round(5).tolist(),motif=z['motif'].astype(int).tolist(),ant=z['ant'].tolist(),density=density.round(4).tolist())
    data=dict(umap=umap_data,colonies=colonies,density_correlation=umap_info['original_radius_spearman_new'])
    template=Path(__file__).with_name('activity_landscape_explorer.html').read_text();(out/'explorer.html').write_text(template.replace('__DATA__',json.dumps(data,allow_nan=False).replace('</','<\\/')))
    if any(stamp(x['path'])!=x for x in sources):raise ValueError('Changed plot source')
    (out/'plot_provenance.json').write_text(json.dumps(sources,indent=2)+'\n')
    frozen=json.loads((out/'ACTIVITY_ONLY_FROZEN.json').read_text())
    if hashlib.sha256((out/'activity_models.joblib').read_bytes()).hexdigest()!=frozen['model_sha256']:raise ValueError('Activity models changed')
    (out/'COMPLETE.json').write_text(json.dumps(dict(complete=True,figures=6,numbering=list(range(1,7)),models_unchanged=True,umap_parameters=umap_info['parameters']))+'\n')
