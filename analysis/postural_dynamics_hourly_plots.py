"""Figures and offline explorer for concatenated hourly ant profiles."""
from __future__ import annotations
import hashlib
import html
import json
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import PowerNorm
from sklearn.decomposition import PCA
from analysis.postural_dynamics_plots import setup, draw_pose, map_plot, INK, COLORS as BASE_COLORS
from analysis.postural_dynamics_hourly import PRIMARY, MIN_CLIPS, MIN_HOURS, project_observed

COLORS=['#128a8b','#d7754e','#6671b7','#b7912d','#ac5e91','#57954b']
FIGURES=[('02_eigenpostures','Discover a posture basis','The fixed body-shape basis, learned from day 1.'),
 ('03_short_dynamics','Form motifs from posture and velocity','Assign one-second histories to the fixed joint dictionary; measure usage hourly.'),
 ('04_ant_profiles','Cluster concatenated hourly activity','One ant = 24 clock-ordered hourly motif-frequency vectors. Display the selected number of groups.'),
 ('05_spatial_outcomes','Reveal the spatial outcomes','Inspect held-out day-2 occupancy only after hourly groups are frozen.'),
 ('06_controls','Account for every tracked ant','All 57 identities per colony, hourly availability, and feature controls.'),
 ('07_persistence','Test the next day','Clock-matched hourly profiles and spatial forecasts, without refitting.')]
PDF='0724_hourly_postural_dynamics.pdf'


def title(fig,n,subtitle):
    fig.suptitle(f'{n:02d}  {FIGURES[n-2][1]}',x=.035,ha='left',fontsize=19,fontweight='bold')
    fig.text(.035,.919,subtitle,fontsize=10,color='#596877',va='top')
    fig.subplots_adjust(top=.82,bottom=.12,left=.075,right=.96,hspace=.60,wspace=.48)


def save(fig,n,out,pdf):
    stem=FIGURES[n-2][0];fig.savefig(out/(stem+'.png'),dpi=180);fig.savefig(out/(stem+'.pdf'));pdf.savefig(fig);plt.close(fig)


def group_name(g,k):
    return 'All eligible ants' if k==1 else f'G{int(g)+1}'


def dots(ax,x,y,groups,k):
    for g in range(k):
        mask=np.asarray(groups)==g
        ax.scatter(np.asarray(x)[mask],np.asarray(y)[mask],color=COLORS[g],s=27,edgecolor='white',lw=.4,label=group_name(g,k),alpha=.9)


def basis(models):
    p=PCA();p.mean_=models['posture_mean'];p.components_=models['posture_components'];p.explained_variance_=models['posture_variance'];p.explained_variance_ratio_=models['posture_variance_ratio'];return p


def render(out,results,qc,profiles,controls,models,dictionary,block,summary):
    setup();pca=basis(models);lengths=models['standard_lengths_mm'];examples=models['motif_example_shape']
    sample=np.load(dictionary/'posture_display_samples.npz');recon=pd.read_csv(dictionary/'posture_reconstruction.csv')
    history=pd.read_csv(dictionary/'history_prediction_validation.csv');first=history.iloc[0]
    fit=dict(rank=int(models['rank']),n_motifs=int(models['n_motifs']),history_table=history,resolution=pd.read_csv(dictionary/'motif_resolution_validation.csv'),persistence_mse=first.mean_mse/(1-first.gain_vs_persistence))
    regions=pd.read_csv(block/'panorama_regions.csv').to_dict('records')
    for reg in regions:
        label=reg['semantic_label'];reg['side']='left' if label.endswith('L') or label.endswith('_left') else 'right';reg['region_type']='arena' if label.startswith('arena') else label[:-1]
    metadata={}
    for side,r in results.items():
        track=Path(r['table'].track_name.iloc[0]).stem
        metadata[side]=json.loads((block/'stitched/grid_occupancy_histograms_arena_0p25mm/per_track'/track/'grid_occupancy_metadata.json').read_text())
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

        fig=plt.figure(figsize=(17,12));gs=fig.add_gridspec(3,4,height_ratios=[.7,1,1.55])
        title(fig,3,f'{fit["rank"]} posture coefficients + signed forward/lateral velocity → 13 samples across 1 s → {fit["n_motifs"]} shared motifs')
        fig.subplots_adjust(top=.83,bottom=.10,hspace=.65,wspace=.48)
        descriptions=[('A  Measure and align','Project tracked anchor motion onto\nthe anterior and lateral body axes.\nAppend both signed velocities (mm/s).'),
          ('B  Scale and stack',f'Equal total weight: posture and velocity.\n13 time samples × {fit["rank"]+2} channels\n= {13*(fit["rank"]+2)} values per one-second history.'),
          ('C  Learn shared centers','KMeans groups similar histories.\nEach center defines a motif; a history\ngets its nearest-center motif label.'),
          ('D  Count usage per hour','Assign clips with the fixed dictionary.\nCount motif frequency in each hour.\nConcatenate 24 hours for each ant.')]
        for j,(heading,description) in enumerate(descriptions):
            ax=fig.add_subplot(gs[0,j]);ax.axis('off')
            ax.text(0,.95,heading,weight='bold',fontsize=11,va='top')
            ax.text(0,.68,description,fontsize=10,va='top',linespacing=1.6)
            if j<3:ax.text(1.08,.50,'→',fontsize=20,color=COLORS[0])
        ax0=fig.add_subplot(gs[1,:2]);ax1=fig.add_subplot(gs[1,2:]);h=fit['history_table']
        for shuffled,color,label in [(False,INK,'Time-ordered history'),(True,COLORS[1],'Past times shuffled')]:
            part=h[h.shuffled.eq(shuffled)];ax0.errorbar(part.history_seconds,part.mean_mse,yerr=part.standard_error,color=color,marker='o',capsize=3,label=label)
        ax0.axhline(fit['persistence_mse'],color='#8f9dac',ls=':',label='Hold current posture + velocity')
        ax0.set(xlabel='Past history (seconds)',ylabel='Joint prediction error (scaled units)',title='Predict 0.25 s ahead · held-out hours · mean ± SE');ax0.legend(frameon=False,fontsize=8)
        res=fit['resolution'];ax1.errorbar(res.motifs,res.mean_mse,yerr=res.standard_error,color=INK,marker='o',capsize=4)
        best=res.loc[res.mean_mse.idxmin()]
        ax1.axhline(best.mean_mse+best.standard_error,color='#8f9dac',ls=':',label='Best mean + 1 SE')
        ax1.axvline(fit['n_motifs'],color=COLORS[0],ls='--',label='Selected count');ax1.set(xlabel='Number of shared motifs',ylabel='Joint prediction error (scaled units)',xticks=res.motifs,title='Resolution · smallest within 1 SE of best');ax1.legend(frameon=False,fontsize=8)
        motion=np.sqrt(np.square(models['motif_center_velocity_mm_s']).sum(axis=-1).mean(axis=1));order=np.argsort(motion)
        for j,index in enumerate(np.linspace(0,len(examples)-1,4).round().astype(int)):
            motif=int(order[index]);sub=gs[2,j].subgridspec(2,1,height_ratios=[1.2,1],hspace=.55)
            ax=fig.add_subplot(sub[0,0])
            for k,t in enumerate([12,16,20,24]):draw_pose(ax,examples[motif,t],lengths,plt.cm.viridis(k/3),alpha=.8,lw=1.3)
            ax.set_title(f'Motif {motif:02d} · measured example');ax.set_xlabel('Anterior axis (mm)')
            ax=fig.add_subplot(sub[1,0]);v=models['motif_example_velocity_mm_s'][motif,12:25]
            for component,name in enumerate(('Forward','Lateral')):ax.plot(np.arange(13)/12,v[:,component],color=COLORS[component],label=name)
            ax.axhline(0,color='#a8b2bd',lw=.7);ax.set(xlabel='Time within history (s)',ylabel='Velocity (mm/s)',xticks=[0,.5,1])
            if j==0:ax.legend(frameon=False,fontsize=8)
        fig.text(.075,.025,'Examples span the motif centers’ velocity range. Skeleton colors run from early purple to late yellow; each measured example is nearest its joint-feature center.',fontsize=9)
        save(fig,3,out,pdf)

        fig,axes=plt.subplots(2,4,figsize=(19,10),gridspec_kw={'width_ratios':[1.6,1,1,1]})
        title(fig,4,f'24 hourly vectors × {fit["n_motifs"]} motifs = {24*fit["n_motifs"]} ordered features per ant · missing hours masked · each ant has equal weight')
        for row,(side,r) in enumerate(results.items()):
            t=r['table'];k=r['selected_k'];order=np.lexsort((t.pc1,t.group));prob=r['profile'][0,order];ax=axes[row,0]
            cmap=plt.get_cmap('magma').copy();cmap.set_bad('#d7dce1')
            im=ax.imshow(prob.reshape(len(t),-1),aspect='auto',cmap=cmap,vmin=0,vmax=max(.1,np.nanquantile(prob,.99)),interpolation='nearest')
            fig.colorbar(im,ax=ax,shrink=.65,label='Hourly motif frequency',pad=.015)
            nm=fit['n_motifs']
            for h in range(1,24):ax.axvline(h*nm-.5,color='white',alpha=.22,lw=.35)
            for boundary in np.cumsum(t.groupby('group').size()).values[:-1]:ax.axhline(boundary-.5,color='white',lw=.7)
            ax.set(xticks=np.arange(0,24,4)*nm+(nm-1)/2,xticklabels=[f'{(10+h)%24:02d}' for h in range(0,24,4)],xlabel='Clock hour · motifs repeat within each hour',ylabel='Ants sorted by selected group, then PC1',title=f'{side.title()} · hourly frequencies · {len(t)}/57 ants')
            ax=axes[row,1];dots(ax,t.pc1,t.pc2,t.group,k);ax.set(xlabel='Hourly-profile PC1',ylabel='Hourly-profile PC2',title=f'Selected K={k}'+(' · no supported split' if k==1 else ''));ax.legend(frameon=False,fontsize=7)
            c=r['candidates'];ax=axes[row,2];ax.plot(c.k,c.silhouette,'o-',color=INK);ax.set(xlabel='Candidate groups',ylabel='Silhouette on shared hours',xticks=range(1,7),title=f'Rule selects K={k}');ax.axvline(k,ls='--',color=COLORS[0],lw=1)
            ax=axes[row,3];ax.plot(c.k,c.bootstrap_ari_median,'o-',color=INK);ax.fill_between(c.k,c.bootstrap_ari_p10,c.bootstrap_ari_p90,color=INK,alpha=.15);ax.axhline(.8,ls=':',color='#94a0ad');ax.set(xlabel='Candidate groups',ylabel='Adjusted Rand agreement',xticks=range(1,7),ylim=(-.05,1.05),title='100 whole-ant bootstraps · 10–90%')
        fig.text(.075,.025,'Select K≥2 only if every group has ≥4 ants and median stability ≥0.8; choose the smallest K within 0.02 of the best eligible silhouette. Otherwise K=1.',fontsize=9)
        fig.text(.075,.008,'Gray = unmeasured hour. Scatter is a display projection; clustering uses all measured hourly features. K=1 displays one group throughout, with no forced P0/P1 split.',fontsize=9)
        save(fig,4,out,pdf)

        max_groups=max(r['selected_k'] for r in results.values());cols=max_groups+2
        fig=plt.figure(figsize=(max(16,cols*3.2),10));gs=fig.add_gridspec(2,cols)
        title(fig,5,'Day-2 occupancy is a withheld outcome · selected hourly groups are used consistently in maps, labels and forecasts')
        for row,(side,r) in enumerate(results.items()):
            t=r['table'];k=r['selected_k'];good=t.position_coverage_day2.ge(.4).to_numpy();means=[]
            for g in range(k):
                eligible=good&t.group.eq(g).to_numpy();means.append(r['spatial_maps'][eligible,1].mean(axis=0) if eligible.any() else np.zeros_like(r['spatial_maps'][0,1]))
            xx,yy=r['spatial_edges'];vmax=max(1e-8,np.quantile(np.sqrt(np.stack(means)/(np.diff(yy)[:,None]*np.diff(xx)[None,:])),.995))
            for g in range(k):
                ax=fig.add_subplot(gs[row,g]);group_title=f'Selected K=1' if k==1 else group_name(g,k)
                im=map_plot(ax,means[g],r,regions,metadata[side],vmax,f'{side.title()} · {group_title}\n{int((good&t.group.eq(g)).sum())}/{int(t.group.eq(g).sum())} position-qualified ants')
                if g==k-1:fig.colorbar(im,ax=ax,shrink=.6,label='√density (mm⁻¹)')
            for j in range(k,max_groups):fig.add_subplot(gs[row,j]).axis('off')
            ax=fig.add_subplot(gs[row,max_groups]);observed=t.loc[good];dots(ax,observed.pc1,observed.day2_colony_percent,observed.group,k);ax.set(xlabel='Day-1 hourly-profile PC1',ylabel='Day-2 nest occupancy (%)',ylim=(-3,103),title='Individual spatial outcome')
            tab=pd.crosstab(t.group,t.spatial_label);ax=fig.add_subplot(gs[row,max_groups+1]);ax.imshow(tab,cmap='Greys',vmin=0,aspect='auto')
            for i in range(len(tab)):
                for j in range(len(tab.columns)):ax.text(j,i,str(tab.iloc[i,j]),ha='center',va='center',color='white' if tab.iloc[i,j]>tab.to_numpy().max()/2 else INK)
            ari=summary[side]['spatial_ari'];ax.set(xticks=range(len(tab.columns)),xticklabels=tab.columns,yticks=range(len(tab)),yticklabels=[group_name(i,k) for i in tab.index],xlabel='Earlier spatial group',ylabel='Selected hourly group',title='K=1: no partition to compare' if ari is None else f'Spatial agreement ARI={ari:.3f}')
        fig.text(.075,.02,'Maps use equal ant weights and ≥40% day-2 position coverage. Gray/teal labeling for K=1 is one unsplit cohort, not a recovered spatial class.',fontsize=9)
        save(fig,5,out,pdf)

        fig,axes=plt.subplots(2,3,figsize=(19,14),gridspec_kw={'width_ratios':[1.5,.8,1.15]})
        title(fig,6,f'All 57 tracked identities per colony · an hour needs ≥{MIN_CLIPS} clips; clustering needs ≥{MIN_HOURS}/24 day-1 hours · original pose-quality rules retained')
        fig.subplots_adjust(left=.065,right=.98,wspace=.58,hspace=.32)
        counts=pd.read_csv(out/'all_ant_hourly_coverage.csv',index_col=0)
        names=[PRIMARY,'instantaneous','posture_history','velocity_history','dynamics_only','velocity_half','velocity_double','camera_only','availability_only']
        labels=['Joint history','Instantaneous joint state','Posture history','Velocity history','Mean-removed history','Half velocity amplitude','Double velocity amplitude','Camera histories','Observed-hour mask']
        for row,(side,r) in enumerate(results.items()):
            all_ants=qc[qc.side.eq(side)].sort_values(['day1_eligible','day1_hours'],ascending=False);t=r['table'];ax=axes[row,0]
            data=counts.loc[all_ants.ant].to_numpy();im=ax.imshow(data,aspect='auto',cmap='viridis',vmin=0,vmax=60,interpolation='nearest',extent=(0,48,len(all_ants),0));ax.axvline(24,color='white',lw=1)
            ax.axhline(all_ants.day1_eligible.sum(),color='#e96f42',lw=1)
            ax.set(yticks=np.arange(len(all_ants))+.5,yticklabels=[a.split(':')[1] for a in all_ants.ant],xticks=range(0,49,6),xlabel='Hours from July 24 10:00; day 2 starts at 24',ylabel='All tracked ant IDs',title=f'{side.title()} · {len(t)} included / {57-len(t)} insufficient hours');ax.tick_params(axis='y',labelsize=5.5)
            fig.colorbar(im,ax=ax,shrink=.65,label='Accepted clips per hour (of 60)')
            ax=axes[row,1];q=qc[qc.side.eq(side)];ax.hist(q.day1_hours,bins=np.arange(-.5,25.5),color=INK);ax.axvline(MIN_HOURS-.5,color=COLORS[1],ls='--');ax.set(xlabel='Usable day-1 hours',ylabel='Tracked identities',xticks=[0,4,8,12,16,20,24],title=f'Old analysis: {int(q.old_day1_eligible.sum())} ants\nHourly analysis: {int(q.day1_eligible.sum())} ants')
            ax=axes[row,2];c=controls[controls.side.eq(side)].set_index('representation').reindex(names);ax.barh(np.arange(len(names)),c.distance_spearman_with_primary,color=[COLORS[1] if n in ('camera_only','availability_only') else COLORS[0] for n in names]);ax.set(yticks=range(len(names)),yticklabels=labels,xlabel='Pair-distance correlation with hourly profile',xlim=(-.15,1.03),title='Descriptive feature / availability controls');ax.invert_yaxis()
        fig.text(.065,.025,'Coverage heat map includes excluded ants below the orange line. Low-count hours remain missing in motif profiles; they are not zero activity.',fontsize=9)
        fig.text(.065,.01,'Right: Spearman correlation of pairwise distances, not independent pair-level significance tests. Fixed feature dictionaries; camera and observation patterns are nuisance comparators.',fontsize=9)
        save(fig,6,out,pdf)

        fig,axes=plt.subplots(2,4,figsize=(19,10),gridspec_kw={'width_ratios':[1,1,1,1.35]})
        title(fig,7,'Test the next 24 clock-matched hours using day-1 models · missing test hours excluded from loss · no automatic 100% retention for K=1')
        for row,(side,r) in enumerate(results.items()):
            t=r['table'];k=r['selected_k'];paired=t.day2_eligible.to_numpy();ax=axes[row,0];dots(ax,t.pc1[paired],t.day2_pc1[paired],t.group[paired],k)
            if paired.any():
                lim=[min(t.pc1[paired].min(),t.day2_pc1[paired].min()),max(t.pc1[paired].max(),t.day2_pc1[paired].max())];ax.plot(lim,lim,':',color='#95a2ae')
            caption='Retention N/A: K=1' if r['repeat_fraction'] is None else f'Retention {r["repeat_fraction"]:.0%}'
            ax.set(xlabel='Day-1 hourly-profile PC1',ylabel='Day-2 hourly-profile PC1',title=f'{side.title()} · {caption}')
            v=r['profile_validation'];ax=axes[row,1];part=v[v.model.str.startswith('K=')];ax.plot(part.complexity,part.explained_vs_mean*100,'o-',color=INK,label='Candidate prototypes');ax.axvline(k,color=COLORS[0],ls=':')
            for j in (1,2):ax.axhline(v.loc[v.model.eq(f'PC{j}'),'explained_vs_mean'].iloc[0]*100,color=COLORS[j-1],ls='--',label=f'Continuous PC{j}')
            ax.set(xlabel='Number of prototypes',ylabel='Day-2 hourly-profile variance explained (%)',xticks=range(1,7),title='Predict measured test-hour features');ax.legend(frameon=True,facecolor='white',framealpha=.95,loc='upper right',fontsize=7)
            ax=axes[row,2];ss=summary[side];lo,hi=ss['spatial_gain_ci'];value=ss['mean_spatial_gain'];ax.errorbar([0],[value],yerr=[[max(0,value-lo)],[max(0,hi-value)]],fmt='o',color=INK,capsize=6);ax.axhline(0,color='#95a2ae',ls=':');ax.scatter(np.random.default_rng(724).uniform(-.18,.18,len(r['forecast_gain'])),r['forecast_gain'],s=10,color=COLORS[0],alpha=.4)
            ax.set(xticks=[0],xticklabels=[f'Selected K={k} forecast'],ylabel='Spatial prediction gain (Hellinger²)',title='K=1 equals colony-mean forecast' if k==1 else f'Whole-ant 95% CI · n={ss["forecast_n"]}')
            ax=axes[row,3];p=r['profile'];entropy=-np.sum(np.where(p>0,p*np.log2(np.maximum(p,1e-12)),0),axis=-1);entropy[~np.isfinite(p).all(axis=-1)]=np.nan
            order=np.lexsort((t.pc1,t.group));matrix=np.concatenate([entropy[0,order],entropy[1,order]],axis=1);cmap=plt.get_cmap('magma').copy();cmap.set_bad('#d7dce1');im=ax.imshow(matrix,aspect='auto',cmap=cmap,vmin=0,vmax=np.log2(fit['n_motifs']),extent=(0,48,len(t),0));ax.axvline(24,color='white',lw=1);ax.set(xlabel='Hours from July 24 10:00',ylabel='Ants sorted by day-1 selected group',title='Hourly motif diversity · gaps retained');fig.colorbar(im,ax=ax,shrink=.65,label='Entropy (bits)')
        fig.text(.075,.022,'Candidate K curves assess alternatives; only the rule-selected K supplies group labels. Positive spatial gain beats the leave-one-ant-out colony mean.',fontsize=9)
        save(fig,7,out,pdf)
    write_report(out,results,qc,models,summary,controls,dictionary)
    write_explorer(out,results,qc,models)
    names=['postural_dynamics.py','postural_dynamics_extract.py','postural_dynamics_plots.py','postural_dynamics_hourly.py','postural_dynamics_hourly_plots.py','postural_dynamics_hourly_explorer.html','postural_dynamics.md','test_postural_dynamics.py','test_postural_dynamics_hourly.py','colony_behavioral_landscape.py']
    (out/'code_provenance.json').write_text(json.dumps({n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names},indent=2)+'\n')


def write_report(out,results,qc,models,summary,controls,dictionary):
    k=int(models['n_motifs']);rank=int(models['rank']);lines=['# 0724: cluster ants from concatenated hourly activity','',
      'The displayed group count now follows model selection. K=1 means one unsplit cohort; there is no forced P0/P1 view. Hourly motif frequencies replace whole-day frequencies everywhere in ant clustering and profile validation.','',
      '## Why the previous figure had 24/28 ants','',
      'There are 57 tracked identities in each colony. The previous detection screen retained 41 left and 45 right; requiring at least 240 accepted day-1 clips then removed another 17 from each, leaving 24/28. These were analysis subsets, not colony sizes. The hourly revision audits all 114 identities, with no detection-fraction prescreen.','',
      '| Colony | Tracked identities | Old detection screen | Old daily cohort | New hourly cohort | Insufficient hours |',
      '|---|---:|---:|---:|---:|---:|']
    for side,r in results.items():
        q=qc[qc.side.eq(side)];lines.append(f'| {side.title()} | {len(q)} | {q.old_detection_pass.sum()} | {q.old_day1_eligible.sum()} | {len(r["table"])} | {len(q)-len(r["table"])} |')
    lines += ['','An hour needs at least five accepted clips; an ant needs at least 16 measured day-1 hours. The 16/24 rule ensures every pair shares at least eight measured clock hours. It replaces the previous 240-clip total, while retaining the same complete-pose, geometry, camera and velocity checks. It is an exploratory coverage rule, not a biological threshold. Figure 6 and `all_ant_quality.csv` account for every identity; `all_ant_hourly_coverage.csv` lists all 48 hourly sample counts. No absence is treated as inactivity.','',
      '## From posture to hourly activity','',
      f'1. Keep the previous day-1 dictionary fixed: {rank} posture coefficients plus signed forward/lateral velocity, 13 times across one second, assigned to {k} learned motif centers. Its basis and centers were learned from the previous 52-ant cohort. Additional ants are scored with exactly the same basis, scaling and centers. This isolates changes to ant inclusion and hourly aggregation; it does not remove possible representation bias from the original cohort.',
      f'2. Within each ant and clock hour, divide motif counts by the number of accepted clips. Low-count hours are NaN. Concatenate the 24 frequency vectors in clock order: {24*k} features per ant. July 24 10:00–July 25 10:00 trains the ant models; the following matched 24 hours are held out.',
      '3. Take square roots of frequencies. Fit KMeans with a missing-hour mask. An ant-to-center distance is the mean squared discrepancy over its measured hours; every ant contributes total weight one. Center updates use reciprocal observed-hour-count weights. An unobserved group-center hour falls back to the training colony mean for that clock hour. A missing hour is never filled with zeros or interpolated in clustering.',
      '4. Compare K=2–6 with 100 whole-ant bootstraps. A split needs at least four ants in every group and median bootstrap adjusted Rand index ≥0.8. Select the smallest K within 0.02 of the best eligible silhouette. Silhouette compares only mutually measured hours. If none qualifies, select K=1 and display one group throughout. K=1 is a failure to support a split under this rule, not a proof of unimodality.',
      '5. Freeze selected groups before opening occupancy outcomes. Hourly timing matters: ants with identical whole-day frequencies can separate if the motifs occur at different hours. Independently shuffled hour order is a timing sensitivity control.','',
      'PCA is used for display and continuous alternatives. Its day-1 basis uses clock-hour colony-mean filling; each ant’s projection coordinates are solved from observed features. This display-only filling never enters KMeans or its distances. Pairwise distances with differing overlap need not form an exact Euclidean metric, so the silhouette is a descriptive selection statistic.','',
      '## Updated results','']
    for side,r in results.items():
        ss=summary[side];n=ss['selected_k'];v=r['profile_validation'].set_index('model');lines += [f'### {side.title()} colony','',f'{ss["n_ants"]}/57 identities qualify. **Selected K={n}**. '+('All eligible ants are displayed as one group; no P0/P1 labels are retained.' if n==1 else 'Group sizes: '+', '.join(f'G{int(g)+1}: {nn}' for g,nn in ss['groups'].items())+'. Group indices follow angular motion, not location.'),'']
        c2=r['candidates'].set_index('k').loc[2]
        lines += [f'The K=2 candidate has silhouette {c2.silhouette:.3f}, median bootstrap ARI {c2.bootstrap_ari_median:.3f}, and smallest group {int(c2.smallest_group)} ants. The stability threshold is 0.8; a high silhouette alone does not qualify a split.','']
        if n>1:
            c=r['candidates'].set_index('k').loc[n];lines += [f'The selected partition has silhouette {c.silhouette:.3f} and median bootstrap ARI {c.bootstrap_ari_median:.3f}. Day-2 retention is {ss["day2_assignment_retention"]:.1%} among {ss["day2_assignment_n"]} qualifying ants. Spatial-group agreement is ARI {ss["spatial_ari"]:.3f} on {ss["spatial_comparison_n"]} ants with previous labels.','']
        else:lines += ['Group retention and agreement with the previous spatial partition are marked not applicable: a one-group model cannot test either. Its spatial forecast is exactly the leave-one-ant-out colony-mean baseline, so group-specific gain is zero.','']
        lines += [f'Day-2 hourly-profile variance explained: selected K={n}: {v.loc[f"K={n}","explained_vs_mean"]:.1%}; continuous PC1: {v.loc["PC1","explained_vs_mean"]:.1%}; PC1+PC2: {v.loc["PC2","explained_vs_mean"]:.1%}. Loss averages only measured test hours, equally per ant.',f'Independent hourly shuffling selects K={r["hour_control"]["selected_k"]}; ARI with ordered assignments is {r["hour_control"]["ari_with_ordered"]:.3f}. Agreement of two one-group results is trivial, not evidence that timing is irrelevant.','',f'Day-2 spatial forecast gain: {ss["mean_spatial_gain"]:.4f} Hellinger² (whole-ant 95% CI {ss["spatial_gain_ci"][0]:.4f} to {ss["spatial_gain_ci"][1]:.4f}; n={ss["forecast_n"]}). Forecasts average only other ants’ day-1 maps. Position coverage must reach 40% on both days.','']
        excluded=qc[qc.side.eq(side)&~qc.day1_eligible];lines += ['Excluded identities (usable day-1 hours): '+', '.join(f'{a.ant.split(":")[1]} ({a.day1_hours})' for a in excluded.itertuples())+'.','']
    lines += ['## Limits and controls','',
      'Five clips is a minimum, not precise estimation of an hourly distribution. Sampling remains one independently randomized 2.5-second clip per minute. Measurements describe visible, quality-passing behavior; complete-pose requirements can selectively lose activity. Temporal samples share an ant and are not independent animals. Day 2 does not select training ants. All fits and choices are exploratory revisions of this recording.',
      'Figure 6 compares hourly pair-distance geometry for frozen posture-only, velocity-only, instantaneous and weighted dictionaries, camera histories, and observed-hour masks. Correlations are descriptive; ant pairs are not independent replicates. No feature weighting or coverage threshold was chosen to recover spatial labels. The previous camera-centering ablation is not reused because its exact global correction was not stored; camera histories remain available as a nuisance comparator.',
      'Whole-ant bootstrap stability keeps the motif dictionary fixed and does not capture all hourly counting noise. Ants belong to only two colonies; these are not population-level replicates. Spatial maps and previous spatial classes are outcomes, not training features. Prior spatial labels cover only their original analysis cohort; missing labels are not treated as classes.','',
      '## Files and provenance','',f'[Six-figure PDF]({PDF}) · [Interactive hourly explorer](explorer.html) · [Figure gallery](index.html)','',
      '`hourly_motif_frequencies.csv` contains all ant × hour observations and motif frequencies; `hourly_motif_profiles.npz` preserves the arrays and counts. `*_hourly_groups.csv` contains only rule-selected groups. `HOURLY_FROZEN.json` records the dictionary hash, concatenation, eligibility and chosen K before spatial validation. Every tracked identity and exclusion reason is in `all_ant_quality.csv`. The dictionary input is '+str(dictionary)+'.','']
    for stem,label,caption in FIGURES:lines += [f'### {stem[:2]}. {label}',caption,f'![{label}]({stem}.png)','']
    text='\n\n'.join(lines);text=re.sub(r'(\|[^\n]*\|)\n\n(?=\|)',r'\1\n',text);(out/'REPORT.md').write_text(re.sub(r'\n{3,}','\n\n',text))
    sections=''.join(f'<section><h2>{stem[:2]}. {html.escape(label)}</h2><p>{html.escape(caption)}</p><a href="{stem}.pdf"><img src="{stem}.png" alt="{html.escape(label)}"></a></section>' for stem,label,caption in FIGURES)
    findings=''.join(f'<li>{side.title()}: {ss["n_ants"]}/57 identities included; selected K={ss["selected_k"]}'+(' — one unsplit cohort.' if ss['selected_k']==1 else '.')+'</li>' for side,ss in summary.items())
    (out/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>0724 · hourly posture and velocity</title><style>body{{font:17px/1.55 system-ui;background:#f3f5f6;color:#263448;margin:0}}main{{max-width:1280px;margin:auto;padding:38px 24px}}h1{{font-size:38px;line-height:1.15}}a{{color:#087e80}}section{{background:white;padding:24px;border-radius:12px;margin:24px 0}}img{{width:100%;height:auto}}nav a{{margin-right:20px}}</style><main><p>20260724 / BLOCK01 · HOURLY ACTIVITY</p><h1>Posture + velocity → motifs → hourly ant profiles</h1><p>Each ant is represented by 24 concatenated hourly motif-frequency vectors. Displayed groups follow model selection; K=1 displays one group. All 57 identities per colony are audited.</p><nav><a href="{PDF}">Six-figure PDF</a><a href="explorer.html">Hourly explorer</a><a href="REPORT.md">Full analysis</a><a href="all_ant_quality.csv">All-ant audit</a></nav><section><h2>Selected groups</h2><ul>{findings}</ul><p>At least five measured clips per hour and 16 measured day-1 hours per ant. Missing hours remain missing. The posture–velocity motif dictionary is fixed from the preceding analysis.</p></section>{sections}</main></html>''')


def write_explorer(out,results,qc,models):
    m=models;payload=dict(lengths=m['standard_lengths_mm'].tolist(),mean=m['posture_mean'].tolist(),components=m['posture_components'].tolist(),sd=np.sqrt(m['posture_variance']).tolist(),variance=m['posture_variance_ratio'].tolist(),examples=m['motif_example_shape'].round(5).tolist(),velocity=m['motif_example_velocity_mm_s'].round(5).tolist(),rank=int(m['rank']),n_motifs=int(m['n_motifs']),colonies={})
    for side,r in results.items():
        ants=[]
        for j,row in enumerate(r['table'].itertuples()):
            profile=r['profile'][:,j];values=np.where(np.isfinite(profile),profile,None).tolist()
            ants.append(dict(ant=row.ant,group=row.group,pc1=row.pc1,pc2=row.pc2,motion=row.angular_motion_rad_s,speed=row.sampled_speed_mm_s,clips=row.day1_clips,hours=row.day1_hours,day2_clips=row.day2_clips,day2_hours=row.day2_hours,day2_group=int(row.day2_group),nest=None if pd.isna(row.day2_colony_percent) else float(row.day2_colony_percent),position_coverage=row.position_coverage_day2,hourly=values,counts=r['counts'][:,j].sum(axis=-1).tolist(),map=r['spatial_maps'][j,1].round(7).tolist()))
        excluded=[dict(ant=row.ant,hours=row.day1_hours,clips=row.day1_clips) for row in qc[qc.side.eq(side)&~qc.day1_eligible].itertuples()]
        payload['colonies'][side]=dict(ants=ants,selected_k=r['selected_k'],excluded=excluded)
    encoded=json.dumps(payload,separators=(',',':'),allow_nan=False).replace('</','<\\/')
    (out/'explorer.html').write_text(Path(__file__).with_name('postural_dynamics_hourly_explorer.html').read_text().replace('__DATA__',encoded))


def render_saved(output):
    """Rebuild displays from frozen hourly fits; never reselect or refit groups."""
    from analysis.postural_dynamics_extract import stamp
    manifest=json.loads((output/'run_manifest.json').read_text());args=manifest['arguments']
    if any(stamp(item['path'])!=item for item in manifest['source_fingerprints']):raise ValueError('Original inputs changed')
    frozen=json.loads((output/'HOURLY_FROZEN.json').read_text())
    if hashlib.sha256((output/'hourly_ant_models.npz').read_bytes()).hexdigest()!=frozen['model_sha256']:raise ValueError('Frozen model changed')
    for side,digest in frozen['groups_sha256'].items():
        if hashlib.sha256((output/f'{side}_hourly_groups.csv').read_bytes()).hexdigest()!=digest:raise ValueError('Frozen groups changed')
    dictionary=Path(args['dictionary']);block=Path(args['block']);atoms=Path(args['spatial_atoms'])
    with np.load(dictionary/'space_blind_models.npz') as z:models={k:z[k] for k in z.files}
    with np.load(output/'hourly_motif_profiles.npz') as z:profiles={k:z[k] for k in z.files}
    with np.load(output/'hourly_ant_models.npz') as z:ant_models={k:z[k] for k in z.files}
    with np.load(output/'heldout_spatial_maps.npz') as z:maps={k:z[k] for k in z.files}
    qc=pd.read_csv(output/'all_ant_quality.csv');summary=json.loads((output/'results_summary.json').read_text());controls=pd.read_csv(output/'hourly_representation_controls.csv');forecasts=pd.read_csv(output/'heldout_spatial_forecasts.csv');results={}
    for side,ss in summary.items():
        t=pd.read_csv(output/f'{side}_groups_with_spatial_outcomes.csv');rows=ant_models[side+'_rows']
        with np.load(atoms/(t.ant.iloc[0].replace(':','_')+'.npz')) as z:edges=(z['x_edges'],z['y_edges'])
        results[side]=dict(table=t,rows=rows,selected_k=ss['selected_k'],profile=profiles[PRIMARY][:,rows],counts=profiles['counts'][:,rows],centers=ant_models[side+'_centers'],candidates=pd.read_csv(output/f'{side}_ant_model_selection.csv'),profile_validation=pd.read_csv(output/f'{side}_heldout_profile_prediction.csv'),spatial_maps=maps[side],spatial_edges=edges,repeat_fraction=ss['day2_assignment_retention'],repeat_n=ss['day2_assignment_n'],forecast_gain=forecasts.loc[forecasts.side.eq(side),'gain'].to_numpy(),hour_control=frozen['hour_shuffle'][side])
    (output/'COMPLETE.json').unlink(missing_ok=True)
    render(output,results,qc,profiles,controls,models,dictionary,block,summary)
    from datetime import datetime, timezone
    manifest['rendered_from_frozen_models_utc']=datetime.now(timezone.utc).isoformat()
    (output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (output/'COMPLETE.json').write_text(json.dumps(dict(complete=True,source=str(block)))+'\n')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description='Render completed hourly analysis without refitting')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();render_saved(a.output)
