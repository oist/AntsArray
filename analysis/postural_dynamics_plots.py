"""Publication figures and offline explorer for the posture-first 0724 analysis."""
from __future__ import annotations

import html
import hashlib
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle, Rectangle
import numpy as np
import pandas as pd

from analysis.postural_dynamics_extract import EDGES, EDGE_NAMES

COLORS=['#128a8b','#d7754e'];INK='#28374a'
FIGURES=[
 ('01_measurement','Begin with body shape','Body-relative joint directions and the cost of requiring complete, measured posture.'),
 ('02_eigenpostures','Discover a posture basis','Learn modes of shape on day 1; test reconstruction on the following day.'),
 ('03_short_dynamics','Add short-time dynamics','Predict posture 0.25 seconds ahead, then describe one-second histories with a shared motif dictionary.'),
 ('04_ant_profiles','Ask whether ants form groups','Group ants by their motif frequencies, without position or previous spatial labels.'),
 ('05_spatial_outcomes','Reveal the spatial outcomes','Only now inspect next-day occupancy and the earlier spatial groups.'),
 ('06_controls','Challenge the interpretation','Compare posture, dynamics and camera controls; inspect measurement selection.'),
 ('07_persistence','Follow the same ants into day 2','Test profile persistence, a continuous alternative and spatial forecasts from posture groups.')]


def skeleton(directions,lengths):
    """Standard-length schematic; input direction cosines, never arena position."""
    vectors=np.asarray(directions).reshape(8,2)
    vectors=vectors/np.maximum(np.linalg.norm(vectors,axis=1,keepdims=True),1e-8)
    xy=np.zeros((10,2));xy[2]=[-lengths[0],0]
    for i,(a,b) in enumerate(EDGES):xy[b]=xy[a]+lengths[i+1]*vectors[i]
    return xy


def draw_pose(ax,directions,lengths,color=INK,alpha=1,lw=1.8,offset=0,labels=False):
    xy=skeleton(directions,lengths);xy[:,0]+=offset
    for a,b in [(0,2),*EDGES]:ax.plot(xy[[a,b],0],xy[[a,b],1],color=color,lw=lw,alpha=alpha)
    ax.scatter(xy[:,0],xy[:,1],s=9,color=color,alpha=alpha)
    if labels:
        for i in (0,1,2,3,6,9):ax.annotate(str(i),xy[i],xytext=(3,5),textcoords='offset points',fontsize=8)
    ax.set_aspect('equal');ax.set(xlabel='Anterior axis (mm)',ylabel='Lateral axis (mm)')


def setup():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,
       'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#a3abb3',
       'axes.labelcolor':INK,'text.color':INK,'figure.facecolor':'white','savefig.facecolor':'white','pdf.fonttype':42})


def title(fig,n,subtitle):
    fig.suptitle(f'{n:02d}  {FIGURES[n-1][1]}',x=.04,ha='left',fontsize=19,fontweight='bold')
    fig.text(.04,.919,subtitle,fontsize=10,color='#596877',va='top')
    fig.subplots_adjust(top=.82,bottom=.12,left=.075,right=.96,hspace=.60,wspace=.48)


def save(fig,n,out,pdf):
    stem=FIGURES[n-1][0];fig.savefig(out/(stem+'.png'),dpi=180);fig.savefig(out/(stem+'.pdf'));pdf.savefig(fig);plt.close(fig)


def dots(ax,x,y,group):
    for g in (0,1):
        m=np.asarray(group)==g
        ax.scatter(np.asarray(x)[m],np.asarray(y)[m],color=COLORS[g],s=29,edgecolor='white',lw=.4,label=f'P{g}',alpha=.9)


def map_plot(ax,m,r,regions,meta,vmax,title_text):
    x,y=r['spatial_edges'];area=np.diff(y)[:,None]*np.diff(x)[None,:]
    im=ax.pcolormesh(x,y,np.sqrt(m/area),cmap='magma',vmin=0,vmax=vmax,rasterized=True)
    side=r['table'].side.iloc[0];ox=meta['arena_bounds_px']['x_min_px'];oy=meta['arena_bounds_px']['y_min_px'];scale=meta['mm_per_px']
    for reg in regions:
        if reg['side']!=side or reg['region_type']=='arena':continue
        color='#6cdbd1' if reg['region_type']=='colony' else '#c9d967'
        if reg['shape']=='rectangle':
            patch=Rectangle(((reg['tracking_x_min_px']-ox)*scale,(reg['tracking_y_min_px']-oy)*scale),
              (reg['tracking_x_max_px']-reg['tracking_x_min_px'])*scale,(reg['tracking_y_max_px']-reg['tracking_y_min_px'])*scale,fill=False,ec=color,lw=.9)
        else:patch=Circle(((reg['tracking_center_x_px']-ox)*scale,(reg['tracking_center_y_px']-oy)*scale),reg['radius_mm'],fill=False,ec=color,lw=.9)
        ax.add_patch(patch)
    ax.set(xlabel='x (mm)',ylabel='y (mm)',title=title_text,aspect='equal',xlim=(x[0],x[-1]),ylim=(y[-1],y[0]));return im


def render(out,results,ants,b,qc,fit,summary,controls,block):
    setup();models=fit['model_arrays'];lengths=models['standard_lengths_mm'];pca=fit['pca']
    examples=models['motif_example_shape'];sample=np.load(out/'posture_display_samples.npz')
    recon=pd.read_csv(out/'posture_reconstruction.csv');temporal=pd.read_csv(out/'temporal_posture_profiles.csv')
    regions=pd.read_csv(block/'panorama_regions.csv').to_dict('records')
    for reg in regions:
        label=reg['semantic_label'];reg['side']='left' if label.endswith('L') or label.endswith('_left') else 'right'
        reg['region_type']='arena' if label.startswith('arena') else label[:-1]
    reference=pd.read_csv(block/'stitched/grid_occupancy_histograms_arena_0p25mm/track_cluster_ids.csv')
    reference['ant']=reference.side+':'+reference.TrackID.astype(int).astype(str).str.zfill(3)
    audit=qc.merge(reference[['ant','cluster_id']],on='ant',how='left',validate='one_to_one')
    audit.to_csv(out/'pose_quality_with_spatial_labels.csv',index=False)
    # Load arena annotations only here, after all representations are frozen.
    meta={}
    for side,r in results.items():
        track=r['table'].track_name.iloc[0]
        path=block/'stitched/grid_occupancy_histograms_arena_0p25mm/per_track'/Path(track).stem/'grid_occupancy_metadata.json'
        meta[side]=json.loads(path.read_text())
    with PdfPages(out/'0724_postural_dynamics.pdf') as pdf:
        fig,ax=plt.subplots(1,3,figsize=(15,6))
        title(fig,1,f'0724/block01 only · two matched 24-hour windows · {len(ants)} / {len(qc)} detection-screened ants pass day-1 posture availability')
        draw_pose(ax[0],pca.mean_,lengths,labels=True)
        ax[0].set_title('Measured landmarks → body-relative angles')
        ax[0].text(.02,-.55,'0 tag · 1 head · 2 petiole · 3 gaster tip\n4–9 antennae; no legs tracked',transform=ax[0].transAxes,fontsize=9)
        for side,color in zip(('left','right'),COLORS):
            part=qc[qc.side.eq(side)];ax[1].scatter(part.day1_clips,part.day2_clips,c=color,label=side.title(),s=25)
        ax[1].axvline(240,color='#8e9ba6',ls=':');ax[1].axhline(240,color='#8e9ba6',ls=':')
        ax[1].set(xlabel='Accepted day-1 clips',ylabel='Accepted day-2 clips',title='Complete 2.5 s clips · 1,440 requested/day');ax[1].legend(frameon=False)
        ordered=qc.sort_values('day1_clips');ax[2].bar(np.arange(len(qc)),ordered.day1_clips/14.4,color=np.where(ordered.day1_eligible,INK,'#cdd5dc'))
        ax[2].set(xlabel='Ants ordered by usable posture',ylabel='Day-1 requested clips accepted (%)',title='No interpolation; one camera per clip')
        fig.text(.075,.015,'Fit eligibility: ≥240 day-1 clips across ≥24 half-hours. Day 2 does not select the training cohort. Geometry bounds reject degenerate segments.',fontsize=9)
        save(fig,1,out,pdf)

        fig,ax=plt.subplots(2,4,figsize=(16,9))
        title(fig,2,f'PCA of 16 direction cosines · equal ant weights · {fit["rank"]} modes retain 95% of day-1 variance · all modes learned without space')
        ax[0,0].bar(np.arange(1,17),100*pca.explained_variance_ratio_,color=INK)
        ax[0,0].set(xlabel='Mode',ylabel='Variance explained (%)',title='Eigenposture spectrum',xticks=[1,4,8,12,16])
        for d,color in [(1,INK),(2,COLORS[1])]:
            part=recon[recon.day.eq(d)];ax[0,1].plot(part.modes,part.explained_variance*100,'o-',ms=3,color=color,label=f'Day {d}')
        ax[0,1].axhline(95,color='#a8b1ba',ls=':');ax[0,1].set(xlabel='Retained modes',ylabel='Reconstructed variance (%)',title='Fixed day-1 basis');ax[0,1].legend(frameon=False)
        z=sample['scores'][:,24];ax[0,2].hexbin(z[:,0],z[:,1],gridsize=38,mincnt=1,cmap='Greys',bins='log')
        ax[0,2].set(xlabel='Mode 1 coefficient',ylabel='Mode 2 coefficient',title='Sampled postures · no ant labels')
        for j in range(3):
            ax[0,3].plot(np.arange(8),np.linalg.norm(pca.components_[j].reshape(8,2),axis=1),'o-',ms=3,label=f'Mode {j+1}')
        ax[0,3].set(xticks=range(8),xticklabels=['Head','Gaster','A1','A2','A3','B1','B2','B3'],ylabel='Loading magnitude',title='Which joints contribute?');ax[0,3].tick_params(axis='x',rotation=45);ax[0,3].legend(frameon=False,fontsize=8)
        for j in range(4):
            for sd,color in [(-1.5,COLORS[0]),(0,'#a8b1ba'),(1.5,COLORS[1])]:draw_pose(ax[1,j],pca.mean_+sd*np.sqrt(pca.explained_variance_[j])*pca.components_[j],lengths,color,alpha=.85)
            ax[1,j].set_title(f'Mode {j+1} · {pca.explained_variance_ratio_[j]:.1%}')
        fig.text(.075,.015,'Bottom: mean (gray) ±1.5 SD (teal/orange), normalized to fixed segment lengths for display. These are schematic modes, not video frames.',fontsize=9)
        save(fig,2,out,pdf)

        fig=plt.figure(figsize=(16,9));gs=fig.add_gridspec(2,4);ax0=fig.add_subplot(gs[0,:2]);ax1=fig.add_subplot(gs[0,2:]);bottom=[fig.add_subplot(gs[1,j]) for j in range(4)]
        title(fig,3,f'Causal 4-frame smoothing → 12 Hz posture histories · {fit["n_motifs"]} motifs chosen using first-day validation only')
        h=fit['history_table']
        for shuffled,color,label in [(False,INK,'Time-ordered history'),(True,COLORS[1],'Past times shuffled')]:
            part=h[h.shuffled.eq(shuffled)];ax0.errorbar(part.history_seconds,part.mean_mse,yerr=part.standard_error,color=color,marker='o',capsize=3,label=label)
        ax0.axhline(fit['persistence_mse'],color='#8f9dac',ls=':',label='Hold current posture')
        ax0.set(xlabel='Past history (seconds)',ylabel='0.25 s future prediction error',title='Held-out hours · mean ± SE across ants');ax0.legend(frameon=False,fontsize=8)
        res=fit['resolution'];ax1.errorbar(res.motifs,res.mean_mse,yerr=res.standard_error,color=INK,marker='o',capsize=4)
        ax1.axvline(fit['n_motifs'],color=COLORS[0],ls='--');ax1.set(xlabel='Number of shared motifs',ylabel='0.25 s future prediction error',xticks=res.motifs,title='Dictionary resolution · smallest within 1 SE of best')
        for j,ax in enumerate(bottom):
            motif=int(np.linspace(0,len(examples)-1,4).round()[j])
            for k,t in enumerate([12,16,20,24]):draw_pose(ax,examples[motif,t],lengths,plt.cm.viridis(k/3),alpha=.8,lw=1.3)
            ax.set_title(f'Motif {motif:02d} · four times over 1 s')
        fig.text(.075,.015,'Examples are actual measured histories nearest each center; colors run from early purple to late yellow. Motifs describe a continuum, not proven behavioral states.',fontsize=9)
        save(fig,3,out,pdf)

        fig,axes=plt.subplots(2,4,figsize=(17,9))
        title(fig,4,'One ant = a distribution over shared posture histories · group fitting uses full square-root frequency vectors, separately by colony')
        for row,(side,r) in enumerate(results.items()):
            t=r['table'];order=np.lexsort((t.pc1,t.group));ax=axes[row,0]
            ax.imshow(r['profile'][0,order],aspect='auto',cmap='magma',vmin=0,vmax=np.quantile(r['profile'][0],.99))
            ax.axhline((t.group==0).sum()-.5,color='white',lw=.8);ax.set(xlabel='Motif index',ylabel='Ants sorted by P0/P1, then PC1',title=f'{side.title()} · day-1 motif frequencies')
            ax=axes[row,1];dots(ax,t.pc1,t.pc2,t.group);ax.set(xlabel='Profile PC1',ylabel='Profile PC2',title=f'Prespecified two-group view · n={len(t)}');ax.legend(frameon=False,fontsize=8)
            c=r['candidates'];ax=axes[row,2];ax.plot(c.k,c.silhouette,'o-',color=INK);ax.set(xlabel='Groups',ylabel='Silhouette',xticks=range(2,7),title=f'Selection rule chooses K={r["selected_k"]}')
            ax=axes[row,3];ax.plot(c.k,c.bootstrap_ari_median,'o-',color=INK);ax.fill_between(c.k,c.bootstrap_ari_p10,c.bootstrap_ari_p90,color=INK,alpha=.15);ax.axhline(.8,ls=':',color='#94a0ad')
            ax.set(xlabel='Groups',ylabel='Adjusted Rand agreement',xticks=range(2,7),ylim=(-.05,1.05),title='100 whole-ant bootstraps · 10–90%')
        fig.text(.075,.015,'P0/P1 are ordered by lower/higher day-1 angular motion, never by location. K=1 means no K=2–6 passed the stability/size rule; it is not a unimodality test.',fontsize=9)
        save(fig,4,out,pdf)

        fig,axes=plt.subplots(2,4,figsize=(17,10))
        title(fig,5,'First use of spatial outcomes: group labels and all posture models are already frozen · occupancy maps use the held-out second day')
        for row,(side,r) in enumerate(results.items()):
            t=r['table'];good=t.position_coverage_day2.ge(.4).to_numpy();means=[r['spatial_maps'][good&t.group.eq(g).to_numpy(),1].mean(axis=0) for g in (0,1)]
            x,y=r['spatial_edges'];vmax=np.quantile(np.sqrt(np.stack(means)/(np.diff(y)[:,None]*np.diff(x)[None,:])),.995)
            for g in (0,1):map_plot(axes[row,g],means[g],r,regions,meta[side],vmax,f'{side.title()} P{g} · {(good&t.group.eq(g)).sum()} ants')
            ax=axes[row,2];observed=t.loc[good];dots(ax,observed.pc1,observed.day2_colony_percent,observed.group);ax.set(xlabel='Day-1 posture-profile PC1',ylabel='Day-2 nest occupancy (%)',ylim=(-3,103),title='Individual spatial outcome')
            tab=pd.crosstab(t.group,t.spatial_label);ax=axes[row,3];ax.imshow(tab,cmap='Greys',vmin=0)
            for i in range(len(tab)):
                for j in range(len(tab.columns)):ax.text(j,i,str(tab.iloc[i,j]),ha='center',va='center',color='white' if tab.iloc[i,j]>tab.to_numpy().max()/2 else INK)
            ax.set(xticks=range(len(tab.columns)),xticklabels=tab.columns,yticks=range(len(tab)),yticklabels=[f'P{i}' for i in tab.index],xlabel='Earlier spatial group',ylabel='Posture group',title=f'Agreement ARI = {summary[side]["spatial_ari"]:.3f}')
        fig.text(.075,.015,'Maps and nest-use points require ≥40% day-2 position coverage. Maps: equal ant weights; shared square-root density scale within colony; nest outline in teal.',fontsize=9)
        save(fig,5,out,pdf)

        fig,axes=plt.subplots(2,3,figsize=(16,9))
        title(fig,6,'Pose-only and dynamics-only ablations; camera adjustment tests sensitivity, while camera-only profiles diagnose a nuisance association')
        fig.subplots_adjust(left=.18,wspace=.60)
        names=['posture_history','posture_only','dynamics_only','camera_centered','camera_only'];labels=['1 s history','Instantaneous pose','Mean-removed dynamics','Camera-centered history','Camera only']
        for row,(side,r) in enumerate(results.items()):
            t=r['table'];c=controls[controls.side.eq(side)].set_index('representation').reindex(names);ax=axes[row,0];y=np.arange(len(names))
            ax.barh(y-.16,c.agreement_with_spatial,height=.3,color=COLORS[1],label='vs spatial groups');ax.barh(y+.16,c.agreement_with_primary,height=.3,color=COLORS[0],label='vs primary posture groups')
            ax.set(yticks=y,yticklabels=labels,xlabel='Adjusted Rand agreement',xlim=(-.12,1.03),title=side.title());ax.invert_yaxis()
            ax=axes[row,1];dots(ax,t.day1_clips/14.4,t.angular_motion_rad_s,t.group);ax.set(xlabel='Requested day-1 clips accepted (%)',ylabel='Angular motion (rad/s)',title='Coverage and apparent movement')
            ax=axes[row,2];dots(ax,t.pc1,t.detection_fraction*100,t.group);ax.set(xlabel='Posture-profile PC1',ylabel='Detected frames in observed span (%)',title='Selection on detectability')
        handles,legend_labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,legend_labels,loc='lower center',bbox_to_anchor=(.47,.047),ncol=2,frameon=False,fontsize=8)
        fig.text(.075,.015,'Camera centering can remove real context-dependent posture as well as measurement bias. Complete-clip selection and absent landmark confidence remain limitations.',fontsize=9)
        save(fig,6,out,pdf)

        fig,axes=plt.subplots(2,4,figsize=(18,9))
        title(fig,7,'Day-2 profiles are projected without refitting · spatial forecasts use other ants’ day-1 maps, grouped only by day-1 posture')
        for row,(side,r) in enumerate(results.items()):
            t=r['table'];paired=t.day2_eligible.to_numpy();z=r['profile_pca'].transform(np.sqrt(r['profile'][1]));ax=axes[row,0]
            dots(ax,t.pc1[paired],z[paired,0],t.group[paired]);lim=[min(t.pc1.min(),z[paired,0].min()),max(t.pc1.max(),z[paired,0].max())];ax.plot(lim,lim,':',color='#95a2ae')
            ax.set(xlabel='Day-1 profile PC1',ylabel='Day-2 profile PC1',title=f'{side.title()} · group retention {r["repeat_fraction"]:.0%}')
            v=r['profile_validation'];ax=axes[row,1];part=v[v.model.str.startswith('K=')];ax.plot(part.complexity,part.explained_vs_mean*100,'o-',color=INK,label='Discrete prototypes')
            for k in (1,2):ax.axhline(v.loc[v.model.eq(f'PC{k}'),'explained_vs_mean'].iloc[0]*100,color=COLORS[k-1],ls='--',label=f'Continuous PC{k}')
            ax.set(xlabel='Number of prototypes',ylabel='Day-2 profile variance explained (%)',xticks=range(1,7),title='Continuous vs grouped profiles');ax.legend(frameon=False,fontsize=7)
            ax=axes[row,2];s=summary[side];lo,hi=s['spatial_gain_ci'];mean=s['mean_spatial_gain'];ax.errorbar([0],[mean],yerr=[[max(0,mean-lo)],[max(0,hi-mean)]],fmt='o',color=INK,capsize=7);ax.axhline(0,color='#95a2ae',ls=':')
            jitter=np.random.default_rng(724).uniform(-.18,.18,len(r['forecast_gain']));ax.scatter(jitter,r['forecast_gain'],s=12,color=COLORS[0],alpha=.4)
            ax.set(xticks=[0],xticklabels=['Posture-group forecast'],ylabel='Spatial prediction gain (Hellinger²)',title=f'Whole-ant 95% CI · n={s["forecast_n"]}')
            ax=axes[row,3];part=temporal[temporal.side.eq(side)];order=t.sort_values(['group','pc1']).ant;matrix=part.pivot(index='ant',columns='halfhour',values='pc1').reindex(index=order,columns=range(96)).to_numpy();count=part.pivot(index='ant',columns='halfhour',values='n_clips').reindex(index=order,columns=range(96)).to_numpy();matrix[count<5]=np.nan
            lim=np.nanquantile(abs(matrix),.98);im=ax.imshow(matrix,cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='auto',extent=(0,48,len(t),0));ax.axvline(24,color=INK,lw=1)
            ax.set(xlabel='Hours from Jul 24 10:00',ylabel='Ants sorted by day-1 group',title='Half-hour profiles · gaps retained');fig.colorbar(im,ax=ax,shrink=.65,label='Profile PC1')
        fig.text(.075,.015,'Positive spatial gain beats the leave-one-ant-out colony mean. Dots are individual ants; bootstrap CI conditions on the fitted groups. Temporal bins need ≥5 measured clips.',fontsize=9)
        save(fig,7,out,pdf)
    write_report(out,results,ants,qc,fit,summary,controls,recon,audit)
    write_explorer(out,results,fit,meta)
    code_files=list(Path(__file__).parent.glob('postural_dynamics*'))+[Path(__file__).with_name('colony_behavioral_landscape.py'),Path(__file__).with_name('test_postural_dynamics.py')]
    (out/'code_provenance.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in code_files if p.is_file()},indent=2)+'\n')


def write_report(out,results,ants,qc,fit,summary,controls,recon,audit):
    ordered=fit['history_table'].query('shuffled == False').set_index('history_seconds')
    hgain=1-ordered.loc[1.,'mean_mse']/ordered.loc[0.,'mean_mse']
    shuffled=fit['history_table'].query('shuffled == True').set_index('history_seconds')
    ordergain=1-ordered.loc[1.,'mean_mse']/shuffled.loc[1.,'mean_mse']
    lines=['# 0724: from posture to individual behavior','',
      'This analysis starts with head, gaster and antenna posture. Neither arena position nor previous spatial labels enters the eigenpostures, motif dictionary or ant clustering. The spatial maps are opened only after model and group hashes are written to `SPACE_BLIND_FROZEN.json`.','',
      '**Interpretation:** posture histories carry information about spatial behavior, but this run does not establish two robust types defined by postural dynamics. Neither colony passes the prespecified cluster-stability rule. Spatial agreement weakens after removing mean posture or camera-associated differences, while camera-only profiles strongly recover spatial membership. This is compatible with real context-dependent posture, imaging effects, or both; the present analysis cannot separate them.','',
      '## What the data say','',
      f'- **Measurement:** {len(ants)} of {len(qc)} detection-screened ants have at least 240 complete day-1 clips across 24 half-hours. The model uses {int(ants.day1_clips.sum()):,} day-1 clips and {int(ants.day2_clips.sum()):,} day-2 clips; training subsets give ants equal weight.',
      f'- **Eigenpostures:** {fit["rank"]} modes capture 95% of day-1 direction-cosine variance. The same rank reconstructs {recon.loc[(recon.day==2)&(recon.modes==fit["rank"]),"explained_variance"].iloc[0]:.1%} on day 2.',
      f'- **Short histories:** one second of ordered posture reduces 0.25-second prediction error by {hgain:.1%} relative to instantaneous posture and by {ordergain:.1%} relative to shuffled past times. Negative gains mean worse prediction. This is internal day-1 validation, with errors averaged across ants.',
      f'- **Motif dictionary:** {fit["n_motifs"]} centers were selected from 12/24/48 using the smallest within one standard error of the best validation prediction. This is the upper end of the tested range, not evidence of exactly 48 natural states; quantized motif forecasts are also less accurate than continuous ridge forecasts. The primary history length was fixed at 1 second before seeing spatial outcomes.','']
    for side,r in results.items():
        s=summary[side];c=r['candidates'].set_index('k').loc[2];v=r['profile_validation'].set_index('model')
        controls_side=controls[controls.side.eq(side)].set_index('representation')
        lines += [f'### {side.title()} colony','',
          f'{s["n_ants"]} ants enter the posture analysis. The prespecified two-group view contains '+', '.join(f'P{k}: {n}' for k,n in s['groups'].items())+f'. Its silhouette is {c.silhouette:.3f}, with median bootstrap ARI {c.bootstrap_ari_median:.3f}. The stability/size/silhouette rule selects **K={s["selected_k"]}**; K=1 means no candidate passed, rather than evidence of a single mode.',
          f'Agreement with the earlier spatial grouping is **ARI {s["spatial_ari"]:.3f}** (ant-label permutation p={s["spatial_permutation_p"]:.3f}). For comparison, instantaneous posture gives ARI {controls_side.loc["posture_only","agreement_with_spatial"]:.3f}, mean-removed dynamics {controls_side.loc["dynamics_only","agreement_with_spatial"]:.3f}, camera-centered history {controls_side.loc["camera_centered","agreement_with_spatial"]:.3f}, and camera-only profiles {controls_side.loc["camera_only","agreement_with_spatial"]:.3f}.',
          f'On day 2, **{s["day2_assignment_retention"]:.1%}** of {s["day2_assignment_n"]} qualifying ants retain their posture group. Two prototypes explain {v.loc["K=2","explained_vs_mean"]:.1%} of held-out motif-profile variance relative to the day-1 mean; a continuous one-dimensional profile projection explains {v.loc["PC1","explained_vs_mean"]:.1%}.',
          f'Using only other ants’ day-1 spatial maps, the posture-group forecast improves day-2 Hellinger² loss by **{s["mean_spatial_gain"]:.4f}** over the colony-mean forecast (whole-ant bootstrap 95% CI {s["spatial_gain_ci"][0]:.4f} to {s["spatial_gain_ci"][1]:.4f}; ant-label permutation p={s["forecast_permutation_p"]:.3f}; n={s["forecast_n"]}). Mean losses: colony mean {s["baseline_spatial_loss"]:.4f}, posture groups {s["group_spatial_loss"]:.4f}, own previous-day map {s["own_previous_day_loss"]:.4f}.','',
          '| Posture group | Ants | Day-2 position-qualified ants | Median day-2 nest occupancy | Median angular motion |',
          '|---|---:|---:|---:|---:|']
        for g,part in r['table'].groupby('group'):
            measured=part[part.position_coverage_day2.ge(.4)]
            lines.append(f'| P{g} | {len(part)} | {len(measured)} | {measured.day2_colony_percent.median():.1f}% | {part.angular_motion_rad_s.median():.2f} rad/s |')
        lines.append('')
        lines.append('Nest-use medians require at least 40% day-2 position coverage; angular motion uses all included ants on day 1.')
        lines.append('')
        included=audit[audit.side.eq(side)].groupby('cluster_id').day1_eligible.agg(['sum','size'])
        lines.append('Posture availability by the previous spatial labels (examined only after fitting): '+', '.join(f'{label}: {int(row["sum"])}/{int(row["size"])} included' for label,row in included.iterrows())+'. This selection can limit recovery of the original spatial split.')
        lines.append('')
    lines+=['## Interpretation and limits','',
      'The eigenworm analogy is the order of inference: intrinsic shape → a compact shape basis → short-time patterns → individual usage → external function. These ants have a branched skeleton with only ten landmarks, not a densely sampled worm centerline. The analysis therefore learns eigenpostures of head, gaster and antennae; it cannot measure leg gait. The number of retained modes is measured, not borrowed from the four-mode worm result.',
      'A correspondence between posture and space is an association in this recording. It does not establish that posture causes nest use, identify a physiological state, or establish two discrete biological types. KMeans always supplies a requested two-way split. Bootstrap agreement, K selection, the continuous-profile alternative and held-out spatial prediction are separate checks on its meaning. Ants from the two colonies are analyzed separately, with only a common pose dictionary.',
      'Fully observed clips are a selective sample. Missing landmarks, extreme segment lengths, duplicate detections and camera switches are excluded; nothing is interpolated or treated as immobility. Per-node confidence is absent from finished tracks, so residual tracking jitter cannot be ruled out. Camera association can reflect both biological location preference and imaging bias. Camera centering is a sensitivity analysis, not a causal correction. The measured variance includes any remaining tracking noise. Ant-bootstrap intervals condition on the learned representation; these are two colonies in one recording, not independent population replicates.',
      'Initial detection screening follows the existing >40% rule computed over each ant’s first-to-last observed span. Posture availability uses full clock-matched days. Position-outcome maps require ≥40% day-2 coverage; spatial forecasting requires it on both days. Unqualified ants remain in `pose_quality.csv`. The earlier spatial labels use the full recording and are descriptive comparators; held-out prediction uses the day split explicitly.','',
      '## Sampling and validation','',
      'The recording spans 2026-07-24 09:31:10 to 2026-07-26 10:40:36.5 at 24 Hz. Training is July 24 10:00 to July 25 10:00, and testing is the following 24 hours. Independently for each ant, one uniformly random 2.5-second clip is sampled per minute (2,880 requested). The petiole-to-tag vector defines the body axis. Eight head, gaster and antennal segment directions yield 16 direction cosines, invariant to translation, global rotation and uniform scale.',
      'Four-frame causal averages are renormalized and sampled at 12 Hz. The prediction endpoint is raw frame 52, and its target is frame 58, so input and target smoothing windows do not overlap. Within day 1, every fourth hour is held out for history and dictionary validation; complete clips never cross these boundaries. PCA retains 95% of training variance. Ridge predictions use the same balanced training clips for ordered, instantaneous and shuffled histories. The dictionary is refitted using day 1, then applied unchanged to day 2. Ant profiles use square-root motif frequencies (Hellinger geometry). K=2–6 are compared with 100 whole-ant bootstraps; the smallest K within 0.02 of the best silhouette among models with median ARI ≥0.8 and smallest group ≥4 is selected. The prespecified K=2 view is reported even when it fails that rule.',
      'Spatial predictions use equal-ant averaged day-1 maps of other members of the same posture group, excluding the target ant. The baseline averages all other ants. The uncertainty interval resamples held-out ants (2,000 samples); the null permutes whole-ant group labels (1,000 samples). Positive prediction gain means lower Hellinger² error. Posture-profile prediction compares the day-1 mean, K=1–6 prototypes and rank-1/2 continuous projections on the same next-day ants.','',
      '## Figures and data','',
      '[Combined PDF](0724_postural_dynamics.pdf) · [Offline interactive explorer](explorer.html) · [Figure gallery](index.html)','']
    for stem,label,caption in FIGURES:lines += [f'### {label}',caption,f'![{label}]({stem}.png)','']
    lines += ['Tables: `pose_quality.csv`, `*_posture_groups.csv`, `*_groups_with_spatial_outcomes.csv`, `representation_spatial_comparison.csv`, `history_prediction_validation.csv`, `*_heldout_profile_prediction.csv`, `heldout_spatial_forecasts.csv`, and `temporal_posture_profiles.csv`. Models, motif examples, frozen hashes, source fingerprints, software versions and code snapshots accompany the figures.','',
      '## Methodological sources','',
      '[Stephens et al. (2008), Dimensionality and Dynamics in the Behavior of C. elegans](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000028) motivates learning intrinsic shape modes before assigning functional meaning.',
      '[Costa, Ahamed, Jordan and Stephens, Maximally predictive states: from partial observations to long timescales](https://arxiv.org/abs/2105.12811) motivates testing short histories by their predictive value. This implementation uses PCA, ridge prediction and a motif dictionary; it does not implement their full transfer-operator method.','']
    report=re.sub(r'(\|[^\n]*\|)\n\n(?=\|)',r'\1\n','\n\n'.join(lines))
    (out/'REPORT.md').write_text(re.sub(r'\n{3,}','\n\n',report))
    sections=''.join(f'<section id="{stem}"><h2>{i+1}. {html.escape(label)}</h2><p>{html.escape(caption)}</p><a href="{stem}.pdf"><img src="{stem}.png" alt="{html.escape(label)}"></a></section>' for i,(stem,label,caption) in enumerate(FIGURES))
    findings=''.join(f'<li>{s.title()}: {v["n_ants"]} ants; selected K={v["selected_k"]}; posture/spatial ARI={v["spatial_ari"]:.3f}; next-day group retention={v["day2_assignment_retention"]:.1%}.</li>' for s,v in summary.items())
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>0724 · posture to behavior</title><style>body{{font:17px/1.55 system-ui;background:#f3f5f6;color:#263448;margin:0}}main{{max-width:1280px;margin:auto;padding:38px 24px}}h1{{font-size:40px;line-height:1.12}}a{{color:#087e80}}section{{background:white;padding:24px;border-radius:12px;margin:24px 0}}img{{width:100%;height:auto}}.lead{{max-width:900px}}nav a{{margin-right:24px}}li{{margin:7px 0}}</style><main><p>20260724 / BLOCK01 · POSTURE FIRST</p><h1>From body shape to individual behavior</h1><p class="lead">Learn body-relative shape and short histories before opening spatial outcomes. The two-group hypothesis is tested against stability, continuous variation, nuisance controls and the following day.</p><nav><a href="0724_postural_dynamics.pdf">Combined PDF</a><a href="explorer.html">Interactive explorer</a><a href="REPORT.md">Full report</a><a href="run_manifest.json">Provenance</a></nav><section><h2>Measured results</h2><p>{len(ants)} eligible ants · {fit['rank']} posture modes retain 95% variance · {fit['n_motifs']} motifs.</p><ul>{findings}</ul><p><strong>The spatial association is substantial, but two robust dynamics-defined types are not established.</strong> Neither colony passes the cluster-stability rule. Removing mean posture or camera-associated differences weakens spatial recovery. Camera-only profiles strongly recover the spatial groups. Read the report and controls before interpreting the split.</p></section>{sections}</main></html>'''
    (out/'index.html').write_text(page)


def write_explorer(out,results,fit,metadata):
    m=fit['model_arrays'];payload=dict(lengths=m['standard_lengths_mm'].tolist(),mean=m['posture_mean'].tolist(),
      components=m['posture_components'].tolist(),sd=np.sqrt(m['posture_variance']).tolist(),variance=m['posture_variance_ratio'].tolist(),
      examples=m['motif_example_shape'].round(5).tolist(),history_seconds=1,colonies={})
    for side,r in results.items():
        t=r['table'];ants=[]
        for j,row in enumerate(t.itertuples()):
            ants.append(dict(ant=row.ant,group=row.group,pc1=row.pc1,pc2=row.pc2,motion=row.angular_motion_rad_s,
              clips=row.day1_clips,day2_clips=row.day2_clips,day2_group=int(row.day2_group),nest=None if pd.isna(row.day2_colony_percent) else float(row.day2_colony_percent),
              position_coverage=row.position_coverage_day2,profile=r['profile'][0,j].round(5).tolist(),
              map=r['spatial_maps'][j,1].round(7).tolist()))
        payload['colonies'][side]=dict(ants=ants,edges=[e.tolist() for e in r['spatial_edges']],selected_k=r['selected_k'])
    encoded=json.dumps(payload,separators=(',',':'),allow_nan=False).replace('</','<\\/')
    template=Path(__file__).with_name('postural_dynamics_explorer.html').read_text()
    (out/'explorer.html').write_text(template.replace('__DATA__',encoded))


def render_saved(output,block):
    """Rebuild figures from frozen numerical results, without refitting models."""
    from datetime import datetime, timezone
    from sklearn.decomposition import PCA
    from analysis.postural_dynamics_extract import stamp
    frozen=json.loads((output/'SPACE_BLIND_FROZEN.json').read_text())
    if hashlib.sha256((output/'space_blind_models.npz').read_bytes()).hexdigest()!=frozen['model_sha256']:
        raise ValueError('Frozen model changed')
    for side,expected in frozen['groups_sha256'].items():
        if hashlib.sha256((output/f'{side}_posture_groups.csv').read_bytes()).hexdigest()!=expected:
            raise ValueError('Frozen groups changed')
    manifest=json.loads((output/'run_manifest.json').read_text())
    if any(stamp(item['path'])!=item for item in manifest['source_fingerprints']):
        raise ValueError('Original analysis inputs changed')
    (output/'COMPLETE.json').unlink(missing_ok=True)
    with np.load(output/'space_blind_models.npz') as z:models={k:z[k] for k in z.files}
    with np.load(output/'ant_motif_profiles.npz') as z:profiles={k:z[k] for k in z.files}
    with np.load(output/'heldout_spatial_maps.npz') as z:maps={k:z[k] for k in z.files}
    def basis(mean,components,variance=None,ratio=None):
        p=PCA();p.mean_=mean;p.components_=components;p.n_features_in_=len(mean)
        if variance is not None:p.explained_variance_=variance
        if ratio is not None:p.explained_variance_ratio_=ratio
        return p
    history=pd.read_csv(output/'history_prediction_validation.csv');first=history.iloc[0]
    fit=dict(pca=basis(models['posture_mean'],models['posture_components'],models['posture_variance'],models['posture_variance_ratio']),
      rank=int(models['rank']),n_motifs=int(models['n_motifs']),model_arrays=models,profiles=profiles,history_table=history,
      resolution=pd.read_csv(output/'motif_resolution_validation.csv'),baseline_mse=first.mean_mse/(1-first.explained_vs_mean),
      persistence_mse=first.mean_mse/(1-first.gain_vs_persistence))
    summary=json.loads((output/'results_summary.json').read_text());results={}
    for side,s in summary.items():
        t=pd.read_csv(output/f'{side}_groups_with_spatial_outcomes.csv');rows=models[side+'_rows']
        profile_mean=models[side+'_profile_pca_mean'];profile_components=models[side+'_profile_pca_components']
        profile_variance=np.var((np.sqrt(profiles['posture_history'][0,rows])-profile_mean)@profile_components.T,axis=0,ddof=1)
        atoms=Path(manifest['arguments']['spatial_atoms'])/(t.ant.iloc[0].replace(':','_')+'.npz')
        with np.load(atoms) as z:edges=(z['x_edges'],z['y_edges'])
        forecast=pd.read_csv(output/'heldout_spatial_forecasts.csv')
        results[side]=dict(table=t,rows=rows,profile=profiles['posture_history'][:,rows],
          profile_pca=basis(profile_mean,profile_components,profile_variance),
          spatial_maps=maps[side],spatial_edges=edges,selected_k=s['selected_k'],repeat_fraction=s['day2_assignment_retention'],
          repeat_n=s['day2_assignment_n'],candidates=pd.read_csv(output/f'{side}_ant_model_selection.csv'),
          profile_validation=pd.read_csv(output/f'{side}_heldout_profile_prediction.csv'),
          forecast_gain=forecast.loc[forecast.side.eq(side),'gain'].to_numpy())
    ants=pd.concat([r['table'] for r in results.values()],ignore_index=True)
    render(output,results,ants,None,pd.read_csv(output/'pose_quality.csv'),fit,summary,
           pd.read_csv(output/'representation_spatial_comparison.csv'),block)
    manifest['rendered_from_frozen_models_utc']=datetime.now(timezone.utc).isoformat()
    manifest['code_sha256'].update(json.loads((output/'code_provenance.json').read_text()))
    (output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (output/'COMPLETE.json').write_text(json.dumps(dict(complete=True,source=str(block)))+'\n')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description='Re-render completed posture results without fitting or changing groups')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--block',type=Path,required=True)
    args=p.parse_args();render_saved(args.output,args.block)
