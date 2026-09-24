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
from matplotlib.colors import PowerNorm
import numpy as np
import pandas as pd

from analysis.postural_dynamics_extract import EDGES, EDGE_NAMES

COLORS=['#128a8b','#d7754e'];INK='#28374a'
FIGURES=[
 ('02_eigenpostures','Discover a posture basis','Learn modes of shape on day 1; test reconstruction on the following day.'),
 ('03_short_dynamics','Form motifs from posture and velocity','Combine posture with forward and lateral velocity; cluster one-second histories into a shared motif dictionary.'),
 ('04_ant_profiles','Ask whether ants form groups','Group ants by their new joint-motif frequencies, without absolute location or previous spatial labels.'),
 ('05_spatial_outcomes','Reveal the spatial outcomes','Only now inspect next-day occupancy and the earlier spatial groups.'),
 ('06_controls','Challenge the interpretation','Compare posture alone, velocity alone, weighting and camera controls.'),
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
    fig.suptitle(f'{n:02d}  {FIGURES[n-2][1]}',x=.04,ha='left',fontsize=19,fontweight='bold')
    fig.text(.04,.919,subtitle,fontsize=10,color='#596877',va='top')
    fig.subplots_adjust(top=.82,bottom=.12,left=.075,right=.96,hspace=.60,wspace=.48)


def save(fig,n,out,pdf):
    stem=FIGURES[n-2][0];fig.savefig(out/(stem+'.png'),dpi=180);fig.savefig(out/(stem+'.pdf'));pdf.savefig(fig);plt.close(fig)


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
    with PdfPages(out/'0724_postural_dynamics_velocity.pdf') as pdf:
        fig,ax=plt.subplots(2,4,figsize=(16,9))
        title(fig,2,f'PCA of 16 direction cosines · equal ant weights · {fit["rank"]} modes retain 95% of day-1 variance · all modes learned without space')
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
          ('D  Count usage per ant','Choose motif count on held-out hours.\nRefit on day 1; freeze for day 2.\nAnt profile = frequency of each motif.')]
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

        fig,axes=plt.subplots(2,4,figsize=(17,9))
        title(fig,4,'One ant = a distribution over joint posture–velocity histories · group fitting uses full square-root frequency vectors, separately by colony')
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
        title(fig,5,'First use of spatial outcomes: group labels and joint-feature models are already frozen · occupancy maps use the held-out second day')
        for row,(side,r) in enumerate(results.items()):
            t=r['table'];good=t.position_coverage_day2.ge(.4).to_numpy();means=[r['spatial_maps'][good&t.group.eq(g).to_numpy(),1].mean(axis=0) for g in (0,1)]
            x,y=r['spatial_edges'];vmax=np.quantile(np.sqrt(np.stack(means)/(np.diff(y)[:,None]*np.diff(x)[None,:])),.995)
            for g in (0,1):map_plot(axes[row,g],means[g],r,regions,meta[side],vmax,f'{side.title()} P{g} · {(good&t.group.eq(g)).sum()} ants')
            ax=axes[row,2];observed=t.loc[good];dots(ax,observed.pc1,observed.day2_colony_percent,observed.group);ax.set(xlabel='Day-1 behavior-profile PC1',ylabel='Day-2 nest occupancy (%)',ylim=(-3,103),title='Individual spatial outcome')
            tab=pd.crosstab(t.group,t.spatial_label);ax=axes[row,3];ax.imshow(tab,cmap='Greys',vmin=0)
            for i in range(len(tab)):
                for j in range(len(tab.columns)):ax.text(j,i,str(tab.iloc[i,j]),ha='center',va='center',color='white' if tab.iloc[i,j]>tab.to_numpy().max()/2 else INK)
            ax.set(xticks=range(len(tab.columns)),xticklabels=tab.columns,yticks=range(len(tab)),yticklabels=[f'P{i}' for i in tab.index],xlabel='Earlier spatial group',ylabel='Posture group',title=f'Agreement ARI = {summary[side]["spatial_ari"]:.3f}')
        fig.text(.075,.015,'Maps and nest-use points require ≥40% day-2 position coverage. Maps: equal ant weights; shared square-root density scale within colony; nest outline in teal.',fontsize=9)
        save(fig,5,out,pdf)

        fig,axes=plt.subplots(2,3,figsize=(17,11))
        title(fig,6,'Refit motifs and groups for each ablation on the same ants; weights and camera correction are sensitivity checks, not spatially optimized choices')
        fig.subplots_adjust(left=.18,wspace=.60)
        names=['posture_velocity_history','instantaneous','posture_history','velocity_history','dynamics_only','velocity_half','velocity_double','camera_centered','camera_only'];labels=['Posture + velocity history','Instantaneous joint state','Posture history only','Velocity history only','Mean-removed joint history','Half velocity weight','Double velocity weight','Camera-centered joint history','Camera only']
        for row,(side,r) in enumerate(results.items()):
            t=r['table'];c=controls[controls.side.eq(side)].set_index('representation').reindex(names);ax=axes[row,0];y=np.arange(len(names))
            ax.barh(y-.16,c.agreement_with_spatial,height=.3,color=COLORS[1],label='vs spatial groups');ax.barh(y+.16,c.agreement_with_primary,height=.3,color=COLORS[0],label='vs primary joint groups')
            ax.set(yticks=y,yticklabels=labels,xlabel='Adjusted Rand agreement',xlim=(-.12,1.03),title=side.title());ax.invert_yaxis()
            ax=axes[row,1];dots(ax,t.angular_motion_rad_s,t.sampled_speed_mm_s,t.group);ax.set(xlabel='Postural angular motion (rad/s)',ylabel='Mean sampled velocity magnitude (mm/s)',title='Postural motion and translation')
            ax=axes[row,2];dots(ax,t.pc1,t.detection_fraction*100,t.group);ax.set(xlabel='Joint-profile PC1',ylabel='Detected frames in observed span (%)',title='Selection on detectability')
        handles,legend_labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,legend_labels,loc='lower center',bbox_to_anchor=(.47,.047),ncol=2,frameon=False,fontsize=8)
        fig.text(.075,.015,'Camera centering can remove real context-dependent posture as well as measurement bias. Complete-clip selection and absent landmark confidence remain limitations.',fontsize=9)
        save(fig,6,out,pdf)

        fig,axes=plt.subplots(2,4,figsize=(18,9))
        title(fig,7,'Day-2 profiles are projected without refitting · spatial forecasts use other ants’ day-1 maps, grouped by day-1 posture and velocity')
        for row,(side,r) in enumerate(results.items()):
            t=r['table'];paired=t.day2_eligible.to_numpy();z=r['profile_pca'].transform(np.sqrt(r['profile'][1]));ax=axes[row,0]
            dots(ax,t.pc1[paired],z[paired,0],t.group[paired]);lim=[min(t.pc1.min(),z[paired,0].min()),max(t.pc1.max(),z[paired,0].max())];ax.plot(lim,lim,':',color='#95a2ae')
            ax.set(xlabel='Day-1 profile PC1',ylabel='Day-2 profile PC1',title=f'{side.title()} · group retention {r["repeat_fraction"]:.0%}')
            v=r['profile_validation'];ax=axes[row,1];part=v[v.model.str.startswith('K=')];ax.plot(part.complexity,part.explained_vs_mean*100,'o-',color=INK,label='Discrete prototypes')
            for k in (1,2):ax.axhline(v.loc[v.model.eq(f'PC{k}'),'explained_vs_mean'].iloc[0]*100,color=COLORS[k-1],ls='--',label=f'Continuous PC{k}')
            ax.set(xlabel='Number of prototypes',ylabel='Day-2 profile variance explained (%)',xticks=range(1,7),title='Continuous vs grouped profiles');ax.legend(frameon=False,fontsize=7)
            ax=axes[row,2];s=summary[side];lo,hi=s['spatial_gain_ci'];mean=s['mean_spatial_gain'];ax.errorbar([0],[mean],yerr=[[max(0,mean-lo)],[max(0,hi-mean)]],fmt='o',color=INK,capsize=7);ax.axhline(0,color='#95a2ae',ls=':')
            jitter=np.random.default_rng(724).uniform(-.18,.18,len(r['forecast_gain']));ax.scatter(jitter,r['forecast_gain'],s=12,color=COLORS[0],alpha=.4)
            ax.set(xticks=[0],xticklabels=['Joint-feature group forecast'],ylabel='Spatial prediction gain (Hellinger²)',title=f'Whole-ant 95% CI · n={s["forecast_n"]}')
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
    shuffled=fit['history_table'].query('shuffled == True').set_index('history_seconds')
    hgain=1-ordered.loc[1.,'mean_mse']/ordered.loc[0.,'mean_mse']
    ordergain=1-ordered.loc[1.,'mean_mse']/shuffled.loc[1.,'mean_mse']
    selection='; '.join(f'{side}: '+('no K=2–6 passed the stability/size rule' if r['selected_k']==1 else f'the rule selects K={r["selected_k"]}') for side,r in results.items())
    overview=f'Updated joint-feature analysis: {selection}. The prespecified two-group view is reported separately from this model-selection result. Spatial agreement is a post hoc association; posture-only, velocity-only and camera controls determine how it should be interpreted.'
    lines=['# 0724: eigenpostures, body velocity and individual behavior','',
      'This revision starts at step 2. The six figures cover eigenpostures, motif construction, ant profiles, withheld spatial outcomes, controls and next-day validation. Measurement QC remains in the tables and methods. All motifs and downstream groups are refitted with signed forward and lateral velocity; no previous group assignments are reused.','',
      overview,'','## Measured results','',
      f'- {len(ants)} of {len(qc)} detection-screened ants qualify: {int(ants.day1_clips.sum()):,} day-1 clips and {int(ants.day2_clips.sum()):,} day-2 clips. Training subsets balance ants.',
      f'- {fit["rank"]} posture modes retain 95% of day-1 direction-cosine variance and reconstruct {recon.loc[(recon.day==2)&(recon.modes==fit["rank"]),"explained_variance"].iloc[0]:.1%} on day 2. The sampled-posture heat map is a rectangular 2D histogram with equal samples per ant; colors show probability per bin.',
      f'- A one-second joint history reduces 0.25-second future **posture-plus-velocity** prediction error by {hgain:.1%} relative to the instantaneous joint state and by {ordergain:.1%} relative to shuffled past samples. Negative gains mean worse prediction. The target and its error use the same balanced feature scaling as the motif model; component-wise errors are also saved.',
      f'- The chosen dictionary has {fit["n_motifs"]} motifs. Selection compares 12/24/48 centers using held-out day-1 hours and chooses the smallest count within one standard error of the best future-state prediction. This tested range does not establish a natural number of discrete states.','',
      '## Step 3: exactly how a motif is formed','',
      '**1. Measure posture and signed velocity.** The eight head, gaster and antennal segment directions are expressed relative to the petiole-to-tag anterior axis. PCA learns an interpretable posture basis. Separately, a five-frame causal least-squares slope estimates the TrackX/TrackY anchor velocity in mm/s. The mean unit anterior axis in those five frames defines the projection: `v_forward = u · a`; `v_lateral = u · (-a_y, a_x)`. Positive forward means motion toward the head. Lateral sign follows the displayed perpendicular axis, not a claim about anatomical left/right in image coordinates. Absolute location and global heading are discarded after these calculations.',
      '**2. Put posture and velocity on a comparable scale.** Center all channels using balanced training data. Divide posture coefficients by the square root of their total variance, preserving the relative weight of posture modes. Divide each velocity channel by its training SD times √2. The posture block and the velocity block each then contribute total variance one. Scaling uses day-1 training data only. Exact centers and divisors are in `feature_scaling.csv`.',
      f'**3. Stack one second of history.** At 12 Hz, 13 timestamps span one second. Each timestamp contains {fit["rank"]} posture coefficients, forward velocity and lateral velocity: {13*(fit["rank"]+2)} values per history. Flatten this time-by-feature matrix and divide by √13, so squared distance is the mean feature discrepancy across those times.',
      '**4. Learn shared centers.** MiniBatchKMeans clusters the training histories in that full joint-feature space. Each center is a motif prototype; every history receives the label of its nearest center. The dictionary is shared by both colonies, learned on day 1 and frozen for day 2. The illustrated skeletons and velocity traces are actual measured histories nearest their respective centers, chosen to span the centers’ velocity range. They are not reconstructed video or hand-labeled behavioral acts.',
      '**5. Move from motifs to ants.** Count the fraction of each ant’s accepted histories assigned to each motif. Ant clustering uses square-root motif frequencies, separately within each colony. An ant group therefore summarizes recurrent combinations of posture and locomotion, rather than an arena-location label.','']
    control_names={'posture_velocity_history':'Posture + velocity history','instantaneous':'Instantaneous joint state','posture_history':'Posture history only','velocity_history':'Velocity history only','dynamics_only':'Mean-removed joint history','velocity_half':'Half velocity amplitude','velocity_double':'Double velocity amplitude','camera_centered':'Camera-centered joint history','camera_only':'Camera only'}
    for side,r in results.items():
        ss=summary[side];c=r['candidates'].set_index('k').loc[2];v=r['profile_validation'].set_index('model')
        lines += [f'## {side.title()} colony','',
          f'{ss["n_ants"]} ants enter the refit. The prespecified K=2 view contains '+', '.join(f'P{k}: {n}' for k,n in ss['groups'].items())+f'. Its silhouette is {c.silhouette:.3f}, with median whole-ant bootstrap ARI {c.bootstrap_ari_median:.3f}. The stability/size/silhouette rule selects K={ss["selected_k"]}; K=1 means no candidate passed, not a test of unimodality.',
          f'Agreement with the previous spatial grouping is **ARI {ss["spatial_ari"]:.3f}** (whole-ant label permutation p={ss["spatial_permutation_p"]:.3f}). Day-2 group retention is {ss["day2_assignment_retention"]:.1%} among {ss["day2_assignment_n"]} qualifying ants.',
          f'Two prototypes explain {v.loc["K=2","explained_vs_mean"]:.1%} of next-day motif-profile variance relative to the day-1 mean. Continuous rank-1 and rank-2 projections explain {v.loc["PC1","explained_vs_mean"]:.1%} and {v.loc["PC2","explained_vs_mean"]:.1%}, respectively.',
          f'The joint-feature group forecast improves day-2 spatial Hellinger² loss by **{ss["mean_spatial_gain"]:.4f}** over the leave-one-ant-out colony-mean forecast (whole-ant bootstrap 95% CI {ss["spatial_gain_ci"][0]:.4f} to {ss["spatial_gain_ci"][1]:.4f}; group-label permutation p={ss["forecast_permutation_p"]:.3f}; n={ss["forecast_n"]}). Both forecasts use only other ants’ day-1 maps. Mean losses: colony mean {ss["baseline_spatial_loss"]:.4f}, joint-feature groups {ss["group_spatial_loss"]:.4f}, own previous-day map {ss["own_previous_day_loss"]:.4f}.','',
          '| Representation | ARI vs spatial groups | ARI vs primary joint groups |','|---|---:|---:|']
        cc=controls[controls.side.eq(side)].set_index('representation')
        for key,label in control_names.items():lines.append(f'| {label} | {cc.loc[key,"agreement_with_spatial"]:.3f} | {cc.loc[key,"agreement_with_primary"]:.3f} |')
        lines += ['','| Group | Ants | Day-2 position-qualified | Median nest occupancy | Median sampled speed |','|---|---:|---:|---:|---:|']
        for g,part in r['table'].groupby('group'):
            measured=part[part.position_coverage_day2.ge(.4)]
            lines.append(f'| P{g} | {len(part)} | {len(measured)} | {measured.day2_colony_percent.median():.1f}% | {part.sampled_speed_mm_s.median():.3f} mm/s |')
        lines+=['','Nest medians require ≥40% day-2 position coverage. Sampled speed is the magnitude of the new body-velocity estimate on accepted day-1 clips; it differs from the pipeline’s full-recording speed estimator. P0/P1 remain ordered by lower/higher median postural angular motion, never by spatial outcome.','']
        included=audit[audit.side.eq(side)].groupby('cluster_id').day1_eligible.agg(['sum','size'])
        lines.append('Availability by previous spatial group, examined only after fitting: '+', '.join(f'{label}: {int(row["sum"])}/{int(row["size"])} included' for label,row in included.iterrows())+'.')
        lines.append('')
    lines += ['## Interpretation and limitations','',
      'This is an exploratory revision following the earlier posture-only analysis. Adding velocity is an explicit scientific change: the motifs now represent posture and locomotion together. Spatial recovery could be driven by posture, velocity or both; the ablations use the same cohort and motif count to expose those contributions. Half/double velocity amplitudes are sensitivity checks, not weights chosen for agreement with spatial labels.',
      'Camera-only profiles are a nuisance comparator. Centering the joint features separately by camera can remove real context-dependent behavior as well as imaging bias, so it is not a causal correction. The position and pose cameras must each remain fixed throughout an accepted clip. This avoids camera-switch jumps but cannot remove all calibration error or tracking jitter. Finished tracks lack per-landmark confidence. The tracked skeleton includes head, gaster and antennae but no leg gait.',
      'KMeans supplies the requested two-way view even when two groups are unsupported. K=2–6 stability, continuous profile predictions, independent day-2 assignments and held-out spatial forecasts must be considered together. Ants share two colonies in one recording; neither ant-bootstrap intervals nor permutation results are population-level replication. Intervals condition on the learned representation. Group correspondence does not establish causation or discrete physiological types.',
      'Complete-clip filtering selects visible behavior. Missing landmarks, duplicate detections, unknown/switching cameras and implausible geometry reject clips. No samples are interpolated or treated as immobility. A broad guard excludes clips whose five-frame estimated velocity magnitude exceeds 20 mm/s. `pose_quality.csv` records posture, velocity and camera acceptance separately. Initial >40% detection screening uses each ant’s first-to-last observed span; fitting requires ≥240 accepted day-1 clips across ≥24 half-hours. Day 2 does not select the training cohort.','',
      '## Sampling, holdouts and provenance','',
      'Only 20260724/block01 is used. Training spans July 24 10:00 to July 25 10:00, with the following matched 24 hours held out. One independently seeded random 2.5-second clip is sampled per minute and ant, exactly matching the previous requested clip schedule. Coordinates are calibrated at 0.016 mm/pixel and 24 Hz. Posture uses a four-frame causal direction average; velocity uses a five-frame causal position slope. Both are sampled at raw frames 4,6,…,58. The history endpoint is frame 52 and the prediction target is frame 58 (0.25 s later); their filter windows do not overlap.',
      'Every fourth first-day hour is withheld for history and dictionary-resolution validation. Shape PCA, normalization and models use balanced training ants. Final models are refitted on day 1 only and then frozen. No absolute location, spatial map, previous spatial label or sleep feature enters the fitted state. The derivatives used for body velocity are explicitly allowed inputs. `SPACE_BLIND_FROZEN.json` records the feature version, dimension, metric and model/group hashes before occupancy outcomes are loaded.',
      'Ant profiles use square-root motif frequencies. Candidate K=2–6 models are resampled over whole ants 100 times. Eligible partitions have smallest group ≥4 and median bootstrap ARI ≥0.8; choose the smallest K within 0.02 of the best eligible silhouette. If none qualifies, report K=1 and retain the prespecified K=2 diagnostic view. All downstream analyses are recomputed, including camera and feature ablations, temporal profiles, next-day profile prediction, spatial maps and forecasts.',
      'Spatial prediction requires ≥40% observed positions on both days. The test ant is excluded when averaging day-1 maps of its own group or the whole colony. Uncertainty resamples held-out ants 2,000 times; 1,000 whole-ant group-label permutations give the null. Previous fine-grid spatial groups are descriptive labels from the full recording, not held-out training targets.','',
      '## Figures and reproducible outputs','',
      '[Combined PDF](0724_postural_dynamics_velocity.pdf) · [Interactive explorer](explorer.html) · [Figure gallery](index.html)','']
    for stem,label,caption in FIGURES:lines += [f'### {stem[:2]}. {label}',caption,f'![{label}]({stem}.png)','']
    lines += ['The tables include feature scaling, quality/selection audits, history prediction by feature block, model selection, group assignments, all representation comparisons, held-out profile/spatial forecasts and temporal profiles. Models include actual motif examples with signed velocities, center velocity histories, PCA, all motif centers and the normalization parameters. Code hashes, source fingerprints and software versions accompany the results.','',
      '## Methodological sources','',
      '[Stephens et al. (2008), Dimensionality and Dynamics in the Behavior of C. elegans](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000028) motivates learning intrinsic shape modes before interpreting external function.',
      '[Costa, Ahamed, Jordan and Stephens, Maximally predictive states](https://arxiv.org/abs/2105.12811) motivates evaluating histories by predictive value. This implementation uses PCA, ridge prediction and KMeans motifs, not their full transfer-operator construction.','']
    report=re.sub(r'(\|[^\n]*\|)\n\n(?=\|)',r'\1\n','\n\n'.join(lines))
    (out/'REPORT.md').write_text(re.sub(r'\n{3,}','\n\n',report))
    sections=''.join(f'<section id="{stem}"><h2>{stem[:2]}. {html.escape(label)}</h2><p>{html.escape(caption)}</p><a href="{stem}.pdf"><img src="{stem}.png" alt="{html.escape(label)}"></a></section>' for stem,label,caption in FIGURES)
    findings=''.join(f'<li>{side.title()}: {ss["n_ants"]} ants; selected K={ss["selected_k"]}; joint-feature/spatial ARI={ss["spatial_ari"]:.3f}; next-day group retention={ss["day2_assignment_retention"]:.1%}.</li>' for side,ss in summary.items())
    page=f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>0724 · posture and velocity</title><style>body{{font:17px/1.55 system-ui;background:#f3f5f6;color:#263448;margin:0}}main{{max-width:1280px;margin:auto;padding:38px 24px}}h1{{font-size:40px;line-height:1.12}}a{{color:#087e80}}section{{background:white;padding:24px;border-radius:12px;margin:24px 0}}img{{width:100%;height:auto}}.lead{{max-width:960px}}nav a{{margin-right:24px}}li{{margin:7px 0}}</style><main><p>20260724 / BLOCK01 · POSTURE + BODY VELOCITY</p><h1>Eigenpostures → joint histories → individual behavior</h1><p class="lead">Start with a posture basis and density heat map. Add signed forward and lateral velocity, learn one-second motifs, and refit every downstream analysis. Spatial occupancy is opened after fitting.</p><nav><a href="0724_postural_dynamics_velocity.pdf">Six-figure PDF</a><a href="explorer.html">Interactive explorer</a><a href="REPORT.md">Full report</a><a href="run_manifest.json">Provenance</a></nav><section><h2>Updated results</h2><p>{len(ants)} eligible ants · {fit['rank']} posture modes + 2 velocity channels · {fit['n_motifs']} motifs.</p><ul>{findings}</ul><p>{html.escape(overview)}</p></section>{sections}</main></html>"""
    (out/'index.html').write_text(page)


def write_explorer(out,results,fit,metadata):
    m=fit['model_arrays'];payload=dict(lengths=m['standard_lengths_mm'].tolist(),mean=m['posture_mean'].tolist(),
      components=m['posture_components'].tolist(),sd=np.sqrt(m['posture_variance']).tolist(),variance=m['posture_variance_ratio'].tolist(),
      examples=m['motif_example_shape'].round(5).tolist(),velocity=m['motif_example_velocity_mm_s'].round(5).tolist(),
      rank=int(m['rank']),n_motifs=int(m['n_motifs']),history_seconds=1,colonies={})
    for side,r in results.items():
        t=r['table'];ants=[]
        for j,row in enumerate(t.itertuples()):
            ants.append(dict(ant=row.ant,group=row.group,pc1=row.pc1,pc2=row.pc2,motion=row.angular_motion_rad_s,speed=row.sampled_speed_mm_s,
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
        profile_variance=np.var((np.sqrt(profiles['posture_velocity_history'][0,rows])-profile_mean)@profile_components.T,axis=0,ddof=1)
        atoms=Path(manifest['arguments']['spatial_atoms'])/(t.ant.iloc[0].replace(':','_')+'.npz')
        with np.load(atoms) as z:edges=(z['x_edges'],z['y_edges'])
        forecast=pd.read_csv(output/'heldout_spatial_forecasts.csv')
        results[side]=dict(table=t,rows=rows,profile=profiles['posture_velocity_history'][:,rows],
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
