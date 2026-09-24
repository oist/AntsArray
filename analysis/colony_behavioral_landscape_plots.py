"""Figures, a linked report and an offline explorer for one colony recording."""
from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle, Rectangle
import numpy as np
import pandas as pd

COLORS=['#168b8b','#dd7652']
INK='#263448'
FIGURES=[
    ('01_measurement','From tracking to spatial behavior','Every observation begins with position and tracking coverage. Gray points and maps precede any group interpretation.'),
    ('02_modes','Find the dominant spatial modes','PCA acts on square-root occupancy, separately for each colony. Signed maps locate the spatial features of each mode.'),
    ('03_groups','Recover the two-group picture','Compare the existing fine-grid labels with a new full-feature two-group fit. The density curves are descriptive alternatives, not a state-discovery test.'),
    ('04_validation','Ask what makes two groups credible','Test more groups, resample whole ants, predict the next day, and change spatial scale. The continuous PC line is an explicit alternative.'),
    ('05_behavior','Interpret the spatial groups','Inspect the physical maps, individual variation, speed and the existing motion-based sleep estimate. Each dot is one ant.'),
    ('06_dynamics','Separate individual differences from dynamics','Project half-hour behavior onto the same spatial axis. Show missing coverage and compare pooled persistence with persistence after removing each ant’s mean.'),
    ('07_reproducibility','Follow the same ants across two days','Compare first-day coordinates with their second-day projections and inspect representative individual traces. These are trajectories through a descriptive landscape.')]


def setup():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,'axes.labelsize':9,
                         'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#a3abb3',
                         'axes.labelcolor':INK,'text.color':INK,'xtick.color':INK,'ytick.color':INK,
                         'figure.facecolor':'white','savefig.facecolor':'white','pdf.fonttype':42})


def spatial(ax, values, result, regions, *, fine=False, signed=False, vmax=None, title=''):
    x,y=result['fine_edges' if fine else 'edges']
    if signed:
        vmax=np.max(np.abs(values)) if vmax is None else vmax
        im=ax.pcolormesh(x,y,values,cmap='RdBu_r',vmin=-vmax,vmax=vmax,rasterized=True)
    else:
        density=values/(np.diff(y)[:,None]*np.diff(x)[None,:])
        view=np.sqrt(density)
        vmax=float(np.quantile(view,.997)) if vmax is None else vmax
        im=ax.pcolormesh(x,y,view,cmap='magma',vmin=0,vmax=vmax,rasterized=True)
    meta=result['metadata']; side=result['summary']['side']
    ox=meta['arena_bounds_px']['x_min_px']; oy=meta['arena_bounds_px']['y_min_px']; scale=meta['mm_per_px']
    for r in regions:
        if r['side']!=side or r['region_type']=='arena': continue
        color='#70d6d0' if r['region_type']=='colony' else ('#b6db4a' if r['region_type']=='food' else '#69aee0')
        if r['shape']=='rectangle':
            xy=((r['tracking_x_min_px']-ox)*scale,(r['tracking_y_min_px']-oy)*scale)
            patch=Rectangle(xy,(r['tracking_x_max_px']-r['tracking_x_min_px'])*scale,(r['tracking_y_max_px']-r['tracking_y_min_px'])*scale,fill=False,ec=color,lw=.8)
        else:
            xy=((r['tracking_center_x_px']-ox)*scale,(r['tracking_center_y_px']-oy)*scale)
            patch=Circle(xy,r['radius_mm'],fill=False,ec=color,lw=.8)
        ax.add_patch(patch)
    ax.set(xlabel='x (mm)',ylabel='y (mm)',title=title,aspect='equal')
    ax.set_xlim(0,meta['grid_width_mm']); ax.set_ylim(meta['grid_height_mm'],0)
    return im


def title(fig, number, subtitle):
    fig.suptitle(f'{number:02d}  {FIGURES[number-1][1]}',x=.035,ha='left',fontsize=19,fontweight='bold')
    fig.text(.035,.925,subtitle,ha='left',va='top',fontsize=10,color='#586675')
    fig.subplots_adjust(top=.84,bottom=.10,left=.065,right=.97,hspace=.50,wspace=.43)


def save(fig, stem, output, pdf):
    fig.savefig(output/(stem+'.png'),dpi=180)
    fig.savefig(output/(stem+'.pdf'))
    pdf.savefig(fig)
    plt.close(fig)


def dots(ax,x,y,groups):
    for k in (0,1):
        keep=np.asarray(groups)==k
        ax.scatter(np.asarray(x)[keep],np.asarray(y)[keep],s=26,c=COLORS[k],edgecolor='white',lw=.3,alpha=.9,label=f'G{k}')


def render(results,inventory,data,info,output):
    setup(); regions=json.loads(info['regions'])
    with PdfPages(output/'0724_behavioral_landscape.pdf') as pdf:
        fig,axes=plt.subplots(2,3,figsize=(15,9))
        title(fig,1,'0724/block01 only  ·  49.16 hours  ·  114 tracked identities  ·  86 ants pass the existing >40% detection rule')
        for row,(side,r) in enumerate(results.items()):
            all_table=inventory[inventory.side.eq(side)].sort_values('speed_detection')
            ax=axes[row,0]; positions=np.arange(len(all_table))
            ax.bar(positions,all_table.speed_detection*100,color=np.where(all_table.selected,INK,'#cbd2d8'),width=.9)
            ax.axhline(40,color=COLORS[1],ls='--',lw=1)
            ax.set(xlabel='Ants ordered by detection',ylabel='Frames detected (%)',ylim=(0,103),title=f'{side.title()}: {len(r["table"])} / {len(all_table)} ants included')
            im=spatial(axes[row,1],r['fine'].mean(axis=0),r,regions,fine=True,title='Mean occupancy · equal ant weights')
            fig.colorbar(im,ax=axes[row,1],shrink=.8,label='√ probability density (1/mm)')
            ax=axes[row,2]; t=r['table']
            ax.scatter(t.speed_detection*100,t.entropy_bits,c=INK,s=25)
            ax.set(xlabel='Frames detected (%)',ylabel='Spatial entropy (bits; 1 mm bins)',title='Coverage and spatial spread')
        save(fig,FIGURES[0][0],output,pdf)

        fig,axes=plt.subplots(2,4,figsize=(17,9))
        title(fig,2,'Separate PCA for each colony; all spatial features enter the fit. A low-dimensional display does not imply two discrete states.')
        for row,(side,r) in enumerate(results.items()):
            p=r['pca']; ax=axes[row,0]; n=min(15,len(p.explained_variance_ratio_))
            ax.plot(np.arange(1,n+1),p.explained_variance_ratio_[:n]*100,'o-',color=INK,ms=4,label='1 mm')
            ax.plot(np.arange(1,n+1),r['pca_fine'].explained_variance_ratio_[:n]*100,'o--',color='#929eab',ms=3,label='0.25 mm')
            ax.set(xlabel='Mode rank',ylabel='Variance explained (%)',title=f'{side.title()} · PC1 {r["summary"]["pc1_variance"]:.0%} at 1 mm')
            ax.legend(frameon=False,fontsize=8)
            for j in (0,1):
                values=p.components_[j,:-1].reshape(r['maps'].shape[1:])
                im=spatial(axes[row,j+1],values,r,regions,signed=True,title=f'PC{j+1} spatial loading')
                fig.colorbar(im,ax=axes[row,j+1],shrink=.8,label='Loading (√ probability)')
            ax=axes[row,3]; z=r['z']; ax.scatter(z[:,0],z[:,1],color=INK,s=25)
            ax.set(xlabel=f'PC1 ({p.explained_variance_ratio_[0]:.0%})',ylabel=f'PC2 ({p.explained_variance_ratio_[1]:.0%})',title='One point per ant · no labels used')
        save(fig,FIGURES[1][0],output,pdf)

        fig,axes=plt.subplots(2,4,figsize=(18,9))
        title(fig,3,'Saved fine-grid Leiden labels are a reference; new G0/G1 labels come from KMeans in the full 1 mm square-root feature space.')
        for row,(side,r) in enumerate(results.items()):
            t=r['table']; ax=axes[row,0]
            for k,label in enumerate(sorted(t.old_fine_label.unique())):
                mask=t.old_fine_label.eq(label); ax.scatter(t.fine_pc1[mask],t.fine_pc2[mask],s=26,c=COLORS[k],label=label)
            ax.legend(frameon=False,fontsize=8); ax.set(xlabel='Fine-grid PC1',ylabel='Fine-grid PC2',title=f'{side.title()} · existing two-group picture')
            ax=axes[row,1]; dots(ax,t.pc1,t.pc2,t.group); ax.legend(frameon=False,fontsize=8)
            ax.set(xlabel='PC1',ylabel='PC2',title='Independent two-group fit')
            order=np.lexsort((t.pc1,t.group)); ax=axes[row,2]
            im=ax.imshow(r['distance'][order][:,order],cmap='viridis',vmin=0,vmax=1,aspect='equal')
            boundary=(t.group==0).sum()-.5; ax.axvline(boundary,color='white',lw=.6);ax.axhline(boundary,color='white',lw=.6)
            ax.set(xlabel='Ants sorted by group, then PC1',ylabel='Same ants',title='Full-space Hellinger distance')
            fig.colorbar(im,ax=ax,shrink=.8)
            ax=axes[row,3]; xx=np.linspace(t.pc1.min()-.12,t.pc1.max()+.12,400)[:,None]
            for model,color,style,name in zip(r['gmm'],[INK,'#929eab'],['-','--'],['1 Gaussian','2 Gaussians']):
                ax.plot(xx[:,0],np.exp(model.score_samples(xx)),style,color=color,label=name)
            dots(ax,t.pc1,np.zeros(len(t)),t.group)
            ax.legend(frameon=False,fontsize=8); ax.set(xlabel='PC1',ylabel='Density',title=f'ΔBIC (1 − 2) = {r["summary"]["pc1_two_gaussian_delta_bic"]:.1f}')
        save(fig,FIGURES[2][0],output,pdf)

        fig,axes=plt.subplots(2,4,figsize=(18,9))
        title(fig,4,'Uncertainty resamples ants, not frames. Next-day prediction uses July 24 10:00–July 25 10:00 for training and the following 24 hours for testing.')
        fig.subplots_adjust(wspace=.65)
        for row,(side,r) in enumerate(results.items()):
            c=r['candidates']; ax=axes[row,0]
            ax.plot(c.k,c.silhouette,'o-',color=INK)
            lo,hi=r['summary']['silhouette_gaussian_null_95']; ax.vlines(2,lo,hi,color=COLORS[1],lw=7,alpha=.4,label='K=2 Gaussian reference 95%')
            ax.set(xlabel='Number of groups',ylabel='Silhouette (full feature space)',title=f'{side.title()} · geometric separation',xticks=range(2,7));ax.legend(frameon=False,fontsize=7,loc='upper right')
            ax=axes[row,1]; ax.plot(c.k,c.bootstrap_ari_median,'o-',color=INK);ax.fill_between(c.k,c.bootstrap_ari_p10,c.bootstrap_ari_p90,color=INK,alpha=.15)
            ax.axhline(.8,color='#a0aab5',ls=':',lw=1);ax.set(xlabel='Number of groups',ylabel='Adjusted Rand agreement',ylim=(-.05,1.05),xticks=range(2,7),title='Ant bootstrap · median and 10–90%')
            ax=axes[row,2]; v=r['validation']; ks=v[v.model.str.startswith('K=')]
            ax.plot(ks.complexity,ks.explained_vs_train_mean*100,'o-',color=INK,label='Discrete prototypes')
            for rank,color in [(1,COLORS[0]),(2,COLORS[1])]:
                value=v.loc[v.model.eq(f'PC{rank}'),'explained_vs_train_mean'].iloc[0]*100
                ax.axhline(value,color=color,ls='--',label=f'Continuous PC{rank} subspace')
            ax.set(xlabel='Number of prototypes',ylabel='Day-2 variance explained (%)',title=f'Temporal holdout · {r["summary"]["temporal_validation_n"]} paired ants',xticks=range(1,7));ax.legend(frameon=False,fontsize=7)
            ax=axes[row,3]; s=r['sensitivity']; names=['Smooth 1 mm','Smooth 2 mm','Fine-grid K=2','Saved fine labels']
            vals=s.ari.iloc[[1,2,3,4]]
            ax.barh(np.arange(4),vals,color=[INK,INK,COLORS[0],COLORS[1]])
            ax.set(yticks=np.arange(4),yticklabels=names,xlim=(0,1.04),xlabel='Adjusted Rand vs primary K=2',title='Sensitivity to representation');ax.invert_yaxis()
        save(fig,FIGURES[3][0],output,pdf)

        fig,axes=plt.subplots(2,5,figsize=(19,9))
        title(fig,5,'G0 is named after its greater median nest occupancy, after clustering. Speed and motion-defined sleep were not used to fit the groups.')
        for row,(side,r) in enumerate(results.items()):
            t=r['table']; means=[r['fine'][t.group.eq(k)].mean(axis=0) for k in (0,1)]
            area=np.diff(r['fine_edges'][1])[:,None]*np.diff(r['fine_edges'][0])[None,:]
            vmax=np.quantile(np.sqrt(np.stack(means)/area),.997)
            for k in (0,1):
                im=spatial(axes[row,k],means[k],r,regions,fine=True,vmax=vmax,title=f'{side.title()} G{k} · n={(t.group==k).sum()}')
            for j,(metric,label) in enumerate([('colony_percent','Inside nest (%)'),('mean_speed_mm_s','Mean speed (mm/s)'),('sleep_percent','Motion-defined sleep (%)')],start=2):
                ax=axes[row,j];dots(ax,t.pc1,t[metric],t.group)
                rho=r['behavior'].set_index('metric').loc[metric,'pc1_spearman']
                ax.set(xlabel='Continuous PC1 coordinate',ylabel=label,title=f'Individual variation · ρ={rho:.2f}')
                if metric.endswith('percent'):ax.set_ylim(-3,103)
                if j==2:ax.legend(frameon=False,fontsize=8)
        fig.text(.065,.025,'Occupancy maps share a color scale within each colony (square-root probability density). Nest outlines: teal; food: green; water: blue.',fontsize=9)
        save(fig,FIGURES[4][0],output,pdf)

        fig,axes=plt.subplots(2,3,figsize=(17,10),gridspec_kw={'width_ratios':[1.35,1.1,1]})
        title(fig,6,'Rows are individual ants, sorted by their whole-recording PC1. Half-hour projections need ≥40% detection and ≥95% recorded exposure; gaps remain blank.')
        for row,(side,r) in enumerate(results.items()):
            order=np.argsort(r['table'].pc1); ant_order=r['table'].ant.iloc[order].to_list();scores=r['scores'][order]
            lim=np.nanquantile(abs(scores),.99);ax=axes[row,0]
            im=ax.imshow(scores,aspect='auto',interpolation='none',cmap='RdBu_r',vmin=-lim,vmax=lim,extent=(0,scores.shape[1]/2,len(scores)-.5,-.5))
            ticks=np.arange(0,len(scores),4);ax.set(yticks=ticks,yticklabels=[a.split(':')[1] for a in np.array(ant_order)[ticks]],ylabel='Ant tag',xlabel='Hours from Jul 24 09:30',title=f'{side.title()} · continuous PC1 through time')
            fig.colorbar(im,ax=ax,shrink=.8,label='PC1')
            cov=np.stack([data[a]['detected']/np.maximum(1,data[a]['recorded']) for a in ant_order]);ax=axes[row,1]
            im=ax.imshow(cov,aspect='auto',interpolation='none',cmap='Greys',vmin=0,vmax=1,extent=(0,scores.shape[1]/2,len(scores)-.5,-.5))
            ax.set(yticks=[],xlabel='Hours from Jul 24 09:30',title='Position coverage · same row order');fig.colorbar(im,ax=ax,shrink=.8,label='Detected fraction')
            ax=axes[row,2];lag=r['lag']
            for key,color,name in [('pooled',INK,'Pooled'),('within_ant',COLORS[0],'Ant means removed')]:
                ax.plot(lag.lag_hours,lag[key],color=color,label=name)
                ax.fill_between(lag.lag_hours,lag[key+'_shuffle_low'],lag[key+'_shuffle_high'],color=color,alpha=.17)
            ax.axhline(0,color='#adb5bf',lw=.7);ax.set(xlabel='Lag (hours)',ylabel='Lag correlation',ylim=(-.35,1.03),title='Persistence and the identity control');ax.legend(frameon=False,fontsize=8)
        fig.text(.065,.027,'Shading: 95% envelope from permuting time within each ant, retaining missing positions. It is a temporal-order control, not a confidence interval.',fontsize=9)
        save(fig,FIGURES[5][0],output,pdf)

        fig,axes=plt.subplots(2,3,figsize=(17,9))
        title(fig,7,'Day 2 is projected using the day-1 PCA. Example ants are chosen near the low, middle and high whole-recording PC1 quantiles, before inspecting trajectories.')
        for row,(side,r) in enumerate(results.items()):
            p=r['paired'];t=r['table']; groups=p.ant.map(t.set_index('ant').group);ax=axes[row,0]
            dots(ax,p.day1_pc1,p.day2_pc1,groups); limits=[min(p.day1_pc1.min(),p.day2_pc1.min()),max(p.day1_pc1.max(),p.day2_pc1.max())]
            ax.plot(limits,limits,':',color='#8998a5');ax.set(xlabel='Day-1 PC1',ylabel='Day-2 PC1 in day-1 basis',title=f'{side.title()} · same-ant ρ={r["summary"]["day_pc1_spearman"]:.2f}')
            order=np.argsort(t.pc1); example=order[np.round(np.array([.1,.5,.9])*(len(order)-1)).astype(int)]
            for i,color in zip(example,['#4477aa','#888888','#aa3377']):
                ant=t.iloc[i].ant;trace=r['time'][r['time'].ant.eq(ant)].set_index('calendar_bin')
                calendar=data[ant]['calendar_bin'];trace=trace.reindex(calendar); h=(calendar-calendar[0])*.5+.25
                y=trace.pc1.where(trace.qualified.eq(True))
                axes[row,1].plot(h,y,color=color,lw=1,label=ant.split(':')[1])
                y=trace.mean_speed_mm_s.where(trace.speed_coverage.ge(.4))
                axes[row,2].plot(h,y,color=color,lw=1,label=ant.split(':')[1])
            axes[row,1].set(xlabel='Hours from Jul 24 09:30',ylabel='Half-hour PC1',title='Representative individual trajectories')
            axes[row,2].set(xlabel='Hours from Jul 24 09:30',ylabel='Mean speed (mm/s)',title='Measured movement · same ants')
            axes[row,1].legend(title='Tag',frameon=False,fontsize=8,ncol=3)
        save(fig,FIGURES[6][0],output,pdf)
    write_report(results,info,output)
    write_explorer(results,data,output)
    write_gallery(output)


def write_report(results,info,output):
    l,r=(results[s] for s in ('left','right'))
    match=[a['sensitivity'].iloc[4].ari for a in (l,r)]
    recovered=(f'An independent full-feature fit exactly recovers all {sum(a["summary"]["n_ants"] for a in (l,r))} saved fine-grid group memberships.'
               if all(np.isclose(x,1) for x in match) else f'Agreement with saved fine-grid groups is ARI {match[0]:.3f} left and {match[1]:.3f} right.')
    validation=[a['validation'].set_index('model') for a in (l,r)]
    main=(f'**Main finding:** {recovered} Individual rankings have day-1/day-2 PC1 rank correlations of '
          f'{l["summary"]["day_pc1_spearman"]:.2f} and {r["summary"]["day_pc1_spearman"]:.2f} (left/right). '
          'The group maps and measurements below describe nest use, movement and motion-defined sleep.')
    dynamics=(f'**The dynamics qualify that picture:** differences between individual ant means account for '
              f'{1-l["summary"]["within_ant_pc1_variance_fraction"]:.0%} (left) and {1-r["summary"]["within_ant_pc1_variance_fraction"]:.0%} (right) of temporal PC1 variation. '
              f'A continuous PC1 axis explains {validation[0].loc["PC1","explained_vs_train_mean"]:.1%}/{validation[1].loc["PC1","explained_vs_train_mean"]:.1%} '
              f'of next-day variation versus {validation[0].loc["K=2","explained_vs_train_mean"]:.1%}/{validation[1].loc["K=2","explained_vs_train_mean"]:.1%} '
              'for two fixed prototypes (left/right). A discrete switching-state interpretation needs additional evidence.')
    lines=['# The 0724 behavioral landscape','',
           'This analysis uses **only 20260724/block01**: July 24 09:31:10 to July 26 10:40:36.5, 49.16 hours at 24 fps. It does not fit on the July 23, July 26 or July 29 recordings. The recording label 0724 includes two nights and the following two mornings.',
           '', '[Open the figure gallery and interactive ant explorer](index.html) · [Download all seven figures as a PDF](0724_behavioral_landscape.pdf)', '',
           'The measurement is an ant’s spatial occupancy distribution, rather than posture or a complete behavioral state. Separate fits are made for the two colony sides, with identity defined by side plus tag. The analysis builds from observations to spatial modes, groups, measured behavior summaries and time dependence.', '',
           main,'',dynamics,'',
           '## What the data show','',
           '| Quantity | Left colony | Right colony |','|---|---:|---:|']
    for label,fn in [
        ('Selected / tracked ants',lambda a:f'{a["summary"]["n_ants"]} / {a["summary"]["all_tracks"]}'),
        ('Saved 0.25 mm Leiden group sizes',lambda a:' / '.join(str(v) for v in a['summary']['saved_fine_groups'].values())),
        ('New 1 mm K=2 group sizes (G0 / G1)',lambda a:' / '.join(map(str,a['summary']['groups']))),
        ('Variance in PC1 / first two PCs',lambda a:f'{a["summary"]["pc1_variance"]:.1%} / {a["summary"]["pc2_cumulative"]:.1%}'),
        ('Dimensions for 90% variance',lambda a:str(a['summary']['dims90'])),
        ('K=2 silhouette',lambda a:f'{a["candidates"].iloc[0].silhouette:.3f}'),
        ('K=2 median bootstrap ARI',lambda a:f'{a["candidates"].iloc[0].bootstrap_ari_median:.3f}'),
        ('Agreement with saved fine-grid groups (ARI)',lambda a:f'{a["sensitivity"].iloc[4].ari:.3f}'),
        ('Paired ants in day-1/day-2 test',lambda a:str(a['summary']['temporal_validation_n'])),
        ('Same-ant day-1/day-2 PC1 rank correlation',lambda a:f'{a["summary"]["day_pc1_spearman"]:.3f}'),
        ('Within-ant share of temporal PC1 variance',lambda a:f'{a["summary"]["within_ant_pc1_variance_fraction"]:.1%}')]:
        lines.append(f'| {label} | {fn(l)} | {fn(r)} |')
    lines+=['','### 1. Observation and coverage','',
            'All 114 available per-ant tracks were audited. The existing fine-grid cohort rule is retained: more than 40% detected frames according to speed metadata (41 left, 45 right). No ant is selected using the result of the new clustering. The excluded tracks remain in `inventory.csv`; this analysis does not characterize poorly observed ants. Tracking coverage itself can correlate with behavior.',
            '', 'Exact half-hour counts are validated against unchanged finished-track fingerprints and must sum exactly to each previously published 1 mm whole-block histogram. The 0.25 mm maps must use the same number of detected position samples. Counts are normalized by detected frames, with a separate outside-arena bin; unobserved frames are never treated as immobility or outside occupancy.',
            '', '### 2. Dominant modes before labels','',
            'For each ant, let p be its spatial probability vector, including the observed outside-arena probability. PCA is fitted to √p without standardizing individual spatial bins. Euclidean distances in these coordinates divided by √2 equal Hellinger distances. Each ant has equal weight. The PC1 sign is oriented toward lower nest occupancy after the fit; this changes no distances or clusters.', '']
    for side,a in results.items():
        s=a['summary']; lines.append(f'In the **{side}** colony, PC1 explains {s["pc1_variance"]:.1%} of 1 mm occupancy variance and the first two modes explain {s["pc2_cumulative"]:.1%}; {s["dims90"]} modes are needed for 90%. At 0.25 mm, PC1 explains {s["fine_pc1_variance"]:.1%}. The signed mode maps show which locations vary together.')
    lines+=['','### 3. How much support is there for two groups?','',
            'The saved fine-grid Leiden labels are displayed without alteration. New KMeans fits compare K=2–6 in the full 1 mm square-root space; the complete PCA score vector is an exact distance-preserving computational basis, not a two-dimensional clustering input. G0 is the fitted group with greater median nest occupancy. G1 is the other group. These names do not assert jobs or biological castes.', '']
    for side,a in results.items():
        c=a['candidates']; best=c.loc[c.silhouette.idxmax()];two=c.iloc[0];s=a['summary'];v=a['validation'].set_index('model')
        lines.append(f'**{side.title()}:** K=2 has silhouette {two.silhouette:.3f} and median whole-ant bootstrap agreement {two.bootstrap_ari_median:.3f} (10th–90th percentile {two.bootstrap_ari_p10:.3f}–{two.bootstrap_ari_p90:.3f}). The best silhouette among K=2–6 is K={int(best.k)} ({best.silhouette:.3f}). A single Gaussian with matching PCA covariance gives K=2 silhouette median {s["silhouette_gaussian_null_median"]:.3f}, with central 95% range {s["silhouette_gaussian_null_95"][0]:.3f}–{s["silhouette_gaussian_null_95"][1]:.3f}. Along PC1, BIC(1 Gaussian) − BIC(2 Gaussians) is {s["pc1_two_gaussian_delta_bic"]:.1f}; positive values favor the two-component density, without proving two dynamical states.')
        lines.append(f'On the next day, two prototypes explain {v.loc["K=2","explained_vs_train_mean"]:.1%} of variation relative to the day-1 mean; a continuous one-dimensional PC line explains {v.loc["PC1","explained_vs_train_mean"]:.1%}. Independent first- and second-day K=2 partitions agree with ARI {v.loc["K=2","day1_day2_partition_ari"]:.3f}, and {v.loc["K=2","same_ant_assignment_fraction"]:.1%} of paired ants retain their first-day nearest-prototype assignment.')
    lines+=['', 'The prototype/subspace comparison evaluates reconstruction in √p coordinates. The subspace is allowed continuous coordinates and is not required to reconstruct a nonnegative normalized probability vector; it is therefore a flexible geometric benchmark, not a parameter-matched likelihood test. Test ants are the same animals on a later day, so this is temporal validation, not generalization to unseen colonies.',
            '', 'The single-Gaussian reference is deliberately simple and does not preserve the probability-simplex constraints; its envelope is a geometric comparison, not a calibrated biological p-value. The PC1 Gaussian-mixture BIC is in-sample and sensitive to the chosen coordinate. Stable clustering and a good two-component density alone do not establish a barrier, a discrete task, or metastability.',
            '', '### 4. Spatial scale is part of the conclusion','',
            '| Comparison (adjusted Rand agreement) | Left | Right |','|---|---:|---:|']
    for i in range(len(l['sensitivity'])):
        if i>=len(r['sensitivity']):continue
        a,b=l['sensitivity'].iloc[i],r['sensitivity'].iloc[i]
        extra_a=f' (n={int(a.n_ants)}; G0/G1={int(a.group0)}/{int(a.group1)})' if pd.notna(a.get('n_ants')) else ''
        extra_b=f' (n={int(b.n_ants)}; G0/G1={int(b.group0)}/{int(b.group1)})' if pd.notna(b.get('n_ants')) else ''
        lines.append(f'| {a.comparison} | {a.ari:.3f}{extra_a} | {b.ari:.3f}{extra_b} |')
    lines+=['','The saved 1 mm Leiden fit has two left-colony groups and **three** right-colony groups, while the saved 0.25 mm fit has two on each side. Smoothing and grid resolution change the representation. A consistent broad separation can coexist with scale-dependent subdivisions; the old and new label numbers are not interchangeable.',
            '', '### 5. What the groups contain','',
            '| Colony / group | Ants | Nest occupancy, median % | Speed, median mm/s | Motion-defined sleep, median % |','|---|---:|---:|---:|---:|']
    for side,a in results.items():
        for group,t in a['table'].groupby('group'):
            lines.append(f'| {side} G{group} | {len(t)} | {t.colony_percent.median():.1f} | {t.mean_speed_mm_s.median():.3f} | {t.sleep_percent.median():.1f} |')
    coverage={side:a['table'].groupby('group').speed_detection.median().to_numpy() for side,a in results.items()}
    lines+=['',f'Tracking coverage differs between groups: median frame detection is {coverage["left"][0]:.0%} versus {coverage["left"][1]:.0%} on the left and {coverage["right"][0]:.0%} versus {coverage["right"][1]:.0%} on the right (G0 versus G1). The high-coverage refits are tabulated above, including their group sizes; very small retained groups provide limited evidence. This check cannot rule out behavior-dependent tracking bias.',
            '', 'Nest occupancy is derived from the same positions as the landscape and provides interpretation, not independent validation. Speed and sleep were not clustering inputs, but share tracking and motion measurements. Sleep here uses the pre-existing **10-second motion-threshold classifier**, not a new model and not direct physiological sleep measurement. Speed uses bodypoint 0, 24 fps, 0.016 mm/px, interpolation gaps ≤5 frames, Gaussian smoothing σ=2 frames and maximum speed 5 mm/s. Metric means use their own valid observed exposure. Group differences with whole-ant bootstrap intervals are in the behavior CSVs and are descriptive within these two colonies.',
            '', '### 6. Individual variation and temporal structure','',
            'The trajectory is each ant’s half-hour map projected into the whole-0724 PCA basis. The basis is descriptive and uses the whole recording; the separate day-1/day-2 test uses only first-day fitting. Temporal panels display all selected ants, with missing or low-coverage half-hours blank. At least 40% position detection and 95% recorded exposure are required for a temporal point. No trajectory crosses an unobserved interval by interpolation.', '']
    for side,a in results.items():
        s=a['summary'];lag=a['lag'].iloc[1]
        lines.append(f'For **{side}**, {s["within_ant_pc1_variance_fraction"]:.1%} of the equal-ant PC1 variance is within individuals over time; the remainder is between ant means. At a 1-hour lag the pooled correlation is {lag.pooled:.3f}, compared with {lag.within_ant:.3f} after removing each ant’s mean. There are {s["sustained_reassignments"]} sustained nearest-prototype reassignments involving {s["sustained_reassignment_ants"]} ants under the predeclared support rule. Of {s["total_halfhours"]} ant-half-hours, {s["qualified_halfhours"]} pass exposure/detection criteria and {s["qualified_margin_halfhours"]} also pass the centroid-margin criterion; a lack of qualifying reassignments does not establish an absence of behavioral changes.')
    lines+=['','A sustained reassignment requires three consecutive qualifying half-hours in the old group followed immediately by three in the new group, each with a relative centroid-distance margin ≥10%. These are descriptive reassignments; a smooth trajectory can cross the decision boundary. They should not be called behavioral-state transitions without a separate predictive dynamical model. Permuting the order of observations within each ant retains individual mean differences and missing-time positions; the shaded lag-correlation bands show that temporal-order control. Removing ant means also removes slow within-recording shifts and can bias long-lag estimates, so only lags through 12 hours are shown.',
            '', '### Reading the figures as a scientific argument','',
            '1. Establish what was observed and how much tracking is available.',
            '2. Show dominant spatial modes and their physical meaning before assigning labels.',
            '3. Locate the existing two-group picture within that space and inspect full-dimensional distances.',
            '4. Challenge the partition with ant resampling, a continuous alternative, new time periods and scale changes.',
            '5. Interpret individual behavior along the continuous axis as well as by group.',
            '6. Separate persistent individual differences from within-ant dynamics before proposing discrete states.',
            '', '### Connection to Stephens’ work','',
            'The progression from quantitative measurement to interpretable modes and then dynamics follows the methodological spirit of [Stephens et al., 2008, *Dimensionality and Dynamics in the Behavior of C. elegans*](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000028). [Ahamed, Costa and Stephens, 2021, *Capturing the continuous complexity of behaviour*](https://www.nature.com/articles/s41567-020-01036-8) motivates preserving continuous variation rather than assuming every pattern is categorical. [Costa et al., 2023, *Maximally predictive states*](https://arxiv.org/abs/2105.12811) motivates testing predictiveness and separating time scales before interpreting long-lived states. This analysis does not implement eigenworms, delay reconstruction or their transfer-operator estimator; occupancy is a coarser and different observable.',
            '', '### Reproducibility and limits','',
            'The fitted basis, ant tables, selection curves, temporal predictions, source fingerprints, code hashes and software versions are included. The run manifest records the exact single-block source. Raw inputs may live inside a previously generated multi-recording cache, but its source index is checked and **no fitted model or reference label from another recording is used**. Only the requested block’s exact counts and measured behavior summaries are read into the analysis.',
            '', 'This is an exploratory analysis of one recording and two colonies. Ants, windows and frames are not independent colony replicates. Missing tracking may be behavior-dependent. Spatial modes do not identify causality, social role, physiological state or a potential-energy landscape. No transition barrier or free energy is inferred from an embedding density.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')


def write_gallery(output):
    panels=''.join(f'<article><h2>{i:02d} · {html.escape(title)}</h2><p>{html.escape(text)}</p><a href="{stem}.png"><img loading="lazy" src="{stem}.png" alt="{html.escape(title)}"></a><p><a href="{stem}.pdf">PDF</a> · <a href="{stem}.png">PNG</a></p></article>' for i,(stem,title,text) in enumerate(FIGURES,1))
    report=html.escape((output/'report.md').read_text())
    (output/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>0724 · Behavioral landscape</title><style>body{{font:17px/1.6 system-ui,sans-serif;color:#263448;margin:0;background:#f3f5f7}}main{{max-width:1450px;margin:auto;padding:40px 5vw}}h1{{font-size:42px;line-height:1.15}}h2{{font-size:25px}}a{{color:#137878}}article{{background:white;padding:25px;margin:28px 0;border-radius:12px}}img{{width:100%;height:auto}}.links{{display:flex;gap:20px;flex-wrap:wrap}}pre{{white-space:pre-wrap;font:14px/1.55 system-ui;max-width:1050px}}</style><main><p>20260724 / BLOCK01 · SINGLE-RECORDING ANALYSIS</p><h1>From measured behavior<br>to the two-group picture</h1><p>49.16 hours · 114 tracked identities · 86 selected ants · two colonies analyzed separately</p><p class="links"><a href="explorer.html">Explore individual ants ↗</a><a href="0724_behavioral_landscape.pdf">All figures · PDF</a><a href="report.md">Analysis · Markdown</a><a href="run_manifest.json">Source manifest</a></p><p>A spatial behavioral landscape, followed by tests of grouping, reproducibility and within-ant dynamics. Read the figures in sequence; the report below contains quantitative results and interpretation limits.</p>{panels}<article><h2>Detailed analysis</h2><pre>{report}</pre></article></main></html>''')


def write_explorer(results,data,output):
    payload={}
    for side,r in results.items():
        members=[]
        for row in r['table'].itertuples():
            d=data[row.ant];trace=r['time'][r['time'].ant.eq(row.ant)].set_index('calendar_bin').reindex(d['calendar_bin'])
            valid=trace.qualified.eq(True).to_numpy(bool)
            members.append(dict(ant=row.ant,group=int(row.group),pc1=row.pc1,pc2=row.pc2,coverage=row.speed_detection,
                                oob=row.oob_agreement,map=np.round(d['coarse'],7).tolist(),
                                series=dict(pc1=trace.pc1.where(valid).round(5).tolist(),
                                            speed=trace.mean_speed_mm_s.where(trace.speed_coverage.ge(.4)).round(5).tolist(),
                                            nest=trace.colony_percent.where(trace.position_coverage.ge(.4)).round(3).tolist(),
                                            sleep=trace.sleep_percent.where(trace.sleep_coverage.ge(.4)).round(3).tolist(),
                                            coverage=(d['detected']/np.maximum(d['recorded'],1)).round(4).tolist())))
        payload[side]=members
    raw=json.dumps(payload,allow_nan=True).replace('NaN','null').replace('Infinity','null')
    template=Path(__file__).with_name('colony_behavioral_landscape_explorer.html').read_text()
    (output/'explorer.html').write_text(template.replace('__LANDSCAPE_DATA__',raw))
