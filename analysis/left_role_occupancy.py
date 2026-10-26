"""Occupancy and diagnostic atlas for every baseline left-colony behavior ant.

The baseline class table defines the cohort. Follow-up coverage never removes
an ant from this report; unavailable identities have explicit empty panels.
Existing common-reference role assignments are reused without refitting.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
import json
from pathlib import Path
import shutil
import hashlib

import numpy as np
import pandas as pd

FPS=24.
BIN_MM=.25


def region_mask(x,y,region):
    if region['shape']=='circle':
        return ((x-region['tracking_center_x_px'])**2+(y-region['tracking_center_y_px'])**2)<=region['radius_px']**2
    return ((x>=region['tracking_x_min_px'])&(x<=region['tracking_x_max_px'])
            &(y>=region['tracking_y_min_px'])&(y<=region['tracking_y_max_px']))


def normalized_occupancy(counts, observed):
    """Do not silently renormalize away detections outside the arena."""
    return counts*100./observed if observed else np.full(counts.shape,np.nan)


def scan_track(task):
    import pyarrow.dataset as ds
    path=Path(task['path']); arena=task['arena'];regions=task['regions']
    scale=task['scale']; x0=arena['tracking_x_min_px'];y0=arena['tracking_y_min_px']
    width=(arena['tracking_x_max_px']-x0)*scale
    height=(arena['tracking_y_max_px']-y0)*scale
    xe=np.arange(0,np.ceil(width/BIN_MM)+1)*BIN_MM
    ye=np.arange(0,np.ceil(height/BIN_MM)+1)*BIN_MM
    hist={(scope,source):np.zeros((len(ye)-1,len(xe)-1),np.int64)
          for scope in task['scopes'] for source in ['track','tag']}
    statistics={(scope,source):dict(observed_frames=0,in_arena_frames=0,outside_colony_frames=0,food_frames=0,water_frames=0)
                for scope,source in hist}
    nsec=int(np.ceil(task['duration']))
    seconds={prefix+name:np.zeros(nsec,np.float64 if name.endswith('sum') else np.int32)
             for prefix in ['track_','tag_'] for name in ['observed','outside','food','water','xsum','ysum']}
    scanner=ds.dataset(path,format='parquet').scanner(columns=['Frame','TrackX','TrackY','ArucoX','ArucoY'],
        filter=ds.field('Bodypoint')==0,batch_size=262144,use_threads=False)
    for batch in scanner.to_batches():
        f,x,y,ax,ay=[batch.column(i).to_numpy(zero_copy_only=False) for i in range(5)]
        for source,xx,yy in [('track',x,y),('tag',ax,ay)]:
            valid=np.isfinite(xx)&np.isfinite(yy)&(f>=0)&(f<task['duration']*FPS)
            frame=f[valid];xx=xx[valid];yy=yy[valid];sec=(frame/FPS).astype(int)
            masks={kind:np.zeros(len(xx),bool) for kind in ['colony','food','water']}
            for region in regions:
                if region['region_type'] in masks:
                    masks[region['region_type']] |= region_mask(xx,yy,region)
            inside_arena=region_mask(xx,yy,arena)
            seconds[source+'_observed']+=np.bincount(sec,minlength=nsec).astype(np.int32)
            for key,mask in [('outside',~masks['colony']),('food',masks['food']),('water',masks['water'])]:
                seconds[source+'_'+key]+=np.bincount(sec[mask],minlength=nsec).astype(np.int32)
            seconds[source+'_xsum']+=np.bincount(sec,weights=(xx-x0)*scale,minlength=nsec)
            seconds[source+'_ysum']+=np.bincount(sec,weights=(yy-y0)*scale,minlength=nsec)
            for scope,(low,high) in task['scopes'].items():
                selected=(frame>=low*FPS)&(frame<high*FPS);valid_grid=selected&inside_arena
                hist[scope,source]+=np.histogram2d((yy[valid_grid]-y0)*scale,(xx[valid_grid]-x0)*scale,
                                                 bins=(ye,xe))[0].astype(np.int64)
                counts=statistics[scope,source]
                counts['observed_frames']+=int(selected.sum())
                counts['in_arena_frames']+=int(valid_grid.sum())
                counts['outside_colony_frames']+=int((selected&~masks['colony']).sum())
                for kind in ['food','water']:counts[kind+'_frames']+=int((selected&masks[kind]).sum())
    output=Path(task['output']);output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/(task['ant'].replace(':','_')+'.npz'),x_edges=xe,y_edges=ye,
        **{scope+'_'+source:value for (scope,source),value in hist.items()},**seconds)
    rows=[]
    for (scope,source),counts in statistics.items():
        low,high=task['scopes'][scope];n=counts['observed_frames']
        assert hist[scope,source].sum()==counts['in_arena_frames']
        assert seconds[source+'_observed'].max(initial=0)<=FPS
        row=dict(ant=task['ant'],phase=task['phase'],scope=scope,source=source,
            available_seconds=high-low,frame_coverage=n/(FPS*(high-low)),observed_seconds_equivalent=n/FPS,**counts)
        for kind in ['outside_colony','food','water']:
            row[kind+'_fraction']=counts[kind+'_frames']/n if n else np.nan
        row['outside_arena_frames']=n-counts['in_arena_frames']
        rows.append(row)
    return rows


def extract(comparison,output,workers):
    from analysis.grid_occupancy_utils import load_panorama_regions
    manifest=json.loads((comparison/'comparison_manifest.json').read_text())
    baseline=pd.read_csv(comparison/'baseline_ant_classes.csv').query('side == "left"')
    output.mkdir(parents=True,exist_ok=True)
    baseline.to_csv(output/'baseline_left_cohort.csv',index=False)
    tasks=[];provenance=dict(comparison=str(comparison),cohort='all baseline left-colony ant-class rows; no follow-up exclusion',
        grid_size_mm=BIN_MM,normalization='100 * bin_count / all finite detected positions in window; out-of-arena detections kept in denominator',
        fps=FPS,annotations={},source_tracks=[],matched_windows=manifest['matched_windows'],light_schedule=manifest['light_schedule'])
    for phase in ['before','after']:
        block=Path(manifest[phase]);inv=pd.read_csv(block/'analysis_outputs/eigenposture/inventory.csv').set_index('ant')
        path=Path(manifest['annotations'][phase]['path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['annotations'][phase]['sha256']
        regions=load_panorama_regions(path).query('side == "left"')
        arena=regions[regions.region_type.eq('arena')].iloc[0].to_dict()
        regions.to_csv(output/(phase+'_regions.csv'),index=False)
        provenance['annotations'][phase]=manifest['annotations'][phase]
        scope={'full':(0,manifest['duration_seconds'][phase]),'matched':manifest['matched_windows'][phase]['elapsed_seconds']}
        for ant in baseline.ant:
            if ant not in inv.index:continue
            path=block/'stitched/per_track'/inv.loc[ant,'track_name']
            tasks.append(dict(ant=ant,phase=phase,path=str(path),arena=arena,regions=regions.to_dict('records'),
                scale=arena['mm_per_pixel'],duration=manifest['duration_seconds'][phase],scopes=scope,output=str(output/'maps'/phase)))
            provenance['source_tracks'].append(dict(phase=phase,ant=ant,path=str(path),size=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns))
    rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,part in enumerate(pool.map(scan_track,tasks),1):
            rows.extend(part);print('OCCUPANCY',i,'/',len(tasks),part[0]['phase'],part[0]['ant'],flush=True)
    pd.DataFrame(rows).to_csv(output/'occupancy_region_metrics.csv',index=False)
    provenance['code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output/'occupancy_manifest.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (output/'measurement_complete.ok').write_text('all available baseline left identities measured\n')


def draw_regions(ax,regions,labels=False):
    from matplotlib.patches import Rectangle,Circle
    arena=regions[regions.region_type.eq('arena')].iloc[0]
    scale=arena.mm_per_pixel;x0=arena.tracking_x_min_px;y0=arena.tracking_y_min_px
    colors={'colony':'#2ca25f','food':'#d34a30','water':'#238aca'}
    for r in regions.to_dict('records'):
        if r['region_type'] not in colors:continue
        color=colors[r['region_type']]
        if r['shape']=='rectangle':
            x=(r['tracking_x_min_px']-x0)*scale;y=(r['tracking_y_min_px']-y0)*scale
            patch=Rectangle((x,y),(r['tracking_x_max_px']-r['tracking_x_min_px'])*scale,
                            (r['tracking_y_max_px']-r['tracking_y_min_px'])*scale,fill=False,lw=1,ec=color)
        else:
            x=(r['tracking_center_x_px']-x0)*scale;y=(r['tracking_center_y_px']-y0)*scale
            patch=Circle((x,y),r['radius_px']*scale,fill=False,lw=1,ec=color)
        ax.add_patch(patch)
        if labels:ax.text(x,y,r['region_type'],fontsize=7,color=color,va='bottom')
    ax.set(xlim=(0,(arena.tracking_x_max_px-x0)*scale),ylim=((arena.tracking_y_max_px-y0)*scale,0))
    ax.set_aspect('equal')


def draw_map(ax,root,metrics,regions,ant,phase,scope='matched',source='track',labels=False):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    path=root/'maps'/phase/(ant.replace(':','_')+'.npz')
    if path.is_file():
        with np.load(path) as z:
            n=metrics.loc[(ant,phase,scope,source),'observed_frames']
            values=normalized_occupancy(z[scope+'_'+source],n)
            cmap=plt.get_cmap('magma_r').copy();cmap.set_bad('white')
            ax.pcolormesh(z['x_edges'],z['y_edges'],np.ma.masked_invalid(np.where(values>0,values,np.nan)),
                          norm=LogNorm(vmin=.0001,vmax=10),cmap=cmap,rasterized=True,shading='flat')
    else:
        ax.set_facecolor('#eee');ax.text(.5,.5,'No matching track\nID unobserved',ha='center',va='center',transform=ax.transAxes,fontsize=9)
    draw_regions(ax,regions,labels)
    ax.tick_params(labelsize=7);return ax


def coarse_series(counts,key,low,high,bin_seconds=1800):
    """Average one-second fractions without treating missing seconds as zero."""
    edges=np.r_[np.arange(int(low),int(high),bin_seconds),int(high)]
    observed=counts['track_observed']
    fraction=np.divide(counts['track_'+key],observed,out=np.zeros(len(observed)),where=observed>0)
    vals=[];coverage=[];tagcoverage=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        known=observed[lo:hi]>0
        vals.append(float(fraction[lo:hi][known].mean()) if known.any() else np.nan)
        coverage.append(float(known.mean()))
        tagcoverage.append(float((counts['tag_observed'][lo:hi]>0).mean()))
    return (edges[:-1]+edges[1:])/2-low,np.array(vals),np.array(coverage),np.array(tagcoverage)


def generate_report(comparison,root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import LogNorm
    from matplotlib.cm import ScalarMappable
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':170})
    manifest=json.loads((comparison/'comparison_manifest.json').read_text())
    cohort=pd.read_csv(root/'baseline_left_cohort.csv')
    all_pairs=pd.read_csv(comparison/'paired_ant_comparison.csv')
    rows=cohort[['ant','ant_class']].merge(all_pairs,on='ant',how='left',validate='one_to_one')
    assert len(rows)==len(cohort) and rows.ant.nunique()==len(cohort)
    rows=rows.sort_values(['ant_class','track_id'],ascending=[False,True])
    rows['followup_track_present']=rows.after_matched_usable_seconds.notna()
    rows['class_changed']=rows.after_matched_class.ne(rows.ant_class)&rows.after_matched_class.notna()
    rows['followup_usable_minutes']=rows.after_matched_usable_seconds/60
    rows['assignment_caution']=np.select([~rows.followup_track_present,rows.after_matched_behavior_coverage.lt(.05),~rows.paired_high_coverage],
        ['Unobserved ID','Very sparse usable behavior (<5%)','Lower coverage; retained'],default='Strong paired coverage')
    rows.to_csv(root/'all_18_left_ant_class_comparison.csv',index=False)
    metrics=pd.read_csv(root/'occupancy_region_metrics.csv').set_index(['ant','phase','scope','source'])
    regions={phase:pd.read_csv(root/(phase+'_regions.csv')) for phase in ['before','after']}
    profiles=pd.read_csv(comparison/'state_profiles.csv')
    columns=[c for c in profiles if c.startswith('S') and c[1:].isdigit()]
    figures=root/'figures';figures.mkdir(exist_ok=True)
    individual=root/'individual_ants';individual.mkdir(exist_ok=True)
    pal={'before':'#3f78b5','after':'#d66c39'}
    role=lambda value:'unobserved' if pd.isna(value) else ('C1 nurse-like' if value==1 else 'C0 forager-like')
    def save(fig,path,pdf=None):
        fig.savefig(path.with_suffix('.png'),bbox_inches='tight')
        fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight')
        if pdf:pdf.savefig(fig,bbox_inches='tight')
        plt.close(fig)
    def colorbar(fig,axes):
        sm=ScalarMappable(norm=LogNorm(vmin=.0001,vmax=10),cmap='magma_r')
        cb=fig.colorbar(sm,ax=axes,pad=.02,fraction=.015,shrink=.75,extend='max')
        cb.set_label('% of detected positions / 0.25 mm cell',fontsize=8)
    # Every baseline ant appears in each atlas, including the missing ID.
    for scope,filename in [('matched','occupancy_matched_12h'),('full','occupancy_full_recordings')]:
        with PdfPages(root/(filename+'.pdf')) as pdf:
            for page,start in enumerate(range(0,len(rows),6),1):
                fig,axes=plt.subplots(3,4,figsize=(14,16))
                for k,row in enumerate(rows.iloc[start:start+6].itertuples()):
                    for j,phase in enumerate(['before','after']):
                        ax=axes[k//2,(k%2)*2+j]
                        draw_map(ax,root,metrics,regions[phase],row.ant,phase,scope=scope,labels=(k==0))
                        cls=row.ant_class if phase=='before' else row.after_matched_class
                        key=(row.ant,phase,scope,'track')
                        coverage=metrics.loc[key,'frame_coverage']*100 if key in metrics.index else 0
                        ax.set_title(f'Ant {row.track_id:02d} | {"9/15" if phase=="before" else "9/16"}\n{role(cls)} | tracked frames {coverage:.1f}%',fontsize=9)
                        ax.set(xlabel='x (mm)',ylabel='y (mm)')
                fig.suptitle(f'All 18 baseline left-colony ants — occupancy before / after ({page}/3)',fontsize=15)
                hours='Matched clock window 22:54–10:54 on each day' if scope=='matched' else 'Full recordings: 27.8 h baseline / 12 h follow-up; unequal clock coverage'
                fig.text(.5,.018,hours+'\nGreen: colony; red: food; blue: water. Conditional occupancy; tracking gaps excluded. Every baseline ID retained.',ha='center',fontsize=9)
                fig.subplots_adjust(left=.06,right=.89,top=.92,bottom=.07,hspace=.35,wspace=.35)
                colorbar(fig,list(axes.flat));save(fig,figures/(filename+f'_page{page}'),pdf)
    start_clock=datetime.fromisoformat(manifest['matched_windows']['before']['start'])
    light=(start_clock.replace(hour=5,minute=30,second=0)+timedelta(days=1)-start_clock).total_seconds()/3600
    def clock_axis(ax):
        ax.axvspan(0,light,color='#606c82',alpha=.09,zorder=-1)
        ax.axvspan(light,12,color='#f1da68',alpha=.12,zorder=-1)
        ax.axvline(light,color='#8d8000',ls=':',lw=1)
        ax.set(xlim=(0,12),xticks=np.arange(0,13,2),xticklabels=[(start_clock+timedelta(hours=i)).strftime('%H:%M') for i in range(0,13,2)])
    with PdfPages(root/'all_18_ant_diagnostics.pdf') as pdf, PdfPages(root/'diagnostic_examples.pdf') as examples:
        for row in rows.itertuples():
            fig=plt.figure(figsize=(15,13));grid=fig.add_gridspec(3,4,height_ratios=[2.2,1,1],hspace=.6,wspace=.4)
            top=[fig.add_subplot(grid[0,i]) for i in range(4)]
            for ax,(phase,source) in zip(top,[('before','track'),('after','track'),('before','tag'),('after','tag')]):
                draw_map(ax,root,metrics,regions[phase],row.ant,phase,source=source,labels=True)
                key=(row.ant,phase,'matched',source)
                cov=metrics.loc[key,'frame_coverage']*100 if key in metrics.index else 0
                ax.set_title(f'{"9/15" if phase=="before" else "9/16"} — {"track" if source=="track" else "direct ArUco"}\nFrames observed: {cov:.1f}%',fontsize=10)
                ax.set(xlabel='x (mm)',ylabel='y (mm)')
            colorbar(fig,top)
            stateax=fig.add_subplot(grid[1,:2]);outax=fig.add_subplot(grid[1,2:])
            noteax=fig.add_subplot(grid[2,:2]);covax=fig.add_subplot(grid[2,2:])
            subset=profiles[profiles.ant.eq(row.ant)&profiles.variant.eq('standard')]
            colors=plt.get_cmap('tab10')(np.arange(len(columns)))
            for y,(phase,scope) in enumerate([('before','full'),('before','matched'),('after','matched')]):
                found=subset[subset.phase.eq(phase)&subset.scope.eq(scope)]
                if found.empty:
                    stateax.text(.02,y,'Unobserved',va='center');continue
                values=found.iloc[0][columns].to_numpy(float);left=0
                for n,value in enumerate(values):
                    stateax.barh(y,value,left=left,color=colors[n],label=columns[n] if y==0 else None);left+=value
            stateax.set(xlim=(0,1),yticks=[0,1,2],yticklabels=['9/15 full','9/15 matched','9/16 matched'],
                        xlabel='Fraction of classified behavior',title='Same 9/15 state dictionary on both days')
            stateax.invert_yaxis();stateax.legend(ncol=7,fontsize=7,loc='upper center',bbox_to_anchor=(.5,-.24))
            for phase in ['before','after']:
                path=root/'maps'/phase/(row.ant.replace(':','_')+'.npz')
                if not path.is_file():continue
                with np.load(path) as z:
                    low,high=manifest['matched_windows'][phase]['elapsed_seconds']
                    x,vals,cov,tag=coarse_series(z,'outside',low,high)
                outax.plot(x/3600,100*vals,c=pal[phase],marker='o',ms=3,label=phase)
                covax.plot(x/3600,100*cov,c=pal[phase],label=phase+' track')
                covax.plot(x/3600,100*tag,c=pal[phase],ls=':',label=phase+' direct tag')
            outax.set(ylabel='Outside / observed seconds (%)',ylim=(-2,102),title='Outside activity in 30-minute bins')
            covax.set(ylabel='Seconds with a detection (%)',ylim=(-2,102),title='Coverage over the same clock window')
            for ax in [outax,covax]:clock_axis(ax);ax.legend(fontsize=8,loc='upper right');ax.grid(axis='y',alpha=.2)
            noteax.axis('off')
            lines=[f'Published baseline class: {role(row.ant_class)}',
                f'Follow-up class (fixed reference): {role(row.after_matched_class)}',
                f'Follow-up class without contacts: {role(row.no_contact_after_matched_class)}',
                f'Usable follow-up behavior: {row.followup_usable_minutes:.1f} min / 720 min' if row.followup_track_present else 'No matching follow-up track',
                row.assignment_caution]
            if row.followup_track_present:
                lines += [f'Outside time: {row.before_outside_fraction*100:.2f}% → {row.after_outside_fraction*100:.2f}% (observed seconds)',
                          f'Completed trips: {row.before_n_completed_trips:.0f} → {row.after_n_completed_trips:.0f}',
                          f'Forager coordinate: {row.before_matched_forager_coordinate:.2f} → {row.after_matched_forager_coordinate:.2f} (boundary 0.5)',
                          f'After coordinate, 95% resampling interval: {row.after_forager_coordinate_low:.2f}–{row.after_forager_coordinate_high:.2f}']
            for phase in ['before','after']:
                key=(row.ant,phase,'matched','track')
                if key in metrics.index:
                    lines.append(f'{phase.capitalize()} food-region occupancy: {metrics.loc[key,"food_fraction"]*100:.2f}% of tracked frames')
            noteax.text(0,1,'\n'.join(lines),va='top',fontsize=10,linespacing=1.4)
            fig.suptitle(f'Left ant {row.track_id:02d}: {role(row.ant_class)} → {role(row.after_matched_class)}',fontsize=16)
            fig.text(.5,.015,'All maps share one log color scale. Food-region occupancy does not prove feeding. Clock shading: nominal dark → light at 05:30; controller log unverified.',ha='center',fontsize=9)
            fig.subplots_adjust(top=.92,bottom=.06)
            pdf.savefig(fig,bbox_inches='tight')
            if row.track_id in [0,39,41,60,68,83,93,95]:examples.savefig(fig,bbox_inches='tight')
            save(fig,individual/f'left_{row.track_id:03d}_diagnostics')
    summary=dict(baseline_ants=len(rows),matched_ids=int(rows.followup_track_present.sum()),
        unobserved_ids=rows.loc[~rows.followup_track_present,'ant'].tolist(),
        main_model_changed=rows.loc[rows.class_changed,'ant'].tolist(),
        unchanged_ids=rows.loc[rows.followup_track_present&~rows.class_changed,'ant'].tolist(),
        very_sparse_ids=rows.loc[rows.after_matched_behavior_coverage.lt(.05),'ant'].tolist(),
        note='No ant is excluded based on follow-up coverage. Class labels from extremely sparse follow-up remain provisional.')
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    text=['ALL BASELINE LEFT-COLONY ANTS: 9/15 to 9/16','',
        'The earlier headline used seven well-tracked baseline nurse-like ants. The baseline left cohort contains 18 ants: 12 nurse-like and 6 forager-like. This report retains all 18 regardless of follow-up coverage.',
        '17 IDs have follow-up tracks; ID 83 is unobserved. Main-model class changes: 41 and 60 (C1 to C0). The other 15 observed IDs keep their main-model class.',
        'Ant 60 has concordant behavioral and spatial evidence. Ant 41 is near the class boundary and stays nurse-like when contact features are omitted. Ant 68 increases outside time but remains nurse-like.',
        'IDs 39 and 95 have only 7.4 and 2.2 minutes of usable follow-up behavior respectively. Their main-model forager labels and contact-free nurse labels disagree; neither should be treated as a reliable role transition.',
        '', 'FILES',
        'occupancy_matched_12h.pdf: all 18 ants, same 22:54:36–10:54:36 clock window each day, three pages.',
        'occupancy_full_recordings.pdf: all 18 ants using full 27.8-hour baseline and 12-hour follow-up (clock coverage differs).',
        'all_18_ant_diagnostics.pdf: one page per baseline ant; track and direct-tag maps, state proportions, outside timeline, coverage, and role uncertainty.',
        'diagnostic_examples.pdf: selected cases 0, 39, 41, 60, 68, 83, 93, 95.',
        'all_18_left_ant_class_comparison.csv: complete ID-linked class table, coverage and uncertainty.',
        'occupancy_region_metrics.csv: frame coverage, colony/food/water occupancy and detections outside arena.',
        '', 'METHODS',
        'Frozen 9/15 behavior dictionary and ant-class centroids are reused from the prior comparison; no independent follow-up labels are compared. Missing follow-up is unknown, never inactivity or confirmed removal.',
        'Occupancy uses 0.25-mm cells in the annotated arena, normalized by all finite observed positions within each interval. Missing frames are excluded; detections outside arena remain in the denominator and are counted in the metrics CSV. All panels use the same log color scale. Raw positions are not interpolated for occupancy.',
        'Direct ArUco panels require the tag to be detected at that frame, corroborating track-based occupancy. Tracking and tag visibility can still vary by location; sparse coverage remains a limitation.',
        'Colony, food and water regions use the existing annotations with the configured earlier-date fallback; their geometry is identical for these two blocks. Region occupancy does not establish brood care or feeding.',
        'Nominal light schedule from the existing analysis: on 05:30, off 19:30. Actual controller transitions were not supplied. Both colonies were manipulated; no untreated control. The behavior dictionary failed some automated stability criteria, so role labels remain exploratory.', '']
    (root/'README.txt').write_text('\n'.join(text))
    (root/'report_complete.ok').write_text('all baseline left ants included\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['extract','report','all']);p.add_argument('--comparison',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=6)
    a=p.parse_args()
    if a.mode in ['extract','all']:extract(a.comparison,a.output,a.workers)
    if a.mode in ['report','all']:generate_report(a.comparison,a.output)
