"""Revisit every left-colony ant in the 9/15 eigenposture class analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _occupancy(track_path, low, high, edges_x, edges_y, bounds, mm_per_px):
    import pyarrow.dataset as ds
    count = np.zeros((len(edges_y)-1, len(edges_x)-1), dtype=np.uint32)
    observed = inside = 0
    dataset = ds.dataset(track_path, format='parquet')
    scanner = dataset.scanner(columns=['Frame','Bodypoint','TrackX','TrackY'],
        filter=(ds.field('Bodypoint')==0)&(ds.field('Frame')>=int(low))&(ds.field('Frame')<int(high)),
        batch_size=262144, use_threads=False)
    xmin,xmax,ymin,ymax = [bounds[x] for x in ['x_min_px','x_max_px','y_min_px','y_max_px']]
    for batch in scanner.to_batches():
        x=batch.column(2).to_numpy(zero_copy_only=False)
        y=batch.column(3).to_numpy(zero_copy_only=False)
        good=np.isfinite(x)&np.isfinite(y)
        observed += int(good.sum())
        in_bounds=good&(x>=xmin)&(x<=xmax)&(y>=ymin)&(y<=ymax)
        inside += int(in_bounds.sum())
        if in_bounds.any():
            xx=(x[in_bounds]-xmin)*mm_per_px
            yy=(y[in_bounds]-ymin)*mm_per_px
            count += np.histogram2d(yy,xx,bins=(edges_y,edges_x))[0].astype(np.uint32)
    return count,observed,inside


def run(before:Path,after:Path,source:Path,output:Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm
    from matplotlib.patches import Rectangle
    from matplotlib.backends.backend_pdf import PdfPages
    from tracking.colony.combine_blocks import recording_start

    output.mkdir(parents=True,exist_ok=True)
    bp=before/'analysis_outputs/eigenposture';ap=after/'analysis_outputs/eigenposture'
    bi=pd.read_csv(bp/'inventory.csv');ai=pd.read_csv(ap/'inventory.csv')
    classes=pd.read_csv(bp/'roles_left/ant_classes.csv').sort_values('track_id')
    assert len(classes)==18 and classes.side.eq('left').all()
    summary=pd.read_csv(source/'paired_ant_comparison.csv')
    paired=summary[summary.side.eq('left')].copy()
    dates={'20260915':bi,'20260916':ai}
    counts={day:{side:int(len(tab[tab.side.eq(side)])) for side in ['left','right']} for day,tab in dates.items()}
    ids={day:{side:set(tab.loc[tab.side.eq(side),'track_id'].astype(int)) for side in ['left','right']} for day,tab in dates.items()}
    ident=dict(tracked_ids_by_day_and_colony=counts,
        same_tag_matches_by_colony={s:len(ids['20260915'][s]&ids['20260916'][s]) for s in ['left','right']},
        only_20260915_by_colony={s:sorted(ids['20260915'][s]-ids['20260916'][s]) for s in ['left','right']},
        only_20260916_by_colony={s:sorted(ids['20260916'][s]-ids['20260915'][s]) for s in ['left','right']},
        eigenposture_selected={d:{s:int(tab[(tab.side==s)&tab.selected].shape[0]) for s in ['left','right']} for d,tab in dates.items()},
        left_baseline_class_cohort_ids=classes.track_id.astype(int).tolist(),
        left_baseline_class_cohort_size=len(classes))
    (output/'identity_counts.json').write_text(json.dumps(ident,indent=2)+'\n')

    rect_path=Path(json.loads((bp/'pipeline_run.json').read_text())['source_block'])
    from camera_cal.region_paths import panorama_regions_path
    from analysis import grid_occupancy_utils as go
    from analysis.trip_phenotyping_utils import colony_rectangles_from_regions
    from analysis.arena_grid_utils import arena_bounds_from_regions
    anno=panorama_regions_path(rect_path,search_earlier_dates=True)
    regions=go.load_panorama_regions(anno,x_split_px=None)
    rect=colony_rectangles_from_regions(regions)['left']
    arena=arena_bounds_from_regions(regions)['left']
    grid=before/'stitched/grid_occupancy_histograms_arena_0p25mm/per_track'
    exemplar=json.loads(next(grid.glob('*left/grid_occupancy_metadata.json')).read_text())
    ex=grid/Path(exemplar['track_name']).stem
    edges_x=np.load(ex/'grid_x_edges_mm.npy');edges_y=np.load(ex/'grid_y_edges_mm.npy')
    scale=float(exemplar['mm_per_px'])
    assert len(edges_x)==169 and len(edges_y)==245
    starts={
        'before':recording_start(before,list((before/'stitched/per_track').glob('*left.parquet')))[0],
        'after':recording_start(after,list((after/'stitched/per_track').glob('*left.parquet')))[0]}
    low_before,high_before=25309*24,68509*24
    low_after,high_after=0,43200*24
    inv_b=bi.set_index('ant');inv_a=ai.set_index('ant')
    paired=paired.set_index('ant')
    # Reject an earlier draft cache which counted only positions inside the colony.
    cached_rows=(pd.read_csv(output/'all_20260915_left_class_ants.csv').set_index('ant')
                 if (output/'all_20260915_left_class_ants.csv').is_file()
                 and 'baseline_arena_occupancy_frames' in pd.read_csv(output/'all_20260915_left_class_ants.csv',nrows=0).columns
                 else None)
    result=[];maps={}
    for row in classes.itertuples():
        ant=row.ant; tid=int(row.track_id)
        bname=inv_b.loc[ant,'track_name']
        path_b=before/'stitched/per_track'/bname
        assert path_b.is_file(),path_b
        after_ant=f'left:{tid:03d}'
        has_after=after_ant in inv_a.index
        ca=na=ia=None
        cache=output/f'arena_occupancy_{tid:03d}.npz'
        if cache.is_file() and cached_rows is not None and ant in cached_rows.index:
            with np.load(cache) as stored:
                cb=stored['before'];ca=stored['after'] if has_after else None
            old=cached_rows.loc[ant]
            nb=int(old.baseline_usable_frames_matched);ib=int(old.baseline_in_arena_frames_matched)
            if has_after and pd.notna(old.followup_usable_frames_matched):
                na=int(old.followup_usable_frames_matched);ia=int(old.followup_in_arena_frames_matched)
            print(f'CACHED OCCUPANCY {ant}',flush=True)
        else:
            cb,nb,ib=_occupancy(path_b,low_before,high_before,edges_x,edges_y,arena,scale)
            if has_after:
                aname=inv_a.loc[after_ant,'track_name']; path_a=after/'stitched/per_track'/aname
                assert path_a.is_file(),path_a
                ca,na,ia=_occupancy(path_a,low_after,high_after,edges_x,edges_y,arena,scale)
            np.savez_compressed(cache,before=cb,after=ca if ca is not None else np.zeros_like(cb),
                x_edges_mm=edges_x,y_edges_mm=edges_y)
        maps[(tid,'before')]=cb
        if ca is not None:maps[(tid,'after')]=ca
        comparison=paired.loc[ant] if ant in paired.index else None
        result.append(dict(ant=ant,side='left',track_id=tid,
            baseline_class=int(row.ant_class),baseline_class_name=('forager-like' if row.ant_class==0 else 'nurse-like'),
            baseline_selected=bool(row.selected),baseline_detection_fraction=float(row.detection_fraction),
            followup_track_found=bool(has_after),followup_detection_fraction=float(inv_a.loc[after_ant,'detection_fraction']) if has_after else np.nan,
            baseline_usable_frames_matched=nb,baseline_arena_occupancy_frames=ib,
            baseline_in_arena_fraction=ib/nb if nb else np.nan,
            followup_usable_frames_matched=na,followup_arena_occupancy_frames=ia,
            followup_in_arena_fraction=ia/na if na else np.nan,
            before_role=int(comparison['before_matched_class']) if comparison is not None and pd.notna(comparison['before_matched_class']) else np.nan,
            after_role=int(comparison['after_matched_class']) if comparison is not None and pd.notna(comparison['after_matched_class']) else np.nan,
            no_contact_after_role=int(comparison['no_contact_after_matched_class']) if comparison is not None and pd.notna(comparison['no_contact_after_matched_class']) else np.nan,
            primary_paired_coverage=bool(comparison['paired_high_coverage']) if comparison is not None else False,
            before_outside_fraction=float(comparison['before_outside_fraction']) if comparison is not None else np.nan,
            after_outside_fraction=float(comparison['after_outside_fraction']) if comparison is not None else np.nan,
            before_completed_trips=float(comparison['before_n_completed_trips']) if comparison is not None else np.nan,
            after_completed_trips=float(comparison['after_n_completed_trips']) if comparison is not None else np.nan,
            direct_aruco_before_fraction=float(comparison['before_aruco_outside_fraction']) if comparison is not None else np.nan,
            direct_aruco_after_fraction=float(comparison['after_aruco_outside_fraction']) if comparison is not None else np.nan))
        print(f'OCCUPANCY {ant} followup={has_after} before_frames={nb} after_frames={na}',flush=True)
    table=pd.DataFrame(result)
    table.to_csv(output/'all_20260915_left_class_ants.csv',index=False)

    arrays=[(arr/arr.sum())[arr>0] for arr in maps.values() if arr.sum()>0]
    vmax=float(np.quantile(np.concatenate(arrays),.995)) if arrays else 1
    vmax=max(vmax,1e-10)
    # Put the changed ants and the ambiguous label-only change first. Every
    # member of the 9/15 left class analysis appears exactly once in the montages.
    order=[tid for tid in [60,41,68,0,12,14,25,26,37,39,47,51,64,65,66,83,93,95] if tid in set(classes.track_id)]
    (output/'montage_order.json').write_text(json.dumps(order,indent=2)+'\n')
    dest=output/'occupancy_plots';dest.mkdir(exist_ok=True)
    pdf=PdfPages(output/'ALL_18_LEFT_ANTS_OCCUPANCY.pdf')
    for page in range(3):
        selected=order[page*6:(page+1)*6]
        fig,axes=plt.subplots(6,2,figsize=(12,17),constrained_layout=True)
        fig.suptitle(f'9/15 left-colony eigenposture ants — matched 12 h occupancy (page {page+1}/3)',fontsize=14)
        for r,tid in enumerate(selected):
            record=table[table.track_id.eq(tid)].iloc[0]
            for c,phase in enumerate(['before','after']):
                ax=axes[r,c]
                arr=maps.get((tid,phase))
                if arr is None:
                    ax.set_facecolor('#dedede');ax.text(.5,.5,'No 9/16 track with this ID',ha='center',va='center',transform=ax.transAxes)
                else:
                    denom=int(arr.sum());prob=arr/denom if denom else arr.astype(float)
                    ax.pcolormesh(edges_x,edges_y,prob,cmap='magma',norm=PowerNorm(.5,vmin=0,vmax=vmax),shading='auto',rasterized=True)
                    ax.add_patch(Rectangle((0,0),(rect['x_max_px']-rect['x_min_px'])*scale,
                        (rect['y_max_px']-rect['y_min_px'])*scale,fill=False,edgecolor='#52e8ff',lw=.8))
                cov=record['baseline_in_arena_fraction' if phase=='before' else 'followup_in_arena_fraction']
                out=record[f'{phase}_outside_fraction']
                classid=record[f'{phase}_role']
                clab=f'C{int(classid)}' if pd.notna(classid) else 'unassigned'
                cls=f"; {record.baseline_class_name}" if phase=='before' else ''
                subtitle=('9/15' if phase=='before' else '9/16')+f' — {clab}{cls}'
                if pd.notna(cov):subtitle+=f' | arena {cov:.0%}'
                if pd.notna(out):subtitle+=f' | outside {out:.1%}'
                ax.set_title(f'ID {tid:02d} {subtitle}',fontsize=8,pad=2)
                ax.set(xlim=(edges_x[0],edges_x[-1]),ylim=(edges_y[0],edges_y[-1]),aspect='equal')
                if r==5:ax.set_xlabel('x (mm)')
                else:ax.set_xticklabels([])
                if c==0:ax.set_ylabel('y (mm)')
                else:ax.set_yticklabels([])
        sm=plt.cm.ScalarMappable(norm=PowerNorm(.5,0,vmax),cmap='magma')
        fig.colorbar(sm,ax=axes,label='Fraction of in-arena detections per 0.25 mm cell (square-root scale)',shrink=.7)
        fig.savefig(dest/f'left_ants_occupancy_page_{page+1}.png',dpi=180,bbox_inches='tight')
        fig.savefig(dest/f'left_ants_occupancy_page_{page+1}.svg',bbox_inches='tight')
        pdf.savefig(fig,bbox_inches='tight')
        plt.close(fig)
    pdf.close()
    (output/'occupancy_manifest.json').write_text(json.dumps(dict(
        matched_clock=('22:54:36–10:54:36 local colony clock'),bin_size_mm=.25,
        normalization='Within each map, detections inside the annotated arena are normalized to sum to 1; out-of-arena detections are excluded and arena-coverage fractions are reported separately.',
        colony_rectangle_px=rect,grid_x_edges_mm=edges_x.tolist(),grid_y_edges_mm=edges_y.tolist(),
        light_on='05:30 nominal from existing colony report settings (controller log unavailable)',
        baseline_start=starts['before'].isoformat(),followup_start=starts['after'].isoformat(),
        page_order=order,model='9/15 left classes transferred using existing fixed state/reference model; see main REPORT.md'),indent=2)+'\n')
    lines=['# All 9/15 left-colony eigenposture ants in 9/16', '',
        f"The 9/15/9/16 tracking inventories contain {counts['20260915']['left']} / {counts['20260916']['left']} left-colony IDs "
        f"and {counts['20260915']['right']} / {counts['20260916']['right']} right-colony IDs. "
        'Each side has 88 IDs in common. Eleven IDs per side appear in the 9/15 inventory and not in 9/16; '
        'the tracking files alone cannot tell whether those ants were removed or lost by tracking. No new tag IDs appear in 9/16.', '',
        'The drop from 99 tracked left-colony IDs to 18 ants in the 9/15 left eigenposture classes comes from the analysis '
        'coverage rule: `selected=true` requires a detected-frame fraction above 60%. The full inventory retains tracks below '
        'that threshold. The 9/15 right colony had 17 class-analysis ants; 9/16 had 24 left and 18 right class-analysis '
        'ants out of 88 tracked IDs per side. See `identity_counts.json` for the complete ID overlap and missing-ID lists.', '',
        'The earlier strict primary paired analysis required >60% track detection on both days, ≥60% observed position seconds, '
        'and ≥20% usable behavior seconds on both days. That stricter set includes nine left and twelve right ants. '
        'The class-transfer table below includes all 17 left-colony baseline ants found again in 9/16, including matches below that threshold.', '',
        'The 9/15 left eigenposture class analysis used 18 ants. Seventeen have a same-side, same-tag track in 9/16; '
        'left ID 83 has no 9/16 track. The table below includes all 18 baseline ants, including weaker-coverage follow-ups. '
        'Role C0 is forager-like and C1 nurse-like in the 9/15 left reference.', '',
        '| ID | 9/15 role | 9/16 track | Role 9/15 → 9/16 | Outside 9/15 → 9/16 | Trips | Primary coverage pair |',
        '|---:|---|---|---|---:|---:|---|']
    for row in table.itertuples():
        roles=f'C{int(row.before_role)} → C{int(row.after_role)}' if pd.notna(row.before_role) and pd.notna(row.after_role) else 'unassigned'
        outside=f'{row.before_outside_fraction:.1%} → {row.after_outside_fraction:.1%}' if pd.notna(row.after_outside_fraction) else '—'
        trips=f'{row.before_completed_trips:.0f} → {row.after_completed_trips:.0f}' if pd.notna(row.after_completed_trips) else '—'
        lines.append(f'| {row.track_id:02d} | {row.baseline_class_name} (C{row.baseline_class}) | {"yes" if row.followup_track_found else "no"} | {roles} | {outside} | {trips} | {"yes" if row.primary_paired_coverage else "no"} |')
    lines += ['', '## Reading the occupancy plots', '',
        'The three plots include all 18 ants, six per page. Both days use identical 0.25 mm bins, arena bounds, and a matched 12-hour clock interval. '
        'Each panel normalizes its in-arena detections to a spatial probability map; the cyan rectangle is the annotated colony area. '
        'Arena coverage and outside time are shown in each title. Gray means that ID has no follow-up track. The cyan rectangle and occupancy map '
        'come from tracked bodypoint positions; ArUco outside fractions in the table provide the independent tag-detection check.', '',
        'The 9/16 analysis tracks only 12 hours, compared with 27.8 hours on 9/15. The plots therefore use only the same 22:54–10:54 clock period on both days. '
        'Light onset at 05:30 is marked nominally; no controller log was available. Use the class uncertainty and model-stability limits in the main report when interpreting role changes.', '']
    (output/'ALL_18_LEFT_ANTS.md').write_text('\n'.join(lines))
    print(json.dumps(ident,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['before','after','source','output']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();run(args.before,args.after,args.source,args.output)
