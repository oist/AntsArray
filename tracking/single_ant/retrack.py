"""Rerun frozen single-cup detections and compare against existing tracking."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

from tracking.single_ant import full_cup_tracking as tracking
from tracking.single_ant.manual_cup_tracking import write_json
from tracking.single_ant.run import freeze_code


def initialize(args):
    old=tracking.manifest(args.source)
    baseline=tracking.root(old)
    if not (baseline/'FULL_RUN_COMPLETE.json').is_file():raise ValueError('Source tracking is incomplete')
    if not (baseline/'mapped').is_dir():raise ValueError('Supply the original flash manifest with cached mapped detections')
    if args.destination.exists():raise FileExistsError(args.destination)
    tracking_only=getattr(args,'tracking_only',False)
    reference=args.reference.resolve() if args.reference else None
    if not tracking_only and (reference is None or not (reference/'models/dictionary.npz').is_file()):
        raise ValueError('A frozen state dictionary is required for the paired state comparison')
    source_block=Path(old['source_block']);label=source_block.parent.name+'_'+source_block.name
    run,code=freeze_code(SimpleNamespace(work_root=args.work_root),'retrack_'+label)
    output=run/source_block.name;output.mkdir()
    for name in ['cup_identities.csv','cup_order.csv','cup_name_mapping.csv','cup_order_source.json',
                 'IDENTITIES_COMPLETE.json','manual_regions.csv','panorama_metadata.json','panorama.png',
                 'detections_manifest.json','full_recording_id_counts.csv']:
        if (baseline/name).is_file():shutil.copy2(baseline/name,output/name)
    (output/'mapped').symlink_to(baseline/'mapped',target_is_directory=True)
    m=dict(old);m.update(output=str(output),title=label+' · pose association fixes',
        native_engine_sha256=hashlib.sha256((code/'tracking/core/tracking_utils.py').read_bytes()).hexdigest(),
        pose_candidate_sha256=hashlib.sha256((code/'tracking/core/pose_candidates.py').read_bytes()).hexdigest(),
        baseline_manifest=str(args.source.resolve()),baseline_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        frozen_mapped_input=str(baseline/'mapped'),reference_states=str(reference) if reference and not tracking_only else None,
        identity_policy='Frozen original cup/resident registry for paired comparison',
        changes=['Geometrically redundant views count as one bridge hypothesis',
                 'Complete same-camera duplicate preferred to partial alias']+
                ([] if tracking_only else ['Locally valid single-ant state prediction independent of discovery cohort']),
        aruco_support_policy='Unchanged: decoded tags without SLEAP support are still rejected')
    m['recording_segments']=[dict(block=label,source_block=m['source_block'],
        start_datetime=m['start_datetime'],frame_offset=0,num_frames=m['n_frames'],gap_before_frames=0)]
    write_json(output/'manifest.json',m)
    # Read source identity/history only. Recording-specific speed columns are
    # refreshed after the rerun; they are not copied as new measurements.
    meta=reference.parent.parent/'analysis/speed_by_metadata/per_ant_summary.csv' if reference else None
    if meta:
        t=pd.read_csv(meta)
        directory=output/'analysis/speed_by_metadata';directory.mkdir(parents=True)
        t.to_csv(directory/'metadata_join.csv',index=False)
    if not tracking_only:
        from tracking.single_ant.state_prediction import initialize_reference
        initialize_reference(output/'analysis/eigenposture',reference,m)
        initialize_reference(output/'analysis/old_tracking_local_prediction',reference,m)
        inventory=pd.read_csv(reference/'inventory_all.csv')
        inventory['n_frames']=m['n_frames'];inventory['prediction_eligible']=False
        inventory.to_csv(output/'analysis/eigenposture/inventory_all.csv',index=False)
    plan=dict(run=str(run),code=str(code),output=str(output),destination=str(args.destination),
        source=str(args.source.resolve()),reference=str(reference) if reference else None,
        metadata=str(meta) if meta else None,python=sys.executable,tracking_only=tracking_only,
        chunks=len(m['chunks']),cups=len(m['cups']),exclude_nodes=args.exclude_nodes)
    write_json(run/'plan.json',plan)
    print(json.dumps(plan,indent=2),flush=True)
    return run/'plan.json'


def submit(path):
    plan=tracking.manifest(path);run=Path(plan['run']);(run/'logs').mkdir(exist_ok=True);(run/'jobs').mkdir(exist_ok=True)
    jobs=tracking.manifest(run/'jobs.json') if (run/'jobs.json').exists() else {}
    def job(name,stage,count=None,deps=(),mem='16G',cpus=2,wall='02:00:00'):
        if name in jobs:return jobs[name]
        command=[plan['python'],'-u','-m','tracking.single_ant.retrack',stage,'--plan',str(path)]
        import shlex
        cmd=shlex.join(command)+(' --index "$SLURM_ARRAY_TASK_ID"' if count else '')
        script=run/'jobs'/f'{name}.sbatch'
        script.write_text(f'''#!/bin/bash
#SBATCH -J cupfix_{name}
#SBATCH -p compute
#SBATCH -c {cpus}
#SBATCH --mem={mem}
#SBATCH -t {wall}
#SBATCH --exclude={plan['exclude_nodes']}
#SBATCH -o {run}/logs/{name}_%A_%a.out
#SBATCH -e {run}/logs/{name}_%A_%a.err
set -euo pipefail
export PYTHONNOUSERSITE=1 MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS={cpus}
cd {shlex.quote(plan['code'])}
{cmd}
''')
        args=['sbatch','--parsable']
        if count:args+=['--array='+f'0-{count-1}']
        if deps:args+=['--dependency=afterok:'+':'.join(deps)]
        jobs[name]=subprocess.check_output(args+[str(script)],text=True).strip().split(';')[0]
        write_json(run/'jobs.json',jobs);return jobs[name]
    track=job('track','track',plan['chunks']*plan['cups'])
    stitch=job('stitch','stitch',plan['cups'],[track])
    diagnose=job('diagnose','diagnose',plan['cups'],[stitch],mem='32G')
    dependencies=[diagnose]
    if not plan.get('tracking_only'):
        dependencies.append(job('states','states',plan['cups'],[stitch],mem='32G'))
    job('report','report',deps=dependencies,mem='32G',cpus=4)
    print(json.dumps(jobs,indent=2),flush=True)


def state_worker(plan,index):
    from tracking.single_ant.state_prediction import predict_track,identity_ok
    output=Path(plan['output']);m=tracking.manifest(output/'manifest.json');old=tracking.manifest(plan['source'])
    cid=int(m['cups'][index]['cup_id']);table=pd.read_csv(output/'cup_identities.csv')
    identity=table.loc[table.CupID.eq(cid)]
    result=dict(cup=cid,track_id=cid,identity_ok=identity_ok(output,cid),discovery_selected=False)
    marker=output/'state_comparison'/f'{cid:04d}.json'
    if marker.exists():return
    if not len(identity) or not bool(identity.Eligible.iloc[0]) or not result['identity_ok']:
        result.update(skipped=True,reason='unassigned_or_identity_excluded');write_json(marker,result);return
    reference=Path(plan['reference']);inventory=pd.read_csv(reference/'inventory_all.csv')
    row=inventory.loc[inventory.track_id.eq(cid)].iloc[0];result['discovery_selected']=bool(row.selected)
    name=f'TrackID_{cid:04d}_all_single.parquet'
    # Ants omitted from discovery are measured on both old and new tracks. This
    # separates recovering local prediction from changing the association code.
    scale=None
    if not row.selected:
        old_result=predict_track(tracking.root(old)/'per_track'/name,output/'analysis/old_tracking_local_prediction',
            reference,m,cid,interpolation_root=output/'analysis/old_tracking_interpolated/per_track')
        result['old_local']=old_result;scale=old_result.get('body_scale_mm')
    result['new']=predict_track(output/'per_track'/name,output/'analysis/eigenposture',reference,m,cid,
        interpolation_root=output/'analysis/posture_interpolated/per_track',scale_mm=scale)
    write_json(marker,result)
    print(json.dumps(result),flush=True)


def compare(plan):
    from tracking.single_ant.full_cup_report import aggregate_report
    output=Path(plan['output']);m=tracking.manifest(output/'manifest.json');old=tracking.manifest(plan['source'])
    aggregate_report(m)
    before=pd.read_csv(tracking.root(old)/'tracking_summary.csv').set_index('CupID')
    after=pd.read_csv(output/'tracking_summary.csv').set_index('CupID')
    metadata=pd.read_csv(plan['metadata']);reference=Path(plan['reference'])
    n=int(m['n_frames']);rows=[]
    for cup in metadata.itertuples():
        cid=int(cup.CupID);record=dict(cup=cid,identity_status=cup.IdentityStatus,colony=cup.Colony,role=cup.Role)
        if cid not in after.index:rows.append(record);continue
        for prefix,table in [('old',before),('new',after)]:
            for key in ['PositionFraction','Full10NodePoseFraction','PoseFraction','ValidSpeedFraction','ArucoOnlyFraction','SleapBridgeFraction','MaxConsecutiveStepMM']:
                record[prefix+'_'+key]=float(table.loc[cid,key])
        if cup.IdentityStatus=='matched':
            baseline=np.load(reference.parent/'sleep_s01_exclusion/per_track'/f'TrackID_{cid:04d}_all_single/labels.npz')['observed'][:n]
            record['old_state_fraction']=float((baseline>=0).mean())
            worker=tracking.manifest(output/'state_comparison'/f'{cid:04d}.json')
            new=worker.get('new',{});record['new_state_fraction']=new.get('assigned_frames',0)/n
            record['old_tracking_local_state_fraction']=worker.get('old_local',{}).get('assigned_frames',int((baseline>=0).sum()))/n
            record['discovery_selected']=worker.get('discovery_selected',False)
            record['new_sleep_fraction']=new.get('sleep_frames',0)/n
        rows.append(record)
    table=pd.DataFrame(rows);comparison=output/'comparison';comparison.mkdir(exist_ok=True)
    table.to_csv(comparison/'per_ant_comparison.csv',index=False)
    matched=table.loc[table.identity_status.eq('matched')]
    summary=dict(n_matched=len(matched),recorded_hours=n/m['fps']/3600,reference_dictionary=str(reference/'models/dictionary.npz'),
        means={col:float(matched[col].mean()) for col in matched if (col.startswith(('old_','new_')) and pd.api.types.is_numeric_dtype(matched[col]))},
        changes=m['changes'],aruco_support_policy=m['aruco_support_policy'],
        state_comparison='Original observed states versus fixed-dictionary locally valid full-posture predictions; motion-only exclusions not mixed into this comparison')
    write_json(comparison/'summary.json',summary)
    # Save viewer-compatible original full-posture vectors for this separate run.
    destination=output/'analysis/sleep_s01_exclusion';destination.mkdir(exist_ok=True)
    definition=dict(mode='fixed_dictionary_full_posture_only',n_frames=n,fps=m['fps'],sleep_states=[0,1],thresholds_mm_s={},
        reference_dictionary_sha256=hashlib.sha256((reference/'models/dictionary.npz').read_bytes()).hexdigest(),
        motion_exclusions=False,description=summary['state_comparison'])
    write_json(destination/'definition.json',definition);definition_hash=hashlib.sha256((destination/'definition.json').read_bytes()).hexdigest()
    inv=pd.read_csv(output/'analysis/eigenposture/inventory_all.csv')
    reference_inventory=pd.read_csv(reference/'inventory_all.csv').set_index('track_id')
    for key in ['n_frames','detected_frames','detection_fraction','source_path']:
        inv['reference_discovery_'+key]=inv.track_id.map(reference_inventory[key])
    for record in rows:
        cid=record['cup'];stem=f'TrackID_{cid:04d}_all_single'
        target=destination/'per_track'/stem;target.mkdir(parents=True,exist_ok=True)
        source=output/'analysis/eigenposture/predictions/per_track'/stem/'labels.npz'
        if source.exists():shutil.copy2(source,target/'labels.npz')
        else:
            unknown=np.full(n,-1,np.int8);np.savez_compressed(target/'labels.npz',observed=unknown,category=unknown,evidence=np.zeros(n,np.uint8))
        write_json(target/'metadata.json',dict(definition_sha256=definition_hash,track_id=cid,identity_ok=record['identity_status']=='matched'))
        inv.loc[inv.track_id.eq(cid),'prediction_eligible']=record.get('new_state_fraction',0)>0
        if cid in after.index:
            import pyarrow.parquet as pq
            raw=output/'per_track'/(stem+'.parquet');stat=raw.stat()
            interpolated=output/'analysis/posture_interpolated/per_track'/stem/raw.name
            values=dict(n_frames=n,detected_frames=int(after.loc[cid,'ObservedFrames']),
                detection_fraction=float(after.loc[cid,'PositionFraction']),original_source_path=str(raw),
                source_path=str(interpolated if interpolated.exists() else raw),
                source_size=stat.st_size,source_mtime_ns=stat.st_mtime_ns,
                input_rows=pq.ParquetFile(raw).metadata.num_rows)
            for key,value in values.items():inv.loc[inv.track_id.eq(cid),key]=value
    inv.to_csv(output/'analysis/eigenposture/inventory_all.csv',index=False)
    inv.to_csv(output/'analysis/eigenposture/inventory.csv',index=False)
    write_json(destination/'COMPLETE.json',dict(fixed_dictionary=True,matched_ants=len(matched)))
    # The identity table comes from a longer recording. Replace every speed
    # measurement with values from this block before using it in the viewer.
    speed_columns=['ValidSpeedFraction','MeanSpeed','MedianSpeed','P95Speed','First6hMean',
                   'First6hCoverage','RawMeanSpeed','RawAbove20Fraction','ValidSpeedFrames']
    metadata=metadata.drop(columns=speed_columns,errors='ignore')
    speed_rows=[]
    for cid in after.index:
        folder=output/'analysis/speed_vectors/per_track'/f'TrackID_{int(cid):04d}_all_single'
        speed=np.load(folder/'speed_mm_s.npy');raw=np.load(folder/'speed_unclipped_mm_s.npy')
        first=speed[:int(6*3600*m['fps'])]
        valid=speed[np.isfinite(speed)];raw_valid=raw[np.isfinite(raw)]
        speed_rows.append(dict(CupID=int(cid),ValidSpeedFraction=float(np.isfinite(speed).mean()),
            MeanSpeed=float(valid.mean()) if len(valid) else np.nan,
            MedianSpeed=float(np.median(valid)) if len(valid) else np.nan,
            P95Speed=float(np.quantile(valid,.95)) if len(valid) else np.nan,
            First6hMean=float(np.nanmean(first)) if np.isfinite(first).any() else np.nan,
            First6hCoverage=float(np.isfinite(first).mean()),
            RawMeanSpeed=float(raw_valid.mean()) if len(raw_valid) else np.nan,
            RawAbove20Fraction=float((raw_valid>20).mean()) if len(raw_valid) else np.nan,
            ValidSpeedFrames=len(valid)))
    metadata=metadata.merge(pd.DataFrame(speed_rows),on='CupID',how='left')
    metadata=metadata.merge(after.reset_index()[['CupID','PositionFraction']],on='CupID',how='left')
    metadata.to_csv(output/'analysis/speed_by_metadata/per_ant_summary.csv',index=False)
    render_comparison(output,table,summary)
    write_json(output/'COMPARISON_COMPLETE.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


def render_comparison(output,table,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    table=table.loc[table.identity_status.eq('matched')].sort_values('cup')
    with PdfPages(output/'comparison/tracking_comparison.pdf') as pdf:
        fig,axes=plt.subplots(1,3,figsize=(16,9),sharey=True,layout='constrained')
        for ax,key,title in zip(axes,['PositionFraction','Full10NodePoseFraction','state_fraction'],
                                ['Position coverage','Complete 10-landmark poses','Assigned full-posture state']):
            for row in table.itertuples():
                old=getattr(row,'old_'+key);new=getattr(row,'new_'+key)
                ax.plot([100*old,100*new],[row.cup,row.cup],color='#999',lw=1)
            ax.scatter(100*table['old_'+key],table.cup,s=22,color='#888',label='Original')
            ax.scatter(100*table['new_'+key],table.cup,s=25,color='#287c8e',label='Rerun')
            ax.set(title=title,xlim=(0,101),xlabel='Recorded time (%)',yticks=table.cup)
            ax.grid(alpha=.12);ax.legend(fontsize=8)
        axes[0].set_ylabel('Cup');pdf.savefig(fig);fig.savefig(output/'comparison/tracking_comparison.png',dpi=150);plt.close(fig)
        fig,ax=plt.subplots(figsize=(12,7));xx=np.arange(len(table))
        ax.bar(xx,100*table.old_state_fraction,color='#888',label='Original state eligibility')
        ax.bar(xx,100*(table.old_tracking_local_state_fraction-table.old_state_fraction),bottom=100*table.old_state_fraction,color='#d99728',label='Local prediction on original tracks')
        ax.scatter(xx,100*table.new_state_fraction,color='#287c8e',s=30,label='New tracks + local prediction')
        ax.set(xticks=xx,xticklabels=table.cup,xlabel='Cup',ylabel='Assigned full-posture state (%)',ylim=(0,103),title='Separate the eligibility change from the tracking change')
        ax.legend(fontsize=9);pdf.savefig(fig);plt.close(fig)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('stage',choices=['prepare','submit','track','stitch','diagnose','states','report','publish'])
    p.add_argument('--source',type=Path);p.add_argument('--reference',type=Path);p.add_argument('--destination',type=Path)
    p.add_argument('--work-root',type=Path);p.add_argument('--exclude-nodes',default='deigo011019')
    p.add_argument('--tracking-only',action='store_true',help='Rerun tracking and diagnostics; fit states after stitching separately')
    p.add_argument('--plan',type=Path);p.add_argument('--index',type=int,default=0);args=p.parse_args()
    if args.stage=='prepare':initialize(args);return
    if args.stage=='submit':submit(args.plan);return
    plan=tracking.manifest(args.plan);m=tracking.manifest(Path(plan['output'])/'manifest.json')
    if args.stage=='track':tracking.track_task(m,args.index)
    elif args.stage=='stitch':tracking.stitch_cup(m,args.index)
    elif args.stage=='diagnose':
        from tracking.single_ant.full_cup_report import diagnose_cup
        diagnose_cup(m,args.index)
    elif args.stage=='states':state_worker(plan,args.index)
    elif args.stage=='report':
        if plan.get('tracking_only'):
            from tracking.single_ant.full_cup_report import aggregate_report
            aggregate_report(m)
            write_json(Path(plan['output'])/'RETRACK_COMPLETE.json',dict(source=plan['source'],tracking_only=True))
        else:compare(plan)
    elif args.stage=='publish':
        marker='RETRACK_COMPLETE.json' if plan.get('tracking_only') else 'COMPARISON_COMPLETE.json'
        if not (Path(plan['output'])/marker).exists():raise ValueError('Rerun is not complete')
        from tracking.single_ant.full_cup_report import publish
        destination=Path(plan['destination'])
        publish(m,destination)
        if (Path(plan['output'])/'comparison').exists():
            subprocess.run(['rsync','-a',str(Path(plan['output'])/'comparison')+'/',
                            str(destination/'comparison')+'/'],check=True)
        code_destination=destination/'tracking_code' if plan.get('tracking_only') else destination/'analysis/eigenposture/code'
        subprocess.run(['rsync','-a','--exclude=__pycache__',plan['code']+'/',
                        str(code_destination)+'/'],check=True)
        for filename in ['inventory.csv','inventory_all.csv']:
            path=destination/'analysis/eigenposture'/filename
            if not path.exists():continue
            table=pd.read_csv(path)
            for column in ['source_path','original_source_path']:
                if column in table:
                    prefix=plan['output']+'/'
                    table[column]=table[column].map(lambda value:
                        str(destination)+'/'+value[len(prefix):]
                        if isinstance(value,str) and value.startswith(prefix) else value)
            table.to_csv(path,index=False)
        write_json(destination/'rerun_plan.json',plan)


if __name__=='__main__':main()
