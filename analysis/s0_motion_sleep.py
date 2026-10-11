"""Calibrate motion sleep from S0, then fan out fresh labels over every track.

Only motion measurements are reused: no historical sleep labels are read.
Thresholds are the equal-ant weighted 95th percentiles of body/antenna motion
during S0. This retains at least 90% jointly in the calibration sample. It is
a motion envelope of S0, not a classifier that can distinguish S0 from S1.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from analysis import sleep_motion_utils as sm
from analysis.compute_sleep_motion_labels import (
    _atomic_save_array, _atomic_write_json, _atomic_write_parquet, _sleep_bouts,
)

PARAMETERS = dict(window_seconds=10., quiet_fraction_threshold=.9,
    min_valid_frame_fraction=.5, require_full_window=True,
    body_bodypoint_ids=list(sm.BODY_BODYPOINT_IDS), antenna_bodypoint_ids=list(sm.ANTENNA_BODYPOINT_IDS),
    body_percentile=75., antenna_percentile=75., min_valid_bodypoint_fraction=.75,
    min_valid_antenna_fraction=.5, max_gap_frames=5, max_bodypoint_speed_mm_s=20.)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weighted_quantile(values, weights, q):
    order=np.argsort(values);values=np.asarray(values)[order];weights=np.asarray(weights)[order]
    if not len(values) or not np.isfinite(values).all() or (weights<=0).any():
        raise ValueError('Expected finite observations with positive weights')
    return float(values[min(np.searchsorted(np.cumsum(weights),q*weights.sum()),len(values)-1)])


def thresholds(sample, quantile=.95):
    selected=sample[sample.state.eq(0)].copy()
    weights=1/selected.groupby('ant').ant.transform('size').to_numpy()
    return {column:weighted_quantile(selected[column].to_numpy(),weights,quantile)
            for column in ['body','antenna']}


def prepare(source,block,work,index):
    ant=pd.read_csv(source/'inventory.csv').iloc[index]
    folder=work/'measurements'/Path(ant.track_name).stem;folder.mkdir(parents=True,exist_ok=True)
    cache_path=block/'stitched/sleep_motion/per_track'/Path(ant.track_name).stem
    cache=sm.load_sleep_motion_cache(cache_path);n=int(cache.metadata['n_frames'])
    vectors={}
    for name,ids,fraction in [('body',sm.BODY_BODYPOINT_IDS,.75),('antenna',sm.ANTENNA_BODYPOINT_IDS,.5)]:
        vectors[name]=sm.aggregate_bodypoint_group_slice(cache,0,n,bodypoint_ids=ids,
            bodypoint_percentile=75,min_valid_bodypoint_fraction=fraction,max_gap_frames=5,
            max_bodypoint_speed_mm_s=20.)
        _atomic_save_array(folder/(name+'.npy'),vectors[name])
    inputs=dict(motion_metadata_sha256=sha(cache_path/sm.METADATA_FILENAME),
        motion_metadata=cache.metadata,ant=ant.ant,selected=bool(ant.selected),
        code_sha256=sha(__file__),aggregation=PARAMETERS)
    if ant.selected:
        name=ant.ant.replace(':','_');starts=np.load(source/'features'/name/'start.npy')
        lengths=np.load(source/'features'/name/'length.npy');states=np.load(source/'assignments'/name/'state.npy')
        dense=np.full((int(ant.n_frames)+1)//2,-1,np.int8)
        for offset in range(int(lengths.max(initial=0))):
            keep=lengths>offset;dense[starts[keep]+offset]=states[keep]
        frames=np.arange(len(dense),dtype=np.int64)*2;local=frames-cache.frame_min
        valid=(local>=0)&(local<n)&(dense>=0);indices=np.flatnonzero(valid)
        valid[indices]&=np.isfinite(vectors['body'][local[indices]])&np.isfinite(vectors['antenna'][local[indices]])
        rng=np.random.default_rng(7241012+index);tables=[]
        # Uniform samples within ant/state; later weights give ants equal weight.
        for state in np.unique(states):
            take=np.flatnonzero(valid&(dense==state));take=np.sort(rng.choice(take,min(10000,len(take)),replace=False))
            tables.append(pd.DataFrame(dict(ant=ant.ant,state=state,frame=frames[take],
                body=vectors['body'][local[take]],antenna=vectors['antenna'][local[take]])))
        _atomic_write_parquet(folder/'calibration.parquet',pd.concat(tables,ignore_index=True))
        inputs['assignments_sha256']=sha(source/'assignments'/name/'state.npy')
        inputs['starts_sha256']=sha(source/'features'/name/'start.npy')
        inputs['lengths_sha256']=sha(source/'features'/name/'length.npy')
    _atomic_write_json(folder/'measurement.json',inputs)
    print('Prepared',ant.ant,flush=True)


def calibrate(source,work,output):
    inventory=pd.read_csv(source/'inventory.csv');samples=[]
    for row in inventory[inventory.selected].itertuples():
        samples.append(pd.read_parquet(work/'measurements'/Path(row.track_name).stem/'calibration.parquet'))
    sample=pd.concat(samples,ignore_index=True);limits=thresholds(sample)
    # Alternating 30-minute blocks diagnose time transfer, without choosing the
    # threshold rule from contact yield, spatial groups, or ant classes.
    train=(sample.frame//(24*1800))%2==0;validation_limits=thresholds(sample[train])
    rows=[]
    for split,subset,cut in [('all',sample,limits),('held_out_blocks',sample[~train],validation_limits)]:
        for state,group in subset.groupby('state'):
            below=(group.body<=cut['body'])&(group.antenna<=cut['antenna'])
            per_ant=group.assign(below=below).groupby('ant').below.mean()
            rows.append(dict(split=split,state=int(state),n_ants=len(per_ant),n_samples=len(group),
                joint_below_fraction=float(per_ant.mean()),body_limit=cut['body'],antenna_limit=cut['antenna']))
    parameters=dict(PARAMETERS,body_speed_threshold_mm_s=limits['body'],antenna_speed_threshold_mm_s=limits['antenna'])
    output.mkdir(parents=True,exist_ok=True)
    sample.to_parquet(output/'calibration_samples.parquet',index=False)
    pd.DataFrame(rows).to_csv(output/'calibration_state_summary.csv',index=False)
    _atomic_write_json(output/'definition.json',dict(definition='S0_calibrated_motion_sleep',
        reference_state=0,quantile=.95,weighting='Equal ants within S0; uniform up to 10000 observations per ant/state',
        rule='95th percentile of each motion metric in S0; no contact-yield or role tuning',
        classifier_parameters=parameters,validation_thresholds=validation_limits,
        n_calibration_ants=int(sample[sample.state.eq(0)].ant.nunique()),
        dictionary_sha256=sha(source/'models/dictionary.npz'),inventory_sha256=sha(source/'inventory.csv'),
        calibration_samples_sha256=sha(output/'calibration_samples.parquet'),code_sha256=sha(__file__),
        note='Motion-only readout calibrated from S0; can include S1 and other low-motion observations. No cached sleep labels.'))
    print('S0 motion thresholds (mm/s):',limits,flush=True)


def label(source,work,output,index):
    row=pd.read_csv(source/'inventory.csv').iloc[index];stem=Path(row.track_name).stem
    measurement=work/'measurements'/stem;origin=json.loads((measurement/'measurement.json').read_text())
    motion=origin['motion_metadata'];definition=json.loads((output/'definition.json').read_text())
    parameters=definition['classifier_parameters']
    body=np.load(measurement/'body.npy',mmap_mode='r');antenna=np.load(measurement/'antenna.npy',mmap_mode='r')
    result=sm.classify_body_antenna_quiet(body,antenna,fps=motion['fps'],
        body_speed_threshold_mm_s=parameters['body_speed_threshold_mm_s'],
        antenna_speed_threshold_mm_s=parameters['antenna_speed_threshold_mm_s'],
        window_seconds=parameters['window_seconds'],quiet_fraction_threshold=parameters['quiet_fraction_threshold'],
        min_valid_fraction=parameters['min_valid_frame_fraction'],require_full_window=True)
    folder=output/'per_track'/stem;folder.mkdir(parents=True,exist_ok=True);state=result['state']
    for name,key,dtype in [('sleep_state','state',np.int8),('quiet_fraction','quiet_fraction',np.float16),('valid_fraction','valid_fraction',np.float16)]:
        _atomic_save_array(folder/(name+'.npy'),result[key].astype(dtype))
    bouts=_sleep_bouts(state,result['quiet_fraction'],result['valid_fraction'],frame_min=motion['frame_min'],fps=motion['fps'])
    _atomic_write_parquet(folder/'sleep_bouts.parquet',bouts)
    summary=dict(track_name=row.track_name,track_id=int(row.track_id),side=row.side,n_frames=len(state),
        n_sleep_frames=int((state==1).sum()),n_classified_frames=int((state>=0).sum()),
        n_unknown_frames=int((state<0).sum()),frame_min=motion['frame_min'],frame_max=motion['frame_max'])
    _atomic_write_json(folder/'sleep_motion_label_metadata.json',dict(fps=motion['fps'],
        frame_min=motion['frame_min'],frame_max=motion['frame_max'],classifier_type=definition['definition'],
        classifier_parameters=parameters,definition_sha256=sha(output/'definition.json'),
        files=dict(sleep_state='sleep_state.npy',quiet_fraction='quiet_fraction.npy',valid_fraction='valid_fraction.npy'),
        summary=summary,measurement=origin))
    print('Labeled',row.ant,summary['n_sleep_frames'],flush=True)


def finalize(source,output):
    rows=[]
    for ant in pd.read_csv(source/'inventory.csv').itertuples():
        folder=output/'per_track'/Path(ant.track_name).stem
        meta=json.loads((folder/'sleep_motion_label_metadata.json').read_text())
        assert meta['definition_sha256']==sha(output/'definition.json')
        state=np.load(folder/'sleep_state.npy',mmap_mode='r')
        assert len(state)==meta['summary']['n_frames'] and np.isin(state,[-1,0,1]).all()
        rows.append(meta['summary'])
    pd.DataFrame(rows).to_csv(output/'track_summary.csv',index=False)
    _atomic_write_json(output/'complete.json',dict(n_tracks=len(rows),definition_sha256=sha(output/'definition.json')))
    print('Complete:',len(rows),'fresh motion label tracks',flush=True)


def replace_context_sleep(context,state,frame_min,width,fps):
    """Replace both sleep-dependent arrays; preserve measured physical context."""
    from analysis.sleep_motion_analysis_utils import sample_dense
    result={k:v for k,v in context.items() if k not in ('state','sleep_age_seconds')}
    frames=np.arange(len(context['inside']),dtype=np.int64)*width+width-1
    result['state']=sample_dense(state,frame_min,frames,missing=-1).astype(np.int8)
    indices=np.arange(len(state));last=np.maximum.accumulate(np.where(state!=1,indices,-1))
    age=np.where(state==1,(indices-last)/fps,0).astype(np.float32)
    result['sleep_age_seconds']=sample_dense(age,frame_min,frames).astype(np.float32)
    return result


def refresh_return_contexts(source,block,tracks,regions,settings,output):
    """Reuse fingerprint-matched positions/motion, never old sleep state or age.

    This is only an optimization. Missing or incompatible physical caches fall
    back to the ordinary raw-input reconstruction in return_sleep_utils.
    """
    from analysis import return_sleep_utils as rs, sleep_motion_analysis_utils as sma
    old_root=source/'return_context_cache';old_labels=source/'sleep_from_states'
    if not old_labels.exists():return
    old_tracks=sma.load_sleep_label_tracks(old_labels);n_bins=int(tracks.frame_max.max())//settings.bin_frames+1
    rows=[];output.mkdir(parents=True,exist_ok=True)
    physical=['body_percentile','antenna_percentile','min_valid_bodypoint_fraction',
        'min_valid_antenna_fraction','max_gap_frames','max_bodypoint_speed_mm_s']
    for row in tracks.itertuples():
        path,crossings=rs.context_cache_paths(row,block,regions,settings,n_bins,output)
        if path.exists() and crossings.exists():continue
        candidates=old_tracks[old_tracks.side.eq(row.side)&old_tracks.track_id.eq(row.track_id)]
        if len(candidates)!=1:continue
        old=next(candidates.itertuples())
        if any(getattr(old,k)!=getattr(row,k) for k in ['track_name','frame_min','frame_max','fps']):continue
        before=json.loads(old.classifier_parameters);after=json.loads(row.classifier_parameters)
        if any(before.get(k)!=after.get(k) for k in physical):continue
        origin,old_crossings=rs.context_cache_paths(old,block,regions,settings,n_bins,old_root)
        if not origin.exists() or not old_crossings.exists():continue
        with np.load(origin) as z:context={key:z[key] for key in z.files}
        state=np.load(row.state_path,mmap_mode='r')
        context=replace_context_sleep(context,state,row.frame_min,settings.bin_frames,settings.fps)
        path.parent.mkdir(parents=True,exist_ok=True);crossings.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,**context);shutil.copy2(old_crossings,crossings)
        rows.append(dict(track_name=row.track_name,physical_source=str(origin),physical_source_sha256=sha(origin),
            new_label_metadata_sha256=sha(row.metadata_path),context=str(path),context_sha256=sha(path)))
    for file in old_root.glob('pair_contact_bouts_*.parquet'):
        if not (output/file.name).exists():shutil.copy2(file,output/file.name)
    manifest=output/'physical_reuse.json'
    if rows:
        previous=json.loads(manifest.read_text()) if manifest.exists() else []
        _atomic_write_json(manifest,previous+rows)
    print('Fresh sleep context from unchanged physical measurements:',len(rows),'tracks',flush=True)


def submit(args):
    args.work.mkdir(parents=True,exist_ok=True);jobs=args.work/'jobs';jobs.mkdir(exist_ok=True)
    count=len(pd.read_csv(args.source/'inventory.csv'));previous=None;ids={}
    stages=['prepare','calibrate','label','finalize']
    for stage in stages[stages.index(args.start_stage):]:
        command=[args.python,'-m','analysis.s0_motion_sleep',stage,'--source',str(args.source),
            '--block',str(args.block),'--work',str(args.work),'--output',str(args.output)]
        array=stage in ('prepare','label')
        body='#!/bin/bash\nset -euo pipefail\numask 0002\nexport PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1\n'
        body+='cd '+shlex.quote(str(Path(__file__).resolve().parents[1]))+'\n'
        body+=shlex.join(command)+(' --index "$SLURM_ARRAY_TASK_ID"' if array else '')+'\n'
        path=jobs/(stage+'.sbatch');path.write_text(body)
        cmd=[args.sbatch,'--parsable','--partition',args.partition,'--cpus-per-task','1','--mem','8G',
            '--time','02:00:00','--job-name','s0motion_'+stage,'--output',str(jobs/(stage+'_%A_%a.log'))]
        if array:cmd+=['--array',f'0-{count-1}%{args.concurrency}']
        if previous:cmd+=['--dependency','afterok:'+previous]
        previous=subprocess.check_output(cmd+[str(path)],text=True).strip().split(';')[0];ids[stage]=previous
        _atomic_write_json(jobs/'submitted.json',ids)
    print(json.dumps(ids),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','calibrate','label','finalize','submit'])
    for name in ['source','block','work','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--index',type=int,default=0);p.add_argument('--concurrency',type=int,default=24)
    p.add_argument('--sbatch',default='sbatch');p.add_argument('--partition',default='compute');p.add_argument('--python',default=sys.executable)
    p.add_argument('--start-stage',choices=['prepare','calibrate','label','finalize'],default='prepare',help='Resume completed preparation without rereading motion measurements')
    a=p.parse_args()
    if a.stage=='submit':submit(a)
    elif a.stage=='prepare':prepare(a.source,a.block,a.work,a.index)
    elif a.stage=='calibrate':calibrate(a.source,a.work,a.output)
    elif a.stage=='label':label(a.source,a.work,a.output,a.index)
    else:finalize(a.source,a.output)
