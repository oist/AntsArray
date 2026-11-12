"""S0/S1 sleep display with one-sided motion exclusions for unclassified time.

Stages prepare/calibrate/label/summary can run independently in Slurm arrays.
Only unknown time may acquire a motion-based negative label. Quiet motion is
never evidence for adding sleep, and learned assignments are never changed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

BIN_WIDTH=.01
EDGES=np.linspace(0,20,2001)
SLEEP_STATES=(0,1)


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recorded_mask(manifest,window_frames=1):
    result=np.zeros(manifest['n_frames'],bool)
    for s in manifest.get('recording_segments') or [dict(frame_offset=0,num_frames=manifest['n_frames'])]:
        lo=int(s['frame_offset'])+window_frames-1;hi=int(s['frame_offset'])+int(s['num_frames'])
        result[lo:hi]=True
    return result


def observed_states(eigen,ant,n,selected):
    dense=np.full((n+1)//2,-1,np.int8)
    # Discovery eligibility is not a permanent prediction exclusion. The
    # single-ant workflow can now assign locally valid windows for other ants.
    if selected or (Path(eigen)/'assignments'/ant.replace(':','_')/'state.npy').is_file():
        name=ant.replace(':','_');folder=Path(eigen)/'features'/name
        start=np.load(folder/'start.npy');length=np.load(folder/'length.npy')
        state=np.load(Path(eigen)/'assignments'/name/'state.npy')
        for offset in range(int(length.max(initial=0))):
            keep=length>offset;dense[start[keep]+offset]=state[keep]
    return np.repeat(dense,2)[:n]


def sustained_high(values,limit,manifest,window_frames=6,minimum_high=3):
    """Require current high evidence plus several recent high observations."""
    high=np.isfinite(values)&(values>limit)&(values<=20)
    total=np.r_[0,np.cumsum(high,dtype=np.int32)]
    count=total[1:]-total[np.maximum(0,np.arange(len(values))+1-window_frames)]
    return high&(count>=minimum_high)&recorded_mask(manifest,window_frames)


def classify(observed,velocity_high,antenna_high,recorded):
    category=np.full(len(observed),-1,np.int8);evidence=np.zeros(len(observed),np.uint8)
    direct=(observed>=0)&recorded
    category[direct]=np.isin(observed[direct],SLEEP_STATES).astype(np.int8)
    evidence[direct]=1
    unknown=(observed<0)&recorded
    evidence[unknown&velocity_high]|=2;evidence[unknown&antenna_high]|=4
    category[(evidence&6)>0]=0
    return category,evidence


def context(args):
    dataset=args.dataset.resolve();eigen=args.eigen_root or dataset/'analysis/eigenposture'
    motion=args.motion_root or dataset/'analysis/sleep_motion'
    manifest=json.loads((dataset/'manifest.json').read_text())
    if manifest['fps']!=24:raise ValueError('Native state assignments require 24-Hz input')
    table=pd.read_csv(eigen/'inventory_all.csv')
    return dataset,eigen,motion,manifest,table


def velocity_path(dataset,row):return dataset/'analysis/speed_vectors/per_track'/Path(row.track_name).stem/'speed_mm_s.npy'


def prepare(args):
    from analysis import sleep_motion_utils as sm
    dataset,eigen,motion,manifest,table=context(args);row=table.iloc[args.index]
    folder=args.work/row.ant.replace(':','_');folder.mkdir(parents=True,exist_ok=True)
    n=manifest['n_frames'];observed=observed_states(eigen,row.ant,n,bool(row.selected))
    support=recorded_mask(manifest);observed[~support]=-1
    antenna=np.full(n,np.nan,np.float32)
    cache_path=motion/'per_track'/Path(row.track_name).stem
    if (cache_path/'sleep_motion_metadata.json').is_file():
        cache=sm.load_sleep_motion_cache(cache_path)
        if cache.metadata.get('motion_version')!=sm.MOTION_VERSION:raise ValueError('Rebuild camera-local motion cache')
        # One partially visible antenna is enough for a negative decision.
        # At least 2/3 landmarks must support its 75th-percentile speed.
        for ids in [(4,5,6),(7,8,9)]:
            values=sm.aggregate_bodypoint_group_slice(cache,0,int(cache.metadata['n_frames']),
                bodypoint_ids=ids,bodypoint_percentile=75,min_valid_bodypoint_fraction=2/3,
                max_gap_frames=5,max_bodypoint_speed_mm_s=20)
            lo=cache.frame_min;hi=lo+len(values)
            antenna[lo:hi]=np.fmax(antenna[lo:hi],values)
    antenna[~support]=np.nan
    np.save(folder/'observed.npy',observed);np.save(folder/'antenna.npy',antenna)
    velocity=np.load(velocity_path(dataset,row),mmap_mode='r')
    if velocity.shape!=(n,):raise ValueError('Velocity timeline differs from state timeline')
    split=(np.arange(n,dtype=np.int64)//(24*1800))%2
    histogram=np.zeros((2,2,2,2000),np.int64)
    for channel,values in enumerate([velocity,antenna]):
        valid=np.isfinite(values)&(values>=0)&(values<=20)
        for state in SLEEP_STATES:
            for parity in [0,1]:
                take=valid&(observed==state)&(split==parity)
                histogram[channel,state,parity]=np.histogram(values[take],EDGES)[0]
    np.savez_compressed(folder/'histogram.npz',histogram=histogram)
    write_json(folder/'prepared.json',dict(ant=row.ant,track_id=int(row.track_id),selected=bool(row.selected),
        velocity_file=str(velocity_path(dataset,row)),motion_metadata=str(cache_path/'sleep_motion_metadata.json'),
        n_antenna_observations=int(np.isfinite(antenna).sum())))
    print('Prepared',row.ant,flush=True)


def calibrate(args):
    dataset,eigen,motion,manifest,table=context(args)
    hist=np.asarray([np.load(args.work/r.ant.replace(':','_')/'histogram.npz')['histogram'] for r in table.itertuples()])
    rows=[];thresholds={}
    for channel,name in enumerate(['velocity','antenna']):
        cuts=[]
        for state in SLEEP_STATES:
            counts=hist[:,channel,state,0];totals=counts.sum(axis=1);valid=totals>0
            if not valid.any():raise ValueError(f'No calibration observations: {name} S{state}')
            distribution=(counts[valid]/totals[valid,None]).mean(axis=0)
            index=min(int(np.searchsorted(np.cumsum(distribution),args.quantile)),1999)
            cutoff=float(EDGES[index+1]);cuts.append(cutoff)
            rows.append(dict(channel=name,state=state,ants=int(valid.sum()),observations=int(totals.sum()),
                             quantile=args.quantile,cutoff_mm_s=cutoff))
        thresholds[name]=max(cuts)
    args.output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output/'calibration.csv',index=False)
    write_json(args.output/'definition.json',dict(mode='S0_S1_with_negative_motion_exclusions',sleep_states=[0,1],
        categories={'1':'S0/S1 (sleep)','0':'not S0/S1','-1':'ambiguous'},
        evidence_bits={'1':'observed state assignment','2':'high trajectory speed','4':'high antennal motion'},
        thresholds_mm_s=thresholds,quantile=args.quantile,
        threshold_rule='Maximum of the per-state equal-ant upper quantiles for S0 and S1; use upper edge of 0.01 mm/s histogram bins; every measured frame contributes',
        training='Even 30-minute blocks',validation='Odd 30-minute blocks; fixed learned state dictionary',
        antenna_rule='Larger of antenna A/B 75th-percentile speeds; at least 2/3 landmarks on one antenna; same camera/recording; gaps <=5 frames; speeds <=20 mm/s',
        velocity_rule='Existing standard colony speed cache, unchanged; finite speeds <=20 mm/s',
        exclusion_rule='Current observation above threshold and at least 3 high observations in trailing 6 frames (0.25 s); reset at recording boundaries',
        positive_rule='Only original observed S0/S1 assignments label sleep; low motion never labels missing time as sleep',
        original_assignments='Preserved, including any motion conflicts; motion rules apply only to unknown frames',
        dataset=str(dataset),n_frames=manifest['n_frames'],fps=24,dictionary_sha256=sha(eigen/'models/dictionary.npz'),
        manifest_sha256=sha(dataset/'manifest.json'),code_sha256=sha(__file__)))
    print('Exclusion thresholds (mm/s)',thresholds,flush=True)


def label(args):
    dataset,eigen,motion,manifest,table=context(args);row=table.iloc[args.index]
    folder=args.work/row.ant.replace(':','_')
    observed=np.load(folder/'observed.npy',mmap_mode='r');antenna=np.load(folder/'antenna.npy',mmap_mode='r')
    velocity=np.load(velocity_path(dataset,row),mmap_mode='r')
    definition=json.loads((args.output/'definition.json').read_text());limits=definition['thresholds_mm_s']
    vh=sustained_high(velocity,limits['velocity'],manifest);ah=sustained_high(antenna,limits['antenna'],manifest)
    # Respect the already audited resident-ID exclusions when present.
    identities=dataset/'analysis/speed_by_metadata/metadata_join.csv';identity_ok=True
    if identities.is_file():
        audit=pd.read_csv(identities);match=audit.loc[audit.TrackID.eq(row.track_id)]
        identity_ok=not len(match) or match.IdentityStatus.eq('matched').all()
    if not identity_ok:vh[:]=False;ah[:]=False
    support=recorded_mask(manifest);category,evidence=classify(observed,vh,ah,support)
    assert np.array_equal(category[observed>=0],np.isin(observed[observed>=0],SLEEP_STATES))
    assert (category[~support]==-1).all()
    target=args.output/'per_track'/Path(row.track_name).stem;target.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target/'labels.npz',category=category,evidence=evidence,observed=observed)
    baseline=(observed<0)&support;inferred=(evidence&6)>0
    stats=dict(track_id=int(row.track_id),ant=row.ant,selected=bool(row.selected),identity_ok=bool(identity_ok),
        recorded_frames=int(support.sum()),sleep_frames=int((category[support]==1).sum()),
        not_sleep_frames=int((category[support]==0).sum()),ambiguous_frames=int((category[support]<0).sum()),
        original_unknown_frames=int(baseline.sum()),inferred_not_frames=int(inferred.sum()),
        velocity_exclusions=int(((evidence&2)>0).sum()),antenna_exclusions=int(((evidence&4)>0).sum()),
        original_sleep_frames=int(np.isin(observed,SLEEP_STATES).sum()),
        definition_sha256=sha(args.output/'definition.json'))
    write_json(target/'metadata.json',stats)
    split=(np.arange(len(observed),dtype=np.int64)//(24*1800))%2
    validation=[]
    for state in range(json.loads((eigen/'models/selection.json').read_text())['n_states']):
        for parity in [0,1]:
            take=(observed==state)&(split==parity);n=int(take.sum())
            if n:validation.append(dict(ant=row.ant,state=state,split='held_out' if parity else 'calibration',n_frames=n,
                velocity_fraction=float(vh[take].mean()),antenna_fraction=float(ah[take].mean()),
                exclusion_fraction=float((vh[take]|ah[take]).mean())))
    pd.DataFrame(validation).to_csv(target/'validation.csv',index=False)
    print(row.ant,'inferred not:',stats['inferred_not_frames'],'ambiguous:',stats['ambiguous_frames'],flush=True)


def summary(args):
    dataset,eigen,motion,manifest,table=context(args)
    paths=[args.output/'per_track'/Path(r.track_name).stem for r in table.itertuples()]
    stats=pd.DataFrame([json.loads((p/'metadata.json').read_text()) for p in paths])
    definition_hash=sha(args.output/'definition.json');assert stats.definition_sha256.eq(definition_hash).all()
    stats.to_csv(args.output/'track_summary.csv',index=False)
    validation=pd.concat([pd.read_csv(p/'validation.csv') for p in paths if (p/'validation.csv').stat().st_size>1],ignore_index=True)
    grouped=validation.groupby(['state','split']).agg(ants=('ant','size'),frames=('n_frames','sum'),
        velocity_fraction=('velocity_fraction','mean'),antenna_fraction=('antenna_fraction','mean'),exclusion_fraction=('exclusion_fraction','mean')).reset_index()
    grouped.to_csv(args.output/'validation_summary.csv',index=False)
    false=grouped.loc[grouped.state.isin(SLEEP_STATES)&grouped.split.eq('held_out')]
    result=dict(tracks=len(stats),definition_sha256=definition_hash,
        original_sleep_assignments_preserved=bool(stats.sleep_frames.eq(stats.original_sleep_frames).all()),
        previously_unknown_frames=int(stats.original_unknown_frames.sum()),inferred_not_frames=int(stats.inferred_not_frames.sum()),
        remaining_ambiguous_frames=int(stats.ambiguous_frames.sum()),max_heldout_sleep_exclusion_fraction=float(false.exclusion_fraction.max()),
        low_false_exclusion=bool(false.exclusion_fraction.max()<=.01))
    assert result['original_sleep_assignments_preserved']
    write_json(args.output/'COMPLETE.json',result)
    print(json.dumps(result,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','calibrate','label','summary'])
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--work',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--eigen-root',type=Path);p.add_argument('--motion-root',type=Path)
    p.add_argument('--index',type=int,default=0);p.add_argument('--quantile',type=float,default=.995)
    a=p.parse_args();globals()[a.stage](a)


if __name__=='__main__':main()
