"""Derive putative sleep exclusively from discovered state membership.

No existing sleep labels are read. Quiet states are interpreted AFTER state
clustering from their body/antenna motion profiles. Unknown intervals stay
unknown and censor bouts rather than becoming apparent awakenings.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.full_behavior_extract import FPS,HZ


def quiet_bouts(state,fps=FPS):
    quiet=np.asarray(state)==1
    edges=np.diff(np.r_[False,quiet,False].astype(np.int8))
    starts=np.flatnonzero(edges==1);stops=np.flatnonzero(edges==-1)
    return pd.DataFrame(dict(start_frame=starts,stop_frame=stops,duration_seconds=(stops-starts)/fps,
        left_censored=(starts==0)|(state[np.maximum(starts-1,0)]<0),
        right_censored=(stops==len(state))|(state[np.minimum(stops,len(state)-1)]<0)))


def sleep_from_windows(starts,lengths,labels,n_frames,quiet_states,minimum_seconds=0):
    """Label observed 12-Hz samples, expand to raw frame intervals; no gap fills."""
    dense=np.full((n_frames+1)//2,-1,np.int16)
    for offset in range(int(lengths.max(initial=0))):
        keep=lengths>offset;dense[starts[keep]+offset]=labels[keep]
    state=np.where(dense<0,-1,np.isin(dense,quiet_states).astype(np.int8)).astype(np.int8)
    state=np.repeat(state,2)[:n_frames]
    bouts=quiet_bouts(state)
    if minimum_seconds:
        for row in bouts[bouts.duration_seconds<minimum_seconds].itertuples():
            # A truncated quiet observation cannot establish a short wake bout.
            state[row.start_frame:row.stop_frame]=-1 if row.left_censored or row.right_censored else 0
    return state,bouts


def select_quiet_states(root,body_limit=.05,antenna_limit=.15):
    with np.load(root/'models/dictionary.npz') as z:raw=z['raw'];state=z['state']
    rows=[]
    for s in range(int(state.max()+1)):
        subset=raw[state==s]
        body=np.linalg.norm(subset[:,24:26],axis=1);antenna=subset[:,26:28].mean(axis=1)
        rows.append(dict(state=s,body_peak_median_mm_s=float(np.median(body)),
            antenna_median_mm_s=float(np.median(antenna)),
            body_peak_p90_mm_s=float(np.quantile(body,.9)),antenna_p90_mm_s=float(np.quantile(antenna,.9)),
            fraction_below_profile_limits=float(np.mean((body<=body_limit)&(antenna<=antenna_limit))),
            antenna_A_median_mm_s=float(np.median(subset[:,26])),antenna_B_median_mm_s=float(np.median(subset[:,27])),
            contact_mean_median=float(np.median(subset[:,29]))))
    table=pd.DataFrame(rows)
    table['quiet_candidate']=(table.body_peak_median_mm_s<=body_limit)&(table.antenna_median_mm_s<=antenna_limit)
    quiet=table.loc[table.quiet_candidate,'state'].to_numpy(int)
    if not len(quiet):raise ValueError('No state meets the stated quiet-state interpretation; inspect profiles, do not force a sleep state.')
    return quiet,table


def build(root,block,quiet_states=None,minimum_seconds=0):
    output=root/'sleep_from_states';output.mkdir(exist_ok=True)
    auto,profiles=select_quiet_states(root)
    if quiet_states is None:quiet_states=auto
    quiet_states=list(map(int,quiet_states));profiles.to_csv(output/'state_motion_profiles.csv',index=False)
    inventory=pd.read_csv(root/'inventory.csv');summaries=[];all_bouts=[]
    # Motion aggregation options are only for downstream response measurements,
    # not a sleep classifier. Sleep membership uses the state IDs above.
    parameters=dict(definition='discovered_behavior_state_membership',quiet_states=quiet_states,
        minimum_seconds=minimum_seconds,body_percentile=75.,antenna_percentile=75.,
        min_valid_bodypoint_fraction=.75,min_valid_antenna_fraction=.5,max_gap_frames=5,
        max_bodypoint_speed_mm_s=20.)
    # Re-running an interactive cell must not invalidate expensive downstream
    # context caches when the assignments and interpretation are unchanged.
    signature=hashlib.sha256(json.dumps(parameters,sort_keys=True).encode())
    signature.update((root/'inventory.csv').read_bytes())
    for ant in inventory.itertuples():
        signature.update((block/'stitched/sleep_motion/per_track'/Path(ant.track_name).stem/'sleep_motion_metadata.json').read_bytes())
        if ant.selected:
            name=ant.ant.replace(':','_')
            for path in (root/'features'/name/'start.npy',root/'features'/name/'length.npy',root/'assignments'/name/'state.npy'):
                signature.update(path.read_bytes())
    fingerprint=signature.hexdigest()
    definition=output/'definition.json'
    required=[output/'quiet_bouts.parquet',output/'ant_quiet_summary.csv']
    required.extend(output/'per_track'/Path(name).stem/file for name in inventory.track_name
                    for file in ('sleep_state.npy','sleep_motion_label_metadata.json'))
    if definition.exists() and json.loads(definition.read_text()).get('inputs_sha256')==fingerprint and all(p.exists() for p in required):
        return output
    for ant in inventory.itertuples():
        name=ant.ant.replace(':','_');n=int(ant.n_frames)
        if ant.selected:
            start=np.load(root/'features'/name/'start.npy');length=np.load(root/'features'/name/'length.npy')
            labels=np.load(root/'assignments'/name/'state.npy')
            state,bouts=sleep_from_windows(start,length,labels,n,quiet_states,minimum_seconds)
            bouts=bouts.assign(ant=ant.ant,side=ant.side,track_id=ant.track_id)
            all_bouts.append(bouts)
        else:state=np.full(n,-1,np.int8)
        # Match the motion cache's global-frame origin for downstream utilities.
        motion=json.loads((block/'stitched/sleep_motion/per_track'/Path(ant.track_name).stem/'sleep_motion_metadata.json').read_text())
        lo,hi=int(motion['frame_min']),int(motion['frame_max'])+1
        folder=output/'per_track'/Path(ant.track_name).stem;folder.mkdir(parents=True,exist_ok=True)
        np.save(folder/'sleep_state.npy',state[lo:hi])
        summary=dict(side=ant.side,track_id=ant.track_id,track_name=ant.track_name,selected=bool(ant.selected),
            classified_seconds=int((state>=0).sum())/FPS,quiet_seconds=int((state==1).sum())/FPS)
        metadata=dict(fps=FPS,frame_min=lo,frame_max=hi-1,files=dict(sleep_state='sleep_state.npy'),
            classifier_type='behavior_cluster_membership',classifier_parameters=parameters,summary=summary,
            interpretation='Putative sleep / quiet behavior; no existing sleep labels or arousal assay used')
        (folder/'sleep_motion_label_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n');summaries.append(summary)
    pd.DataFrame(summaries).to_csv(output/'ant_quiet_summary.csv',index=False)
    pd.concat(all_bouts,ignore_index=True).to_parquet(output/'quiet_bouts.parquet',index=False)
    definition.write_text(json.dumps(dict(**parameters,inputs_sha256=fingerprint,body_interpretation_limit_mm_s=.05,
        antenna_interpretation_limit_mm_s=.15,bouts='Unknown gaps censor bouts; no gap filling',
        note='Limits interpret learned clusters after fitting, never label training data; inspect and change quiet_states explicitly if desired'),indent=2)+'\n')
    print('STATE-DERIVED SLEEP',quiet_states,'minimum seconds',minimum_seconds,flush=True)
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--block',type=Path,required=True);p.add_argument('--minimum-seconds',type=float,default=0)
    a=p.parse_args();build(a.root,a.block,minimum_seconds=a.minimum_seconds)
