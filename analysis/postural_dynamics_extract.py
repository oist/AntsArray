"""Sample short, body-relative pose sequences without loading spatial labels.

The output contains intrinsic segment directions and signed body velocities.
Absolute position, global heading, sleep and spatial cluster labels are excluded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

# Branches connect occiput to the two proximal antennal landmarks. The fixed
# petiole-to-tag direction defines the anterior body axis and is not a feature.
EDGES = ((0,1),(2,3),(1,4),(4,5),(5,6),(1,7),(7,8),(8,9))
EDGE_NAMES = ('head','gaster','antenna_A_base','antenna_A_mid','antenna_A_tip',
              'antenna_B_base','antenna_B_mid','antenna_B_tip')
FPS = 24
CLIP_FRAMES = 60
RECORDING_START = '2026-07-24T09:31:10'
DAY1_FRAME = (28*60+50)*FPS  # first complete clock-aligned day: July 24 10:00
DAY_FRAMES = 86400*FPS
MM_PER_PX = .016
SAMPLE_ENDS = np.arange(4,59,2)


def body_velocity(xy, position, *, fps=FPS, mm_per_px=MM_PER_PX):
    """Causal five-frame anchor-velocity fit, projected onto the body axes.

    Input [clips,60,10,2] landmarks and [clips,60,2] TrackX/TrackY.
    Output [clips,28,2] in mm/s: positive anterior and signed lateral.
    At raw frame t, only t-4 through t enter either velocity or orientation.
    The lateral unit vector is (-anterior_y, anterior_x), in image coordinates.
    """
    xy=np.asarray(xy,dtype=float);position=np.asarray(position,dtype=float)
    if xy.shape[1:]!=(60,10,2) or position.shape!=(len(xy),60,2):
        raise ValueError('Expected complete 60-frame landmark and anchor clips')
    axis=xy[:,:,0]-xy[:,:,2]
    norm=np.linalg.norm(axis,axis=-1,keepdims=True)
    axis=np.divide(axis,norm,out=np.full_like(axis,np.nan),where=norm>1e-9)
    indices=SAMPLE_ENDS[:,None]+np.arange(-4,1)
    positions=position[:,indices]*mm_per_px
    # Center before differencing to retain precision under large translations.
    positions=positions-positions[:,:,2:3]
    velocity=np.sum(positions*np.arange(-2,3)[None,None,:,None],axis=2)*(fps/10.)
    direction=np.mean(axis[:,indices],axis=2)
    norm=np.linalg.norm(direction,axis=-1,keepdims=True)
    direction=np.divide(direction,norm,out=np.full_like(direction,np.nan),where=norm>.1)
    lateral=np.stack([-direction[...,1],direction[...,0]],axis=-1)
    return np.stack([(velocity*direction).sum(axis=-1),(velocity*lateral).sum(axis=-1)],axis=-1).astype(np.float32)


def stamp(path):
    p=Path(path);s=p.stat()
    return dict(path=str(p),size=s.st_size,mtime_ns=s.st_mtime_ns)


def schedule(seed, clips_per_halfhour=30):
    """One random 2.5s clip per minute, independently seeded for each ant.

    Clips cannot cross minute, half-hour, training/validation-hour or day edges.
    """
    if clips_per_halfhour != 30:
        raise ValueError('This protocol uses exactly one clip in each minute')
    rng=np.random.default_rng(seed)
    starts=DAY1_FRAME+np.arange(2880,dtype=np.int64)*60*FPS+rng.integers(0,60*FPS-CLIP_FRAMES+1,2880)
    return starts


def intrinsic_directions(xy):
    """Remove translation, rotation and scale; preserve head articulation.

    Input [...,10,2]; output [...,8,2] direction cosines, plus QC lengths in px.
    Reflection is deliberately not removed: antenna identity is retained.
    """
    xy=np.asarray(xy,dtype=float)
    axis=xy[...,0,:]-xy[...,2,:]
    axis_length=np.linalg.norm(axis,axis=-1)
    anterior=np.divide(axis,axis_length[...,None],out=np.full_like(axis,np.nan),where=axis_length[...,None]>1e-9)
    left=np.stack([-anterior[...,1],anterior[...,0]],axis=-1)
    vectors=np.stack([xy[...,b,:]-xy[...,a,:] for a,b in EDGES],axis=-2)
    lengths=np.linalg.norm(vectors,axis=-1)
    direction=np.divide(vectors,lengths[...,None],out=np.full_like(vectors,np.nan),where=lengths[...,None]>1e-9)
    body=np.stack([(direction*anterior[...,None,:]).sum(axis=-1),(direction*left[...,None,:]).sum(axis=-1)],axis=-1)
    return body, np.concatenate([axis_length[...,None],lengths],axis=-1)


def prepare(block,run,all_tracks=False):
    """Cohort is determined from detection coverage, never spatial membership."""
    if block.parent.name!='20260724' or block.name!='block01':
        raise ValueError('This protocol is scoped to 20260724/block01')
    run.mkdir(parents=True,exist_ok=True)
    inventory=[];tasks=[]
    for i,p in enumerate(sorted((block/'stitched/per_track').glob('*.parquet'))):
        name=p.stem;side=name.rsplit('_',1)[1];tag=int(re.search(r'TrackID_(\d+)',name).group(1))
        sm_path=block/'stitched/speed_vectors/per_track'/name/'speed_metadata.json'
        sm=json.loads(sm_path.read_text());coverage=sm['n_observed_frames']/sm['n_frames']
        ant=f'{side}:{tag:03d}'
        row=dict(ant=ant,side=side,track_id=tag,track_name=p.name,detection_fraction=coverage,selected=all_tracks or coverage>.4,
                 source=stamp(p),speed_metadata=stamp(sm_path),n_frames=sm['n_frames'])
        inventory.append(row)
        if row['selected']:
            tasks.append(dict(**row,seed=724000+tag+(10000 if side=='right' else 0)))
    if len(inventory)!=114:
        raise ValueError(f'Expected 114 complete tracks; got {len(inventory)}')
    (run/'inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    (run/'tasks.json').write_text(json.dumps(tasks,indent=2)+'\n')
    print(json.dumps(dict(tracks=len(inventory),selected=len(tasks),sampled_clips_per_ant=2880)),flush=True)


def extract(task, output):
    import pyarrow.parquet as pq
    source=Path(task['source']['path'])
    if stamp(source)!=task['source']:
        raise ValueError(f'Track changed: {source}')
    code_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    signature=dict(task=task,code_sha256=code_hash,clip_frames=CLIP_FRAMES,fps=FPS,feature_version='posture_velocity_v2',sampling='one uniform random clip per minute for two clock-matched days')
    output.mkdir(parents=True,exist_ok=True)
    target=output/(task['ant'].replace(':','_')+'.npz')
    meta_path=target.with_suffix('.json')
    if target.exists() and meta_path.exists() and json.loads(meta_path.read_text()).get('signature')==signature:
        print('CACHE_HIT',task['ant'],flush=True);return
    starts=schedule(task['seed'])
    frames=(starts[:,None]+np.arange(CLIP_FRAMES)).ravel()
    parquet=pq.ParquetFile(source)
    n_frames=int(parquet.schema_arrow.metadata[b'num_frames'])
    # Speed arrays cover each ant's first-to-last observation span; parquet
    # metadata records the complete global recording clock used for sampling.
    if n_frames!=4247196:raise ValueError('Unexpected complete 0724 recording span')
    if frames[-1]>=n_frames:raise ValueError('Recording is shorter than two complete test days')
    lookup=np.full(n_frames,-1,dtype=np.int32);lookup[frames]=np.arange(len(frames),dtype=np.int32)
    sums=np.zeros((len(frames),10,2),np.float64)
    counts=np.zeros((len(frames),10),np.uint16)
    cameras=np.full(len(frames),np.nan,dtype=np.float32)
    positions=np.full((len(frames),2),np.nan,dtype=np.float64)
    position_cameras=np.full(len(frames),np.nan,dtype=np.float32)
    conflicting_camera=np.zeros(len(frames),bool)
    scanned=0
    for batch in parquet.iter_batches(batch_size=262144,columns=['Frame','Bodypoint','X','Y','SleapCam','TrackX','TrackY','CameraID'],use_threads=False):
        frame=batch.column(0).to_numpy(zero_copy_only=False);bp=batch.column(1).to_numpy(zero_copy_only=False)
        valid=(frame>=0)&(frame<n_frames)&(bp>=0)&(bp<10)
        selected=np.flatnonzero(valid)
        offsets=lookup[frame[selected]]
        keep=offsets>=0;selected=selected[keep];offsets=offsets[keep]
        x=batch.column(2).to_numpy(zero_copy_only=False)[selected];y=batch.column(3).to_numpy(zero_copy_only=False)[selected]
        node=bp[selected].astype(np.intp)
        finite=np.isfinite(x)&np.isfinite(y)
        np.add.at(sums[:,:,0],(offsets[finite],node[finite]),x[finite])
        np.add.at(sums[:,:,1],(offsets[finite],node[finite]),y[finite])
        np.add.at(counts,(offsets[finite],node[finite]),1)
        anchor=node==0
        offsets0=offsets[anchor];cam=batch.column(4).to_numpy(zero_copy_only=False)[selected[anchor]]
        conflict=np.isfinite(cameras[offsets0])&np.isfinite(cam)&(cameras[offsets0]!=cam)
        conflicting_camera[offsets0[conflict]]=True
        cameras[offsets0]=cam
        for j in (0,1):positions[offsets0,j]=batch.column(5+j).to_numpy(zero_copy_only=False)[selected[anchor]]
        position_cameras[offsets0]=batch.column(7).to_numpy(zero_copy_only=False)[selected[anchor]]
        scanned+=len(batch)
    xy=np.divide(sums,counts[...,None],out=np.full_like(sums,np.nan),where=counts[...,None]>0)
    duplicated=(counts>1).any(axis=1)
    xy[duplicated|conflicting_camera]=np.nan
    shape,lengths=intrinsic_directions(xy)
    velocity=body_velocity(xy.reshape(len(starts),CLIP_FRAMES,10,2),positions.reshape(len(starts),CLIP_FRAMES,2))
    shape=shape.reshape(len(starts),CLIP_FRAMES,16).astype(np.float32)
    lengths=(lengths.reshape(len(starts),CLIP_FRAMES,9)*MM_PER_PX).astype(np.float32)
    cameras=cameras.reshape(len(starts),CLIP_FRAMES)
    # No absolute positions or global orientations are written to this cache.
    np.savez_compressed(target,shape=shape,lengths_mm=lengths,frames=starts,cameras=cameras,
                        velocity_mm_s=velocity,position_cameras=position_cameras.reshape(len(starts),CLIP_FRAMES),
                        duplicate_frame=duplicated.reshape(len(starts),CLIP_FRAMES))
    if stamp(source)!=task['source']:raise ValueError('Input changed during extraction')
    complete=np.isfinite(shape).all(axis=(1,2))
    result=dict(signature=signature,scanned_rows=scanned,clips=len(starts),complete_clips=int(complete.sum()),
                complete_fraction=float(complete.mean()),duplicate_frames=int(duplicated.sum()),
                length_percentiles=np.nanquantile(lengths,[.01,.5,.99],axis=(0,1)).tolist(),
                velocity_percentiles_mm_s=np.nanquantile(velocity,[.001,.01,.5,.99,.999],axis=(0,1)).tolist(),
                output=stamp(target))
    meta_path.write_text(json.dumps(result,indent=2)+'\n')
    print('EXTRACTED',task['ant'],json.dumps({k:result[k] for k in ('scanned_rows','clips','complete_clips','duplicate_frames')}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--block',type=Path)
    p.add_argument('--prepare',type=Path)
    p.add_argument('--all-tracks',action='store_true',help='Audit all tracked identities; use hourly availability for inclusion')
    p.add_argument('--tasks',type=Path)
    p.add_argument('--task-index',type=int)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.prepare:prepare(a.block,a.prepare,a.all_tracks)
    else:extract(json.loads(a.tasks.read_text())[a.task_index],a.output)


if __name__=='__main__':main()
