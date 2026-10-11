"""Full-recording body-aligned posture and exact simultaneous contacts.

Use LOCAL scratch for the large intermediate arrays (remote home quotas are
small). No sampled-hour restriction, cached sleep labels or spatial labels.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.signal import savgol_filter
from threadpoolctl import threadpool_limits

FPS, HZ, MM_PER_PX, SEED = 24, 12, .016, 7241012


def stamp(path):
    p=Path(path);s=p.stat()
    return dict(path=str(p),size=s.st_size,mtime_ns=s.st_mtime_ns)


def inventory(block,output,side='both'):
    rows=[]
    for path in sorted((block/'stitched/speed_vectors/per_track').glob('*/speed_metadata.json')):
        meta=json.loads(path.read_text());track=block/'stitched/per_track'/meta['track_name']
        total=int(pq.ParquetFile(track).schema_arrow.metadata[b'num_frames'])
        colony=track.stem.rsplit('_',1)[1];fraction=meta['n_observed_frames']/total
        rows.append(dict(ant=f'{colony}:{int(meta["track_id"]):03d}',side=colony,track_id=int(meta['track_id']),
            track_name=track.name,n_frames=total,detected_frames=meta['n_observed_frames'],detection_fraction=fraction,
            selected=fraction>.6 and (side=='both' or colony==side),**{'source_'+k:v for k,v in stamp(track).items()}))
    table=pd.DataFrame(rows).sort_values(['side','track_id']).reset_index(drop=True)
    assert not table.duplicated(['side','track_id']).any() and table.n_frames.nunique()==1
    output.mkdir(parents=True,exist_ok=True);table.to_csv(output/'inventory.csv',index=False)
    return table


def count_contacts(frames,a,b,ids,n_frames):
    """Deduplicate pair/frame hits; count both ants; never join contact gaps."""
    pairs=np.sort(np.column_stack([a,b]),axis=1)
    data=np.unique(np.column_stack([frames,pairs]),axis=0);data=data[data[:,1]!=data[:,2]]
    counts=np.zeros((len(ids),n_frames),np.uint8);lookup={int(ant):i for i,ant in enumerate(ids)}
    for node in (1,2):
        for ant in np.intersect1d(data[:,node],ids):
            np.add.at(counts[lookup[int(ant)]],data[data[:,node]==ant,0].astype(int),1)
    return counts


def contacts(block,output,ants):
    from analysis import grid_occupancy_utils as go
    from analysis.return_sleep_utils import resolve_interaction_root,load_published_interaction_chunks
    speed_tracks=go.load_speed_tracks(block/'stitched/speed_vectors')
    if not np.allclose(speed_tracks.fps,FPS):
        raise ValueError(f'Full-body extraction requires {FPS} fps')
    context=go.recording_context(block,speed_tracks)
    chunks,run=load_published_interaction_chunks(resolve_interaction_root(block),block/'tracks',fps=FPS,
        start_clock_seconds=context['start_clock_seconds'])
    root=output/'contacts';root.mkdir(exist_ok=True);n_frames=int(ants.n_frames.iloc[0])
    fingerprint=dict(run_id=run['run_id'],files=[stamp(c.interaction_path) for c in chunks],
        ants=ants.ant.tolist(),definition='unique concurrent partners, no temporal joining')
    meta=root/'manifest.json'
    if meta.exists() and json.loads(meta.read_text())==fingerprint:return
    arrays={r.ant:np.zeros(n_frames,np.uint8) for r in ants.itertuples()}
    covered={s:np.zeros(n_frames,bool) for s in ants.side.unique()}
    for c in chunks:
        if c.side not in covered:continue
        selected=ants[ants.side.eq(c.side)]
        d=pd.read_parquet(c.interaction_path,columns=['Frame','ant_a','ant_b'])
        local=count_contacts(d.Frame.to_numpy(),d.ant_a.to_numpy(),d.ant_b.to_numpy(),
            selected.track_id.to_numpy(),c.chunk_frame_count)
        lo,hi=c.chunk_global_frame_offset,c.chunk_global_frame_offset+c.chunk_frame_count
        assert hi<=n_frames and not covered[c.side][lo:hi].any()
        covered[c.side][lo:hi]=True
        for i,r in enumerate(selected.itertuples()):arrays[r.ant][lo:hi]=local[i]
    for ant,arr in arrays.items():np.save(root/(ant.replace(':','_')+'.npy'),arr)
    for side,arr in covered.items():np.save(root/(side+'_covered.npy'),arr)
    meta.write_text(json.dumps(fingerprint,indent=2)+'\n');print('Exact contacts ready',flush=True)


def align_body(xy,scale):
    """All ten points; tag origin, petiole->tag anterior, reflection retained."""
    axis=xy[:,0]-xy[:,2];length=np.linalg.norm(axis,axis=1)
    anterior=axis/np.maximum(length[:,None],1e-12);lateral=np.column_stack([-anterior[:,1],anterior[:,0]])
    relative=xy-xy[:,0:1]
    aligned=np.stack([(relative*anterior[:,None]).sum(axis=2),
        (relative*lateral[:,None]).sum(axis=2)],axis=2)*MM_PER_PX/scale
    good=np.isfinite(aligned).all(axis=(1,2))&(length*MM_PER_PX>=.1)&(length*MM_PER_PX<=1.25)
    good&=(np.linalg.norm(relative,axis=2)*MM_PER_PX<=5).all(axis=1)
    aligned[~good]=np.nan
    return aligned,anterior,good


def extract_one(task):
    block,output,row=task;threadpool_limits(limits=1)
    root=output/'sequences'/row['ant'].replace(':','_');root.mkdir(parents=True,exist_ok=True)
    source=block/'stitched/per_track'/row['track_name']
    signature=dict(source=stamp(source),measurement_version='full_body_v1')
    meta=root/'metadata.json'
    if meta.exists() and meta.stat().st_size and json.loads(meta.read_text())['signature']==signature:
        return dict(ant=row['ant'],cached=True)
    n=int(row['n_frames']);xy=np.full((n,10,2),np.nan,np.float32)
    position=np.full((n,2),np.nan,np.float32);cam=np.full(n,-1,np.int16);pcam=cam.copy()
    seen=np.zeros((n,10),np.uint8)
    columns=['Frame','Bodypoint','X','Y','SleapCam','TrackX','TrackY','CameraID']
    for batch in pq.ParquetFile(source).iter_batches(batch_size=262144,columns=columns,use_threads=False):
        fr,bp=[batch.column(i).to_numpy(zero_copy_only=False) for i in (0,1)]
        selected=np.flatnonzero((fr>=0)&(fr<n)&(bp>=0)&(bp<10))
        f,p=fr[selected].astype(int),bp[selected].astype(int);np.add.at(seen,(f,p),1)
        for j in (0,1):xy[f,p,j]=batch.column(2+j).to_numpy(zero_copy_only=False)[selected]
        anchor=p==0;at=selected[anchor];f=f[anchor]
        for j in (0,1):position[f,j]=batch.column(5+j).to_numpy(zero_copy_only=False)[at]
        for column,target in ((4,cam),(7,pcam)):
            values=batch.column(column).to_numpy(zero_copy_only=False)[at]
            target[f]=np.where(np.isfinite(values),values,-1).astype(np.int16)
    duplicate=(seen>1).any(axis=1);xy[duplicate]=np.nan;cam[duplicate]=-1;del seen
    lengths=np.linalg.norm(xy[:,0]-xy[:,2],axis=1)*MM_PER_PX
    scale=float(np.nanmedian(lengths[(lengths>=.1)&(lengths<=1.25)]));m=(n+1)//2
    pose=np.full((m,20),np.nan,np.float32);motion=np.full((m,5),np.nan,np.float32);cameras=np.full(m,-1,np.int16)
    count=0;first=np.zeros(20);second=np.zeros((20,20))
    for lo in range(2,n-2,120000):
        hi=min(n-2,lo+120000);centers=np.arange(lo,hi,2);ix=centers[:,None]+np.arange(-2,3)
        shape,anterior,valid=align_body(xy[lo-2:hi+2],scale);offsets=centers-lo+2
        # Only centered five-frame windows are retained. Padding is never
        # sampled; avoid fitting unused edge polynomials to missing poses.
        aligned=savgol_filter(shape,5,2,axis=0,mode='constant',cval=np.nan)[offsets]
        derivative=savgol_filter(shape,5,2,deriv=1,delta=1/FPS,axis=0,mode='constant',cval=np.nan)[offsets]*scale
        good=valid[ix-(lo-2)].all(axis=1)&(cam[ix]>=0).all(axis=1)&(np.ptp(cam[ix],axis=1)==0)
        velocity=((position[ix]-position[ix[:,2:3]])*np.arange(-2,3)[None,:,None]).sum(axis=1)*(FPS*MM_PER_PX/10)
        orientation=anterior[offsets];lateral=np.column_stack([-orientation[:,1],orientation[:,0]])
        body=np.abs(np.column_stack([(velocity*orientation).sum(axis=1),(velocity*lateral).sum(axis=1)]))
        antenna=np.linalg.norm(derivative[:,4:10]-derivative[:,1:2],axis=2).reshape(-1,2,3).mean(axis=2)
        directions=anterior[ix-(lo-2)]
        delta=np.arctan2(directions[:,4,1]*directions[:,0,0]-directions[:,4,0]*directions[:,0,1],
            (directions[:,4]*directions[:,0]).sum(axis=1))
        values=np.column_stack([body,antenna,np.abs(delta)*FPS/4])
        motion_good=good&(pcam[ix]==cam[ix]).all(axis=1)&np.isfinite(values).all(axis=1)&(np.linalg.norm(velocity,axis=1)<=20)
        aligned[~good]=np.nan;values[~motion_good]=np.nan;indices=centers//2
        pose[indices]=aligned.reshape(-1,20);motion[indices]=values;cameras[indices]=np.where(good,cam[centers],-1)
        data=aligned[good].reshape(-1,20).astype(float);count+=len(data);first+=data.sum(axis=0);second+=data.T@data
    for name,values in (('pose',pose),('motion',motion),('camera',cameras)):np.save(root/(name+'.npy'),values)
    assert stamp(source)==signature['source']
    np.savez(root/'moments.npz',count=count,first=first,second=second)
    result=dict(signature=signature,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),ant=row['ant'],
        body_scale_mm=scale,valid_pose_samples=count,valid_motion_samples=int(np.isfinite(motion).all(axis=1).sum()),
        total_samples=m,fps=FPS,sampling_hz=HZ,landmarks='tag, occiput, petiole, gaster tip, six antennal points')
    meta.write_text(json.dumps(result,indent=2)+'\n')
    return {k:result[k] for k in ('ant','valid_pose_samples','valid_motion_samples','total_samples')}


def fit_pca(output,ants):
    count=0;first=np.zeros(20);second=np.zeros((20,20))
    for ant in ants.ant:
        with np.load(output/'sequences'/ant.replace(':','_')/'moments.npz') as z:
            count+=int(z['count']);first+=z['first'];second+=z['second']
    center=first/count;cov=second/count-np.outer(center,center)
    eigenvalues,modes=np.linalg.eigh(cov);order=np.argsort(eigenvalues)[::-1]
    eigenvalues=np.maximum(eigenvalues[order],0);modes=modes[:,order].T
    for mode in modes:
        if mode[np.argmax(np.abs(mode))]<0:mode*=-1
    np.savez(output/'full_body_pca.npz',center=center,modes=modes,eigenvalues=eigenvalues,
        variance=eigenvalues/eigenvalues.sum(),n_samples=count)
    print('PCA full recording:',count,'first 8 variance',eigenvalues[:8].sum()/eigenvalues.sum(),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--block',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--side',choices=['both','left','right'],default='both');a=p.parse_args()
    table=inventory(a.block,a.output,a.side);ants=table[table.selected]
    print('Selected',ants.groupby('side').size().to_dict(),flush=True);contacts(a.block,a.output,ants)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for r in pool.map(extract_one,[(a.block,a.output,row) for row in ants.to_dict('records')]):print(json.dumps(r),flush=True)
    fit_pca(a.output,ants)


if __name__=='__main__':main()
