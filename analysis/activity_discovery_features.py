"""Activity features without arena location, spatial classes or camera identity."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import warnings
import numpy as np
from analysis.postural_dynamics_extract import DAY1_FRAME, DAY_FRAMES, FPS, stamp

SPEED_NAMES=['log_mean_speed','log_median_speed','log_p90_speed','log_p99_speed',
             'fraction_above_005','fraction_above_02','fraction_above_1',
             'rest005_run_median','rest005_run_p90','active005_run_median',
             'fraction_rest005_runs30s','fraction_rest02_runs120s','switches_per_minute',
             'logspeed_acf1s','logspeed_acf10s','logspeed_acf60s']
VIEWS=('day1','split_a','split_b','day2')


def state_runs(speed,threshold):
    """Observed one-second runs, truncated at missing data and block edges."""
    state=np.where(np.isfinite(speed),(speed>threshold).astype(int),-1)
    boundaries=np.r_[0,np.flatnonzero(np.diff(state)!=0)+1,len(state)]
    return state[boundaries[:-1]],np.diff(boundaries)


def block_features(speed):
    valid=np.isfinite(speed);n=valid.sum()
    if n<180:return np.full(len(SPEED_NAMES),np.nan)
    v=speed[valid];log=np.log1p(speed/.1);values=[np.log1p(v.mean()/.1),*np.log1p(np.quantile(v,[.5,.9,.99])/.1),*(np.mean(v>t) for t in (.05,.2,1.))]
    states,durations=state_runs(speed,.05);rest=durations[states==0];active=durations[states==1]
    values += [np.log1p(np.median(rest)/30) if len(rest) else 0.,np.log1p(np.quantile(rest,.9)/30) if len(rest) else 0.,np.log1p(np.median(active)/30) if len(active) else 0.,float(rest[rest>=30].sum()/n)]
    st,du=state_runs(speed,.2);values.append(float(du[(st==0)&(du>=120)].sum()/n))
    pair=valid[1:]&valid[:-1];values.append(float((((speed[1:]>.05)!=(speed[:-1]>.05))&pair).sum()/(n/60)))
    for lag in (1,10,60):
        ok=valid[:-lag]&valid[lag:]
        if ok.sum()<20 or np.std(log[:-lag][ok])<1e-6 or np.std(log[lag:][ok])<1e-6:values.append(0.)
        else:values.append(float(np.corrcoef(log[:-lag][ok],log[lag:][ok])[0,1]))
    return np.asarray(values)


def hourly_from_blocks(blocks,seconds_count):
    """Four views with clock-matched hours; split on alternating five-minute blocks."""
    blocks=blocks.reshape(2,24,12,-1);seconds_count=seconds_count.reshape(2,24,12)
    result=[];coverage=[]
    for day,parity in ((0,None),(0,0),(0,1),(1,None)):
        choose=np.ones(12,bool) if parity is None else np.arange(12)%2==parity
        part=blocks[day][:,choose,:]
        count=seconds_count[day][:,choose].sum(axis=-1)
        minimum=3 if parity is None else 2
        keep=(count>=.5*300*choose.sum())&(np.isfinite(part).all(axis=-1).sum(axis=-1)>=minimum)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning);hourly=np.nanmean(part,axis=1)
        hourly[~keep]=np.nan;result.append(hourly);coverage.append(count/(300*choose.sum()))
    return np.asarray(result),np.asarray(coverage)


def extract_speed(task,block,output):
    name=Path(task['track_name']).stem;root=block/'stitched/speed_vectors/per_track'/name
    metadata=root/'speed_metadata.json';path=root/'speed_mm_s.npy';meta=json.loads(metadata.read_text())
    if meta['fps']!=24 or meta['mm_per_px']!=.016 or meta['bodypoint_filter']!=0:raise ValueError('Unexpected speed calibration')
    if (meta['max_interp_gap_frames'],meta['smooth_sigma_frames'],meta['max_speed_mm_s'])!=(5,2.,5.):raise ValueError('Unexpected speed estimator')
    output.mkdir(parents=True,exist_ok=True);target=output/(task['ant'].replace(':','_')+'.npz')
    signature=dict(sources=[stamp(path),stamp(metadata)],code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),ant=task['ant'])
    if target.exists() and target.with_suffix('.json').exists() and json.loads(target.with_suffix('.json').read_text()).get('signature')==signature:return
    data=np.load(path,mmap_mode='r');window=np.full(2*DAY_FRAMES,np.nan,np.float32)
    if len(data)!=meta['n_frames']:raise ValueError('Speed vector span mismatch')
    lo=max(DAY1_FRAME,meta['frame_min']);hi=min(DAY1_FRAME+2*DAY_FRAMES,meta['frame_max']+1)
    if hi>lo:window[lo-DAY1_FRAME:hi-DAY1_FRAME]=data[lo-meta['frame_min']:hi-meta['frame_min']]
    window[(window<0)|(window>meta['max_speed_mm_s'])]=np.nan
    frames=window.reshape(-1,FPS);n=np.isfinite(frames).sum(axis=1)
    second=np.divide(np.nansum(frames,axis=1),n,out=np.full(len(frames),np.nan),where=n>=18)
    pieces=second.reshape(-1,300);counts=np.isfinite(pieces).sum(axis=1)
    blocks=np.stack([block_features(piece) for piece in pieces]);hourly,coverage=hourly_from_blocks(blocks,counts)
    np.savez_compressed(target,hourly=hourly,coverage=coverage,blocks=blocks,valid_seconds=counts,feature_names=np.array(SPEED_NAMES))
    if [stamp(path),stamp(metadata)]!=signature['sources']:raise ValueError('Source changed during extraction')
    target.with_suffix('.json').write_text(json.dumps(dict(signature=signature,output=stamp(target)),indent=2)+'\n')
    print('SPEED_ACTIVITY',task['ant'],np.isfinite(hourly).all(axis=-1).sum(axis=-1).tolist(),flush=True)


def aggregate_clips(values,ids,minutes,n_ants,min_full=5,min_split=2):
    """Per-hour mean measured clip features, preserving A/B block separation."""
    out=np.full((4,n_ants,24,values.shape[-1]),np.nan);counts=np.zeros((4,n_ants,24),int)
    day=minutes//1440;hour=(minutes//60)%24;parity=(minutes//5)%2
    for view,(d,s) in enumerate(((0,None),(0,0),(0,1),(1,None))):
        valid=(day==d) if s is None else (day==d)&(parity==s)
        for ant in range(n_ants):
            for h in range(24):
                mask=valid&(ids==ant)&(hour==h);counts[view,ant,h]=mask.sum()
                if mask.sum()>=(min_full if s is None else min_split):out[view,ant,h]=values[mask].mean(axis=0)
    return out,counts


def hourly_quantiles(hourly):
    """Clock-independent distribution of hourly measurements, not pooled clips."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        return np.nanquantile(hourly,[.25,.5,.75],axis=2).transpose(1,2,0,3).reshape(hourly.shape[0],hourly.shape[1],-1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks',type=Path,required=True);p.add_argument('--task-index',type=int,required=True)
    p.add_argument('--block',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();extract_speed(json.loads(a.tasks.read_text())[a.task_index],a.block,a.output)

if __name__=='__main__':main()
