"""Removable short-recording follow-up: movement allocation and contact timing.

No inferred nest/resource labels and no assumption that tags identify the same
individual across recordings. All outputs are confined to the requested output.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.compute_track_speed_vector import load_track_xy, speed_vector, track_id_from_name


def mean(a, axis=None):
    n = np.isfinite(a).sum(axis=axis)
    return np.divide(np.nansum(a, axis=axis), n, out=np.full(np.shape(n), np.nan), where=n > 0)


def save(fig, output, name):
    fig.tight_layout()
    fig.savefig(output / (name + ".png"), dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def read_ant(path, nframes, fps):
    xy = load_track_xy(path, frame_col="Frame", x_col="TrackX", y_col="TrackY", bodypoint=0)
    frames = xy.Frame.to_numpy(int)
    if frames.max() >= nframes:
        raise ValueError(f"Track exceeds audited recording duration: {path}")
    x = np.full(nframes, np.nan); y = x.copy()
    x[frames] = xy.X; y[frames] = xy.Y
    raw = speed_vector(x, y, fps=fps, mm_per_px=1., max_interp_gap_frames=5,
                       smooth_sigma_frames=2., max_speed_mm_s=312.5)
    width = int(fps * 5); nbins = int(np.ceil(nframes / width))
    padded = np.full(nbins * width, np.nan); padded[:nframes] = raw
    blocks = padded.reshape(-1, width)
    coverage = np.isfinite(blocks).mean(axis=1)
    s = mean(blocks, axis=1); s[coverage < .25] = np.nan
    observed = np.full(nbins * width, False); observed[frames] = True
    detection = observed.reshape(-1, width).mean(axis=1)
    side = "left" if "_left" in path.name else "right"
    ant = track_id_from_name(path)
    p = xy[xy.Frame % int(fps) == 0].rename(columns={"Frame":"frame", "X":"x_px", "Y":"y_px"}).copy()
    p["side"] = side; p["track_id"] = ant; p["track_name"] = path.name
    row = {"track_name":path.name, "track_id":ant, "side":side, "detection_fraction":len(xy)/nframes,
           "mean_speed_px_s":float(mean(s)), "frame_min":int(frames.min()), "frame_max":int(frames.max())}
    return row, s, coverage, detection, p


def load(dataset, output):
    diag = json.loads(next(dataset.glob("cam01*.diag.json")).read_text())
    fps = float(diag["context"]["fps"])
    nframes = int(diag["capture"]["framesEmitted"])
    paths = sorted((dataset / "stitched/per_track").glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No per-track trajectories under {dataset}")
    fingerprint = {"version":1, "fps":fps, "nframes":nframes,
                   "tracks":[[str(p), p.stat().st_mtime_ns] for p in paths]}
    settings = output / "cache_metadata.json"
    if settings.exists() and json.loads(settings.read_text()) == fingerprint:
        tracks = pd.read_csv(output / "track_audit.csv")
        arrays = np.load(output / "sampled_tracks.npz")
        return tracks, arrays["speed"], arrays["coverage"], arrays["detection"], pd.read_parquet(output / "positions_1hz.parquet"), fps, nframes
    data = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks = [pool.submit(read_ant, p, nframes, fps) for p in paths]
        for k, task in enumerate(as_completed(tasks), 1):
            result = task.result(); data[result[0]["track_name"]] = result
            if k % 20 == 0 or k == len(paths):
                print(f"Read {k}/{len(paths)} tracks", flush=True)
    ordered = [data[p.name] for p in paths]
    tracks = pd.DataFrame([x[0] for x in ordered])
    speed, coverage, detection = [np.stack([x[i] for x in ordered]) for i in (1,2,3)]
    positions = pd.concat([x[4] for x in ordered], ignore_index=True)
    tracks.to_csv(output / "track_audit.csv", index=False)
    np.savez_compressed(output / "sampled_tracks.npz", speed=speed, coverage=coverage, detection=detection)
    positions.to_parquet(output / "positions_1hz.parquet", index=False)
    settings.write_text(json.dumps(fingerprint, indent=2))
    return tracks, speed, coverage, detection, positions, fps, nframes


def allocation(tracks, speed, coverage, detection, positions, output, fps, nframes, rng):
    results = {}; epoch_rows = []
    minutes = (np.arange(speed.shape[1]) + .5) / 12
    duration = nframes / fps / 60
    fig, axes = plt.subplots(3,2,figsize=(13,10))
    detail, dax = plt.subplots(2,2,figsize=(12,9))
    for col, side in enumerate(["left", "right"]):
        idx = np.flatnonzero(tracks.side == side); s = speed[idx]
        a = mean(s[:,minutes < duration/2],axis=1); b = mean(s[:,minutes >= duration/2],axis=1)
        good = (np.isfinite(s[:,minutes < duration/2]).mean(axis=1) >= .7) & (np.isfinite(s[:,minutes >= duration/2]).mean(axis=1) >= .7)
        order = np.argsort(a)
        im = axes[0,col].imshow(s[order],aspect="auto",extent=[0,duration,len(idx),0],vmin=0,vmax=100,cmap="magma")
        axes[0,col].set_title(f"{side}: {len(idx)} ants, ordered by first-half speed")
        axes[0,col].set_ylabel("Ant rank"); fig.colorbar(im,ax=axes[0,col],label="Speed (px/s)")
        axes[1,col].plot(minutes,mean(s,axis=0),lw=.6)
        axes[1,col].set_ylabel("Colony mean speed (px/s)")
        axes[2,col].plot(minutes,mean(detection[idx],axis=0),label="Raw position detection")
        axes[2,col].plot(minutes,mean(coverage[idx],axis=0),label="Valid speed",alpha=.7)
        axes[2,col].set_ylim(0,1); axes[2,col].legend(fontsize=8)
        axes[2,col].set_xlabel("Minutes since recording start")
        for ant,x,y in zip(tracks.iloc[idx[good]].track_id,a[good],b[good]):
            dax[0,col].scatter(x,y,s=20,color="C0"); dax[0,col].annotate(str(ant),(x,y),fontsize=6)
        rho = float(spearmanr(a[good],b[good]).statistic)
        dax[0,col].set_title(f"{side}: first/second-half rank rho={rho:.2f}, n={good.sum()}")
        dax[0,col].set_xlabel("First-half speed (px/s)"); dax[0,col].set_ylabel("Second-half speed (px/s)")
        # Fixed cohort and early-defined groups for later investment comparisons.
        baseline = mean(s[:,minutes < 20],axis=1)
        adequate = np.isfinite(s[:,minutes < 20]).mean(axis=1) >= .7
        high = baseline > np.nanmedian(baseline[adequate])
        for lo in np.arange(20, duration - 9, 10):
            mask = (minutes >= lo) & (minutes < lo+10)
            values = mean(s[:,mask],axis=1)
            ok = adequate & (np.isfinite(s[:,mask]).mean(axis=1) >= .7)
            for i in np.flatnonzero(ok):
                epoch_rows.append({"side":side,"track_id":int(tracks.iloc[idx[i]].track_id),"start_min":lo,
                                   "baseline_speed":baseline[i],"epoch_speed":values[i],"earlier_high_half":bool(high[i])})
        p = positions[positions.side == side]
        bins = [np.linspace(*np.quantile(p.x_px,[.001,.999]),35),np.linspace(*np.quantile(p.y_px,[.001,.999]),45)]
        early=[];late=[]
        for name, ant in p.groupby("track_name"):
            halves=[]
            for keep in [ant.frame < nframes/2, ant.frame >= nframes/2]:
                q=ant[keep]
                if len(q) < 600: break
                h=np.histogram2d(q.x_px,q.y_px,bins=bins)[0].ravel()+1e-12
                halves.append(h/h.sum())
            if len(halves)==2: early.append(halves[0]);late.append(halves[1])
        distances=np.array([[jensenshannon(a,b,base=2) for b in late] for a in early])
        same=np.diag(distances);other=distances[~np.eye(len(early),dtype=bool)]
        null=[distances[np.arange(len(early)),rng.permutation(len(early))].mean() for _ in range(2000)]
        dax[1,col].hist(other,bins=25,density=True,color=".8",label="Different ants")
        dax[1,col].hist(same,bins=15,density=True,alpha=.7,label="Same ant")
        dax[1,col].set_xlabel("First/second-half spatial-profile distance");dax[1,col].legend()
        results[side]={"selected_ants":len(idx),"matched_ants":int(good.sum()),"movement_rank_rho":rho,
                       "mean_first":float(a[good].mean()),"mean_second":float(b[good].mean()),
                       "spatial_same":float(same.mean()),"spatial_other":float(other.mean()),
                       "spatial_identity_p":float((1+np.sum(np.array(null)<=same.mean()))/2001)}
    save(fig,output,"01_activity_and_detection");save(detail,output,"02_individual_persistence")
    pd.DataFrame(epoch_rows).to_csv(output / "ant_epochs.csv",index=False)
    return results


def contacts(dataset, tracks, speed, positions, output, fps, rng):
    """Contact-associated changes with same-ant, time/space/history controls.

    Contacts are detector candidates, not manually verified social interactions.
    Analyze contact to an ant's body; antenna/body labels are geometric roles.
    """
    results={}; matched_rows=[]; event_rows=[]
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    lags=np.arange(-12,25); pre=(lags>=-6)&(lags<=-2); post=(lags>=1)&(lags<=12)
    for col,side in enumerate(["left","right"]):
        idx=np.flatnonzero(tracks.side==side); chosen=tracks.iloc[idx]; s=speed[idx]
        id_to_i={int(a):i for i,a in enumerate(chosen.track_id)}
        nbins=s.shape[1]; contact=np.zeros(s.shape,dtype=bool)
        path=next((dataset/"interactions").glob(f"*_{side}.parquet"))
        df=pd.read_parquet(path)
        # Collapse repeated frames/body contacts to a five-second presence flag.
        pairs=df[(df.antenna_track_id!=df.body_track_id)&df.body_track_id.isin(id_to_i)].copy()
        ri=pairs.body_track_id.map(id_to_i).to_numpy(int)
        ci=(pairs.Frame.to_numpy(int)//int(fps*5))
        good=(ci>=0)&(ci<nbins);contact[ri[good],ci[good]]=True
        p=positions[(positions.side==side)&(positions.frame%int(fps*5)==0)]
        x=np.full(s.shape,np.nan);y=x.copy()
        ri=p.track_id.map(id_to_i).to_numpy(int);ci=(p.frame.to_numpy(int)//int(fps*5))
        x[ri,ci]=p.x_px;y[ri,ci]=p.y_px
        distance2=(x[:,None,:]-x[None,:,:])**2+(y[:,None,:]-y[None,:,:])**2
        neighbors=((distance2<150**2)&(distance2>0)).sum(axis=1).astype(float)
        neighbors[~np.isfinite(x)]=np.nan
        curves=[];controls=[]
        n_candidates=0
        for i,ant in enumerate(chosen.track_id):
            centers=np.arange(12,nbins-24)
            windows=s[i,centers[:,None]+lags]
            adequate=(np.isfinite(windows[:,pre]).mean(axis=1)>=.8)&(np.isfinite(windows[:,post]).mean(axis=1)>=.8)
            baseline=mean(windows[:,pre],axis=1)
            quiet=(baseline<10)&adequate
            no_recent=(contact[i,centers[:,None]+np.arange(-6,0)].sum(axis=1)==0)
            eligible=quiet&no_recent&np.isfinite(x[i,centers])
            events=np.flatnonzero(eligible&contact[i,centers])
            available=np.flatnonzero(eligible&~contact[i,centers])
            n_candidates+=len(events)
            used=set();last_event=-1000
            for e in events:
                ce=centers[e]
                # At least two minutes between accepted events for one ant.
                if ce-last_event<24: continue
                space=np.hypot(x[i,centers[available]]-x[i,ce],y[i,centers[available]]-y[i,ce])
                lag=np.abs(centers[available]-ce)
                balance=np.abs(baseline[available]-baseline[e])
                density=np.abs(neighbors[i,centers[available]]-neighbors[i,ce])
                allowed=(lag>=24)&(lag<=180)&(space<=300)&(balance<=3)&(density<=2)
                allowed&=np.array([all(abs(int(centers[a])-u)>=24 for u in used) for a in available])
                candidates=np.flatnonzero(allowed)
                if not len(candidates): continue
                # Match only on past state, position, density, and clock time.
                score=balance[candidates]/3+space[candidates]/300+density[candidates]/2+lag[candidates]/180
                c=available[candidates[np.argmin(score)]];cc=centers[c]
                used.add(int(cc));last_event=ce
                actual=windows[e];control=windows[c]
                delta=float(mean(actual[post])-baseline[e]);control_delta=float(mean(control[post])-baseline[c])
                row={"side":side,"track_id":int(ant),"event_seconds":ce*5,"control_seconds":cc*5,
                     "baseline":baseline[e],"control_baseline":baseline[c],"delta":delta,"control_delta":control_delta,
                     "matched_difference":delta-control_delta,"space_distance_px":float(np.hypot(x[i,ce]-x[i,cc],y[i,ce]-y[i,cc])),
                     "neighbors":neighbors[i,ce],"control_neighbors":neighbors[i,cc]}
                matched_rows.append(row);curves.append(actual-baseline[e]);controls.append(control-baseline[c])
        side_matches=pd.DataFrame([r for r in matched_rows if r['side']==side])
        if len(side_matches):
            ant_means=side_matches.groupby('track_id').matched_difference.mean()
            vals=ant_means.to_numpy();boot=vals[rng.integers(len(vals),size=(2000,len(vals)))].mean(axis=1)
            axes[0,col].plot(lags*5,mean(np.array(curves),axis=0),label="Body-contact onset")
            axes[0,col].plot(lags*5,mean(np.array(controls),axis=0),label="Same-ant matched non-contact time")
            axes[0,col].axvspan(0,5,color='.8',alpha=.5);axes[0,col].axhline(0,color='.6',lw=.6)
            axes[0,col].legend(fontsize=8);axes[0,col].set_title(f"{side}: {len(side_matches)} matched events, {len(vals)} ants")
            axes[0,col].set_xlabel("Seconds relative to five-second contact bin")
            axes[0,col].set_ylabel("Change from pre-event speed (px/s)")
            axes[1,col].scatter(side_matches.control_delta,side_matches.delta,s=15,alpha=.6)
            low=min(side_matches[['delta','control_delta']].min());high=max(side_matches[['delta','control_delta']].max())
            axes[1,col].plot([low,high],[low,high],'--',color='.5')
            axes[1,col].set_xlabel("Speed change at matched control (px/s)");axes[1,col].set_ylabel("Speed change after body contact (px/s)")
            axes[1,col].set_title(f"Ant-weighted excess change: {vals.mean():.1f} px/s")
            results[side]={"candidate_events":n_candidates,"matched_events":len(side_matches),"matched_ants":len(vals),
                           "ant_weighted_excess_delta":float(vals.mean()),"ant_bootstrap_95ci":np.quantile(boot,[.025,.975]).tolist(),
                           "median_space_match_px":float(side_matches.space_distance_px.median()),
                           "mean_baseline_difference":float((side_matches.baseline-side_matches.control_baseline).mean()),
                           "event_mean_delta":float(side_matches.delta.mean()),"control_mean_delta":float(side_matches.control_delta.mean())}
        else:
            axes[0,col].text(.5,.5,"No adequately matched events",ha='center');results[side]={"candidate_events":n_candidates,"matched_events":0}
        print('Contact results',side,results[side],flush=True)
    pd.DataFrame(matched_rows).to_csv(output/'matched_contact_events.csv',index=False)
    fig.suptitle("Do contacts precede movement in previously quiet ants? Observational, not causal")
    save(fig,output,"03_contact_associated_activation")
    return results


def workforce(tracks,speed,output,rng):
    """Hold cohort fixed across epochs and define groups before comparison."""
    results={};fig,axes=plt.subplots(2,2,figsize=(12,8))
    rows=[]
    masks=[(np.arange(speed.shape[1])>=a*12)&(np.arange(speed.shape[1])<b*12) for a,b in [(0,20),(20,40),(40,60),(60,80)]]
    for col,side in enumerate(['left','right']):
        idx=np.flatnonzero(tracks.side==side);s=speed[idx]
        ok=np.array([np.isfinite(s[:,m]).mean(axis=1)>=.7 for m in masks]).all(axis=0)
        m=np.array([mean(s[ok][:,mask],axis=1) for mask in masks]).T
        ids=tracks.iloc[idx[ok]].track_id.to_numpy()
        high=m[:,0]>np.median(m[:,0]);shares=m.sum(axis=0)
        higher_share=m[high].sum(axis=0)/shares
        k=int(np.ceil(len(m)*.2));early_top=set(np.argsort(m[:,0])[-k:])
        late_top=set(np.argsort(m[:,-1])[-k:]);overlap=len(early_top&late_top)/k
        axes[0,col].plot([10,30,50,70],m[high].mean(axis=0),'o-',label='Earlier more-active half')
        axes[0,col].plot([10,30,50,70],m[~high].mean(axis=0),'o-',label='Earlier less-active half')
        axes[0,col].set_title(f'{side}: fixed cohort of {len(m)} ants');axes[0,col].legend(fontsize=8)
        axes[0,col].set_xlabel('Minutes since recording start');axes[0,col].set_ylabel('Mean speed (px/s)')
        im=axes[1,col].imshow((m/m.sum(axis=0))[np.argsort(m[:,0])],aspect='auto',cmap='viridis')
        axes[1,col].set_xticks(range(4),['0–20','20–40','40–60','60–80'])
        axes[1,col].set_xlabel('Minutes since recording start');axes[1,col].set_ylabel('Ants ordered by first-epoch investment')
        fig.colorbar(im,ax=axes[1,col],label='Share of cohort movement')
        results[side]={'fixed_cohort_ants':len(m),'higher_half_share_by_epoch':higher_share.tolist(),
                       'top20_first_last_overlap':overlap,'early_late_rho':float(spearmanr(m[:,0],m[:,-1]).statistic),
                       'mean_speed_by_epoch':m.mean(axis=0).tolist()}
        for i,ant in enumerate(ids):
            for j in range(4):rows.append({'side':side,'track_id':int(ant),'epoch':j,'mean_speed':m[i,j],'movement_share':m[i,j]/shares[j],'earlier_high_half':bool(high[i])})
    pd.DataFrame(rows).to_csv(output/'fixed_cohort_allocation.csv',index=False)
    fig.suptitle('Changing colony movement: does the same workforce continue to contribute?')
    save(fig,output,'04_workforce_allocation')
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    output=args.output or args.dataset / "analysis_outputs/collective_task_probe_20260908"
    output.mkdir(parents=True,exist_ok=True)
    tracks,speed,coverage,detection,positions,fps,nframes=load(args.dataset,output)
    choose=tracks.detection_fraction>=.4
    speed=speed[choose];coverage=coverage[choose];detection=detection[choose]
    tracks=tracks[choose].reset_index(drop=True)
    positions=positions[positions.track_name.isin(tracks.track_name)]
    tracks.to_csv(output / "selected_ants.csv",index=False)
    results={"dataset":str(args.dataset),"minutes":nframes/fps/60,"fps":fps,"scope":"Within-recording movement and contact proxies; no validated resource/task labels"}
    results["allocation"]=allocation(tracks,speed,coverage,detection,positions,output,fps,nframes,np.random.default_rng(20260908))
    results["workforce"]=workforce(tracks,speed,output,np.random.default_rng(20260908))
    results["contacts"]=contacts(args.dataset,tracks,speed,positions,output,fps,np.random.default_rng(20260908))
    (output / "results.json").write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2),flush=True)


if __name__=="__main__": main()
