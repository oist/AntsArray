"""Removable May 15 block02 follow-up: allocation across days and trip history.

One-Hz position sampling uses the current panorama polygons, all sufficiently
observed ants, and missing-aware denominators. Original analysis outputs remain
untouched. A region visit is not a demonstrated feeding event.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from scipy.stats import spearmanr

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from analysis.grid_occupancy_utils import load_panorama_regions, panorama_regions_path
from analysis.trip_phenotyping_utils import _bridge_short_state_gaps, _remove_short_state_flicker, _run_length_encoding


def save(fig,output,name):
    fig.tight_layout();fig.savefig(output/(name+'.png'),dpi=170,bbox_inches='tight',facecolor='white');plt.close(fig)


def ci_mean(x,rng):
    x=np.asarray(x);x=x[np.isfinite(x)]
    return np.quantile(x[rng.integers(len(x),size=(2000,len(x)))].mean(axis=1),[.025,.975]).tolist() if len(x) else [np.nan,np.nan]


def observed_mean(values,axis=None):
    counts=np.isfinite(values).sum(axis=axis)
    return np.divide(np.nansum(values,axis=axis),counts,out=np.full(np.shape(counts),np.nan),where=counts>0)


def scan_ant(row,regions,dataset,nseconds,fps):
    x=np.full(nseconds,np.nan,dtype='float32');y=x.copy()
    path=dataset/'stitched/per_track'/row.track_name
    scanner=ds.dataset(path).scanner(columns=['Frame','TrackX','TrackY'],filter=ds.field('Bodypoint')==0,batch_size=262144,use_threads=False)
    for batch in scanner.to_batches():
        frames=batch.column(0).to_numpy();keep=frames%int(fps)==0
        t=(frames[keep]//int(fps)).astype(int)
        x[t]=batch.column(1).to_numpy()[keep];y[t]=batch.column(2).to_numpy()[keep]
    observed=np.isfinite(x)&np.isfinite(y)
    inside=np.zeros(nseconds,dtype=bool);food=inside.copy();water=inside.copy()
    for region in regions[regions.side==row.side].itertuples():
        if region.shape=='rectangle':
            hit=(x>=region.tracking_x_min_px)&(x<=region.tracking_x_max_px)&(y>=region.tracking_y_min_px)&(y<=region.tracking_y_max_px)
        else:
            hit=(x-region.tracking_center_x_px)**2+(y-region.tracking_center_y_px)**2<=region.radius_px**2
        if region.region_type=='colony':inside|=hit
        elif region.region_type=='food':food|=hit
        elif region.region_type=='water':water|=hit
    state=np.full(nseconds,-1,dtype='int8');state[observed]=inside[observed]
    return state,food,water


def load(dataset,output):
    rows=[]
    for p in (dataset/'stitched/speed_vectors').rglob('speed_metadata.json'):
        row=json.loads(p.read_text());row['side']='left' if '_left' in row['track_name'] else 'right';rows.append(row)
    table=pd.DataFrame(rows);nframes=int(table.frame_max.max())+1;fps=float(table.fps.median())
    table['recording_detection']=table.n_observed_frames/nframes
    tracks=table[table.recording_detection>=.4].sort_values(['side','track_id','track_name']).reset_index(drop=True)
    if tracks.duplicated(['side','track_id']).any():raise ValueError('Duplicate sufficiently observed identities require resolution')
    clock=re.search(r'_all_(\d{6})_',tracks.track_name.iloc[0])[1]
    start=int(clock[:2])*3600+int(clock[2:4])*60+int(clock[4:])
    region_path=panorama_regions_path(dataset)
    nseconds=int(np.ceil(nframes/fps));regions=load_panorama_regions(region_path)
    fingerprint={'version':1,'nseconds':nseconds,'fps':fps,'regions':region_path.read_text(),
                 'sources':[[r.track_name,(dataset/'stitched/per_track'/r.track_name).stat().st_mtime_ns] for r in tracks.itertuples()]}
    settings=output/'cache_metadata.json';cache=output/'states_1hz.npz'
    if settings.exists() and cache.exists() and json.loads(settings.read_text())==fingerprint:
        with np.load(cache) as a:state=a['state'];food=a['food'];water=a['water']
    else:
        data={}
        with ThreadPoolExecutor(max_workers=4) as pool:
            jobs={pool.submit(scan_ant,r,regions,dataset,nseconds,fps):r.track_name for r in tracks.itertuples()}
            for k,job in enumerate(as_completed(jobs),1):
                data[jobs[job]]=job.result()
                if k%8==0 or k==len(jobs):print(f'Position sampling: {k}/{len(jobs)} ants',flush=True)
        state,food,water=[np.stack([data[n][i] for n in tracks.track_name]) for i in range(3)]
        np.savez_compressed(cache,state=state,food=food,water=water);settings.write_text(json.dumps(fingerprint,indent=2))
    table.to_csv(output/'track_audit.csv',index=False);tracks.to_csv(output/'selected_ants.csv',index=False);regions.to_csv(output/'resolved_regions.csv',index=False)
    return tracks,state,food,water,start,fps


def extract_trips(tracks,state,food,water,start,max_gap=30,min_seconds=30):
    rows=[]
    for i,ant in enumerate(tracks.itertuples()):
        smooth=_remove_short_state_flicker(_bridge_short_state_gaps(state[i],max_gap),5)
        starts,ends,values=_run_length_encoding(smooth)
        for j in range(1,len(values)):
            a=int(starts[j]);b=int(ends[j]+1)
            if values[j]!=0 or b-a<min_seconds or values[j-1]!=1 or ends[j-1]-starts[j-1]+1<5:continue
            complete=j+1<len(values) and values[j+1]==1 and ends[j+1]-starts[j+1]+1>=5
            observed=state[i,a:b]>=0;coverage=observed.mean()
            if coverage<.5:continue
            rows.append({'side':ant.side,'track_id':ant.track_id,'track_name':ant.track_name,'exit_seconds':a,'return_seconds':b,
                         'complete':bool(complete),'duration_seconds':b-a,'coverage':coverage,
                         'food_seconds':int(food[i,a:b].sum()),'water_seconds':int(water[i,a:b].sum()),
                         'observed_seconds':int(observed.sum()),'clock_hour':((start+a)%86400)/3600,'day':int((start+a)//86400)})
    trips=pd.DataFrame(rows)
    trips['resource_visit']=(trips.food_seconds+trips.water_seconds)>=5
    trips['food_visit']=trips.food_seconds>=5;trips['water_visit']=trips.water_seconds>=5
    return trips.sort_values(['side','track_id','exit_seconds']).reset_index(drop=True)


def make_tables(tracks,state,food,water,trips,start):
    clock=start+np.arange(state.shape[1]);day=clock//86400;hour=clock//3600
    daily=[];hourly=[]
    for i,ant in enumerate(tracks.itertuples()):
        for d in sorted(np.unique(day)):
            mask=day==d;valid=state[i,mask]>=0;n=int(valid.sum());out=int((state[i,mask]==0).sum())
            tt=trips[(trips.track_name==ant.track_name)&(trips.day==d)]
            complete=tt[tt.complete]
            daily.append({'side':ant.side,'track_id':ant.track_id,'track_name':ant.track_name,'day':int(d),
                          'available_seconds':int(mask.sum()),'observed_seconds':n,'outside_seconds':out,'outside_fraction':out/n if n else np.nan,
                          'food_seconds':int(food[i,mask].sum()),'water_seconds':int(water[i,mask].sum()),
                          'departures':len(tt),'trip_rate_per_observed_day':len(tt)*86400/n if n else np.nan,
                          'completed_trips':len(complete),'mean_trip_seconds':complete.duration_seconds.mean(),
                          'resource_trip_fraction':complete.resource_visit.mean()})
        for h in sorted(np.unique(hour)):
            mask=hour==h;valid=state[i,mask]>=0;n=int(valid.sum())
            hourly.append({'side':ant.side,'track_id':ant.track_id,'track_name':ant.track_name,'day':int(h//24),'hour':int(h%24),
                           'clock_hour_continuous':int(h),'available_seconds':int(mask.sum()),'observed_seconds':n,
                           'outside_seconds':int((state[i,mask]==0).sum()),'resource_seconds':int((food[i,mask]|water[i,mask]).sum())})
    return pd.DataFrame(daily),pd.DataFrame(hourly)


def daily_allocation(daily,hourly,output,rng):
    results={};fig,axes=plt.subplots(2,3,figsize=(15,9));heat,haxes=plt.subplots(1,2,figsize=(13,9))
    features=[('outside_fraction','Fraction observed outside'),('trip_rate_per_observed_day','Departures / observed day'),('mean_trip_seconds','Mean completed-trip duration (s)')]
    for row,side in enumerate(['left','right']):
        data=daily[(daily.side==side)&daily.day.isin([1,2])].copy()
        pivot=data.pivot(index='track_name',columns='day')
        matched=(pivot['observed_seconds']>=.7*86400).all(axis=1)
        names=pivot.index[matched];p=pivot.loc[names]
        result={'matched_ants':len(names)}
        first_foragers=p.index[p['departures'][1]>=3]
        for col,(f,label) in enumerate(features):
            a=p[f][1];b=p[f][2];good=a.notna()&b.notna()
            if f=='mean_trip_seconds':good&=(p['completed_trips']>=3).all(axis=1)
            rho=float(spearmanr(a[good],b[good]).statistic)
            axes[row,col].scatter(a[good],b[good],s=24,color='C0',alpha=.7)
            changed=(b[good]-a[good]).abs().nlargest(5).index
            for name in changed:
                ant_id=int(p.loc[name,('track_id',1)])
                axes[row,col].annotate(str(ant_id),(a[name],b[name]),fontsize=7)
            limit=float(max(a.max(),b.max()));axes[row,col].plot([0,limit],[0,limit],'--',color='.5')
            axes[row,col].set_xlabel('May 16: '+label);axes[row,col].set_ylabel('May 17: '+label)
            axes[row,col].set_title(f'{side}: rho={rho:.2f}, n={good.sum()}')
            result[f+'_rho']=rho
            fa=p.loc[first_foragers,f][1];fb=p.loc[first_foragers,f][2];ok=fa.notna()&fb.notna()
            if f=='mean_trip_seconds':ok&=(p.loc[first_foragers,'completed_trips']>=3).all(axis=1)
            result[f+'_day1_forager_rho']=float(spearmanr(fa[ok],fb[ok]).statistic)
        a=p['outside_fraction'][1];b=p['outside_fraction'][2]
        k=int(np.ceil(len(a)*.2));top1=set(a.nlargest(k).index);top2=set(b.nlargest(k).index)
        result.update({'day1_foragers':len(first_foragers),'top20_overlap':len(top1&top2)/k,
                       'day1_top20_share_day1':float(a.loc[list(top1)].sum()/a.sum()),
                       'day1_top20_share_day2':float(b.loc[list(top1)].sum()/b.sum()),
                       'mean_outside_fraction_day1':float(a.mean()),'mean_outside_fraction_day2':float(b.mean()),
                       'day1_low_outside_ants':int((a<.01).sum()),'day1_low_to_day2_above10pct':int(((a<.01)&(b>.1)).sum())})
        h=hourly[(hourly.side==side)&hourly.track_name.isin(names)&hourly.day.isin([1,2])].copy()
        h['fraction']=h.outside_seconds/h.observed_seconds;h.loc[h.observed_seconds<.5*h.available_seconds,'fraction']=np.nan
        matrix=h.pivot(index='track_name',columns='clock_hour_continuous',values='fraction').reindex(a.sort_values().index)
        im=haxes[row].imshow(matrix,aspect='auto',vmin=0,vmax=1,cmap='magma',extent=[0,48,len(matrix),0])
        haxes[row].axvline(24,color='cyan',lw=1);haxes[row].set_xticks([0,6,12,18,24,30,36,42,48],['00','06','12','18','00','06','12','18','00'])
        haxes[row].set_xlabel('Clock hour, May 16 then May 17');haxes[row].set_ylabel('Ants ordered by May 16 outside fraction')
        stride=max(1,int(np.ceil(len(matrix)/25)))
        ordered_ids=[int(p.loc[n,('track_id',1)]) for n in matrix.index]
        haxes[row].set_yticks(np.arange(len(matrix))[::stride]+.5,[str(n) for n in ordered_ids[::stride]],fontsize=7)
        haxes[row].set_title(f'{side}: outside fraction, fixed ant ordering')
        heat.colorbar(im,ax=haxes[row],label='Fraction of observed time outside')
        results[side]=result
    fig.suptitle('Does individual trip investment persist across two complete calendar days?')
    save(fig,output,'01_day_to_day_persistence');save(heat,output,'02_workforce_over_days')
    return results


def timing(tracks,state,food,water,trips,start,daily,output,rng):
    clock=start+np.arange(state.shape[1]);days=clock//86400;hours=(clock%86400)/3600
    dark=(hours>=19.5)|(hours<5.5)  # Repository's configured schedule, not a measured light log.
    rows=[];results={};fig,axes=plt.subplots(2,2,figsize=(12,9))
    for i,ant in enumerate(tracks.itertuples()):
        for d in [1,2]:
            row={'side':ant.side,'track_id':ant.track_id,'track_name':ant.track_name,'day':d}
            for label,phase in [('dark',dark),('light',~dark)]:
                mask=(days==d)&phase;observed=state[i,mask]>=0;n=int(observed.sum())
                row[label+'_observed_seconds']=n;row[label+'_outside_seconds']=int((state[i,mask]==0).sum())
                row[label+'_resource_seconds']=int((food[i,mask]|water[i,mask]).sum())
                row[label+'_coverage']=n/int(mask.sum())
            row['outside_dark_light_ratio']=((row['dark_outside_seconds']+.5)/row['dark_observed_seconds'])/((row['light_outside_seconds']+.5)/row['light_observed_seconds']) if min(row['dark_observed_seconds'],row['light_observed_seconds']) else np.nan
            rows.append(row)
    phase=pd.DataFrame(rows);phase.to_csv(output/'day_night_by_ant.csv',index=False)
    for col,side in enumerate(['left','right']):
        first=daily[(daily.side==side)&(daily.day==1)&(daily.departures>=3)]
        p=phase[(phase.side==side)&phase.track_name.isin(first.track_name)]
        pivot=p.pivot(index='track_name',columns='day')
        valid=(pivot['dark_coverage']>=.7).all(axis=1)&(pivot['light_coverage']>=.7).all(axis=1)
        p=pivot[valid];a=np.log2(p['outside_dark_light_ratio'][1]);b=np.log2(p['outside_dark_light_ratio'][2])
        rho=float(spearmanr(a,b).statistic)
        axes[0,col].scatter(a,b,s=25);axes[0,col].axhline(0,color='.6');axes[0,col].axvline(0,color='.6')
        lim=max(abs(np.r_[a,b]));axes[0,col].plot([-lim,lim],[-lim,lim],'--',color='.7')
        axes[0,col].set_xlabel('May 16: log2(dark / light outside fraction)');axes[0,col].set_ylabel('May 17: same measure')
        axes[0,col].set_title(f'{side}: timing bias repeatability, rho={rho:.2f}, n={len(a)}')
        # Plot colony profiles among an identical set of first-day foragers.
        ids=np.flatnonzero(tracks.track_name.isin(p.index))
        profile=[]
        for d in [1,2]:
            for h in range(24):
                mask=(days==d)&(hours>=h)&(hours<h+1)
                obs=(state[ids][:,mask]>=0).sum(axis=1);out=(state[ids][:,mask]==0).sum(axis=1)
                res=(food[ids][:,mask]|water[ids][:,mask]).sum(axis=1)
                good=obs>=.7*3600
                profile.append({'side':side,'day':d,'hour':h,'outside_fraction':float(np.mean(out[good]/obs[good])),
                                'resource_fraction':float(np.mean(res[good]/obs[good]))})
        profile=pd.DataFrame(profile)
        for d,style in [(1,'-'),(2,'--')]:
            q=profile[profile.day==d];axes[1,col].plot(q.hour+.5,100*q.outside_fraction,style,label=f'May {15+d}')
        axes[1,col].axvspan(0,5.5,color='.8',alpha=.5);axes[1,col].axvspan(19.5,24,color='.8',alpha=.5)
        axes[1,col].set_xlabel('Clock hour');axes[1,col].set_ylabel('Observed time outside (%)');axes[1,col].legend()
        profile.to_csv(output/f'{side}_daily_clock_profiles.csv',index=False)
        results[side]={'matched_day1_foragers':len(a),'day_night_bias_rho':rho,
                       'median_dark_light_ratio_day1':float(np.median(2**a)),'median_dark_light_ratio_day2':float(np.median(2**b))}
    fig.suptitle('Does timing preference persist? Dark/light rates normalized by observed exposure')
    save(fig,output,'03_day_night_repeatability')
    return results


def morning(tracks,state,start,output):
    clock=start+np.arange(state.shape[1]);rows=[];results=[]
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for col,side in enumerate(['left','right']):
        ids=np.flatnonzero(tracks.side==side)
        for d in [1,2,3]:
            # Windows specified from the configured lights-on time, before inspecting this recording.
            history=(clock>=d*86400+.5*3600)&(clock<d*86400+4.5*3600)
            before=(clock>=d*86400+5*3600)&(clock<d*86400+(5+1/3)*3600)
            after=(clock>=d*86400+(5+2/3)*3600)&(clock<d*86400+6*3600)
            def fraction(mask):
                sub=state[ids][:,mask];obs=(sub>=0).sum(axis=1);out=(sub==0).sum(axis=1)
                return out/np.maximum(obs,1),obs/max(1,mask.sum())
            h,hcov=fraction(history);a,acov=fraction(before);b,bcov=fraction(after)
            valid=(hcov>=.7)&(acov>=.7)&(bcov>=.7)
            # Classify contribution using an independent history, not the pre/post comparison itself.
            for i in np.flatnonzero(valid):rows.append({'side':side,'track_id':int(tracks.iloc[ids[i]].track_id),'day':d,
                                                       'history_outside_fraction':h[i],'before_outside_fraction':a[i],'after_outside_fraction':b[i]})
            hh=h[valid];aa=a[valid];bb=b[valid];high=hh>np.median(hh)
            results.append({'side':side,'day':d,'n_ants':int(valid.sum()),'before_outside_fraction':float(aa.mean()),
                            'after_outside_fraction':float(bb.mean()),'change':float((bb-aa).mean()),
                            'earlier_high_share_before':float(aa[high].sum()/aa.sum()),'earlier_high_share_after':float(bb[high].sum()/bb.sum())})
            profile=[]
            for hclock in np.arange(4,7,.1):
                mask=(clock>=d*86400+hclock*3600)&(clock<d*86400+(hclock+.1)*3600)
                v,c=fraction(mask);ok=valid&(c>=.7);profile.append(float(np.mean(v[ok])))
            axes[0,col].plot(np.arange(4,7,.1)+.05,100*np.array(profile),label=f'May {15+d}')
        q=pd.DataFrame([r for r in rows if r['side']==side])
        for d,color in [(1,'C0'),(2,'C1'),(3,'C2')]:
            v=q[q.day==d];axes[1,col].scatter(v.history_outside_fraction,100*(v.after_outside_fraction-v.before_outside_fraction),s=17,alpha=.6,label=f'May {15+d}')
        axes[0,col].axvline(5.5,color='.5',ls='--');axes[0,col].set_title(side);axes[0,col].legend()
        axes[0,col].set_xlabel('Clock hour');axes[0,col].set_ylabel('Observed time outside (%)')
        axes[1,col].axhline(0,color='.5');axes[1,col].set_xlabel('Earlier outside fraction, 00:30–04:30')
        axes[1,col].set_ylabel('Post minus pre lights-on (percentage points)')
    fig.suptitle('Three morning transitions: does outside investment rise, and who changes?')
    save(fig,output,'04_morning_transitions');pd.DataFrame(rows).to_csv(output/'morning_by_ant.csv',index=False)
    return results


def trip_history(tracks,state,trips,start,output,rng):
    import statsmodels.formula.api as smf
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import Ridge,LogisticRegression
    from sklearn.metrics import log_loss,roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder,StandardScaler
    rows=[];index={n:i for i,n in enumerate(tracks.track_name)}
    for name,ant in trips.groupby('track_name'):
        a=ant.sort_values('exit_seconds');records=list(a.itertuples());i=index[name]
        for previous,nxt in zip(records[:-1],records[1:]):
            if not previous.complete or not nxt.complete:continue
            left=int(previous.return_seconds);right=int(nxt.exit_seconds)
            if right-left<5:continue
            interval=state[i,left:right];known=interval>=0
            if known.mean()<.7 or (interval[known]==1).mean()<.9:continue
            st,en,va=_run_length_encoding(interval)
            if np.any((va==-1)&(en-st+1>30)):continue
            clock=(start+left)%86400/3600
            row={'side':previous.side,'track_id':previous.track_id,'track_name':name,'day':int((start+left)//86400),
                 'next_day':nxt.day,'return_seconds':left,'wait_seconds':right-left,'duration_seconds':previous.duration_seconds,
                 'next_duration_seconds':nxt.duration_seconds,'resource_visit':int(previous.resource_visit),'next_resource_visit':int(nxt.resource_visit),
                 'food_visit':int(previous.food_visit),'water_visit':int(previous.water_visit),'clock_hour':clock}
            for k in [1,2]:row[f'sin{k}']=np.sin(2*np.pi*k*clock/24);row[f'cos{k}']=np.cos(2*np.pi*k*clock/24)
            row['log_wait']=np.log1p(row['wait_seconds']);row['log_duration']=np.log1p(row['duration_seconds'])
            rows.append(row)
    seq=pd.DataFrame(rows);seq.to_csv(output/'trip_sequences.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(12,9));results={}
    for col,side in enumerate(['left','right']):
        data=seq[(seq.side==side)&seq.day.isin([1,2])].copy()
        # Within-ant comparison, controlling trip duration, clock time, and day.
        fit=smf.ols('log_wait ~ resource_visit + log_duration + sin1 + cos1 + sin2 + cos2 + C(day) + C(track_id)',data).fit(cov_type='cluster',cov_kwds={'groups':data.track_id})
        interval=fit.conf_int().loc['resource_visit'].tolist()
        result={'sequence_pairs':len(data),'ants':int(data.track_id.nunique()),'resource_log_wait_coefficient':float(fit.params.resource_visit),
                'resource_ant_clustered_95ci':interval,'adjusted_wait_multiplier':float(np.exp(fit.params.resource_visit)),
                'previous_duration_log_wait_coefficient':float(fit.params.log_duration)}
        ant_summary=data.groupby(['track_id','resource_visit']).wait_seconds.agg(['median','count']).unstack()
        valid=(ant_summary['count']>=3).all(axis=1);a=ant_summary['median'].loc[valid,0]/60;b=ant_summary['median'].loc[valid,1]/60
        axes[0,col].scatter(a,b,s=24);limit=max(a.max(),b.max());axes[0,col].plot([0,limit],[0,limit],'--',color='.6')
        axes[0,col].set_xlabel('Median nest interval after non-resource trip (min)');axes[0,col].set_ylabel('After resource-reaching trip (min)')
        axes[0,col].set_title(f'{side}: within-ant raw medians; n={len(a)}')
        train=data[(data.day==1)&(data.next_day==1)];test=data[(data.day==2)&(data.next_day==2)&data.track_id.isin(train.track_id)].copy()
        numeric=['log_duration','sin1','cos1','sin2','cos2']
        predictions={}
        for name,extra in [('identity_clock_duration',[]),('plus_resource_history',['resource_visit'])]:
            transform=ColumnTransformer([('numeric',StandardScaler(),numeric+extra),('identity',OneHotEncoder(handle_unknown='ignore'),['track_id'])])
            model=make_pipeline(transform,Ridge(alpha=10.));model.fit(train,train.log_wait)
            predictions[name]=model.predict(test);test[name+'_prediction']=predictions[name]
        err0=np.abs(test.log_wait-predictions['identity_clock_duration']);err1=np.abs(test.log_wait-predictions['plus_resource_history'])
        test['absolute_error_improvement']=err0-err1
        improvement=test.groupby('track_id').absolute_error_improvement.mean()
        result.update({'heldout_day2_pairs':len(test),'heldout_ant_weighted_MAE_improvement':float(improvement.mean()),
                       'heldout_MAE_improvement_ant_bootstrap_95ci':ci_mean(improvement.to_numpy(),rng),
                       'heldout_baseline_MAE':float(err0.mean()),'heldout_plus_history_MAE':float(err1.mean())})
        test.to_csv(output/f'{side}_heldout_history_predictions.csv',index=False)
        axes[1,col].scatter(improvement.index,improvement,s=22);axes[1,col].axhline(0,color='.5')
        axes[1,col].set_xlabel('Ant ID');axes[1,col].set_ylabel('Held-out log-wait absolute-error improvement')
        axes[1,col].set_title(f'Train May 16, test May 17: mean gain={improvement.mean():.3f}')
        # A second prospective endpoint: reaching a resource on the next trip.
        losses={}
        for name,extra in [('identity_clock_duration',[]),('plus_resource_history',['resource_visit'])]:
            transform=ColumnTransformer([('numeric',StandardScaler(),numeric+extra),('identity',OneHotEncoder(handle_unknown='ignore'),['track_id'])])
            model=make_pipeline(transform,LogisticRegression(C=1.,max_iter=2000))
            model.fit(train,train.next_resource_visit);prob=model.predict_proba(test)[:,1]
            test[name+'_next_resource_probability']=prob
            losses[name]=-(test.next_resource_visit*np.log(np.clip(prob,1e-8,1))+(1-test.next_resource_visit)*np.log(np.clip(1-prob,1e-8,1)))
            result[name+'_next_resource_log_loss']=float(log_loss(test.next_resource_visit,prob))
            result[name+'_next_resource_auc']=float(roc_auc_score(test.next_resource_visit,prob))
        test['resource_log_loss_improvement']=losses['identity_clock_duration']-losses['plus_resource_history']
        gains=test.groupby('track_id').resource_log_loss_improvement.mean()
        result['next_resource_ant_weighted_log_loss_improvement']=float(gains.mean())
        result['next_resource_gain_ant_bootstrap_95ci']=ci_mean(gains.to_numpy(),rng)
        test.to_csv(output/f'{side}_heldout_history_predictions.csv',index=False)
        results[side]=result
    fig.suptitle('Does reaching a resource predict when the same ant leaves again?')
    save(fig,output,'05_trip_history_and_next_departure')
    return results


def resource_allocation(tracks,state,food,water,daily,output,rng):
    results={};fig,axes=plt.subplots(2,2,figsize=(12,9))
    for col,side in enumerate(['left','right']):
        table=daily[(daily.side==side)&daily.day.isin([1,2])].pivot(index='track_name',columns='day')
        valid=(table['observed_seconds']>=.7*86400).all(axis=1)&(table['departures'][1]>=3)
        p=table[valid]
        resource=p['food_seconds']+p['water_seconds']
        conditional=resource/p['outside_seconds']
        food_share=p['food_seconds']/resource
        enough=(resource>=30).all(axis=1)
        a=conditional[1];b=conditional[2]
        axes[0,col].scatter(100*a,100*b,s=25);lim=float(100*max(a.max(),b.max()));axes[0,col].plot([0,lim],[0,lim],'--',color='.5')
        axes[0,col].set_xlabel('May 16: resource seconds / outside seconds (%)');axes[0,col].set_ylabel('May 17: same measure (%)')
        rho=float(spearmanr(a,b).statistic);axes[0,col].set_title(f'{side}: resource focus rho={rho:.2f}, n={len(a)}')
        axes[1,col].scatter(food_share.loc[enough,1],food_share.loc[enough,2],s=25)
        axes[1,col].plot([0,1],[0,1],'--',color='.5');axes[1,col].set_xlim(-.03,1.03);axes[1,col].set_ylim(-.03,1.03)
        axes[1,col].set_xlabel('May 16: food share of resource time');axes[1,col].set_ylabel('May 17: food share of resource time')
        frho=float(spearmanr(food_share.loc[enough,1],food_share.loc[enough,2]).statistic)
        axes[1,col].set_title(f'Food/water allocation rho={frho:.2f}, n={enough.sum()}')
        results[side]={'day1_foragers':len(p),'resource_focus_rho':rho,'food_share_rho':frho,'food_share_eligible_ants':int(enough.sum()),
                       'median_resource_fraction_of_outside_day1':float(a.median()),'median_resource_fraction_of_outside_day2':float(b.median())}
    fig.suptitle('Does resource allocation persist after accounting for time outside?')
    save(fig,output,'06_resource_allocation_across_days')
    return results


def coverage_sensitivity(daily,hourly):
    rows=[]
    for side in ['left','right']:
        p=daily[(daily.side==side)&daily.day.isin([1,2])].pivot(index='track_name',columns='day')
        for threshold in [.5,.7,.9]:
            valid=(p['observed_seconds']>=threshold*86400).all(axis=1)&(p['departures'][1]>=3)
            q=p[valid]
            rows.append({'side':side,'test':'daily_coverage','threshold':threshold,'n':len(q),
                         'outside_rho':float(spearmanr(q['outside_fraction'][1],q['outside_fraction'][2]).statistic)})
        h=hourly[(hourly.side==side)&hourly.day.isin([1,2])].copy()
        h['outside_fraction']=h.outside_seconds/h.observed_seconds
        v=h.pivot(index=['track_name','hour'],columns='day')
        valid=(v['observed_seconds']>=.7*3600).all(axis=1)
        balanced=v.loc[valid,'outside_fraction'].groupby(level='track_name').agg(['mean','count'])
        keep=(balanced[(1,'count')]>=18)&balanced.index.isin(p.index[p['departures'][1]>=3])
        b=balanced[keep]
        rows.append({'side':side,'test':'same_clock_hours_on_both_days','threshold':.7,'n':len(b),
                     'outside_rho':float(spearmanr(b[(1,'mean')],b[(2,'mean')]).statistic)})
    return rows


def focused_foragers(dataset,daily,output,rng):
    from scipy.stats import rankdata
    old_path=dataset/'stitched/grid_occupancy_histograms_0p5mm_inferred_bounds/panorama_region_analysis/optional_trip_phenotyping/trip_summary.csv'
    names=set(pd.read_csv(old_path).track_name)
    phase=pd.read_csv(output/'day_night_by_ant.csv');results={}
    fig,axes=plt.subplots(2,3,figsize=(15,9))
    for row,side in enumerate(['left','right']):
        d=daily[(daily.side==side)&daily.day.isin([1,2])&daily.track_name.isin(names)].pivot(index='track_name',columns='day')
        d=d[(d.observed_seconds>=.7*86400).all(axis=1)]
        p=phase[(phase.side==side)&phase.track_name.isin(names)].pivot(index='track_name',columns='day')
        p=p[(p.dark_coverage>=.7).all(axis=1)&(p.light_coverage>=.7).all(axis=1)]
        q=d.reindex(p.index)
        rho1=float(spearmanr(q.outside_fraction[1],q.outside_fraction[2]).statistic)
        rho2=float(spearmanr(p.outside_dark_light_ratio[1],p.outside_dark_light_ratio[2]).statistic)
        result={'daily_matched_original_foragers':len(d),'phase_matched_original_foragers':len(p),
                'daily_outside_rho':float(spearmanr(d.outside_fraction[1],d.outside_fraction[2]).statistic),
                'daily_trip_rate_rho':float(spearmanr(d.trip_rate_per_observed_day[1],d.trip_rate_per_observed_day[2]).statistic),
                'same_cohort_outside_rho':rho1,'same_cohort_timing_rho':rho2}
        ix=rng.integers(len(p),size=(3000,len(p)))
        def bootstrap_rho(a,b):
            x=rankdata(np.asarray(a)[ix],axis=1);y=rankdata(np.asarray(b)[ix],axis=1)
            x-=x.mean(axis=1,keepdims=True);y-=y.mean(axis=1,keepdims=True)
            return (x*y).sum(axis=1)/np.sqrt((x*x).sum(axis=1)*(y*y).sum(axis=1))
        difference=bootstrap_rho(q.outside_fraction[1],q.outside_fraction[2])-bootstrap_rho(p.outside_dark_light_ratio[1],p.outside_dark_light_ratio[2])
        result['outside_minus_timing_rho_paired_ant_bootstrap_95ci']=np.quantile(difference[np.isfinite(difference)],[.025,.975]).tolist()
        resources=d.food_seconds+d.water_seconds;foodshare=d.food_seconds/resources;enough=(resources>=30).all(axis=1)
        result['food_share_rho']=float(spearmanr(foodshare.loc[enough,1],foodshare.loc[enough,2]).statistic)
        result['food_share_n']=int(enough.sum())
        panels=[(q.outside_fraction[1],q.outside_fraction[2],'Fraction of time outside',rho1),
                (np.log2(p.outside_dark_light_ratio[1]),np.log2(p.outside_dark_light_ratio[2]),'log2(dark / light outside fraction)',rho2),
                (foodshare.loc[enough,1],foodshare.loc[enough,2],'Food share of resource time',result['food_share_rho'])]
        for col,(a,b,label,rho) in enumerate(panels):
            axes[row,col].scatter(a,b,s=25,color='C0');lo=min(a.min(),b.min());hi=max(a.max(),b.max())
            axes[row,col].plot([lo,hi],[lo,hi],'--',color='.6')
            for name in (b-a).abs().nlargest(4).index:
                axes[row,col].annotate(str(int(d.loc[name,('track_id',1)])),(a[name],b[name]),fontsize=7)
            axes[row,col].set_xlabel('May 16: '+label);axes[row,col].set_ylabel('May 17: same measure')
            axes[row,col].set_title(f'{side}: rho={rho:.2f}, n={len(a)}')
        results[side]=result
    fig.suptitle('Previously identified roaming ants: investment, timing, and resource destination')
    save(fig,output,'07_forager_investment_timing_and_resources')
    return results


def return_wave(dataset,tracks,state,start,output):
    clock=start+np.arange(state.shape[1]);results=[];rows=[]
    speed_paths={json.loads(p.read_text())['track_name']:p.parent/'speed_mm_s.npy' for p in (dataset/'stitched/speed_vectors').rglob('speed_metadata.json')}
    # Return events include any sufficiently long observed outside run followed
    # by an inside anchor, even if the departure was censored.
    returns=np.zeros(state.shape,dtype=bool);departures=returns.copy()
    for i in range(len(tracks)):
        smooth=_remove_short_state_flicker(_bridge_short_state_gaps(state[i],30),5)
        st,en,va=_run_length_encoding(smooth)
        for j in range(1,len(va)):
            if va[j]==1 and en[j]-st[j]+1>=5 and va[j-1]==0 and en[j-1]-st[j-1]+1>=30:returns[i,st[j]]=True
            if va[j]==0 and en[j]-st[j]+1>=30 and va[j-1]==1 and en[j-1]-st[j-1]+1>=5:departures[i,st[j]]=True
    fig,axes=plt.subplots(3,2,figsize=(13,12))
    for col,side in enumerate(['left','right']):
        ids=np.flatnonzero(tracks.side==side)
        for d in [1,2,3]:
            span=(clock>=d*86400+4.5*3600)&(clock<d*86400+6.5*3600)
            sec=np.flatnonzero(span);h=(clock[span]-d*86400)/3600
            before=(h>=5)&(h<5+1/3);after=(h>=5+2/3)&(h<6)
            ss=np.full((len(ids),len(sec)),np.nan)
            for k,i in enumerate(ids):
                ant=tracks.iloc[i];raw=np.load(speed_paths[ant.track_name],mmap_mode='r')
                indices=sec*24-int(ant.frame_min);good=(indices>=0)&(indices<len(raw));ss[k,good]=raw[indices[good]]
            st=state[ids][:,span];observed=st>=0
            valid=(observed[:,before].mean(axis=1)>=.7)&(observed[:,after].mean(axis=1)>=.7)
            valid_speed=valid&(np.isfinite(ss[:,before]).mean(axis=1)>=.7)&(np.isfinite(ss[:,after]).mean(axis=1)>=.7)
            pre=float(np.nanmean(np.nanmean(ss[valid_speed][:,before],axis=1)));post=float(np.nanmean(np.nanmean(ss[valid_speed][:,after],axis=1)))
            outside_pre=(st[valid][:,before]==0).sum(axis=1)/observed[valid][:,before].sum(axis=1)
            outside_post=(st[valid][:,after]==0).sum(axis=1)/observed[valid][:,after].sum(axis=1)
            speed_cohort_pre=(st[valid_speed][:,before]==0).sum(axis=1)/observed[valid_speed][:,before].sum(axis=1)
            speed_cohort_post=(st[valid_speed][:,after]==0).sum(axis=1)/observed[valid_speed][:,after].sum(axis=1)
            record={'side':side,'day':d,'speed_matched_ants':int(valid_speed.sum()),'pre_speed_mm_s':pre,'post_speed_mm_s':post,'speed_fold':post/pre,
                    'state_matched_ants':int(valid.sum()),'outside_fraction_pre':float(outside_pre.mean()),'outside_fraction_post':float(outside_post.mean())}
            record['same_speed_cohort_outside_pre']=float(speed_cohort_pre.mean())
            record['same_speed_cohort_outside_post']=float(speed_cohort_post.mean())
            profile=[]
            for low in np.arange(4.5,6.5,1/12):
                mask=(h>=low)&(h<low+1/12)
                ms=observed_mean(observed_mean(ss[valid_speed][:,mask],axis=1))
                obs=(st[valid][:,mask]>=0).sum();out=(st[valid][:,mask]==0).sum()
                ret=int(returns[ids[valid]][:,sec[mask]].sum());dep=int(departures[ids[valid]][:,sec[mask]].sum())
                item={'side':side,'day':d,'clock_hour':low+1/24,'speed':float(ms),'outside_fraction':out/max(obs,1),'returns':ret,'departures':dep,
                      'return_hazard_per_outside_min':ret/max(out/60,1e-6)}
                profile.append(item);rows.append(item)
            p=pd.DataFrame(profile);color=f'C{d-1}'
            axes[0,col].plot(p.clock_hour,p.speed,label=f'May {15+d}',color=color)
            axes[1,col].plot(p.clock_hour,100*p.outside_fraction,color=color)
            axes[2,col].plot(p.clock_hour,p.returns-p.departures,color=color)
            baseline=p[(p.clock_hour>=4.75)&(p.clock_hour<5.25)]
            onset=p[(p.clock_hour>=5.5)&(p.clock_hour<5.75)]
            record['baseline_net_returns_per_5min']=float((baseline.returns-baseline.departures).mean())
            record['onset_net_returns_per_5min']=float((onset.returns-onset.departures).mean())
            record['onset_returns']=int(onset.returns.sum());record['onset_departures']=int(onset.departures.sum())
            results.append(record)
        for ax in axes[:,col]:ax.axvline(5.5,color='.5',ls='--');ax.set_xlabel('Clock hour')
        axes[0,col].set_title(side);axes[0,col].set_ylabel('Mean speed (mm/s)');axes[0,col].legend()
        axes[1,col].set_ylabel('Observed time outside (%)')
        axes[2,col].set_ylabel('Returns minus departures / five minutes');axes[2,col].axhline(0,color='.6')
    fig.suptitle('Movement rises while ants return to the nest near configured lights-on')
    save(fig,output,'08_morning_speed_and_return_wave');pd.DataFrame(rows).to_csv(output/'morning_speed_and_return_profiles.csv',index=False)
    return results


def phase_boundary_sensitivity(dataset,tracks,state,start):
    old=pd.read_csv(dataset/'stitched/grid_occupancy_histograms_0p5mm_inferred_bounds/panorama_region_analysis/optional_trip_phenotyping/trip_summary.csv')
    clock=np.arange(state.shape[1])+start;rows=[]
    for side in ['left','right']:
        ids=np.flatnonzero((tracks.side==side)&tracks.track_name.isin(old.track_name));values=[];coverage=[]
        for cycle in [0,1]:
            boundary=(19.5+24*cycle)*3600;parts=[];cov=[]
            for low,high in [(boundary,boundary+10*3600),(boundary+10*3600,boundary+24*3600)]:
                sub=state[ids][:,(clock>=low)&(clock<high)];observed=(sub>=0).sum(axis=1);out=(sub==0).sum(axis=1)
                parts.append((out+.5)/np.maximum(observed,1));cov.append(observed/(high-low))
            values.append(parts[0]/parts[1]);coverage.append(np.min(cov,axis=0))
        for threshold in [.5,.7]:
            valid=(np.array(coverage)>=threshold).all(axis=0)
            rows.append({'side':side,'period':'19:30-to-19:30 complete dark-light cycles','min_phase_coverage':threshold,
                         'n_ants':int(valid.sum()),'timing_rank_rho':float(spearmanr(np.array(values)[0,valid],np.array(values)[1,valid]).statistic)})
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dataset',type=Path,required=True);parser.add_argument('--output',type=Path)
    args=parser.parse_args();output=args.output or args.dataset/'analysis_outputs/multiday_task_allocation_20260908';output.mkdir(parents=True,exist_ok=True)
    tracks,state,food,water,start,fps=load(args.dataset,output)
    trips=extract_trips(tracks,state,food,water,start);trips.to_csv(output/'trips.csv',index=False)
    daily,hourly=make_tables(tracks,state,food,water,trips,start);daily.to_csv(output/'ant_days.csv',index=False);hourly.to_csv(output/'ant_hours.csv',index=False)
    results={'dataset':str(args.dataset),'hours':state.shape[1]/3600,'selected_ants':tracks.groupby('side').size().to_dict(),
             'departures':len(trips),'completed_trips':int(trips.complete.sum()),'start_clock_seconds':start}
    results['daily_allocation']=daily_allocation(daily,hourly,output,np.random.default_rng(20260908))
    results['timing']=timing(tracks,state,food,water,trips,start,daily,output,np.random.default_rng(20260908))
    results['morning']=morning(tracks,state,start,output)
    results['trip_history']=trip_history(tracks,state,trips,start,output,np.random.default_rng(20260908))
    results['resource_allocation']=resource_allocation(tracks,state,food,water,daily,output,np.random.default_rng(20260908))
    results['coverage_sensitivity']=coverage_sensitivity(daily,hourly)
    results['focused_foragers']=focused_foragers(args.dataset,daily,output,np.random.default_rng(20260908))
    results['return_wave']=return_wave(args.dataset,tracks,state,start,output)
    results['phase_boundary_sensitivity']=phase_boundary_sensitivity(args.dataset,tracks,state,start)
    (output/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':main()
