"""Export S0/S1 versus other observed states for every physical ant cup."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tracking.single_ant.manual_cup_tracking import write_json
from tracking.single_ant.sleep_states import observed_states,recorded_mask


def recording_bins(manifest,seconds=60):
    width=max(1,round(seconds*manifest['fps']));bins=[];previous=0
    for segment in manifest['recording_segments']:
        lo=int(segment['frame_offset']);hi=lo+int(segment['num_frames'])
        if lo>previous:bins.append((previous,lo,False))
        bins.extend((s,min(s+width,hi),True) for s in range(lo,hi,width))
        previous=hi
    return np.asarray(bins,dtype=np.int64)


def export_labels(packet,dataset,output=None):
    packet,dataset=Path(packet),Path(dataset);output=Path(output or packet/'s01_labels')
    manifest=json.loads((dataset/'manifest.json').read_text());n=int(manifest['n_frames'])
    support=recorded_mask(manifest);recorded=int(support.sum())
    inventory=pd.read_csv(packet/'inventory_all.csv').set_index('track_id')
    audit=dataset/'analysis/speed_by_metadata/metadata_join.csv'
    if audit.exists():metadata=pd.read_csv(audit)
    else:
        identities=pd.read_csv(dataset/'cup_identities.csv').set_index('CupID')
        metadata=pd.DataFrame([dict(CupID=c['cup_id'],TrackID=c['cup_id'] if c['cup_id'] in identities.index else np.nan,
            IdentityStatus='matched' if c['cup_id'] in identities.index else 'no track')
            for c in manifest.get('physical_cups',manifest['cups'])])
    digest=hashlib.sha256((packet/'models/dictionary.npz').read_bytes()).hexdigest()
    selection_path=packet/'models/selection.json'
    selection=json.loads(selection_path.read_text()) if selection_path.exists() else {}
    definition=dict(mode='fixed_dictionary_full_posture_only',sleep_states=[0,1],
        categories={'1':'S0/S1','0':'other observed state','-1':'ambiguous'},
        n_frames=n,fps=manifest['fps'],motion_exclusions=False,
        reference_dictionary_sha256=digest,manifest_sha256=hashlib.sha256((dataset/'manifest.json').read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model_supported=bool(selection.get('supported',False)),model_selection=selection,
        positive_rule='Observed S0 or S1 membership in the newly fitted dictionary',
        negative_rule='Observed membership in any other fitted state',
        unknown_rule='Missing or invalid full-body features, unrecorded gaps, or excluded resident identity',
        prediction_rule='Every locally valid full-body window; discovery coverage eligibility is unchanged')
    output.mkdir(parents=True,exist_ok=True)
    if (output/'definition.json').exists():
        previous=json.loads((output/'definition.json').read_text())
        if previous!=definition:raise ValueError('State labels belong to different inputs; use a fresh output')
    write_json(output/'definition.json',definition)
    definition_sha=hashlib.sha256((output/'definition.json').read_bytes()).hexdigest()
    bins=recording_bins(manifest);starts,stops,is_recorded=bins.T;lengths=stops-starts
    fractions=np.empty((len(metadata),len(bins),3),np.float32)
    speeds=np.full((len(metadata),len(bins)),np.nan,np.float32)
    coverage=speeds.copy();rows=[]
    for index,row in enumerate(metadata.to_dict('records')):
        cid=int(row['CupID']);tid=int(row['TrackID']) if pd.notna(row.get('TrackID')) else None
        identity_ok=row['IdentityStatus']=='matched';observed=np.full(n,-1,np.int8)
        selected=False;predicted=False
        if identity_ok and tid is not None and tid in inventory.index:
            item=inventory.loc[tid];selected=bool(item.selected)
            predicted=(packet/'assignments'/item.ant.replace(':','_')/'state.npy').is_file()
            observed=observed_states(packet,item.ant,n,selected)
        observed[~support]=-1
        category=np.full(n,-1,np.int8);known=observed>=0
        category[known]=np.isin(observed[known],[0,1]).astype(np.int8)
        stem=f'TrackID_{tid if tid is not None else cid:04d}_all_single'
        folder=output/'per_track'/stem;folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/'labels.npz',observed=observed,category=category,evidence=known.astype(np.uint8))
        result=dict(row,StateModelSupported=definition['model_supported'],
            DiscoverySelected=selected,PredictionEligible=bool(predicted and known.any()),
            StateCohort=bool(known.any()),RecordedFrames=recorded,ObservedStateFrames=int(known.sum()),
            ObservedStatePercent=100*float(known.sum())/recorded,
            SleepPercent=100*float((category[support]==1).sum())/recorded,
            NotS01Percent=100*float((category[support]==0).sum())/recorded,
            AmbiguousPercent=100*float((category[support]<0).sum())/recorded)
        for k,value in enumerate([1,0,-1]):
            fractions[index,:,k]=np.add.reduceat((category==value).astype(np.int32),starts)/lengths
        # Refresh the displayed speed summary from the corrected stitched data.
        stats=dict(ValidSpeedFrames=0,ValidSpeedFraction=np.nan,MeanSpeed=np.nan,MedianSpeed=np.nan,
            P95Speed=np.nan,First6hMean=np.nan,First6hCoverage=np.nan,RawMeanSpeed=np.nan,RawAbove20Fraction=np.nan)
        speed_folder=dataset/'analysis/speed_vectors/per_track'/stem
        if identity_ok and tid is not None:
            speed=np.load(speed_folder/'speed_mm_s.npy',mmap_mode='r')
            if speed.shape!=(n,):raise ValueError('Speed and state timelines differ')
            valid=np.isfinite(speed)&support;values=speed[valid];first=valid[:round(6*3600*manifest['fps'])]
            first_coverage=float(first.sum()/support[:len(first)].sum()) if support[:len(first)].any() else 0.
            counts=np.add.reduceat(valid.astype(np.int32),starts)
            means=np.divide(np.add.reduceat(np.where(valid,speed,0),starts,dtype=np.float64),counts,
                            out=np.full(len(bins),np.nan),where=counts>0)
            cov=np.where(is_recorded,counts/lengths,np.nan);coverage[index]=cov
            speeds[index]=np.where(cov>=.5,means,np.nan)
            stats.update(ValidSpeedFrames=int(valid.sum()),ValidSpeedFraction=float(valid.sum())/recorded,
                MeanSpeed=float(values.mean(dtype=np.float64)) if len(values) else np.nan,
                MedianSpeed=float(np.median(values)) if len(values) else np.nan,
                P95Speed=float(np.quantile(values,.95)) if len(values) else np.nan,
                First6hCoverage=first_coverage,
                First6hMean=float(speed[:len(first)][first].mean(dtype=np.float64)) if first_coverage>=.5 else np.nan)
            raw_path=speed_folder/'speed_unclipped_mm_s.npy'
            if raw_path.exists():
                raw=np.load(raw_path,mmap_mode='r');finite=raw[np.isfinite(raw)&support]
                if len(finite):stats.update(RawMeanSpeed=float(finite.mean(dtype=np.float64)),RawAbove20Fraction=float((finite>20).mean()))
        result.update(stats);rows.append(result)
        write_json(folder/'metadata.json',dict(cup=cid,track_id=tid,identity_ok=identity_ok,
            model_supported=definition['model_supported'],
            discovery_selected=selected,prediction_eligible=bool(predicted and known.any()),
            recorded_frames=recorded,assigned_frames=int(known.sum()),definition_sha256=definition_sha))
    table=pd.DataFrame(rows);table.to_csv(output/'per_ant_summary.csv',index=False)
    np.savez_compressed(output/'binned_data.npz',cup_ids=table.CupID.to_numpy(),
        start_frame=starts,stop_frame=stops,recorded=is_recorded.astype(bool),
        time_h=(starts+stops)/2/manifest['fps']/3600,fractions=fractions,speed=speeds,coverage=coverage)
    render(output,manifest,table,bins,fractions,speeds)
    matched=table.loc[table.IdentityStatus.eq('matched')]
    complete=dict(cups=len(table),matched_ants=len(matched),predicted_ants=int(table.PredictionEligible.sum()),
        discovery_ants=int(table.DiscoverySelected.sum()),n_frames=n,recorded_frames=recorded,
        dictionary_sha256=digest,definition_sha256=definition_sha,model_supported=definition['model_supported'],
        mean_matched_ambiguous_percent=float(matched.AmbiguousPercent.mean()),
        mean_matched_s01_percent=float(matched.SleepPercent.mean()),
        mean_matched_other_percent=float(matched.NotS01Percent.mean()))
    write_json(output/'COMPLETE.json',complete);return complete


def render(output,manifest,table,bins,fractions,speeds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    colors=['#8452b8','#d99728','#b8bcc6'];fps=manifest['fps']
    definition=json.loads((output/'definition.json').read_text())
    status='State model meets stability criteria' if definition['model_supported'] else 'Exploratory: state model does not meet stability criteria'
    edges=np.r_[bins[:,0],bins[-1,1]]/fps/3600;time=(edges[1:]+edges[:-1])/2
    labels=[f"Cup {int(r.CupID):02d} · {r.get('Colony','')} {r.get('Role','')}" for _,r in table.iterrows()]
    with PdfPages(output/'s01_every_ant.pdf') as pdf:
        fig,axes=plt.subplots(1,2,figsize=(17,12),sharey=True,layout='constrained')
        for ax,k,title in zip(axes,[0,2],['S0/S1','Ambiguous']):
            values=fractions[:,:,k]*100;values=values.copy();values[:,~bins[:,2].astype(bool)]=np.nan
            mesh=ax.pcolormesh(edges,np.arange(len(table)+1),values,cmap='Purples' if k==0 else 'Greys',vmin=0,vmax=100,rasterized=True)
            ax.set(xlabel='Hours from first trigger',title=title,yticks=np.arange(len(table))+.5,yticklabels=labels)
            fig.colorbar(mesh,ax=ax,label='Recorded time (%)')
        axes[0].invert_yaxis()
        fig.suptitle(f"Single-ant S0/S1 labels · {manifest['start_datetime']}\n{status}",fontsize=14)
        pdf.savefig(fig);fig.savefig(output/'s01_every_ant_overview.png',dpi=140);plt.close(fig)
        for i,row in table.iterrows():
            fig,axes=plt.subplots(2,1,figsize=(15,6),sharex=True,layout='constrained')
            axes[0].stackplot(time,fractions[i].T*100,colors=colors,labels=['S0/S1','Other observed state','Ambiguous'])
            axes[0].set(ylim=(0,100),ylabel='Recorded time (%)');axes[0].legend(ncol=3,fontsize=9)
            axes[1].plot(time,speeds[i],lw=.7,color='#287c8e');axes[1].set(ylabel='Speed (mm/s)',xlabel='Hours from first trigger',ylim=(0,20))
            for lo,hi,recorded in bins:
                if not recorded:
                    for ax in axes:ax.axvspan(lo/fps/3600,hi/fps/3600,color='white',hatch='//',ec='#999',zorder=10)
            fig.suptitle(f"{labels[i]} · {row.IdentityStatus}\nS0/S1 {row.SleepPercent:.1f}% · other {row.NotS01Percent:.1f}% · ambiguous {row.AmbiguousPercent:.1f}%\n{status}")
            pdf.savefig(fig);plt.close(fig)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--eigen-root',type=Path,required=True);p.add_argument('--output',type=Path)
    args=p.parse_args();print(json.dumps(export_labels(args.eigen_root,args.dataset,args.output),indent=2))


if __name__=='__main__':main()
