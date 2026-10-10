"""Audit wavelet coverage and refit short windows without the slow-wavelet gate.

This is a prespecified follow-up to the common-support comparison: test whether
requiring slow-frequency context discards fast, fragmented movement samples.
Only the kinematics and ordered-trajectory representations are comparable here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.behavior_landscape_features import load_contacts
from analysis.fine_behavior_features import WIDTHS, BASE_NAMES, window_features, validate_sequence
from analysis.fine_behavior_extract import START, FPS, N_HOURS
from analysis import fine_behavior_states as models


def prepare(block,source,sequences,output):
    output.mkdir(parents=True,exist_ok=True)
    ants=pd.read_csv(source/'all_ant_coverage.csv')
    info=dict(start_time='2026-07-24 10:00:00',frame_start=START,
              frame_stop=START+N_HOURS*3600*FPS,fps=24.,context_settings=dict(fps=24.))
    bouts,coverage,sources,run_id=load_contacts(block,info)
    pieces=[];audit=[];provenance=[]
    for ai,ant in enumerate(ants.itertuples()):
        file=sequences/(ant.ant.replace(':','_')+'.npz')
        meta=validate_sequence(file,block,ant.track_name)
        provenance.append(meta)
        with np.load(file) as z:d=dict(z)
        signals,camera=d['signals'],d['camera']
        frames=d['starts'][:,None]+d['sample_offsets'][None,:]
        contacts=bouts[bouts.side.eq(ant.side)&(bouts.ant_a.eq(ant.track_id)|bouts.ant_b.eq(ant.track_id))]
        flat=frames.ravel();lo=np.searchsorted(flat,contacts.start_frame);hi=np.searchsorted(flat,contacts.end_frame+1)
        delta=np.zeros(len(flat)+1);np.add.at(delta,lo,1);np.add.at(delta,hi,-1)
        partners=np.cumsum(delta[:-1]).reshape(frames.shape)
        partners[~coverage[ant.side][frames//FPS]]=np.nan
        onsets=np.sort(contacts.loc[contacts.is_new_onset,'start_frame'].to_numpy())
        dummy=np.zeros((*signals.shape[:2],24),np.float32)
        variants=[window_features(signals,dummy,camera,frames,partners,onsets,w,False) for w in WIDTHS]
        # Two-second support gives identical examples for the width comparison.
        common=np.logical_and.reduce([v[3] for v in variants])
        hour,ci=np.where(common);centers=variants[0][4];sample=centers[ci]
        pieces.append(dict(base=np.stack([v[0][common] for v in variants]),
            trajectory=np.stack([v[1][common] for v in variants]),
            spectrum=np.stack([v[2][common] for v in variants]),
            ant=np.full(common.sum(),ai,np.int16),hour=hour.astype(np.int8),sample=sample.astype(np.int16),
            frame=frames[hour,sample],camera=camera[hour,sample].astype(np.int16),
            interpolation=d['interpolated_fraction'][hour,sample],residual=d['residual_mm'][hour,sample]))
        for width,v in zip(WIDTHS,variants):
            for use_common in (False,True):
                valid=common if use_common else v[3]
                raw=v[0][valid]
                audit.append(dict(ant=ant.ant,width=width,common=use_common,n_windows=len(raw),
                    valid_hours=int(valid.any(axis=1).sum()),
                    median_forward=float(np.median(raw[:,8])) if len(raw) else np.nan,
                    mean_forward=float(np.mean(raw[:,8])) if len(raw) else np.nan,
                    median_antenna=float(np.median(raw[:,10:12])) if len(raw) else np.nan))
        print('SHORT',ant.ant,int(common.sum()),flush=True)
    arrays={key:np.concatenate([p[key] for p in pieces],axis=1 if key in ('base','trajectory','spectrum') else 0) for key in pieces[0]}
    np.savez_compressed(output/'features.npz',**arrays,widths=WIDTHS,base_names=BASE_NAMES)
    pd.DataFrame(audit).to_csv(output/'coverage_audit.csv',index=False)
    (output/'extraction_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (output/'feature_protocol.json').write_text(json.dumps(dict(common_support='All widths use 2-second finite kinematic context; no wavelet gate',
        contact_run=run_id,contact_cache=str(sources[-1])),indent=2)+'\n')


def run(features,source,output,wavelet=False):
    from threadpoolctl import threadpool_limits
    models.REPRESENTATIONS=('spectrum',) if wavelet else ('kinematics','trajectory')
    models.PROTOCOL=dict(models.PROTOCOL,representations=models.REPRESENTATIONS,
        support='Full finite wavelet context' if wavelet else 'All widths on common 2-second kinematics support; no slow-wavelet requirement',
        leiden_iterations='Full convergence (-1), unlike preliminary three-iteration screen')
    # At fine resolution a fixed number of iterations can stop before convergence.
    import leidenalg as la
    def converged(graph,resolution,seed=models.SEED):
        return np.asarray(la.find_partition(graph,la.RBConfigurationVertexPartition,weights='weight',
            resolution_parameter=resolution,n_iterations=-1,seed=seed).membership,dtype=int)
    models.partition=converged
    with threadpool_limits(limits=4):models.run(features,source,output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','run'))
    p.add_argument('--wavelet',action='store_true',help='Convergence audit of the initial wavelet comparison')
    for key in ('block','source','sequences','features','output'):p.add_argument('--'+key,type=Path)
    a=p.parse_args()
    if a.action=='prepare':prepare(a.block,a.source,a.sequences,a.output)
    else:run(a.features,a.source,a.output,a.wavelet)
