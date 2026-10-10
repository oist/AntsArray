"""Short-window kinematics, ordered trajectories and normalized motion spectra."""
import argparse
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import minimum_filter1d, maximum_filter1d
from scipy.signal import convolve

from analysis.fine_behavior_extract import HZ, FPS, START, N_HOURS, SECONDS

WIDTHS = (.5, 1., 2.)
FREQUENCIES = (.5, .8, 1.2, 1.8, 2.7, 4.)
STRIDE = 3  # one state assignment per 0.25 s
BASE_NAMES = ['PC1','PC2','PC3','PC4','head_cos','head_sin','gaster_cos','gaster_sin',
              'forward_peak','lateral_peak','antenna_A_mean','antenna_B_mean','turn_mean',
              'mean_contact_partners','contact_bout_fraction','new_contact_bouts']


def validate_sequence(file, block, track_name):
    metadata=json.loads(file.with_suffix('.json').read_text())
    if hashlib.sha256(file.read_bytes()).hexdigest()!=metadata['output_sha256']:
        raise ValueError(f'Changed sequence: {file}')
    raw=block/'stitched/per_track'/track_name
    source=metadata['signature']['source']
    if (raw.stat().st_size,raw.stat().st_mtime_ns)!=(source['size'],source['mtime_ns']):
        raise ValueError(f'Changed raw tracking: {raw}')
    return metadata


def spectrum(scores, camera, frequencies=FREQUENCIES):
    """Finite Morlet coefficients; every kernel sample must be observed/filled.

    omega0=5, +/-2 sigma, zero DC, unit L2 energy. The slowest band has a
    ~6.5-second support; shorter summary windows do not remove that context.
    """
    finite = np.isfinite(scores).all(axis=2) & np.isfinite(camera)
    outputs = []
    for f in frequencies:
        sigma = 5/(2*np.pi*f)
        radius = int(np.ceil(2*sigma*HZ))
        time = np.arange(-radius, radius+1)/HZ
        envelope = np.exp(-.5*(time/sigma)**2)
        carrier = np.exp(2j*np.pi*f*time)
        kernel = envelope*(carrier-(envelope*carrier).sum()/envelope.sum())
        kernel /= np.sqrt((np.abs(kernel)**2).sum())
        support = convolve(finite.astype(float), np.ones((1,len(kernel))), mode='same', method='direct') >= len(kernel)-.5
        cams = np.nan_to_num(camera, nan=-999.)
        support &= maximum_filter1d(cams,len(kernel),axis=1,mode='nearest') == minimum_filter1d(cams,len(kernel),axis=1,mode='nearest')
        coefficients = convolve(np.nan_to_num(scores), kernel.conj()[None,::-1,None], mode='same', method='fft')
        power = np.abs(coefficients)**2
        power[~support] = np.nan
        outputs.append(power)
    return np.stack(outputs,axis=3).reshape(*scores.shape[:2],-1).astype(np.float32)


def window_features(signals, power, cameras, frames, partners, onsets, seconds, require_spectrum=True):
    half = int(round(seconds*HZ/2))
    centers = np.arange(12,signals.shape[1]-12,STRIDE)
    indices = centers[:,None] + np.arange(-half,half+1)
    observed = signals[:,indices]
    powers = power[:,indices]
    valid = np.isfinite(observed).all(axis=(2,3))
    if require_spectrum:
        valid &= np.isfinite(powers).all(axis=(2,3))
    valid &= np.isfinite(partners[:,indices]).all(axis=2)
    valid &= np.ptp(cameras[:,indices],axis=2) == 0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        base = np.concatenate([np.mean(observed[:,:,:,:8],axis=2),
            np.max(observed[:,:,:,8:10],axis=2), np.mean(observed[:,:,:,10:13],axis=2),
            np.mean(partners[:,indices],axis=2)[:,:,None],
            np.mean(partners[:,indices]>0,axis=2)[:,:,None]],axis=2)
        spectral = np.mean(powers,axis=2)
    start,stop = frames[:,centers-half],frames[:,centers+half]+1
    events = np.searchsorted(onsets,stop)-np.searchsorted(onsets,start)
    base = np.concatenate([base,events[:,:,None]],axis=2)
    taps = centers[:,None] + np.round(np.linspace(-half,half,5)).astype(int)
    trajectory = signals[:,taps,:8]
    trajectory = trajectory-trajectory.mean(axis=2,keepdims=True)
    trajectory = trajectory.reshape(*trajectory.shape[:2],-1)
    return base.astype(np.float32),trajectory.astype(np.float32),spectral,valid,centers


def prepare(block, source, sequences, output):
    from analysis.behavior_landscape_features import load_contacts
    output.mkdir(parents=True,exist_ok=True)
    ants = pd.read_csv(source/'all_ant_coverage.csv')
    info = dict(start_time='2026-07-24 10:00:00',frame_start=START,
                frame_stop=START+N_HOURS*3600*FPS,fps=24.,context_settings=dict(fps=24.))
    bouts, coverage, sources, run_id = load_contacts(block, info)
    pieces,quality,provenance = [],[],[]
    for ant_index,ant in enumerate(ants.itertuples()):
        file = sequences/(ant.ant.replace(':','_')+'.npz')
        provenance.append(validate_sequence(file,block,ant.track_name))
        with np.load(file) as z: saved=dict(z)
        assert str(saved['ant']) == ant.ant and int(saved['ant_index']) == ant_index
        signals,cameras = saved['signals'],saved['camera']
        frames = saved['starts'][:,None]+saved['sample_offsets'][None,:]
        contact = bouts[bouts.side.eq(ant.side) & (bouts.ant_a.eq(ant.track_id)|bouts.ant_b.eq(ant.track_id))]
        flat = frames.ravel()
        lo=np.searchsorted(flat,contact.start_frame.to_numpy()); hi=np.searchsorted(flat,contact.end_frame.to_numpy()+1)
        delta=np.zeros(len(flat)+1);np.add.at(delta,lo,1);np.add.at(delta,hi,-1)
        partners=np.cumsum(delta[:-1]).reshape(frames.shape)
        partners[~coverage[ant.side][frames//FPS]]=np.nan
        onsets=np.sort(contact.loc[contact.is_new_onset,'start_frame'].to_numpy())
        power=spectrum(signals[:,:,:4],cameras)
        variants=[window_features(signals,power,cameras,frames,partners,onsets,w) for w in WIDTHS]
        common=np.logical_and.reduce([v[3] for v in variants])
        centers=variants[0][4]
        hour,center_index=np.where(common)
        selected=centers[center_index]
        bases=np.stack([v[0][common] for v in variants])
        trajectories=np.stack([v[1][common] for v in variants])
        spectra=np.stack([v[2][common] for v in variants])
        pieces.append(dict(base=bases,trajectory=trajectories,spectrum=spectra,
            ant=np.full(common.sum(),ant_index,np.int16),hour=hour.astype(np.int8),
            sample=selected.astype(np.int16),frame=frames[common.nonzero()[0],selected],
            camera=cameras[hour,selected].astype(np.int16),
            interpolation=saved['interpolated_fraction'][hour,selected],
            residual=saved['residual_mm'][hour,selected]))
        quality.append(dict(ant=ant.ant,ant_index=ant_index,candidate_centers=int(common.size),
            common_centers=int(common.sum()),valid_hours=len(np.unique(hour)),
            complete_samples=int(np.isfinite(signals).all(axis=2).sum()),
            **{f'valid_{w:g}s':int(v[3].sum()) for w,v in zip(WIDTHS,variants)}))
        print('FEATURES',quality[-1],flush=True)
    arrays={key:np.concatenate([p[key] for p in pieces],axis=1 if key in ('base','trajectory','spectrum') else 0) for key in pieces[0]}
    np.savez_compressed(output/'features.npz',**arrays,widths=WIDTHS,frequencies=FREQUENCIES,base_names=BASE_NAMES)
    pd.DataFrame(quality).to_csv(output/'coverage.csv',index=False)
    (output/'extraction_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (output/'feature_protocol.json').write_text(json.dumps(dict(width_seconds=WIDTHS,stride_seconds=STRIDE/HZ,
        wavelet_frequencies=FREQUENCIES,wavelet_support='Zero-DC unit-energy Morlet, omega0=5, +/-2 sigma; full finite same-camera support. Slowest band ~6.5 s, plus summary-window width.',
        common_support='All three representations and widths use identical centers; no camera/gap crossing.',
        contact_run=run_id,contact_cache=str(sources[-1]),
        contact='Mean concurrent partners, observed merged-bout fraction, and new pair-bout onsets; merged gaps up to 2 s.',
        families='Pose (PC1–4 plus head/gaster articulation); one motion block; social context; optional temporal representation. No repeated equal weights for correlated motion summaries.',
        no_labels_used=True),indent=2)+'\n')


def encode(bundle, width_index, representation, fit_rows, normalizer=None):
    """Fit all scales on the training dictionary; retain every coordinate."""
    base=np.asarray(bundle['base'][width_index],dtype=float).copy()
    base[:,8:12]=np.log1p(base[:,8:12]/.1)
    base[:,12]=np.log1p(base[:,12]/.1)
    base[:,13]=np.log1p(base[:,13])
    base[:,15]=np.log1p(base[:,15])
    blocks=[base[:,:8],base[:,8:13],base[:,13:]]
    block_names=['pose','motion','social']
    floor=0.
    if representation=='trajectory':
        blocks.append(bundle['trajectory'][width_index]);block_names.append('ordered_posture_variation')
    elif representation=='spectrum':
        power=np.maximum(bundle['spectrum'][width_index],0).astype(float)
        floor=max(float(np.quantile(power[fit_rows].sum(axis=1),.05)),1e-10) if normalizer is None else normalizer['spectral_floor']
        probability=np.column_stack([power,np.full(len(power),floor)])
        probability/=probability.sum(axis=1,keepdims=True)
        blocks.append(np.sqrt(probability));block_names.append('relative_spectrum')
    elif representation!='kinematics':
        raise ValueError(representation)
    if normalizer is None:
        centers=[b[fit_rows].mean(axis=0) for b in blocks]
        scales=[max(float(np.sqrt(b[fit_rows].var(axis=0).sum())),1e-6) for b in blocks]
        normalizer=dict(centers=[c.tolist() for c in centers],scales=scales,blocks=block_names,spectral_floor=floor)
    x=np.concatenate([(b-np.asarray(c))/s for b,c,s in zip(blocks,normalizer['centers'],normalizer['scales'])],axis=1)
    return x.astype(np.float32),normalizer


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('block','source','sequences','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();prepare(a.block,a.source,a.sequences,a.output)
