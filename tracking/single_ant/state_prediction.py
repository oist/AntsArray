"""Predict locally supported full-posture states outside the discovery cohort.

Discovery remains subject to the existing position-coverage rule. Prediction
uses the fixed PCA/dictionary and exactly the same full-body feature requirements.
No partial vector, quiet missing frame or motion-only observation adds sleep.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from tracking.single_ant.manual_cup_tracking import write_json


def initialize_reference(output, reference, manifest):
    output,reference=Path(output),Path(reference)
    output.mkdir(parents=True,exist_ok=True)
    if output.resolve()!=reference.resolve():
        choice=json.loads((reference/'models/selection.json').read_text())
        for name in ['geometry.json','full_body_pca.npz','models/dictionary.npz','models/selection.json',
                     'models/'+choice['representation']+'_normalizer.json']:
            target=output/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(reference/name,target)
    write_json(output/'analysis_protocol.json',dict(mode='single_ant',social_features=False,
        state_cohort='Discovery: existing >60% coverage cohort; prediction: every locally valid full-posture window',
        reference=str(reference),partial_posture_inference=False))
    timing=dict(manifest)
    timing.setdefault('recording_segments',[dict(frame_offset=0,num_frames=timing['n_frames'])])
    write_json(output/'tracking_manifest.json',timing)


def identity_ok(dataset, track_id):
    path=Path(dataset)/'analysis/speed_by_metadata/metadata_join.csv'
    if path.exists():
        t=pd.read_csv(path);row=t.loc[t.TrackID.eq(track_id)]
        return not len(row) or bool(row.IdentityStatus.eq('matched').all())
    return True


def _reference_scale(reference,ant):
    path=Path(reference)/'sequences'/ant.replace(':','_')/'metadata.json'
    return float(json.loads(path.read_text())['body_scale_mm']) if path.exists() else None


def predict_track(raw, output, reference, manifest, track_id, *, interpolation_root,
                  scale_mm=None):
    from analysis.compute_track_posture_interpolation import run as interpolate
    from analysis.full_behavior_extract import extract_one
    from analysis.full_behavior_states import make_features,encode,vote
    from tracking.single_ant.eigenposture import geometry_sample
    from tracking.single_ant.sleep_states import observed_states,recorded_mask
    raw,output,reference=Path(raw),Path(output),Path(reference)
    ant=f'single:{track_id:03d}';name=ant.replace(':','_')
    marker=output/'prediction_qc'/f'{track_id:04d}.json'
    if marker.exists():return json.loads(marker.read_text())
    import pyarrow.parquet as pq
    if not pq.ParquetFile(raw).metadata.num_rows:
        result=dict(track_id=track_id,prediction_eligible=False,reason='no_track_rows',assigned_frames=0)
        write_json(marker,result);return result
    try:axis=geometry_sample(raw)
    except ValueError as error:
        if 'No body-axis observations' not in str(error):raise
        result=dict(track_id=track_id,prediction_eligible=False,reason='no_body_axis',assigned_frames=0)
        write_json(marker,result);return result
    folder=Path(interpolation_root)/raw.stem
    if not (folder/'_SUCCESS').is_file():interpolate(raw,folder,fps=float(manifest['fps']),max_gap_seconds=1.)
    source=folder/raw.name
    row=dict(ant=ant,track_name=raw.name,source_path=str(source),n_frames=int(manifest['n_frames']))
    extract_one((raw.parent.parent,output,row))
    meta_path=output/'sequences'/name/'metadata.json';meta=json.loads(meta_path.read_text())
    source_scale=float(meta['body_scale_mm'])
    target_scale=scale_mm or _reference_scale(reference,ant) or source_scale
    if 'reference_normalization' not in meta:
        # A comparison must hold the reference ant's length normalization fixed.
        # Physical motion channels are already scale-independent; normalized
        # posture coordinates and their rates require this explicit conversion.
        ratio=source_scale/target_scale
        for filename in ['pose.npy','pose_rate.npy']:
            values=np.load(meta_path.parent/filename,mmap_mode='r+')
            for lo in range(0,len(values),100000):values[lo:lo+100000]*=ratio
            values.flush()
        meta.update(body_scale_mm=target_scale,reference_normalization=dict(source_scale_mm=source_scale,
            reference_scale_mm=target_scale,reference=str(reference)))
        write_json(meta_path,meta)
    make_features((output,ant))
    features=output/'features'/name
    raw_features=np.load(features/'raw.npy',mmap_mode='r')
    model=np.load(reference/'models/dictionary.npz')
    choice=json.loads((reference/'models/selection.json').read_text())
    normalizer=json.loads((reference/'models'/f"{choice['representation']}_normalizer.json").read_text())
    tree=cKDTree(model['x']);labels=model['state']
    state=np.empty(len(raw_features),np.int16);confidence=np.empty(len(raw_features),np.float32)
    for lo in range(0,len(state),20000):
        hi=min(len(state),lo+20000);x=encode(raw_features[lo:hi],normalizer)
        distance,indices=tree.query(x,k=5,workers=1)
        state[lo:hi],confidence[lo:hi]=vote(labels,indices,distance)
    assignment=output/'assignments'/name;assignment.mkdir(parents=True,exist_ok=True)
    np.save(assignment/'state.npy',state);np.save(assignment/'confidence.npy',confidence)
    observed=observed_states(output,ant,int(manifest['n_frames']),True)
    recorded=recorded_mask(manifest);observed[~recorded]=-1
    category=np.full(len(observed),-1,np.int8);known=observed>=0
    category[known]=np.isin(observed[known],[0,1]).astype(np.int8)
    destination=output/'predictions/per_track'/raw.stem;destination.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(destination/'labels.npz',observed=observed,category=category,
                        evidence=known.astype(np.uint8))
    result=dict(track_id=track_id,prediction_eligible=True,reason='locally_valid_full_posture',
        windows=len(state),assigned_frames=int(known.sum()),recorded_frames=int(recorded.sum()),
        sleep_frames=int((category==1).sum()),body_scale_mm=target_scale,axis_sample=axis,
        dictionary_sha256=hashlib.sha256((reference/'models/dictionary.npz').read_bytes()).hexdigest())
    write_json(marker,result);return result


def predict_additional(plan):
    """Single-ant hook after native discovery/assignment; colony code is untouched."""
    output=Path(plan['output']);table=pd.read_csv(output/'inventory.csv')
    all_table=pd.read_csv(output/'inventory_all.csv')
    all_table['prediction_eligible']=all_table.selected.astype(bool)
    source=Path(plan['source']);manifest=json.loads((source/'manifest.json').read_text())
    for row in table.loc[~table.selected].itertuples():
        if not identity_ok(source,int(row.track_id)):continue
        result=predict_track(Path(row.original_source_path),output,output,manifest,int(row.track_id),
            interpolation_root=Path(plan['block'])/'stitched/posture_interpolated/per_track')
        all_table.loc[all_table.track_id.eq(row.track_id),'prediction_eligible']=result['prediction_eligible']
    all_table.to_csv(output/'inventory_all.csv',index=False)
