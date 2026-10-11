"""View offsets must not look like movement or discard valid overlap posture."""
import json
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from analysis.camera_motion import camera_local_fit, recording_keys
from analysis.sleep_motion_utils import compute_sleep_motion_cache, load_sleep_motion_cache
from analysis.full_behavior_extract import extract_one
from analysis.s0_motion_sleep import PARAMETERS


def test_camera_fit_recovers_motion_through_alternating_offset_views():
    frame=np.arange(40);camera=frame%2;recording=np.zeros(40,int)
    values=(.2*frame**2+3*frame+100*camera)[:,None]
    centers=np.arange(4,36)
    fitted,rate=camera_local_fit(values,camera,recording,centers)
    np.testing.assert_allclose(fitted[:,0],values[centers,0],atol=1e-10)
    np.testing.assert_allclose(rate[:,0],.4*centers+3,atol=1e-10)
    # Consecutive observations reproduce the original SG filter exactly.
    fitted,rate=camera_local_fit(values,np.zeros(40,int),recording,centers)
    np.testing.assert_allclose(fitted[:,0],savgol_filter(values[:,0],5,2)[centers],atol=1e-10)
    np.testing.assert_allclose(rate[:,0],savgol_filter(values[:,0],5,2,deriv=1)[centers],atol=1e-10)


def test_camera_fit_requires_observed_center_short_support_and_same_recording():
    frame=np.arange(40);values=frame[:,None].astype(float);camera=np.zeros(40,int)
    rec=(frame>=20).astype(int);values[10:15]=np.nan
    fitted,_=camera_local_fit(values,camera,rec,frame)
    assert np.isnan(fitted[10:15]).all()
    assert np.isnan(fitted[9]).all()  # Earlier support cannot bridge six frames.
    assert np.isnan(fitted[18:22]).all()  # No fit across recording boundary.
    assert np.isfinite(fitted[4:8]).all()


def test_motion_cache_pairs_within_view_and_recording_and_records_actual_gap(tmp_path):
    frame=np.arange(16);camera=frame%2
    source=tmp_path/'ant.parquet'
    pd.DataFrame(dict(Frame=frame,Bodypoint=0,X=frame+100*camera,Y=0,
        SleapCam=camera,source_file=np.where(frame<8,'r1_chunk000','r2_chunk000'))).to_parquet(source)
    compute_sleep_motion_cache(source,tmp_path/'cache',fps=1,mm_per_px=1,cache_max_gap_frames=5)
    cache=load_sleep_motion_cache(tmp_path/'cache')
    assert cache.metadata['format_version']==3
    assert cache.metadata['motion_version']=='same_camera_v1'
    assert np.isnan(cache.speed_mm_s[[0,1,8,9],0]).all()
    use=np.setdiff1d(frame,[0,1,8,9])
    np.testing.assert_allclose(cache.speed_mm_s[use,0],1)
    np.testing.assert_array_equal(cache.frame_gap[use,0],2)
    assert cache.metadata['n_valid_speed_values']==12


def test_extraction_retains_mixed_camera_posture_without_handoff_motion(tmp_path):
    skeleton=np.array([[0,0],[12,0],[-20,0],[-45,4],[16,5],[24,12],[33,12],
                       [16,-5],[24,-12],[33,-12]],float)
    frame=np.repeat(np.arange(40),10);point=np.tile(np.arange(10),40)
    camera=(frame//2)%2;track=tmp_path/'track.parquet'
    pd.DataFrame(dict(Frame=frame,Bodypoint=point,X=skeleton[point,0]+100*camera,
        Y=skeleton[point,1],SleapCam=camera,TrackX=50.,TrackY=80.,CameraID=2,
        source_file='r1_chunk000')).to_parquet(track,index=False)
    row=dict(ant='right:001',track_name=track.name,source_path=str(track),n_frames=40)
    extract_one((tmp_path,tmp_path/'out',row))
    root=tmp_path/'out/sequences/right_001'
    motion=np.load(root/'motion.npy');pose=np.load(root/'pose.npy')
    assert np.isfinite(motion[2:18]).all()
    np.testing.assert_allclose(motion[2:18],0,atol=1e-8)
    np.testing.assert_allclose(pose[2:18],np.repeat(pose[2:3],16,axis=0),atol=1e-8)
    np.testing.assert_allclose(np.load(root/'pose_rate.npy')[2:18],0,atol=1e-8)
    # At 12 Hz the retained camera alternates every sample. Window support and
    # PC rates must survive that alternation without differentiating offsets.
    from analysis.full_behavior_states import make_features
    output=tmp_path/'out';contacts=output/'contacts';contacts.mkdir()
    np.save(contacts/'right_001.npy',np.zeros(40,np.uint8))
    np.save(contacts/'right_covered.npy',np.ones(40,bool))
    np.savez(output/'full_body_pca.npz',center=np.zeros(20),modes=np.eye(20))
    result=make_features((output,'right:001'))
    assert result['n_assigned_samples']==16
    features=np.load(output/'features/right_001/raw.npy')
    np.testing.assert_allclose(features[:,16:29],0,atol=1e-8)


def test_unknown_camera_has_no_motion_but_legacy_single_view_still_works(tmp_path):
    source=tmp_path/'ant.parquet'
    rows=pd.DataFrame(dict(Frame=np.arange(4),Bodypoint=0,X=np.arange(4),Y=0))
    rows.assign(SleapCam=[0,np.nan,-1,0]).to_parquet(source)
    compute_sleep_motion_cache(source,tmp_path/'known',fps=1,mm_per_px=1)
    speed=load_sleep_motion_cache(tmp_path/'known').speed_mm_s[:,0]
    assert np.isnan(speed[:3]).all() and speed[3]==1
    rows.to_parquet(source)
    compute_sleep_motion_cache(source,tmp_path/'legacy',fps=1,mm_per_px=1)
    cache=load_sleep_motion_cache(tmp_path/'legacy')
    assert cache.metadata['camera_provenance']=='legacy_single_view'
    np.testing.assert_array_equal(cache.speed_mm_s[1:,0],1)


def test_adjacent_chunks_are_one_recording_and_sleep_is_one_second():
    rows=pd.DataFrame(dict(source_file=['r1_chunk000','r1_chunk001','r2_chunk000']))
    keys=recording_keys(rows)
    assert keys[0]==keys[1] and keys[1]!=keys[2]
    assert PARAMETERS['window_seconds']==1.
