"""Checks on invariance, leakage, missing-data handling and time boundaries."""
import numpy as np
import pytest

from analysis.postural_dynamics_extract import intrinsic_directions,body_velocity,schedule,DAY1_FRAME,DAY_FRAMES,CLIP_FRAMES,FPS
from analysis.postural_dynamics import causal_pose,clip_quality,history_features,distributions,joint_state,CURRENT,FUTURE


def moving_skeleton(velocity):
    position=np.arange(60)[None,:,None]/FPS*np.asarray(velocity)[None,None,:]
    xy=np.zeros((1,60,10,2));xy[:,:,2,0]=-1
    xy+=position[:,:,None,:]
    return xy,position


def test_velocity_recovers_signed_motion_and_is_rotation_translation_invariant():
    xy,position=moving_skeleton([2.,-.5])
    actual=body_velocity(xy,position,mm_per_px=1)
    np.testing.assert_allclose(actual,np.broadcast_to([2.,-.5],(1,28,2)),atol=1e-6)
    angle=2.1;rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    transformed=body_velocity(xy@rot.T+[1743,-229],position@rot.T+[1743,-229],mm_per_px=1)
    np.testing.assert_allclose(actual,transformed,atol=1e-6)
    # Coordinates in pixels and physical calibration must give the same answer.
    np.testing.assert_allclose(body_velocity(xy/.016,position/.016),actual,atol=1e-6)


def test_velocity_is_causal_and_cannot_span_a_missing_anchor():
    xy,position=moving_skeleton([-1.,.25]);base=body_velocity(xy,position,mm_per_px=1)
    position[:,53:]+=100
    changed=body_velocity(xy,position,mm_per_px=1)
    np.testing.assert_allclose(changed[:,:CURRENT+1],base[:,:CURRENT+1])
    assert not np.allclose(changed[:,25],base[:,25])
    position[:,10]=np.nan
    missing=body_velocity(xy,position,mm_per_px=1)
    assert not np.isfinite(missing[:,3:6]).any()


def test_joint_scaling_balances_blocks_and_does_not_fit_on_heldout_data():
    rng=np.random.default_rng(721)
    pose=rng.normal(size=(8,28,5))*np.arange(1,6);v=rng.normal(size=(8,28,2))*[2,.3]+[.2,-.1]
    train=np.array([0,2,4,6]);state,norm=joint_state(pose,v,train)
    var=state[train,:CURRENT+1].var(axis=(0,1))
    np.testing.assert_allclose([var[:5].sum(),var[5:].sum()],[1,1],atol=1e-6)
    pose[1::2]*=100;v[1::2]+=1000
    other,other_norm=joint_state(pose,v,train)
    for key in norm:np.testing.assert_allclose(norm[key],other_norm[key])
    np.testing.assert_allclose(state[train],other[train])


def test_shape_removes_translation_rotation_scale_but_keeps_head_motion():
    rng=np.random.default_rng(5);xy=rng.normal(size=(12,10,2));base,lengths=intrinsic_directions(xy)
    angle=1.231;rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    moved,other=intrinsic_directions(7.3*(xy@rot.T)+[725,-190])
    np.testing.assert_allclose(moved,base,atol=2e-13);np.testing.assert_allclose(other,7.3*lengths)
    xy[:,1]+=[.7,.8];changed,_=intrinsic_directions(xy)
    assert not np.allclose(changed[...,0,:],base[...,0,:])


def test_causal_smoothing_does_not_leak_future_into_history():
    rng=np.random.default_rng(9);angles=rng.uniform(-1,1,(3,60,8));raw=np.stack([np.cos(angles),np.sin(angles)],axis=-1).reshape(3,60,16)
    original=causal_pose(raw);perturbed=raw.copy();perturbed[:,53:]*=-1
    changed=causal_pose(perturbed)
    np.testing.assert_allclose(history_features(original,2),history_features(changed,2))
    assert not np.allclose(original[:,FUTURE],changed[:,FUTURE])
    np.testing.assert_allclose(np.linalg.norm(original.reshape(3,28,8,2),axis=-1),1,atol=1e-6)


def test_shuffle_keeps_present_and_past_values():
    scores=np.random.default_rng(1).normal(size=(5,28,3))
    ordered=history_features(scores,1).reshape(5,13,3);shuffled=history_features(scores,1,shuffle=True).reshape(5,13,3)
    np.testing.assert_allclose(ordered[:,-1],shuffled[:,-1])
    np.testing.assert_allclose(np.sort(ordered[:,:-1],axis=1),np.sort(shuffled[:,:-1],axis=1))
    assert not np.allclose(ordered,shuffled)
    centered=history_features(scores,1,dynamics_only=True).reshape(5,13,3)
    np.testing.assert_allclose(centered.mean(axis=1),0,atol=1e-15)


def test_missing_point_camera_switch_duplicate_and_bad_geometry_excluded():
    shape=np.ones((5,60,16));lengths=np.full((5,60,9),.4);camera=np.ones((5,60));duplicate=np.zeros((5,60),bool)
    shape[1,20,3]=np.nan;camera[2,20]=2;duplicate[3,20]=True;lengths[4,20,3]=.001
    good,_=clip_quality(shape,lengths,camera,duplicate)
    np.testing.assert_array_equal(good,[True,False,False,False,False])


def test_sampling_never_crosses_minute_or_day_boundaries():
    a=schedule(724);np.testing.assert_array_equal(a,schedule(724));assert not np.array_equal(a,schedule(725))
    starts=(a-DAY1_FRAME)//(60*FPS);ends=(a+CLIP_FRAMES-1-DAY1_FRAME)//(60*FPS)
    np.testing.assert_array_equal(starts,np.arange(2880));np.testing.assert_array_equal(starts,ends)
    assert a[1439]+CLIP_FRAMES<=DAY1_FRAME+DAY_FRAMES<=a[1440]


def test_profiles_preserve_counts_and_missing_days():
    p,n=distributions(np.array([0,1,1,2]),np.array([0,0,1,0]),np.array([0,0,0,1]),2,3)
    np.testing.assert_allclose(p[0,0],[.5,.5,0]);np.testing.assert_array_equal(n.sum(axis=-1),[[2,1],[1,0]])
    np.testing.assert_array_equal(p[1,1],0)
