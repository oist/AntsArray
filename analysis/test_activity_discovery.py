"""Regression checks for activity-only selection, timing, and missingness."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score
from analysis.activity_discovery_features import state_runs,block_features,hourly_from_blocks,aggregate_clips,hourly_quantiles
from analysis.activity_discovery import ActivityTransform,choose_activity_method,evaluate


def test_runs_end_at_missing_data_and_never_bridge_unknown_intervals():
    v=np.r_[np.zeros(60),np.full(15,np.nan),np.zeros(50),np.ones(40)]
    state,length=state_runs(v,.05)
    np.testing.assert_array_equal(state,[0,-1,0,1]);np.testing.assert_array_equal(length,[60,15,50,40])
    f=block_features(np.r_[np.zeros(190),np.full(110,np.nan)])
    assert np.isfinite(f).all() and f[4]==0 and f[10]==1
    assert np.isnan(block_features(np.r_[np.zeros(179),np.full(121,np.nan)])).all()


def test_ab_blocks_are_disjoint_and_future_day_cannot_change_training_hours():
    blocks=np.tile(np.arange(12),(48,1)).reshape(576,1).astype(float)
    count=np.full(576,300)
    h,c=hourly_from_blocks(blocks,count)
    np.testing.assert_allclose(h[1],5);np.testing.assert_allclose(h[2],6)
    changed=blocks.copy();changed[288:]+=10000
    again,_=hourly_from_blocks(changed,count)
    np.testing.assert_allclose(h[:3],again[:3]);assert not np.allclose(h[3],again[3])


def test_clip_hour_estimates_follow_five_minute_split_and_no_daily_pool():
    minutes=np.array([0,1,2,3,4,5,6,7,8,9,60,61,62,63,64,65,66,67,68,69])
    values=np.where(minutes%10<5,1.,0.)[:,None]
    h,n=aggregate_clips(values,np.zeros(len(minutes),int),minutes,1)
    np.testing.assert_allclose(h[0,0,:2,0],.5);np.testing.assert_allclose(h[1,0,:2,0],1);np.testing.assert_allclose(h[2,0,:2,0],0)
    assert np.isnan(h[0,0,2:]).all()
    q=hourly_quantiles(h)
    np.testing.assert_allclose(q[0,0],[.5,.5,.5])


def test_transform_has_no_access_to_validation_or_spatial_values():
    rng=np.random.default_rng(42);train=rng.normal(size=(20,6));heldout=rng.normal(size=(20,6))
    transform=ActivityTransform().fit(train,[0,0,0,1,1,1]);center=transform.center.copy();components=transform.pca.components_.copy()
    transform.transform(heldout*1000);np.testing.assert_equal(transform.center,center);np.testing.assert_equal(transform.pca.components_,components)
    with pytest.raises(ValueError):ActivityTransform().fit(np.ones((10,3)),[0,0,0])


def test_method_selection_ignores_spatial_agreement_and_day2():
    t=pd.DataFrame(dict(scope=['common']*4,representation=['a','a','b','b'],algorithm=['gmm']*4,selected_k=[2,2,2,1],activity_gain=[.3,.4,.9,0],spatial_ari=[0,0,1,1],day2_profile_gain=[0,0,100,100]))
    _,chosen=choose_activity_method(t);assert chosen['representation']=='a'
    t.spatial_ari=[1,1,0,0];t.day2_profile_gain=[-100,-100,1000,1000]
    assert choose_activity_method(t)[1]==chosen


def test_stable_activity_mixture_can_choose_two_groups_without_labels():
    rng=np.random.default_rng(11);truth=np.repeat([0,1],30);x=rng.normal(scale=.15,size=(60,3));x[:,0]+=(truth*2-1)*2
    raw=np.stack([x,x+rng.normal(scale=.02,size=x.shape),x+rng.normal(scale=.02,size=x.shape),x+rng.normal(scale=.04,size=x.shape)])
    model,result,_=evaluate(raw,[0,0,0],'gmm',bootstrap=5)
    assert result['selected_k']==2 and adjusted_rand_score(model['labels'],truth)==1
    assert result['replicate_retention']>.95 and result['activity_gain']>0
