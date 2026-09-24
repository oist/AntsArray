"""Clock-order, missing-hour and selected-partition regression checks."""
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from analysis.postural_dynamics_hourly import (hourly_profiles, distances,
    pairwise_distances, fit_masked, cluster_hourly, select_partition, observed_loss)


def test_hourly_profiles_preserve_clock_order_counts_and_unknown_hours():
    labels=np.array([0]*5+[1]*5+[1]*4+[0]*5)
    ants=np.zeros(len(labels),int)
    minutes=np.array([1]*5+[61]*5+[121]*4+[1441]*5)
    p,n=hourly_profiles(labels,ants,minutes,1,2)
    np.testing.assert_equal(n.sum(),len(labels))
    np.testing.assert_equal(p[0,0,0],[1,0]);np.testing.assert_equal(p[0,0,1],[0,1])
    assert np.isnan(p[0,0,2]).all() and np.isnan(p[0,0,3]).all()
    np.testing.assert_equal(p[1,0,0],[1,0])


def test_missing_hours_are_masked_not_encoded_as_inactivity():
    x=np.tile([.8,.6],(2,24,1));x[1,:8]=np.nan
    centers=np.tile([.8,.6],(1,24,1))
    np.testing.assert_allclose(distances(x,centers),0)
    np.testing.assert_allclose(pairwise_distances(x),0)
    other=centers.copy();other[:,0]=[10,10]
    assert distances(x,other)[0,0]>0 and distances(x,other)[1,0]==0
    loss=observed_loss(x,other.repeat(2,axis=0))
    assert loss[0]>0 and loss[1]==0


def test_equal_daily_frequencies_can_have_different_hourly_activity():
    a=np.tile([1.,0.],(24,1));a[12:]=[0.,1.]
    b=a[::-1].copy();x=np.stack([a]*8+[b]*8)
    np.testing.assert_allclose(x[:8].mean(axis=1),x[8:].mean(axis=1))
    fit=fit_masked(np.sqrt(x),2,n_init=4)
    assert adjusted_rand_score([0]*8+[1]*8,fit['labels'])==1
    assert pairwise_distances(np.sqrt(x))[0,8]>0


def test_displayed_partition_uses_the_selected_k():
    # Identical ants cannot supply a supported split; this used to display K=2.
    x=np.tile([.8,.6],(12,24,1));x[0,:6]=np.nan
    r=cluster_hourly(x,np.arange(12),bootstrap=2)
    assert r['selected_k']==1 and r['centers'].shape[0]==1
    np.testing.assert_array_equal(r['labels'],np.zeros(12,int))
    assert np.isnan(r['oob_agreement']).all()


def test_stability_and_size_rule_selects_actual_supported_partition():
    c=pd.DataFrame(dict(k=[1,2,3,4],smallest_group=[20,10,5,2],
       bootstrap_ari_median=[np.nan,.91,.93,.99],silhouette=[np.nan,.31,.32,.6]))
    assert select_partition(c)==2
    c.loc[c.k>=2,'bootstrap_ari_median']=.79
    assert select_partition(c)==1


def test_masked_kmeans_centers_use_only_observed_training_hours():
    x=np.tile([.8,.6],(3,24,1));x[0,:8]=np.nan;x[1,8:16]=np.nan
    fit=fit_masked(x,1,n_init=1)
    np.testing.assert_allclose(fit['centers'][0],np.tile([.8,.6],(24,1)))
    assert fit['objective']<1e-20
