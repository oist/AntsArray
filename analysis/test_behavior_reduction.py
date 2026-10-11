"""State merges conserve observations; class selection rewards reproducibility."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from analysis.full_behavior_states import merge_map, between_state_variance
from analysis.full_behavior_roles import merge_counts, proportions, select_model


def test_merging_preserves_every_ant_time_bin_and_unknown_bins():
    counts=np.array([[[4,2,1],[0,0,0]],[[0,3,5],[1,0,2]]])
    merged=merge_counts(counts,[0,0,1])
    np.testing.assert_array_equal(merged,[[[6,1],[0,0]],[[3,5],[1,2]]])
    np.testing.assert_array_equal(merged.sum(-1),counts.sum(-1))
    assert np.isnan(proportions(merged)[0,1]).all()
    with pytest.raises(ValueError):merge_counts(counts,[0,2,2])


@pytest.mark.parametrize('weighting',['state','observation'])
def test_state_merges_ignore_arbitrary_ids_and_preserve_variation_order(weighting):
    x=np.array([[0,0],[.1,0],[.2,0],[.3,0],[8,1],[8.2,1],[8,9],[8.2,9]])
    labels=np.repeat(np.arange(4),2)
    permutation=np.array([2,0,3,1])
    previous=between_state_variance(x,labels)
    for k in (4,3,2,1):
        result=merge_map(x,labels,k,weighting)[labels]
        renamed=merge_map(x,permutation[labels],k,weighting)[permutation[labels]]
        assert adjusted_rand_score(result,renamed)==1
        variation=between_state_variance(x,result)
        assert variation<=previous+1e-10
        previous=variation
    assert previous==pytest.approx(0)


def test_weighted_ward_uses_observation_counts_not_equal_centroid_votes():
    # Equal centroid weights merge the two nearby common states. Actual sample
    # mass instead merges the rare outlier, losing less total feature variation.
    x=np.r_[np.zeros(100),np.full(100,2.),[10.]][:,None]
    labels=np.r_[np.zeros(100,int),np.ones(100,int),[2]]
    mapping=merge_map(x,labels,2,'observation')
    assert mapping[0]!=mapping[1] and mapping[1]==mapping[2]
    equal=merge_map(x,labels,2,'state')
    assert equal[0]==equal[1] and equal[1]!=equal[2]
    merged=mapping[labels]
    direct=sum(((x[merged==s]-x[merged==s].mean(0))**2).sum() for s in np.unique(merged))
    assert direct==pytest.approx(100*64/101)


def test_more_classes_do_not_win_just_by_passing_thresholds():
    table=pd.DataFrame([
        dict(model='four',k=4,qualifies=True,stability_score=.65,silhouette=.4),
        dict(model='two',k=2,qualifies=True,stability_score=.9,silhouette=.5),
        dict(model='unsupported',k=3,qualifies=False,stability_score=1.,silhouette=.9)])
    assert select_model(table)['model']=='two'
    table.loc[0,'stability_score']=.9
    assert select_model(table)['model']=='two'
    assert select_model(table.assign(qualifies=False))['k']==1
