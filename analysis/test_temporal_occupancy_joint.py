import numpy as np
import pandas as pd
import pytest
from analysis.temporal_occupancy_joint import occupancy_features, align_labels, choose_k


def test_hellinger_features_preserve_outside_mass_and_neighbor_geometry():
    maps=np.array([[[.5,0],[0,0]],[[0,.5],[0,0]],[[0,0],[0,1.]]])
    x=occupancy_features(maps)
    np.testing.assert_allclose(np.sum(x*x,axis=1),1)
    np.testing.assert_allclose(x[:2,-1]**2,.5)
    assert np.linalg.norm(x[0]-x[1])/np.sqrt(2)==pytest.approx(np.sqrt(.5))
    smoothed=occupancy_features(maps,1)
    np.testing.assert_allclose(smoothed[:2,-1]**2,.5)
    assert np.linalg.norm(smoothed[0]-smoothed[1])<np.linalg.norm(x[0]-x[1])
    with pytest.raises(ValueError):occupancy_features(np.zeros((1,2,2)))


def test_bootstrap_label_alignment_cannot_be_driven_by_held_out_ants():
    # In-bag labels are reversed; the larger held-out cohort must not set alignment.
    baseline=np.array([0,1]+[0]*20+[1]*20)
    alternate=np.array([1,0]+[0]*20+[1]*20)
    weights=np.array([1.,1.]+[0.]*40)
    aligned=align_labels(baseline,alternate,weights,2)
    np.testing.assert_array_equal(aligned[:2],baseline[:2])
    np.testing.assert_array_equal(aligned[2:],1-baseline[2:])


def test_model_selection_requires_stability_and_avoids_unsupported_extra_clusters():
    table=pd.DataFrame(dict(k=[2,3,4,5],weighted_silhouette=[.3,.31,.5,.6],
                            bootstrap_ari_median=[.95,.94,.4,.95],
                            min_distinct_ants=[20,8,4,1],min_observed_hours=10))
    assert choose_k(table)==(2,"")
    table.loc[table.k==3,'weighted_silhouette']=.35
    assert choose_k(table)==(3,"")
