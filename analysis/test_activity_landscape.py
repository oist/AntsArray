"""Checks for label alignment and the density diagnostic, without changing fits."""
import numpy as np
import pandas as pd
from analysis.activity_landscape import log_neighbor_radius
from analysis.activity_landscape_plots import spatial_alignment


def test_spatial_names_are_aligned_without_changing_any_membership():
    table=pd.DataFrame(dict(group=[0,0,0,1,1,1],spatial_label=['z','z','a','a','a','a']))
    before=table.copy(deep=True)
    result=spatial_alignment(table)
    assert result['spatial_to_activity_color']=={'z':0,'a':1}
    assert result['matched']==5 and result['n']==6
    assert result['counts']==[[1,2],[3,0]]
    pd.testing.assert_frame_equal(table,before)
    renamed=table.replace({'z':'arbitrary_first','a':'arbitrary_second'})
    assert spatial_alignment(renamed)['matched']==5


def test_missing_spatial_labels_are_not_counted_as_matches():
    table=pd.DataFrame(dict(group=[0,0,1,1,1],spatial_label=['a','a','b','b',None]))
    result=spatial_alignment(table)
    assert result['matched']==4 and result['n']==4


def test_local_log_radius_respects_translation_rotation_and_scale():
    x=np.random.default_rng(724).normal(size=(40,3));rotation=np.array([[0.,1,0],[-1,0,0],[0,0,1]])
    radius=log_neighbor_radius(x)
    np.testing.assert_allclose(log_neighbor_radius(4*x@rotation+7),radius+np.log(4),atol=1e-10)
