"""Scientific invariants for normalization, holdout and identity controls."""
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist

from analysis.colony_behavioral_landscape import align, features, interval_frames, lag_statistics, matched_days


def test_hellinger_preserves_outside_mass_and_rejects_missing():
    p=np.array([[[.5,.25]],[[.2,.6]]])
    x=features(p)
    np.testing.assert_allclose(x*x,[[.5,.25,.25],[.2,.6,.2]])
    np.testing.assert_allclose((x*x).sum(axis=1),1)
    expected=np.sqrt(np.sum((np.sqrt([.5,.25,.25])-np.sqrt([.2,.6,.2]))**2)/2)
    np.testing.assert_allclose(pdist(x)/np.sqrt(2),expected)
    for bad in (np.zeros((1,2,2)),np.array([[[1.1]]]),np.array([[[-.1,.5]]]),np.array([[[np.nan]]])):
        with pytest.raises(ValueError):features(bad)


def test_observed_outside_only_is_distinct_from_missing():
    outside=features(np.zeros((1,1,2)),observed=np.array([24]))
    np.testing.assert_array_equal(outside,[[0,0,1]])
    with pytest.raises(ValueError):features(np.zeros((1,1,2)),observed=np.array([0]))


def test_partial_edges_count_frames_exactly_and_holdout_is_clock_matched():
    start=pd.Timestamp('2026-07-24 09:31:10')
    first=start.value//(1800*10**9)
    bins=np.arange(first,first+99)
    n=4_247_196
    exposure=interval_frames(bins,start,n)
    assert exposure.sum()==n
    assert exposure[0]==(1800-70)*24
    day1,day2=matched_days(bins,exposure)
    assert not (day1 & day2).any()
    assert exposure[day1].sum()==exposure[day2].sum()==24*3600*24
    np.testing.assert_array_equal(bins[day2]-bins[day1],48)
    assert pd.Timestamp(int(bins[day1][0])*1800*10**9)==pd.Timestamp('2026-07-24 10:00')


def test_holdout_rejects_incomplete_second_day():
    bins=np.arange(80)
    with pytest.raises(ValueError):matched_days(bins,np.full(80,43200))


def test_alignment_does_not_use_absent_ant_labels():
    reference=np.array([0,1,0,0,0,0,0])
    prediction=np.array([1,0,0,0,0,0,0])
    aligned=align(reference,prediction,np.array([1,1,0,0,0,0,0],bool))
    np.testing.assert_array_equal(aligned,[0,1,1,1,1,1,1])


def test_identity_difference_does_not_masquerade_as_within_ant_persistence():
    rng=np.random.default_rng(31)
    scores=np.repeat(np.linspace(-3,3,40)[:,None],300,axis=1)+rng.normal(0,.15,(40,300))
    scores[rng.random(scores.shape)<.1]=np.nan
    lag=lag_statistics(scores,4)
    assert (lag.pooled>.98).all()
    assert (abs(lag.within_ant)<.04).all()
    assert (lag.n_ants==40).all()


def test_lags_do_not_close_missing_time_gaps():
    scores=np.tile(np.array([0.,np.nan,1.,np.nan,0.,np.nan,1.,np.nan,0.,np.nan,1.,np.nan,0.,np.nan,1.,np.nan,0.]),(4,1))
    with np.errstate(invalid='ignore'):
        result=lag_statistics(scores,2)
    assert result.iloc[0].n_pairs==0
    assert result.iloc[1].n_pairs==32
