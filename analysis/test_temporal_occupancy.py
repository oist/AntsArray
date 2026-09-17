import numpy as np
import pandas as pd

from analysis.temporal_occupancy_cache import temporal_counts
from analysis.temporal_occupancy import calendar_windows, continuity, fit_change_models
from analysis.compute_track_grid_occupancy import compute_grid_occupancy


def test_atomic_counts_preserve_arena_clipping_and_last_bin_edge():
    p=pd.DataFrame(dict(Frame=[0,24,48,72,96,120],X=[10,11,12,12.1,9.9,11],Y=[20,21,22,21,21,20]))
    bounds=dict(x_min_px=10.,x_max_px=12.,y_min_px=20.,y_max_px=22.)
    m=dict(arena_bounds_px=bounds,input_x_origin_px=10.,y_origin_px=20.,mm_per_px=1.)
    original,x,y,_=compute_grid_occupancy(p,side='left',mm_per_px=1.,grid_size_mm=1.,x_min_px=0.,x_split_px=15.,x_max_px=30.,y_min_px=0.,y_max_px=30.,input_x_is_side_local=False,same_shape_sides=False,grid_pad_mm=0.,arena_bounds_px=bounds)
    counts,n=temporal_counts(p,m,x,y,0,24,0,3,width=2)
    assert n.tolist()==[2,2,2] and counts.sum()==4
    np.testing.assert_array_equal((counts.sum(axis=0)/n.sum()).astype('float32'),original)


def test_calendar_window_combines_handoff_without_filling_gap():
    m=dict(windows=[dict(start='2026-07-24 08:00:00',stop='2026-07-24 09:29:59'),dict(start='2026-07-24 09:31:00',stop='2026-07-24 11:59:59')],sources=[dict(info=dict(fps=1)),dict(info=dict(fps=1))])
    windows=calendar_windows(m,4)
    assert len(windows)==1
    assert windows.start.iloc[0]==pd.Timestamp('2026-07-24 08:00:00')
    assert windows.stop.iloc[0]==pd.Timestamp('2026-07-24 12:00:00')
    assert windows.recorded_hours.iloc[0]==(4*3600-60)/3600
    assert windows.max_recording_gap_seconds.iloc[0]==60


def test_recording_gap_prevents_adjacent_feature_jump_claim():
    p=pd.DataFrame([dict(side='left',ant='left:001',bin=b,first_observed=pd.Timestamp(start),last_observed=pd.Timestamp(stop)) for b,start,stop in [(1,'2026-07-23 12:00','2026-07-23 14:57'),(2,'2026-07-23 19:31','2026-07-23 20:00')]])
    steps,partitions=continuity(p,{})
    assert steps.empty and partitions.empty


def test_gradual_and_step_models_are_distinguished_in_original_continuous_score():
    t=np.arange(72,dtype=float)
    rhythm=.1*np.sin(2*np.pi*t/24)
    for y,expected in [(rhythm+.5*t/71,'gradual drift'),(rhythm+.5*(t>=35),'single step')]:
        fits=fit_change_models(t,y,np.ones(len(t)))
        assert min(fits,key=lambda r:r['bic'])['model']==expected


def test_behavior_join_preserves_microsecond_parquet_timestamps():
    from analysis.temporal_occupancy import temporal_behavior
    b=pd.DataFrame(dict(timestamp=pd.to_datetime(['2026-07-24 08:15','2026-07-24 09:45']).as_unit('us'),
                        source_block=['early','later'],ant='left:001',n_expected_frames=43200,
                        colony_percent=[80.,20.],food_percent=0.,water_percent=0.,mean_speed_mm_s=1.,sleep_percent=0.,
                        position_coverage=1.,resource_coverage=1.,speed_coverage=1.,sleep_coverage=1.,
                        trip_available=False,trip_count=np.nan,trip_observed_hours=np.nan,trip_duration_sum_minutes=np.nan))
    result=temporal_behavior(b,4)
    assert len(result)==1
    assert result.bin.iloc[0]==pd.Timestamp('2026-07-24 08:00').value//int(4*3600*1e9)
    assert result.colony_percent.iloc[0]==50.


def test_recording_behavior_keeps_original_labels_and_missing_values():
    from analysis.temporal_occupancy_plots import recording_cohort_behavior
    reference=pd.DataFrame(dict(side=['left','right'],ant=['001','001'],
                                original_cluster=['left_cluster_0','right_cluster_1']))
    whole=pd.DataFrame(dict(side=['left','left','right','left'],ant=['001','001','001','999'],
                           block_index=[1,2,2,2],start=pd.to_datetime(['2026-07-23','2026-07-24','2026-07-24','2026-07-24']),
                           original_cluster=['left_cluster_0','left_cluster_1','right_cluster_0','left_cluster_1'],
                           mean_speed_mm_s=[1.,np.nan,2.,3.]))
    result=recording_cohort_behavior(dict(reference=reference,whole_projected=whole))
    assert len(result)==3
    assert result.baseline_cluster.tolist()==['left_cluster_0','left_cluster_0','right_cluster_1']
    assert pd.isna(result.iloc[1].mean_speed_mm_s)
    assert result.iloc[1].original_cluster=='left_cluster_1'
    assert 'baseline_cluster' not in whole.columns


def test_recording_behavior_rejects_ambiguous_reference_identity():
    import pytest
    from analysis.temporal_occupancy_plots import recording_cohort_behavior
    reference=pd.DataFrame(dict(side=['left','left'],ant=['001','001'],
                                original_cluster=['left_cluster_0','left_cluster_1']))
    with pytest.raises(ValueError,match='one original cluster'):
        recording_cohort_behavior(dict(reference=reference))
