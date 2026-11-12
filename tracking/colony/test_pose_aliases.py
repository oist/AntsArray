"""Equivalent camera/NN observations must not behave like additional ants."""
import numpy as np
import pandas as pd

from tracking.core.tracking_utils import get_complete_tracks
from tracking.core.pose_candidates import duplicate_groups

SHAPE=np.array([[0,0],[12,0],[-20,0],[-45,4],[16,5],[24,12],[33,12],
                [16,-5],[24,-12],[33,-12]],float)


def observations(frames=6, offsets=(0.,), cameras=(0,), sizes=(10,)):
    return pd.DataFrame([(f,100+j,bp,x+SHAPE[bp,0],SHAPE[bp,1],cam)
        for f in range(frames) for j,(x,cam,n) in enumerate(zip(offsets,cameras,sizes)) for bp in range(n)],
        columns=['Frame','Instance','Bodypoint','X','Y','Cam'])


def tags(frames=(0,5)):
    return pd.DataFrame([(f,7,0.,0.,0) for f in frames],columns=['Frame','Instance','X','Y','Cam'])


def test_equivalent_camera_views_bridge_marker_dropout(tmp_path):
    p=tmp_path/'tracked.parquet'
    get_complete_tracks(p,tags(),observations(offsets=(0.,.5),cameras=(0,1),sizes=(9,10)),num_frames=6,stream_output=True)
    d=pd.read_parquet(p)
    assert d.Frame.unique().tolist()==list(range(6))
    assert d.groupby('Frame').Bodypoint.nunique().eq(10).all()
    assert d.TrackID.eq(7).all()


def test_complete_same_camera_duplicate_wins_over_closer_partial(tmp_path):
    p=tmp_path/'tracked.parquet'
    get_complete_tracks(p,tags(range(6)),observations(offsets=(0.,.5),cameras=(0,0),sizes=(7,10)),num_frames=6,stream_output=True)
    d=pd.read_parquet(p)
    assert d.groupby('Frame').Bodypoint.nunique().eq(10).all()
    np.testing.assert_allclose(d.loc[d.Bodypoint.eq(0),'X'],.5)


def test_distinct_nearby_small_ants_do_not_become_aliases(tmp_path):
    p=tmp_path/'tracked.parquet'
    get_complete_tracks(p,tags(),observations(offsets=(0.,5.),cameras=(0,1),sizes=(10,10)),num_frames=6,stream_output=True)
    # Both remain plausible at the missing-tag frames: do not pick either.
    assert pd.read_parquet(p).Frame.unique().tolist()==[0,5]


def test_redundant_view_cannot_be_claimed_by_another_lost_identity(tmp_path):
    a=pd.DataFrame([(0,7,0.,0.,0),(0,8,60.,0.,0)],columns=['Frame','Instance','X','Y','Cam'])
    initial=observations(frames=1,offsets=(0.,60.),cameras=(0,0),sizes=(10,10))
    later=observations(frames=1,offsets=(0.,.5),cameras=(0,1),sizes=(10,10));later.Frame=1
    p=tmp_path/'tracked.parquet'
    get_complete_tracks(p,a,pd.concat([initial,later]),num_frames=2,stream_output=True)
    d=pd.read_parquet(p)
    assert set(d.loc[d.Frame.eq(0),'TrackID'])=={7,8}
    assert set(d.loc[d.Frame.eq(1),'TrackID'])=={7}


def test_unsupported_aruco_tags_still_do_not_keep_tracks_alive(tmp_path):
    p=tmp_path/'tracked.parquet';s=observations();s=s.loc[s.Frame.isin([0,5])]
    get_complete_tracks(p,tags(range(6)),s,num_frames=6,stream_output=True)
    assert pd.read_parquet(p).Frame.unique().tolist()==[0,5]


def test_alias_groups_do_not_chain_or_merge_conflicting_barcodes():
    nodes={i:{bp:tuple(point+[2*i,0]) for bp,point in enumerate(SHAPE)} for i in range(3)}
    candidates=[(0.,i) for i in nodes]
    groups=duplicate_groups(candidates,nodes.__getitem__,{i:0 for i in nodes},{},association_distance=90)
    assert sorted(map(len,groups))==[1,2]
    groups=duplicate_groups(candidates[:2],nodes.__getitem__,{0:0,1:0},{0:7,1:8},association_distance=90)
    assert list(map(len,groups))==[1,1]


def test_matching_anchors_with_different_anatomy_are_not_duplicates():
    nodes={0:{bp:tuple(point) for bp,point in enumerate(SHAPE)},
           1:{bp:tuple(-point) for bp,point in enumerate(SHAPE)}}
    groups=duplicate_groups([(0.,0),(0.,1)],nodes.__getitem__,{0:0,1:1},{},association_distance=90)
    assert len(groups)==2


def test_partial_pose_nan_rows_do_not_hide_six_valid_matching_landmarks():
    nodes={i:{bp:tuple(point) for bp,point in enumerate(SHAPE)} for i in [0,1]}
    nodes[0][9]=(np.nan,np.nan)
    groups=duplicate_groups([(0.,0),(0.,1)],nodes.__getitem__,{0:0,1:1},{},association_distance=90)
    assert len(groups)==1
