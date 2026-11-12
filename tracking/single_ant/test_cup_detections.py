import numpy as np
import pytest
from tracking.single_ant.cup_detections import FrameTable,TrackTable,project_points,cup_coordinates,draw_overlay,DEFAULT_EDGES,DEFAULT_NAMES


def test_projection_uses_board_calibration_and_half_pixel_centers():
    cup=dict(center_mm=[10,20],radius_mm=5)
    points=project_points([[100,200]],np.diag([2.,2.,1.]))*.05
    np.testing.assert_allclose(points,[[10,20]])
    np.testing.assert_allclose(cup_coordinates(points,cup,101),[[50,50]])


@pytest.mark.parametrize('frames',[[0,0,3],[3,0,0]])
def test_lossless_frame_index_keeps_repeated_ids_and_empty_frames(tmp_path,frames):
    import h5py
    rows=np.array([(f,8,float(i),2.) for i,f in enumerate(frames)],dtype=[('Frame','i4'),('Instance','i4'),('X','f4'),('Y','f4')])
    path=tmp_path/'a.h5'
    with h5py.File(path,'w') as f:f.attrs['num_frames']=4;f.create_dataset('aruco_detections',data=rows)
    table=FrameTable(path,'aruco_detections',4,'num_frames')
    try:
        assert len(table.frame(0))==2
        assert (table.frame(0)['Instance']==8).all()
        assert len(table.frame(1))==0
        assert len(table.frame(3))==1
    finally:table.close()
    with pytest.raises(ValueError,match='span changed'):FrameTable(path,'aruco_detections',5,'num_frames')


def test_track_row_group_boundaries_and_missing_frames(tmp_path):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    rows=pd.DataFrame(dict(Frame=[0,0,1,1,4,4],Bodypoint=[0,1]*3,X=[1.]*6,Y=[2.]*6,TrackX=[1.]*6,TrackY=[2.]*6))
    path=tmp_path/'track.parquet';pq.write_table(pa.Table.from_pandas(rows,preserve_index=False),path,row_group_size=3)
    table=TrackTable(path)
    try:
        frame,trail=table.frame(1,5)
        assert len(frame)==2 and set(frame.Bodypoint)=={0,1}
        assert np.array_equal(trail[:,0],[0,1])
        missing,trail=table.frame(2,0)
        assert missing is None or len(missing)==0
        assert trail.shape==(0,3)
    finally:table.close()


def test_partial_pose_does_not_join_across_a_missing_node_or_change_video():
    image=np.zeros((100,100),np.uint8);before=image.copy()
    pose=dict(camera=0,instance=1,bodypoints=np.array([4,6]),xy=np.array([[20,50],[80,50]]),scores=np.array([.8,.8]))
    payload=dict(aruco=[],poses=[pose],track=None,trail=np.empty((0,3)),edges=DEFAULT_EDGES,names=DEFAULT_NAMES)
    result=np.asarray(draw_overlay(image,payload,dict(sleap=True,min_score=0)))
    assert (result[50,20]>0).any() and (result[50,80]>0).any()
    assert (result[50,50]==0).all()  # No invented antenna joint/connection.
    np.testing.assert_array_equal(image,before)


def test_camera_ownership_masks_and_all_view_toggle():
    image=np.zeros((100,100),np.uint8)
    payload=dict(aruco=[dict(camera=1,id=8,xy=np.array([50,50]))],poses=[],track=None,trail=np.empty((0,3)),edges=[],names=[])
    mask={1:np.zeros((100,100),bool)}
    hidden=np.asarray(draw_overlay(image,payload,dict(aruco=True),mask))
    shown=np.asarray(draw_overlay(image,payload,dict(aruco=True,all_cameras=True),mask))
    assert not hidden.any() and shown.any()


def test_aruco_only_position_cannot_generate_a_tracked_skeleton():
    image=np.zeros((100,100),np.uint8)
    payload=dict(aruco=[],poses=[],track=dict(bodypoints=np.array([],int),xy=np.empty((0,2)),position=np.array([50,50])),
                 trail=np.empty((0,3)),edges=DEFAULT_EDGES,names=DEFAULT_NAMES)
    pixels=np.asarray(draw_overlay(image,payload,dict(track=True)))
    assert pixels.any()
    # Yellow position marker, no cyan posture pixels.
    assert not ((pixels[:,:,0]<100)&(pixels[:,:,1]>150)&(pixels[:,:,2]>150)).any()
