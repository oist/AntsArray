import json
import numpy as np
import pytest
from tracking.single_ant.state_overlay import intervals, clip_intervals, load_s0


def test_intervals_preserve_unknowns_gaps_and_half_open_edges():
    values=np.array([-1,1,1,0,1,-1,-1,1,0,1])
    np.testing.assert_array_equal(intervals(values==1,100),[[101,103],[104,105],[107,108],[109,110]])
    np.testing.assert_array_equal(intervals(values>=0,100),[[101,105],[107,110]])
    np.testing.assert_array_equal(clip_intervals(intervals(values==1,100),102,109),[[102,103],[104,105],[107,108]])
    assert clip_intervals(intervals(values==1),3,4).shape==(0,2)


def test_only_direct_s0_labels_can_overlay_speed(tmp_path):
    folder=tmp_path/'analysis/eigenposture/sleep_from_states/per_track/TrackID_0002_all_single'
    folder.mkdir(parents=True)
    path=folder/'sleep_motion_label_metadata.json'
    meta=dict(frame_min=10,frame_max=14,fps=24,files=dict(sleep_state='sleep_state.npy'),
        classifier_type='behavior_cluster_membership',classifier_parameters=dict(quiet_states=[0],minimum_seconds=0))
    path.write_text(json.dumps(meta));np.save(folder/'sleep_state.npy',[-1,1,1,0,-1])
    result=load_s0(tmp_path,2)
    np.testing.assert_array_equal(result['s0'],[[11,13]])
    np.testing.assert_array_equal(result['known'],[[11,14]])
    assert load_s0(tmp_path,None) is None and load_s0(tmp_path,3) is None
    meta['classifier_type']='S0_calibrated_motion_sleep';path.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='direct S0'):load_s0(tmp_path,2)


def test_speed_shading_aligns_with_exact_frames_and_clears_on_toggle():
    from matplotlib.figure import Figure
    from types import SimpleNamespace
    from tracking.single_ant.speed_viewer import SpeedViewer
    viewer=object.__new__(SpeedViewer)
    viewer.ax=Figure().subplots();viewer.data=SimpleNamespace(fps=24)
    viewer.state_artists=[];viewer.show_s0=SimpleNamespace(get=lambda:True)
    labels=dict(s0=np.array([[12,24],[36,48]]),known=np.array([[0,24],[36,60]]))
    viewer.overlay_s0(labels,0,3/3600)
    shade=viewer.state_artists[-1]
    assert shade.get_label()=='S0' and len(shade.get_paths())==2
    for path,expected in zip(shade.get_paths(),[[12,24],[36,48]]):
        np.testing.assert_allclose([path.vertices[:,0].min(),path.vertices[:,0].max()],np.array(expected)/24/3600)
    viewer.show_s0.get=lambda:False;viewer.overlay_s0(labels,0,3/3600)
    assert not viewer.state_artists


def test_three_categories_retain_observed_and_inferred_evidence(tmp_path):
    import hashlib
    from tracking.single_ant.state_overlay import load_sleep_categories,state_status
    root=tmp_path/'analysis/sleep_s01_exclusion';folder=root/'per_track/TrackID_0002_all_single'
    folder.mkdir(parents=True);(root/'COMPLETE.json').write_text('{}')
    definition=dict(n_frames=6,fps=24,thresholds_mm_s=dict(velocity=.5,antenna=3.))
    (root/'definition.json').write_text(json.dumps(definition))
    digest=hashlib.sha256((root/'definition.json').read_bytes()).hexdigest()
    (folder/'metadata.json').write_text(json.dumps(dict(definition_sha256=digest)))
    np.savez_compressed(folder/'labels.npz',category=np.array([1,1,0,-1,0,-1],np.int8),
        evidence=np.array([1,1,1,0,2,0],np.uint8),observed=np.array([0,1,2,-1,-1,-1],np.int8))
    labels=load_sleep_categories(tmp_path,2)
    np.testing.assert_array_equal(labels['sleep'],[[0,2]])
    np.testing.assert_array_equal(labels['not_sleep'],[[2,3],[4,5]])
    assert 'observed S1' in state_status(labels,1)
    assert 'observed S2' in state_status(labels,2)
    assert 'inferred from high velocity' in state_status(labels,4)
    assert state_status(labels,3).startswith('Ambiguous')
