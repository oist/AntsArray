from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from tracking.single_ant.cup_video import CameraDecoder, MatroskaSource, cluster_description, vint_bytes
from tracking.single_ant.video_viewer import CupData, cup_maps, full_posture_categories, segment_at, timeline_image


def test_stitched_frames_keep_source_offset_and_unrecorded_gap():
    segments = [dict(frame_offset=0, num_frames=100), dict(frame_offset=120, num_frames=200)]
    assert segment_at(segments, 99) == (0, 99)
    assert segment_at(segments, 100) == (None, None)
    assert segment_at(segments, 119) == (None, None)
    assert segment_at(segments, 120) == (1, 0)
    assert segment_at(segments, 319) == (1, 199)
    assert segment_at(segments, 320) == (None, None)


def test_full_classifier_means_original_s0_s1_with_unknown_preserved():
    assert np.array_equal(full_posture_categories(np.array([-1, 0, 1, 2, 3, 4, 5])), [-1, 1, 1, 0, 0, 0, 0])


def test_loader_ignores_motion_fills_and_abandoned_partial_classifier(tmp_path):
    import json
    root = tmp_path
    np.savez(root / 'H.npz', H=np.eye(3)[None])
    (root / 'manifest.json').write_text(json.dumps(dict(fps=24, n_frames=4, start_datetime='2026-10-27T00:00:00+09:00',
        recording_segments=[dict(frame_offset=0, num_frames=4, source_block=str(root))],
        cups=[dict(cup_id=1, center_mm=[1, 1], radius_mm=1)])))
    (root / 'panorama_metadata.json').write_text(json.dumps(dict(mm_per_pixel=1, homographies=str(root / 'H.npz'))))
    (root / 'cup_identities.csv').write_text('CupID,TrackID,ArucoID,Eligible\n1,1,7,True\n')
    for source in ['sleep_s01_exclusion', 'sleep_s01_relaxed_candidate']:
        folder = root / 'analysis' / source / 'per_track/TrackID_0001_all_single';folder.mkdir(parents=True)
        np.savez(folder / 'labels.npz', observed=np.array([-1, 0, 1, 4], np.int8), category=np.zeros(4, np.int8))
    assert np.array_equal(CupData(root).labels(0)['category'], [-1, 1, 1, 0])


def test_cup_projection_uses_calibration_mm_per_pixel_not_track_output_scale():
    # A 10-mm cup radius at 1 mm/calibration-pixel, inverse H scale 2.
    cup = dict(center_mm=[25, 25], radius_mm=10)
    plans, needed, _ = cup_maps(cup, np.array([np.diag([.5, .5, 1])]), 1., 21, width=100, height=100)
    assert needed == [0]
    assert plans[0]['map_x'][10, 10] == pytest.approx(50)
    assert plans[0]['map_y'][10, 10] == pytest.approx(50)
    assert plans[0]['coverage'] == 1


def test_timeline_counts_short_events_instead_of_sampling_them_away():
    labels = np.array([1] * 5 + [0] * 3 + [-1] * 2)
    pixels = np.array(timeline_image(labels, 0, 10, 1, 10))[:, 0]
    assert np.array_equal(pixels[:5], np.tile([132, 82, 184], (5, 1)))
    assert np.array_equal(pixels[5:8], np.tile([217, 151, 40], (3, 1)))
    assert np.array_equal(pixels[8:], np.tile([168, 173, 183], (2, 1)))


def test_ebml_vint_and_invalid_prefix():
    assert vint_bytes(bytes.fromhex('1f43b675'), identifier=True) == (0x1F43B675, 4)
    assert vint_bytes(bytes.fromhex('4080')) == (128, 2)
    with pytest.raises(ValueError):
        vint_bytes(b'\0')


def test_frame_exact_random_seek_reverse_and_cache_with_b_frames(tmp_path):
    pytest.importorskip('av')
    if not shutil.which('ffmpeg'):
        pytest.skip('ffmpeg required for a deterministic video fixture')
    path = tmp_path / 'source.mkv'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=96x64:rate=24',
                    '-frames:v', '240', '-c:v', 'libx264', '-g', '48', '-bf', '2', '-crf', '15', str(path)], check=True)
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'rawvideo', '-pix_fmt', 'gray', 'pipe:1'])
    expected = np.frombuffer(raw, np.uint8).reshape(-1, 64, 96)
    yy, xx = np.mgrid[:64, :96].astype(np.float32)
    reader = CameraDecoder(path, 24, 240, xx, yy)
    try:
        for frame in [0, 1, 201, 200, 199, 2, 239, 48, 47, 51]:
            actual = reader.read(frame)
            assert np.abs(actual.astype(float) - expected[frame]).max() <= 1, frame
    finally:
        reader.close()


def test_video_motion_overlay_never_adds_sleep_or_overwrites_original(tmp_path):
    import json
    root=tmp_path
    np.savez(root/'H.npz',H=np.eye(3)[None])
    (root/'manifest.json').write_text(json.dumps(dict(fps=24,n_frames=5,start_datetime='2026-10-27T00:00:00+09:00',
        recording_segments=[dict(frame_offset=0,num_frames=4,source_block=str(root))],
        cups=[dict(cup_id=1,center_mm=[1,1],radius_mm=1)])))
    (root/'panorama_metadata.json').write_text(json.dumps(dict(mm_per_pixel=1,homographies=str(root/'H.npz'))))
    (root/'cup_identities.csv').write_text('CupID,TrackID,ArucoID,Eligible\n1,1,7,True\n')
    original=root/'analysis/sleep_s01_exclusion/per_track/TrackID_0001_all_single';original.mkdir(parents=True)
    np.savez(original/'labels.npz',observed=np.array([-1,0,1,4,-1],np.int8))
    motion=root/'analysis/sleep_s01_video_energy';(motion/'per_cup/cup_0001').mkdir(parents=True)
    (motion/'application.json').write_text(json.dumps({'released':True}))
    # Even a malformed overlay cannot turn original sleep negative, nor fill a recording gap.
    np.savez(motion/'per_cup/cup_0001/labels.npz',category=np.zeros(5,np.int8),evidence=np.full(5,32,np.uint8))
    data=CupData(root);data.use_video_motion=True
    np.testing.assert_array_equal(data.labels(0)['category'],[0,1,1,0,-1])
    assert data.status(0,0)[1]=='NOT S0/S1 · MOTION'
    data.use_video_motion=False
    np.testing.assert_array_equal(data.labels(0)['category'],[-1,1,1,0,-1])
    data.use_video_motion=True;data.state_cache.clear()
    (motion/'application.json').write_text(json.dumps({'released':False}))
    np.testing.assert_array_equal(data.labels(0)['category'],[-1,1,1,0,-1])
