"""Scientific boundary conditions and lossless publication of interpolated pose."""
import hashlib
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from analysis.compute_track_posture_interpolation import interpolate_table, run


def pose(n=30):
    frame = np.repeat(np.arange(n), 10)
    bp = np.tile(np.arange(10), n)
    return pd.DataFrame(dict(Frame=frame, Bodypoint=bp, X=frame * 2. + bp,
        Y=frame * 3. - bp, TrackID=1, TrackX=20., TrackY=30., SleapCam=2.,
        CameraID=2, score_node=.9, source_file='20260928_135717_chunk000_right.parquet'))


def fill(df, maximum=24):
    table, meta = interpolate_table(pa.Table.from_pandas(df, preserve_index=False), maximum)
    return table.to_pandas(), meta


def test_one_second_boundary_sparse_rows_and_observed_coordinates():
    original = pose(30)
    missing = ((original.Bodypoint == 6) & original.Frame.between(1, 24)
               | (original.Bodypoint == 9) & original.Frame.between(1, 25)
               | (original.Bodypoint == 3) & original.Frame.between(0, 2)
               | (original.Bodypoint == 2) & original.Frame.between(28, 29))
    raw = original[~missing]
    result, meta = fill(raw)
    pd.testing.assert_frame_equal(result[~result.PostureInterpolated][raw.columns].reset_index(drop=True),
                                  raw.reset_index(drop=True))
    new = result[result.PostureInterpolated]
    assert len(new) == 24
    assert (new.Bodypoint == 6).all() and (new.PostureGapFrames == 24).all()
    np.testing.assert_allclose(new.X, new.Frame * 2 + 6)
    np.testing.assert_allclose(new.Y, new.Frame * 3 - 6)
    assert new.score_node.isna().all()  # interpolation is not detector confidence
    assert meta['filled_points'] == 24 and meta['filled_gaps'] == 1


@pytest.mark.parametrize('boundary', ['camera_return', 'missing_pose', 'missing_identity',
                                      'absent_frame', 'recording', 'wrong_endpoint_camera'])
def test_never_bridge_camera_identity_or_recording_discontinuities(boundary):
    df = pose(8)
    df = df[~((df.Bodypoint == 6) & df.Frame.between(2, 4))].copy()
    if boundary == 'camera_return':
        df.loc[df.Frame == 3, 'SleapCam'] = 3.
    elif boundary == 'missing_pose':
        df.loc[df.Frame == 3, 'SleapCam'] = -1.
    elif boundary == 'missing_identity':
        df.loc[df.Frame == 3, 'TrackX'] = np.nan
    elif boundary == 'absent_frame':
        df = df[df.Frame != 3]
    elif boundary == 'recording':
        df.loc[df.Frame >= 3, 'source_file'] = '20260929_135717_chunk000_right.parquet'
    else:
        df.loc[(df.Frame == 1) & (df.Bodypoint == 6), 'SleapCam'] = 3.
    result, meta = fill(df)
    assert meta['filled_points'] == 0 and len(result) == len(df)


def test_partial_nan_chunk_boundary_and_coverage_counts():
    df = pose(8)
    df.loc[(df.Frame == 3) & (df.Bodypoint == 6), 'Y'] = np.nan
    df.loc[(df.Frame == 3) & (df.Bodypoint == 6), 'X'] = 99.
    df.loc[df.Frame >= 4, 'source_file'] = '20260928_135717_chunk001_right.parquet'
    result, meta = fill(df)
    row = result[(result.Frame == 3) & (result.Bodypoint == 6)].iloc[0]
    assert row.X == 99. and row.Y == 3. and row.PostureInterpolated
    assert meta['complete_frames_before'] == 7 and meta['complete_frames_after'] == 8
    assert len(result) == len(df)


def test_duplicate_input_rejected_and_unsorted_supported():
    df = pose(4)
    result, _ = fill(df.iloc[::-1])
    pd.testing.assert_frame_equal(result[df.columns], df)
    with pytest.raises(ValueError, match='Duplicate'):
        fill(pd.concat([df, df.iloc[:1]]))
    empty, meta = fill(df.iloc[:0])
    assert len(empty) == 0 and meta['filled_points'] == 0


def test_disk_output_preserves_timing_raw_data_and_invalidates_completion(tmp_path):
    df = pose()
    df = df[~((df.Bodypoint == 6) & df.Frame.between(1, 24))]
    source = tmp_path / 'TrackID_0001_all_right.parquet'
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table.replace_schema_metadata({b'num_frames': b'100', b'fps': b'24'}), source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    out = tmp_path / 'output'
    meta = run(source, out)
    result = pq.read_table(out / source.name)
    assert result.schema.metadata[b'num_frames'] == b'100'
    assert result.schema.metadata[b'fps'] == b'24'
    assert meta['filled_points'] == 24 and meta['num_frames'] == 100
    assert json.loads((out / 'posture_interpolation_metadata.json').read_text()) == meta
    assert (out / '_SUCCESS').is_file()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    with pytest.raises(ValueError, match='fps differs'):
        run(source, out, fps=12.)
    assert not (out / '_SUCCESS').exists()
    with pytest.raises(ValueError, match='separate'):
        run(source, tmp_path)


def test_generated_worker_runs_and_writes_real_output(tmp_path):
    from pathlib import Path
    import subprocess
    import sys
    repo = Path(__file__).resolve().parents[1]
    tracks = tmp_path / 'tracks'; tracks.mkdir()
    source = tracks / 'TrackID_0001_all_right.parquet'
    df = pose(8)
    df = df[~((df.Bodypoint == 6) & df.Frame.between(2, 4))]
    df.to_parquet(source, index=False)
    output = tmp_path / 'output'
    subprocess.run(['bash', str(repo / 'scripts/per_track_slurm_fanout.sh'),
        '--per_track_dir', str(tracks), '--flash_output_dir', str(output),
        '--bucket_output_dir', str(tmp_path / 'bucket'), '--no_transfer_to_bucket',
        '--operation_script', 'analysis/compute_track_posture_interpolation.py',
        '--operation_args', '--fps 24 --max_gap_seconds 1', '--run_workdir', str(repo),
        '--worker_python_bin', sys.executable, '--python_bin', sys.executable, '--dry_run'],
        check=True, capture_output=True)
    worker = next((output / 'jobs/workers').glob('*.sbatch'))
    subprocess.run(['bash', str(worker)], check=True, capture_output=True)
    task = output / 'per_track' / source.stem
    assert (task / '_SUCCESS').is_file()
    assert json.loads((task / 'posture_interpolation_metadata.json').read_text())['filled_points'] == 3
