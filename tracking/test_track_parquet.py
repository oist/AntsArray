"""Storage changes must preserve poses, temporal queries and stitched clocks."""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tracking.core.track_parquet import AntParquetWriter, ant_row_groups, repack
from tracking import stitch_tracks as st


def poses():
    df = pd.DataFrame([(f, ant, bp, float(f + ant + bp), float(f - ant))
                       for f in range(23) for ant in (7, 2, 42) for bp in (0, 1)],
                      columns=["Frame", "TrackID", "Bodypoint", "X", "Y"])
    df.loc[3, "X"] = np.nan
    df["CameraID"] = pd.array([None if i % 3 == 0 else 4 for i in range(len(df))], dtype="Int64")
    table = pa.Table.from_pandas(df, preserve_index=False)
    return table.replace_schema_metadata({**table.schema.metadata, b"num_frames": b"30"})


def write_grouped(path, table):
    with AntParquetWriter(path, table.schema, frame_window=5, row_group_size=8) as writer:
        for batch in table.to_batches(max_chunksize=17):
            writer.write_table(pa.Table.from_batches([batch]))


def test_rows_order_nulls_and_time_windows(tmp_path):
    table = poses()
    path = tmp_path / "grouped.parquet"
    write_grouped(path, table)
    pf = pq.ParquetFile(path)
    assert pf.schema_arrow.metadata[b"num_frames"] == b"30"
    assert pf.metadata.num_rows == table.num_rows
    groups = ant_row_groups(pf.metadata)
    assert set(groups) == {2, 7, 42}
    for ant, indices in groups.items():
        expected = table.to_pandas().query("TrackID == @ant").reset_index(drop=True)
        pd.testing.assert_frame_equal(pf.read_row_groups(indices).to_pandas(), expected)
        for index in indices:
            stats = pf.metadata.row_group(index).column(0).statistics
            assert stats.min // 5 == stats.max // 5
    # Interaction queries use Frame predicates and sort poses after reading.
    for start, stop in ((0, 3), (4, 16), (22, 30)):
        result = pq.read_table(path, filters=[("Frame", ">=", start), ("Frame", "<", stop)]).to_pandas()
        expected = table.to_pandas().query("@start <= Frame < @stop")
        keys = ["Frame", "TrackID", "Bodypoint"]
        pd.testing.assert_frame_equal(result.sort_values(keys).reset_index(drop=True),
                                      expected.sort_values(keys).reset_index(drop=True))


def test_reader_does_not_decode_other_ants(tmp_path, monkeypatch):
    path = tmp_path / "grouped.parquet"
    table = poses()
    write_grouped(path, table)
    expected_groups = ant_row_groups(pq.read_metadata(path))[7]
    read = pq.ParquetFile.read_row_group
    calls = []

    def record(self, index, *args, **kwargs):
        calls.append(index)
        return read(self, index, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", record)
    result = st.read_track_rows_streaming(path, columns=table.column_names,
                                          track_col="TrackID", track_id=7, engine="pyarrow")
    assert calls == expected_groups
    pd.testing.assert_frame_equal(result, table.to_pandas().query("TrackID == 7").reset_index(drop=True))


@pytest.mark.parametrize("statistics", [True, False])
def test_legacy_mixed_groups_still_work(tmp_path, statistics):
    table = poses()
    path = tmp_path / "legacy.parquet"
    pq.write_table(table, path, row_group_size=19, write_statistics=statistics)
    assert ant_row_groups(pq.read_metadata(path)) is None
    result = st.read_track_rows_streaming(path, columns=table.column_names,
                                          track_col="TrackID", track_id=7, engine="pyarrow")
    pd.testing.assert_frame_equal(result, table.to_pandas().query("TrackID == 7").reset_index(drop=True))


def test_missing_track_column_legacy_zero(tmp_path):
    table = poses().drop(["TrackID"])
    path = tmp_path / "legacy.parquet"
    pq.write_table(table, path)
    for ant in (0, 7):
        result = st.read_track_rows_streaming(path, columns=table.column_names,
                                              track_col="TrackID", track_id=ant, engine="pyarrow")
        assert len(result) == (len(table) if ant == 0 else 0)


def test_stitch_indexes_ids_from_footer_without_column_scan(tmp_path, monkeypatch):
    path = tmp_path / "arena_chunk000_left.parquet"
    write_grouped(path, poses())
    output = tmp_path / "stitched"
    output.mkdir()

    def reject_scan(*args, **kwargs):
        raise AssertionError("Indexing ant-homogeneous groups must not scan a column")

    monkeypatch.setattr(pd, "read_parquet", reject_scan)
    st.stitch_group([path], "left", output, poses().column_names, chunk_frames=30)
    assert len(list(output.glob("*.parquet"))) == 3


def test_failed_write_preserves_previous_output(tmp_path):
    path = tmp_path / "chunk.parquet"
    table = poses()
    pq.write_table(table, path)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="ordered by Frame"):
        with AntParquetWriter(path, table.schema, frame_window=5) as writer:
            writer.write_table(table)
            writer.write_table(table)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_empty_file_is_valid(tmp_path):
    path = tmp_path / "empty.parquet"
    with AntParquetWriter(path, poses().schema):
        pass
    assert pq.read_metadata(path).num_rows == 0
    assert ant_row_groups(pq.read_metadata(path)) == {}


def test_repack_stitch_and_resume_keep_values_and_global_frames(tmp_path):
    table = poses()
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    for chunk in (5, 8):
        name = f"arena_chunk{chunk:03d}_left.parquet"
        pq.write_table(table, old / name, row_group_size=19)
        repack(old / name, new / name, frame_window=5)
    outputs = []
    for directory in (old, new):
        output = directory / "stitched"
        output.mkdir()
        # This is exactly what an interrupted legacy stitch leaves behind.
        st.stitched_parquet_path(output, 7, "left").write_bytes(b"PAR1unfinished")
        st.stitch_group(sorted(directory.glob("*.parquet")), "left", output,
                        table.column_names, chunk_frames=30, skip_existing=True)
        outputs.append(output)
    for ant in (2, 7, 42):
        paths = [st.stitched_parquet_path(out, ant, "left") for out in outputs]
        pd.testing.assert_frame_equal(*(pd.read_parquet(p) for p in paths))
        assert st.parquet_num_frames(paths[1]) == 270
        assert pd.read_parquet(paths[1]).Frame.min() == 150


def test_stitched_write_failure_is_atomic(tmp_path, monkeypatch):
    path = tmp_path / "stitched.parquet"
    pq.write_table(poses(), path)
    original = path.read_bytes()

    def fail(table, destination, **kwargs):
        from pathlib import Path
        Path(destination).write_bytes(b"PAR1partial")
        raise OSError("test disk failure")

    monkeypatch.setattr(pq, "write_table", fail)
    with pytest.raises(OSError, match="disk failure"):
        st.write_parquet_with_num_frames(poses().to_pandas(), path, num_frames=30,
                                         engine="pyarrow", compression="zstd")
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("empty", [False, True])
def test_tracker_stream_and_memory_outputs_agree(tmp_path, empty):
    from tracking.core.tracking_utils import get_complete_tracks
    aruco = pd.DataFrame([(f, ant, float(ant * 100 + f), 100.)
                          for f in range(6) for ant in (2, 7)],
                         columns=["Frame", "Instance", "X", "Y"])
    sleap = pd.DataFrame([(f, ant, bp, float(ant * 100 + f + bp), 100.)
                          for f in range(6) for ant in (2, 7) for bp in (0, 1)],
                         columns=["Frame", "Instance", "Bodypoint", "X", "Y"])
    if empty:
        aruco, sleap = aruco.iloc[:0], sleap.iloc[:0]
    outputs = []
    for stream in (False, True):
        path = tmp_path / f"stream_{stream}.parquet"
        get_complete_tracks(path, aruco, sleap, num_frames=8, stream_output=stream,
                            output_batch_rows=7)
        assert st.parquet_num_frames(path) == 8
        groups = ant_row_groups(pq.read_metadata(path))
        assert set(groups) == (set() if empty else {2, 7})
        outputs.append(pd.read_parquet(path).astype(float))
    pd.testing.assert_frame_equal(*outputs, check_dtype=False)
