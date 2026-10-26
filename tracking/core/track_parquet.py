"""Ant-selective chunk storage, retaining short time windows for interactions.

The file/column schema is unchanged. Each row group contains one TrackID and
at most 3000 frames (125 seconds at 24 Hz). Parquet statistics can therefore
exclude other ants *before* decoding their coordinates. Input must be ordered
by Frame, as emitted by the tracker. Order within each ant is preserved.

Existing chunks can be reorganized without rerunning tracking::

    python -m tracking.core.track_parquet old.parquet new.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path
import uuid
import weakref

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _discard(writer, path):
    try:
        writer.close()
    finally:
        path.unlink(missing_ok=True)


class AntParquetWriter:
    """Bounded time-window buffer; only publish a closed, successful file."""

    def __init__(self, path, schema, *, frame_window=3000, row_group_size=65536,
                 compression="snappy"):
        if frame_window <= 0 or row_group_size <= 0:
            raise ValueError("frame_window and row_group_size must be positive")
        for name in ("Frame", "TrackID"):
            if not pa.types.is_integer(schema.field(name).type):
                raise ValueError(f"{name} must have an integer type")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temp = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.partial")
        metadata = dict(schema.metadata or {})
        metadata.update({b"track_row_group_layout": b"ant_time_v1",
                         b"track_row_group_frame_window": str(frame_window).encode()})
        self.schema = schema.with_metadata(metadata)
        self.frame_window = int(frame_window)
        self.row_group_size = int(row_group_size)
        self.writer = pq.ParquetWriter(self.temp, self.schema, compression=compression,
                                       write_statistics=True)
        self._cleanup = weakref.finalize(self, _discard, self.writer, self.temp)
        self.parts = []
        self.window = None
        self.last_frame = None
        self.closed = False

    def write_table(self, table):
        if self.closed:
            raise ValueError("Writer is closed")
        if not table.schema.equals(self.schema, check_metadata=False):
            raise ValueError("Table schema differs from output schema")
        if not table.num_rows:
            return
        if table["Frame"].null_count or table["TrackID"].null_count:
            raise ValueError("Frame and TrackID cannot contain nulls")
        frames = table["Frame"].to_numpy()
        if ((self.last_frame is not None and frames[0] < self.last_frame)
                or np.any(frames[1:] < frames[:-1])):
            raise ValueError("AntParquetWriter requires input ordered by Frame")
        self.last_frame = int(frames[-1])
        windows = frames // self.frame_window
        starts = np.r_[0, np.flatnonzero(windows[1:] != windows[:-1]) + 1]
        stops = np.r_[starts[1:], len(frames)]
        for start, stop in zip(starts, stops):
            window = int(windows[start])
            if self.window != window:
                self._flush()
                self.window = window
            self.parts.append(table.slice(int(start), int(stop - start)))

    def _flush(self):
        if not self.parts:
            return
        table = pa.concat_tables(self.parts)
        self.parts = []
        ids = table["TrackID"].to_numpy()
        order = np.argsort(ids, kind="stable")
        table = table.take(pa.array(order))
        ids = ids[order]
        starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
        stops = np.r_[starts[1:], len(ids)]
        for start, stop in zip(starts, stops):
            self.writer.write_table(table.slice(int(start), int(stop - start)),
                                    row_group_size=self.row_group_size)

    def close(self):
        if self.closed:
            return
        try:
            self._flush()
            self.writer.close()
            self.temp.replace(self.path)
        finally:
            self.closed = True
            self._cleanup()

    def abort(self):
        self.parts = []
        self.closed = True
        self._cleanup()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close() if exc_type is None else self.abort()


def ant_row_groups(metadata, track_col="TrackID"):
    """Exact ID -> row groups if ALL groups have unambiguous integer IDs.

    Return None for legacy/mixed/missing-statistic files; callers must scan
    their ID column. No layout marker is trusted without checking statistics.
    """
    names = metadata.schema.names
    if track_col not in names:
        return None
    col = names.index(track_col)
    groups = {}
    for i in range(metadata.num_row_groups):
        column = metadata.row_group(i).column(col)
        stats = column.statistics
        if (column.physical_type not in ("INT32", "INT64") or stats is None
                or not stats.has_min_max or stats.min != stats.max
                or stats.null_count != 0):
            return None
        groups.setdefault(int(stats.min), []).append(i)
    return groups


def candidate_row_groups(metadata, track_col, track_id):
    """Conservatively prune integer ID statistics; otherwise retain a group."""
    names = metadata.schema.names
    if track_col not in names:
        return list(range(metadata.num_row_groups)) if track_id == 0 else []
    col = names.index(track_col)
    result = []
    for i in range(metadata.num_row_groups):
        column = metadata.row_group(i).column(col)
        stats = column.statistics
        if (column.physical_type in ("INT32", "INT64") and stats is not None
                and stats.has_min_max and not stats.min <= track_id <= stats.max):
            continue
        result.append(i)
    return result


def repack(source, destination, *, frame_window=3000):
    """One pass over a legacy frame-ordered chunk; original stays untouched."""
    if Path(source).resolve() == Path(destination).resolve():
        raise ValueError("Choose a separate destination to preserve the original")
    with pq.ParquetFile(source) as pf:
        with AntParquetWriter(destination, pf.schema_arrow, frame_window=frame_window) as writer:
            for batch in pf.iter_batches(batch_size=200_000):
                writer.write_table(pa.Table.from_batches([batch]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--frame-window", type=int, default=3000)
    args = parser.parse_args()
    repack(args.source, args.destination, frame_window=args.frame_window)
