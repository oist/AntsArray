# Chunk storage and stitching

Tracking writes one Parquet per chunk and colony. Each row group contains a
single ant within a 3000-frame window, subdivided further at 65,536 rows if
needed. At 24 Hz a window spans 125 seconds. These are storage boundaries;
they do not bin, average, interpolate, or otherwise change the observations.

The stitcher indexes TrackID statistics once per chunk, reuses the footer,
and reads only row groups containing the requested ant. It discovers IDs from
the footer when every group has an unambiguous integer ID. Legacy files with
mixed ants or missing statistics still work, but require scanning. Merely
updating the reader cannot make those old, mixed row groups selective.

Time-window interaction queries still use Frame predicate pushdown. A short
query can decode up to one storage window per boundary; the default interaction
batch is also 3000 frames. The physical order across ants changes, so consumers
requiring global time order must sort by Frame. Order within each ant, all
column values/nulls, and `num_frames` metadata are preserved.

Tracking and stitching publish files by atomic rename after the Parquet footer
has been written. Stitching's `--skip_existing` checks the footer instead of
accepting a filename left by an interrupted write. Hidden `.partial` files
from a hard-killed process are not treated as outputs.

## Existing chunks

Reorganization reads each old chunk once, without repeating detection/tracking.
Use a separate output directory and preserve each filename:

```bash
python -m tracking.core.track_parquet \
  old_tracks/20261001_121948_chunk000_left.parquet \
  indexed_tracks/20261001_121948_chunk000_left.parquet
```

The input must be ordered by Frame, as emitted by the tracker. The command
rejects an in-place destination and unordered input. Point stitching at the new
directory, keeping the original pipeline state/chunk duration so frame offsets
remain anchored to the block. Future tracking runs write this layout directly.

## Checks

`tracking/test_track_parquet.py` exercises row preservation, nulls, selective
reads, Frame queries, legacy layouts, interrupted writes, tracker integration,
and stitching/resuming across gaps in chunk numbers. Existing offset,
interaction-cache, ArUco identity and multi-block combination tests also apply.
