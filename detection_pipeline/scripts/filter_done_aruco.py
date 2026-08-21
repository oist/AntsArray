#!/usr/bin/env python3
"""Filter a worklist down to the chunks whose ArUco output is not already on the bucket.

The ArUco counterpart of ``filter_done_chunks.py``. Until this existed the two
legs were asymmetric in a way that only showed up on a re-run: SLEAP skipped
chunks already complete on the bucket, while ArUco's only skip lived in
``aruco_array.sbatch`` and looked at ``$FLASH_ROOT`` -- so once ``cleanup`` freed
/flash, any later run recomputed ArUco for the entire block. On a 4,950-chunk
block that is days of `compute` time spent redoing work already sitting in
``data/``.

A chunk is COMPLETE (dropped from the output) when BOTH ``_aruco_tracks.h5`` and
``_aruco_detections.h5`` exist and are non-empty on the bucket, AND:

  * shape-verified (preferred): ``aruco_tracks`` is dense ``(frames, ids, 2)``
    (run_aruco_mp.py allocates it as ``np.zeros((num_frames, dict_size, 2))``),
    so ``shape[0]`` IS the frame count the chunk was processed at. Comparing it
    with the worklist's expected_frames proves the existing output came from the
    same chunking. This is stronger than the SLEAP side, which needs a written
    ``expected_frames`` attr because a .slp carries no such dimension.
  * presence fallback: when h5py is unavailable the check degrades to size-only
    presence, which is still bucket-aware. Warned about, never silent.

Safety bias, same contract as filter_done_chunks.py: any uncertainty KEEPS the
row (recompute). An unreadable bucket dir, a corrupt h5, or a shape that cannot
be read all result in the chunk being kept. "Cannot verify" must never become a
verdict of "done" -- that would silently leave a hole in the block.

Kept python 3.6-compatible: this runs in chunk_finalize on a deigo compute node,
whose system python3 is 3.6.8.
"""
import argparse
import os
import sys

try:
    import h5py  # type: ignore

    _HAVE_H5PY = True
except Exception:  # ImportError, or a broken native lib on this node
    _HAVE_H5PY = False


def _nonempty(path):
    try:
        return os.path.getsize(path) > 0
    except OSError:
        return False


def _tracks_frames(h5_path):
    """Return (ok, frames) for the tracks h5.

    ok=False means the file could not be opened or holds no usable dataset ->
    treat as incomplete. frames is None for a valid file whose dataset shape
    cannot be interpreted (presence-fallback case).
    """
    if not _HAVE_H5PY:
        return True, None
    try:
        with h5py.File(h5_path, "r") as h5:
            if "aruco_tracks" not in h5:
                return False, None  # truncated / not a real tracks h5
            shape = h5["aruco_tracks"].shape
            if not shape:
                return False, None
            return True, int(shape[0])
    except Exception:
        return False, None


def _is_complete(data_dir, vname, chunk, expected, stats):
    trk = os.path.join(data_dir, "%s_%s_aruco_tracks.h5" % (vname, chunk))
    det = os.path.join(data_dir, "%s_%s_aruco_detections.h5" % (vname, chunk))

    if not _nonempty(trk) or not _nonempty(det):
        return False

    ok, frames = _tracks_frames(trk)
    if not ok:
        stats["corrupt_h5"] += 1
        return False

    if frames is None:
        stats["presence_fallback"] += 1
        return True

    if expected > 0 and frames != expected:
        stats["shape_mismatch"] += 1
        return False  # produced under a different chunking -> recompute

    stats["shape_verified"] += 1
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worklist", required=True, help="full pipeline worklist in")
    ap.add_argument("--data-dir", required=True, help="block bucket data/ dir")
    ap.add_argument("--out", required=True, help="filtered worklist out (rows to run)")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.worklist):
        sys.stderr.write("[ERR] worklist not found: %s\n" % args.worklist)
        return 2

    # An unreadable bucket dir cannot confirm anything is done -> keep every row.
    bucket_ok = os.path.isdir(args.data_dir)
    if not bucket_ok:
        sys.stderr.write("[WARN] bucket data dir absent/unreadable (%s); keeping ALL "
                         "chunks (no bucket skip).\n" % args.data_dir)
    if not _HAVE_H5PY:
        sys.stderr.write("[WARN] h5py unavailable on this node; bucket skip uses "
                         "size-only presence (no frame-count verification).\n")

    stats = {"shape_verified": 0, "presence_fallback": 0,
             "shape_mismatch": 0, "corrupt_h5": 0}
    total = kept = skipped = 0

    with open(args.worklist) as fin, open(args.out, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            total += 1
            parts = line.rstrip("\n").split("\t")
            vname = parts[0] if parts else ""
            chunk = parts[1] if len(parts) > 1 else ""
            try:
                expected = int(parts[2]) if len(parts) > 2 and parts[2] else 0
            except ValueError:
                expected = 0

            if vname and bucket_ok and _is_complete(args.data_dir, vname, chunk,
                                                    expected, stats):
                skipped += 1
            else:
                kept += 1
                fout.write(line if line.endswith("\n") else line + "\n")

    sys.stderr.write(
        "[INFO] aruco bucket skip: %d total, %d skipped (already complete), %d to run "
        "[verified=%d presence=%d | recomputed: mismatch=%d corrupt=%d]\n"
        % (total, skipped, kept, stats["shape_verified"], stats["presence_fallback"],
           stats["shape_mismatch"], stats["corrupt_h5"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
