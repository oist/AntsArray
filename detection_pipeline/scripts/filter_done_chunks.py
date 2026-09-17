#!/usr/bin/env python3
"""Filter a SLEAP worklist down to the chunks that still need processing.

Reads the full pipeline worklist (``vname\\tNNN\\texpected_frames`` per line) and
the block's bucket ``data/`` dir, and writes a filtered worklist containing only
the rows whose SLEAP output is NOT already complete on the bucket. This is what
gives ``--only-sleap`` (and any full re-run) a bucket-scoped skip: a delayed
re-run recomputes only the gaps instead of the whole block.

A chunk is considered COMPLETE (dropped from the output) when BOTH
``<vname>_<NNN>.slp`` and ``<vname>_<NNN>_sleap_data.h5`` exist and are non-empty
on the bucket, AND the completeness check below passes:

  * attr-verified (preferred): the h5's ``expected_frames`` attribute equals the
    worklist's expected_frames for that row. This proves the existing output was
    produced under the SAME chunking (same --chunk-sec), so skipping is safe.
    ``frame_count`` is deliberately NOT used -- it is the labeled-frame count and
    excludes empty frames on the legacy ``sleap-nn track --no_empty_frames`` path.
  * presence fallback: outputs written before the ``expected_frames`` attr existed
    have both files present but no attr. They are treated as complete, which is
    safe as long as the re-run uses the same --chunk-sec as the original (the
    catalog emits the inferred --chunk-sec for exactly this reason).

Safety bias: any uncertainty keeps the row (recompute) rather than dropping it.
An unreadable bucket dir, an unreadable/corrupt h5, or a mismatched attr all
result in the chunk being KEPT. Missing h5py degrades to a size-only presence
check (still bucket-aware) with a warning -- attr verification is skipped.

Kept compatible with python 3.6 (f-strings only, no 3.7+ syntax such as
`from __future__ import annotations`): it runs on deigo in the bridge, whose base
python3 may be 3.6, and MUST parse there so the skip still happens (degrading to
presence-only when h5py is absent) rather than erroring and forcing a full
recompute.
"""
import argparse
import os
import sys

try:
    import h5py  # type: ignore

    _HAVE_H5PY = True
except Exception:  # ImportError, or a broken native lib on this node
    _HAVE_H5PY = False


def _nonempty(path: str) -> bool:
    try:
        return os.path.getsize(path) > 0
    except OSError:
        return False


def _h5_expected_frames(h5_path: str):
    """Return (ok, expected) for the bucket h5.

    ok=False means the file could not be opened/validated -> treat as incomplete.
    expected is the stored expected_frames attr, or None if the file is a valid
    h5 that simply predates the attribute (presence-fallback case).
    """
    if not _HAVE_H5PY:
        # Can't introspect; caller does size-only presence check.
        return True, None
    try:
        with h5py.File(h5_path, "r") as h5:
            if "sleap_data" not in h5:
                # Truncated / not a real sleap_data.h5.
                return False, None
            val = h5.attrs.get("expected_frames")
            if val is None:
                return True, None
            return True, int(val)
    except Exception:
        return False, None


def _is_complete(data_dir: str, vname: str, chunk: str, expected: int, stats: dict) -> bool:
    slp = os.path.join(data_dir, f"{vname}_{chunk}.slp")
    h5 = os.path.join(data_dir, f"{vname}_{chunk}_sleap_data.h5")

    if not _nonempty(slp) or not _nonempty(h5):
        return False  # missing .slp or .h5 -> not complete (also covers slp2h5 gotcha)

    ok, stored = _h5_expected_frames(h5)
    if not ok:
        stats["corrupt_h5"] += 1
        return False  # unreadable/truncated h5 -> recompute

    if stored is None:
        stats["presence_fallback"] += 1
        return True  # valid h5, no attr -> trust presence (same-chunking assumption)

    if expected > 0 and stored != expected:
        stats["attr_mismatch"] += 1
        return False  # produced under a different chunking -> recompute

    stats["attr_verified"] += 1
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worklist", required=True, help="full pipeline worklist in")
    ap.add_argument("--data-dir", required=True, help="block bucket data/ dir")
    ap.add_argument("--out", required=True, help="filtered worklist out (rows to run)")
    args = ap.parse_args()

    if not os.path.isfile(args.worklist):
        print(f"[ERR] worklist not found: {args.worklist}", file=sys.stderr)
        return 2

    # If the bucket dir is unreadable we cannot confirm anything is done -> keep
    # every row (recompute all) rather than risk dropping unfinished chunks.
    bucket_ok = os.path.isdir(args.data_dir)
    if not bucket_ok:
        print(
            f"[WARN] bucket data dir absent/unreadable ({args.data_dir}); "
            "keeping ALL chunks (no bucket skip).",
            file=sys.stderr,
        )
    if not _HAVE_H5PY:
        print(
            "[WARN] h5py unavailable on this node; bucket skip uses size-only "
            "presence (no expected_frames attr verification).",
            file=sys.stderr,
        )

    stats = {"attr_verified": 0, "presence_fallback": 0, "attr_mismatch": 0, "corrupt_h5": 0}
    total = kept = skipped = 0

    with open(args.worklist) as fin, open(args.out, "w") as fout:
        for line in fin:
            row = line.rstrip("\n")
            if not row.strip():
                continue
            total += 1
            parts = row.split("\t")
            vname = parts[0] if parts else ""
            chunk = parts[1] if len(parts) > 1 else ""
            try:
                expected = int(parts[2]) if len(parts) > 2 and parts[2] else 0
            except ValueError:
                expected = 0

            complete = bool(vname) and bucket_ok and _is_complete(
                args.data_dir, vname, chunk, expected, stats
            )
            if complete:
                skipped += 1
            else:
                kept += 1
                fout.write(line if line.endswith("\n") else line + "\n")

    print(
        "[INFO] bucket skip: %d total, %d skipped (already complete), %d to run "
        "[verified=%d presence=%d | recomputed: mismatch=%d corrupt=%d]"
        % (
            total,
            skipped,
            kept,
            stats["attr_verified"],
            stats["presence_fallback"],
            stats["attr_mismatch"],
            stats["corrupt_h5"],
        ),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
