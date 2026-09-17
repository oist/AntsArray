#!/usr/bin/env python3
"""Build the /bucket/ReiterU/Ants/basler filming-session metadata catalog.

Emits, into <root>/_catalog/ (or --outdir):
  catalog.csv        one row per block / flat-session / aux entry (the "one-go" sheet)
  videos.csv         one row per grid video (per-camera recording health)
  trials.csv         one row per vibration pulse (CSV_PULSE) with cam frame ranges
  catalog_run.json   run summary + ignored/unknown entries
  logs/catalog_*.log timestamped run log

Usage:
  python detection_pipeline/catalog.py all \\
      --root Z:/ReiterU/Ants/basler --outdir Z:/ReiterU/Ants/basler/_catalog

Subcommands:
  scan    walk the tree and update the incremental cache (no CSV emitted)
  build   emit CSVs from the existing cache only (no filesystem walk)
  all     scan + emit  (default)

Notes:
  * Sidecar-first: fps/frames come from *.diag.json. Old sidecar-less .avi get
    blank fps/frames unless --allow-ffprobe is passed (slow on /bucket).
  * A refresh re-reads only blocks whose fingerprint changed; use --force to
    rescan everything.
"""
import argparse
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
from catalog import build, const, provenance, recover  # noqa: E402
import tracking_state  # noqa: E402


def _make_logger(logdir, stamp):
    os.makedirs(logdir, exist_ok=True)
    logpath = os.path.join(logdir, f"catalog_{stamp}.log")
    fh = open(logpath, "a", encoding="utf-8")

    def log(msg):
        line = str(msg)
        sys.stderr.write(line + "\n")
        sys.stderr.flush()
        fh.write(line + "\n")
        fh.flush()

    return log, logpath, fh


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", nargs="?",
                    choices=["scan", "build", "all", "recover", "state-init", "track-init"],
                    default="all")
    ap.add_argument("target", nargs="?", default=None,
                    help="for 'recover'/'state-init'/'track-init': the block id, "
                         "e.g. 20260623/block03")
    ap.add_argument("--hmats", default=None,
                    help="track-init: the homography .npz the block was tracked "
                         "with (as passed to submit_blocks_pipeline.sh --hmats)")
    ap.add_argument("--x-threshold", type=float, default=None,
                    help="track-init: the panorama split used (--x_threshold)")
    ap.add_argument("--tracked-at", default=None,
                    help="track-init: when the tracking ran, ISO date/time "
                         "(the backfill's own timestamp is recorded separately)")
    ap.add_argument("--note", default=None,
                    help="track-init: where the hmats value came from, e.g. "
                         "'rendered sbatch on /flash, 2026-09-11'")
    ap.add_argument("--overwrite", action="store_true",
                    help="track-init: replace a record the pipeline wrote itself "
                         "(the displaced record is kept in history)")
    ap.add_argument("--allow-missing-tracks", action="store_true",
                    help="track-init: record even though the block has no tracks/ "
                         "directory (creates it)")
    ap.add_argument("--chunk-sec", type=int, default=None,
                    help="state-init: the --chunk-sec the ORIGINAL run used. "
                         "Required, and cross-checked against the archived "
                         "manifest -- guessing it is the weakness the contract "
                         "exists to remove.")
    ap.add_argument("--chunk-ext", default="mkv",
                    help="state-init: chunk container of the original run")
    ap.add_argument("--dry-run", action="store_true",
                    help="state-init/track-init: print what would be recorded, "
                         "write nothing")
    ap.add_argument("--root", default="/bucket/ReiterU/Ants/basler",
                    help="basler root to scan (Windows: Z:/ReiterU/Ants/basler)")
    ap.add_argument("--outdir", default=None,
                    help="output dir (default: <root>/_catalog)")
    ap.add_argument("--workers", type=int, default=const.DEFAULT_WORKERS,
                    help="parallel sidecar-probe workers")
    ap.add_argument("--only", default=None,
                    help="comma-separated top-level names to restrict scanning")
    ap.add_argument("--force", action="store_true",
                    help="ignore cache; rescan every block")
    ap.add_argument("--allow-ffprobe", action="store_true",
                    help="ffprobe sidecar-less videos (slow on /bucket)")
    ap.add_argument("--parquet", action="store_true",
                    help="also emit .parquet mirrors (needs pandas+pyarrow)")
    ap.add_argument("--check-sizes", action="store_true",
                    help="stat data files to flag truncated artifacts (slower)")
    ap.add_argument("--labels-file", default=None,
                    help="per-block label overlay CSV keyed by '<date>/block**' "
                         "(default: <outdir>/block_labels.csv)")
    args = ap.parse_args(argv)

    outdir = args.outdir or os.path.join(args.root, "_catalog")
    only = {s.strip() for s in args.only.split(",")} if args.only else None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    scanned_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    log, logpath, fh = _make_logger(os.path.join(outdir, "logs"), stamp)
    log(f"[catalog] mode={args.mode} root={args.root} outdir={outdir} "
        f"only={only} force={args.force} allow_ffprobe={args.allow_ffprobe}")
    try:
        if args.mode == "recover":
            if not args.target:
                log("[catalog] recover needs a block id, e.g. "
                    "'recover 20260623/block03' or 'recover 20260623::block03'")
                return
            recover.run_recover(args.root, outdir, args.target, log)
            log("[catalog] done. log: %s" % logpath)
            return
        if args.mode == "track-init":
            # Backfill tracks/TRACKING_STATE.json for a block tracked before the
            # mapper recorded its homography. Evidence comes from the operator
            # (a rendered sbatch, a lab note); a record the pipeline wrote itself
            # is never replaced without --overwrite. (--force keeps its scan
            # meaning and does nothing here.)
            if not args.target or not args.hmats:
                log("[catalog] track-init needs a block id and --hmats, e.g. "
                    "'track-init 20260716/block01 --hmats /bucket/.../aruco_H_mats.npz "
                    "--x-threshold 2475'")
                return
            session_id, block = recover.parse_block_id(args.target)
            tracks_dir = os.path.join(recover.block_dir(args.root, session_id, block),
                                      "tracks")
            if not os.path.isdir(tracks_dir) and not args.allow_missing_tracks:
                log("[catalog] %s has no tracks/: nothing was tracked here "
                    "(--allow-missing-tracks records anyway)" % os.path.dirname(tracks_dir))
                return
            if args.overwrite:
                log("[catalog] --overwrite: a record written by the pipeline itself "
                    "will be replaced (kept in history)")
            try:
                state = tracking_state.backfill(
                    tracks_dir, args.hmats, x_threshold=args.x_threshold,
                    tracked_at=args.tracked_at, note=args.note,
                    force=args.overwrite, dry_run=args.dry_run)
            except ValueError as e:
                log("[catalog] track-init failed: %s" % e)
                return
            m = state["map"]
            log("[catalog] %s %s: hmats=%s sha256=%s x_threshold=%s tracked_at=%s"
                % ("would record" if args.dry_run else "recorded",
                   tracking_state.state_path(tracks_dir), m["hmats_calib_id"],
                   (m.get("hmats_sha256") or "file-not-readable")[:12],
                   m.get("x_threshold"), m.get("tracked_at") or "-"))
            if not m.get("hmats_sha256"):
                log("[catalog] note: %s is not readable from here, so the record "
                    "carries the path only, not a content hash" % args.hmats)
            log("[catalog] done. log: %s" % logpath)
            return
        if args.mode == "state-init":
            # Backfill a processing contract onto a block that predates them, so
            # the catalog can report it against a declared denominator and a
            # later run is held to the settings it was actually processed with.
            if not args.target:
                log("[catalog] state-init needs a block id, e.g. "
                    "'state-init 20260716/block01 --chunk-sec 1800'")
                return
            if not args.chunk_sec:
                log("[catalog] state-init needs --chunk-sec: the value the "
                    "ORIGINAL run used. It is cross-checked against the archived "
                    "manifest, and catalog.csv's chunk_sec column carries the "
                    "inferred value if you need a starting point.")
                return
            session_id, block = recover.parse_block_id(args.target)
            bd = recover.block_dir(args.root, session_id, block)
            try:
                provenance.backfill_state(bd, args.chunk_sec, args.chunk_ext,
                                          dry_run=args.dry_run, log=log)
            except ValueError as e:
                log("[catalog] state-init failed: %s" % e)
            log("[catalog] done. log: %s" % logpath)
            return
        build.run(root=args.root, outdir=outdir, scanned_at=scanned_at,
                  workers=args.workers, only=only, force=args.force,
                  allow_ffprobe=args.allow_ffprobe, parquet=args.parquet,
                  check_sizes=args.check_sizes, mode=args.mode,
                  labels_file=args.labels_file, log=log)
        log(f"[catalog] done. log: {logpath}")
    except Exception as e:  # never leave a half-written log without a reason
        log(f"[catalog][FATAL] {type(e).__name__}: {e}")
        raise
    finally:
        fh.close()


if __name__ == "__main__":
    main()
