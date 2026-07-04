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
from catalog import build, const, recover  # noqa: E402


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
    ap.add_argument("mode", nargs="?", choices=["scan", "build", "all", "recover"],
                    default="all")
    ap.add_argument("target", nargs="?", default=None,
                    help="for 'recover': the block id, e.g. 20260623/block03")
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
        build.run(root=args.root, outdir=outdir, scanned_at=scanned_at,
                  workers=args.workers, only=only, force=args.force,
                  allow_ffprobe=args.allow_ffprobe, parquet=args.parquet,
                  check_sizes=args.check_sizes, mode=args.mode, log=log)
        log(f"[catalog] done. log: {logpath}")
    except Exception as e:  # never leave a half-written log without a reason
        log(f"[catalog][FATAL] {type(e).__name__}: {e}")
        raise
    finally:
        fh.close()


if __name__ == "__main__":
    main()
