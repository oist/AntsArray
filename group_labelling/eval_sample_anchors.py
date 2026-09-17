#!/usr/bin/env python3
"""Stage 0 sampling: deterministic anchor + burst manifests for every camera-run.

Writes one manifest per camera plus a strata index. Nothing downstream chooses
its own frames -- every later stage reads these files, so the ArUco reference,
both models' predictions and the adjudication packages are guaranteed to
describe the same pixels.

Contract: group_labelling/EVAL_HARNESS_DESIGN.md sections 4 and 5.

    python group_labelling/eval_sample_anchors.py --out-dir /work/ReiterU/centroid_eval
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_common import (  # noqa: E402
    ANCHORS_PER_REPLICATE,
    BLOCKS,
    BURST_HALF_WIDTH,
    N_REPLICATES,
    burst_frames,
    discover_camera_runs,
    sample_anchors,
    setup_logging,
    write_json,
)


def probe_video(path: Path) -> dict:
    """Container facts. FRAME_COUNT is the container's claim, not ground truth;
    the burst window keeps us BURST_HALF_WIDTH frames clear of the tail so an
    over-reported count cannot produce an unreadable anchor."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    facts = {
        "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return facts


def sidecar_clock(video: Path) -> dict:
    """Wall-clock facts from the recorder's .diag.json sidecar.

    Day/night binning must never come from frame_idx / F: this project has
    documented drop events (19-26 frames lost on cam08-17; a 13-minute dead tail
    costing ~24k frames/camera), so index does not track time. Cameras without a
    usable sidecar are flagged NO-CLOCK and drop out of the day/night slice only.
    """
    side = video.with_suffix(video.suffix + ".diag.json")
    if not side.exists():
        return {"clock": "NO-CLOCK", "clock_reason": "no .diag.json sidecar"}
    try:
        with side.open(encoding="utf-8") as fh:
            d = json.load(fh)
        ctx = d.get("context", {})
        start_ms = ctx.get("startEpochMs")
        fps = ctx.get("fps")
        if start_ms is None or not fps:
            return {"clock": "NO-CLOCK",
                    "clock_reason": "sidecar lacks startEpochMs/fps"}
        return {
            "clock": "OK",
            "start_epoch_ms": int(start_ms),
            "sidecar_fps": float(fps),
            "duration_ms": ctx.get("durationMs"),
            "frames_emitted": d.get("capture", {}).get("framesEmitted"),
        }
    except Exception as exc:  # a malformed sidecar is NO-CLOCK, not a crash
        return {"clock": "NO-CLOCK", "clock_reason": f"unreadable sidecar: {exc}"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--blocks", type=str, default=",".join(BLOCKS))
    p.add_argument("--cameras", type=str, default=None,
                   help="comma-separated subset, e.g. cam10,cam22")
    p.add_argument("--replicates", type=int, default=N_REPLICATES)
    p.add_argument("--per-replicate", type=int, default=ANCHORS_PER_REPLICATE)
    args = p.parse_args()

    out: Path = args.out_dir.resolve()
    anchors_dir = out / "anchors"
    anchors_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out / "logs", "eval_sample_anchors")

    blocks = tuple(b.strip() for b in args.blocks.split(",") if b.strip())
    cams = tuple(c.strip() for c in args.cameras.split(",")) if args.cameras else None
    runs = discover_camera_runs(blocks, cams)
    logging.info("discovered %d camera-runs across %s", len(runs), ",".join(blocks))

    strata: dict[str, list[str]] = {"S1": [], "S2": [], "S3": []}
    summary = []
    for run in runs:
        try:
            facts = probe_video(run.video)
        except Exception:
            logging.exception("[%s] probe failed - skipping", run.key)
            continue
        n_frames = facts["n_frames"]
        if n_frames <= 2 * BURST_HALF_WIDTH + 1:
            logging.error("[%s] only %d frames - skipping", run.key, n_frames)
            continue

        anchors, reps, n_shifted = sample_anchors(
            run.block, run.cam, n_frames,
            n_replicates=args.replicates, per_replicate=args.per_replicate)
        if len(anchors) == 0:
            logging.error("[%s] no anchors - skipping", run.key)
            continue
        frames, owner = burst_frames(anchors)
        clock = sidecar_clock(run.video)

        write_json({
            "block": run.block,
            "cam": run.cam,
            "key": run.key,
            "stratum": run.stratum,
            "video": str(run.video),
            **facts,
            **clock,
            "n_replicates": args.replicates,
            "per_replicate": args.per_replicate,
            "burst_half_width": BURST_HALF_WIDTH,
            "n_anchors": int(len(anchors)),
            "n_burst_frames": int(len(frames)),
            "n_shifted": int(n_shifted),
            "anchors": anchors,
            "replicate": reps,
            "burst_frames": frames,
            "anchor_of_frame": owner,
        }, anchors_dir / f"{run.key}.json")

        strata[run.stratum].append(run.key)
        summary.append({
            "key": run.key, "stratum": run.stratum, "n_frames": n_frames,
            "n_anchors": int(len(anchors)), "n_burst": int(len(frames)),
            "n_shifted": int(n_shifted), "clock": clock["clock"],
        })
        logging.info("[%s] %s F=%d anchors=%d burst=%d shifted=%d %s",
                     run.key, run.stratum, n_frames, len(anchors), len(frames),
                     n_shifted, clock["clock"])

    write_json({
        "strata": strata,
        "counts": {k: len(v) for k, v in strata.items()},
        "summary": summary,
        "banners": {
            "S1": ("SAME-CAMERA / DIFFERENT UNSEEN FOOTAGE -- camera-identity "
                   "overlap only, NOT memorisation. The ch01-06 training footage "
                   "(20260515 13-09-56) has been deleted; block00's *_000 symlinks "
                   "to it all dangle. No block01/02/03 frame was in training."),
            "S2": "UNDERPOWERED, n=6 -- reported as six individual camera values.",
            "S3": ("n_blocks = 2, n_colonies = 1, n_recording_days = 1 (20260515). "
                   "'Never seen' means never-seen-cameras-within-one-recording-day."),
        },
    }, out / "strata.json")

    logging.info("strata: S1=%d S2=%d S3=%d",
                 len(strata["S1"]), len(strata["S2"]), len(strata["S3"]))
    n_clockless = sum(1 for s in summary if s["clock"] != "OK")
    if n_clockless:
        logging.warning("%d camera-runs are NO-CLOCK (excluded from the day/night "
                        "slice only)", n_clockless)
    if not summary:
        logging.error("no manifests written")
        return 1
    total_anchors = sum(s["n_anchors"] for s in summary)
    logging.info("total anchors=%d (=%d model-frames across 2 models), burst=%d",
                 total_anchors, 2 * total_anchors,
                 sum(s["n_burst"] for s in summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
