#!/usr/bin/env python3
"""Split a basler recording block into a clean head + a droppy tail at a
keyframe, losslessly (stream copy, single pass), and regenerate sidecars.

Motivation
----------
In a hardware-triggered multi-PC session every camera shares one trigger, so
`file frame N == trigger pulse N` on a healthy camera. When one PC (e.g. PC2 /
drive-E) stalls late in a long block, its cameras drop frames and then stop
early, while the other PCs keep recording to the scheduled end. The simplest
way to keep the surviving footage frame-synchronised across ALL cameras is to
trim every camera to the last pulse before the first drop.

This tool:
  * finds the first dropped pulse across all cameras (from the *.diag.json
    `provenance.gaps`), rounds DOWN to the previous keyframe (GOP boundary) so
    the cut is a clean stream-copy boundary common to every camera, and
  * writes `<block>-1/` (clean aligned head, frames [0, K)) and
    `<block>-2/` (the disposable tail, frames [K, end)),
  * regenerates a CLEAN sidecar for every head video (framesEmitted ==
    framesEncoded == K, no gaps, healthy, closed) and a derived sidecar for
    every tail video, and copies the session log into both.

The originals are never touched. Cut is lossless (no re-encode) and single-pass
(each source read once) -- required because these live-muxed MKVs have no seek
index, so random `-ss` would linear-scan the whole file.

Usage
-----
  python split_block_at_drop.py "Z:/ReiterU/Ants/basler/20260716/block03"
  python split_block_at_drop.py <block> --dry-run          # plan only
  python split_block_at_drop.py <block> --split-frame N     # force cut frame
  python split_block_at_drop.py <block> --jobs 2 --force
"""
import argparse
import concurrent.futures as cf
import copy
import datetime as dt
import glob
import json
import os
import shutil
import subprocess
import sys
import threading

TOOL = "makoto_play/split_block_at_drop.py"
DEFAULT_FPS = 24
DEFAULT_GOP = 1440          # fallback if probing fails; verified 60s @24fps

_print_lock = threading.Lock()
_LOG_FH = None


def log(msg):
    line = f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}"
    with _print_lock:
        print(line, flush=True)
        if _LOG_FH:
            _LOG_FH.write(line + "\n")
            _LOG_FH.flush()


# --------------------------------------------------------------------------- #
# sidecar discovery / split-point detection
# --------------------------------------------------------------------------- #
def sidecar_for(mkv):
    return mkv + ".diag.json"


def load_sidecar(mkv):
    p = sidecar_for(mkv)
    try:
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def first_drop_pulse(sc):
    """Earliest missing pulse for one camera, or None if no gaps."""
    gaps = ((sc or {}).get("provenance") or {}).get("gaps") or []
    firsts = [g["previousBlockId"] + 1 for g in gaps
              if isinstance(g, dict) and "previousBlockId" in g]
    return min(firsts) if firsts else None


def probe_gop(mkv, scan=3200):
    """Keyframe interval (frames) from the first `scan` frames. None on failure."""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-read_intervals", f"%+#{scan}",
           "-show_entries", "frame=key_frame", "-of", "csv=p=0", mkv]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=180).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    idx = [i for i, line in enumerate(out.splitlines()) if line.strip().startswith("1")]
    if len(idx) < 2:
        return None
    diffs = [b - a for a, b in zip(idx, idx[1:])]
    return max(set(diffs), key=diffs.count)   # most common interval


# --------------------------------------------------------------------------- #
# ffmpeg split
# --------------------------------------------------------------------------- #
def split_one(mkv, K, head_dir, tail_dir, staging_root, verify_decode=False):
    """Stream-copy split `mkv` at frame K. Returns dict(result)."""
    name = os.path.basename(mkv)
    stage = os.path.join(staging_root, name + ".stage")
    if os.path.isdir(stage):
        shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(stage, exist_ok=True)
    seg_pat = os.path.join(stage, "seg_%03d.mkv")
    seg_list = os.path.join(stage, "seg_list.csv")

    cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-y",
           "-i", mkv, "-map", "0:v:0", "-c", "copy",
           "-f", "segment", "-segment_frames", str(K),
           "-reset_timestamps", "1",
           "-segment_list", seg_list, "-segment_list_type", "csv",
           seg_pat]
    t0 = dt.datetime.now()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dur = (dt.datetime.now() - t0).total_seconds()
    if proc.returncode != 0:
        return {"name": name, "ok": False,
                "err": (proc.stderr or "ffmpeg failed")[-500:]}

    seg0 = os.path.join(stage, "seg_000.mkv")
    seg1 = os.path.join(stage, "seg_001.mkv")
    if not os.path.exists(seg0):
        return {"name": name, "ok": False, "err": "no head segment produced"}
    head_frames = None
    if verify_decode:
        head_frames = _decode_count(seg0)
        if head_frames != K:
            return {"name": name, "ok": False,
                    "err": f"decoded head={head_frames}, expected {K}"}

    os.replace(seg0, os.path.join(head_dir, name))   # same volume => instant
    tail_present = os.path.exists(seg1)
    if tail_present:
        os.replace(seg1, os.path.join(tail_dir, name))
    return {"name": name, "ok": True, "sec": dur, "head_frames": head_frames,
            "tail_present": tail_present, "stage": stage}


def _decode_count(mkv):
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-count_frames", "-show_entries", "stream=nb_read_frames",
           "-of", "csv=p=0", mkv]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
        return int(out)
    except (subprocess.SubprocessError, ValueError, OSError):
        return None


# --------------------------------------------------------------------------- #
# sidecar regeneration
# --------------------------------------------------------------------------- #
def _fps_of(sc):
    try:
        return float((sc.get("context") or {}).get("fps") or DEFAULT_FPS)
    except (TypeError, ValueError):
        return float(DEFAULT_FPS)


def clean_head_sidecar(orig, K, block_name):
    """Sidecar accurately describing the clean [0,K) head: closed, healthy, no gaps."""
    sc = copy.deepcopy(orig)
    fps = _fps_of(sc)
    ctx = sc.setdefault("context", {})
    cap = sc.setdefault("capture", {})
    rec = sc.setdefault("recorder", {})
    sdk = sc.setdefault("sdk", {})
    prov = sc.setdefault("provenance", {})
    health = sc.setdefault("health", {})

    t0 = (prov.get("firstEncodedFrame") or {}).get("hostEpochMs") or ctx.get("startEpochMs")
    dur_ms = round(K / fps * 1000)
    last_ms = round((K - 1) / fps * 1000)

    cap["framesEmitted"] = K
    if cap.get("emitIntervalCount") is not None:
        cap["emitIntervalCount"] = K - 1
    if cap.get("emitIntervalAvgMs") is not None:          # max during head was healthy
        cap["emitIntervalMaxMs"] = cap["emitIntervalAvgMs"]
    cap.pop("emitIntervalMaxEpochMs", None)

    rec["framesEncoded"] = K
    for k in ("currentPendingQueue", "compressedQueueCurrentBytes",
              "compressedQueueDiscardedBytes", "compressedQueuePeakBytes"):
        if k in rec:
            rec[k] = 0
    if "compressedWriterError" in rec:
        rec["compressedWriterError"] = False
    if "compressedWriterErrorReason" in rec:
        rec["compressedWriterErrorReason"] = ""
    if rec.get("encodeTimeAvgMs") is not None:
        rec["encodeTimeMaxMs"] = rec["encodeTimeAvgMs"]
    rec.pop("encodeTimeMaxEpochMs", None)

    sdk["Statistic_Missed_Frame_Count"] = 0
    sdk["Statistic_Failed_Buffer_Count"] = 0
    if "Statistic_Last_Failed_Buffer_Status" in sdk:
        sdk["Statistic_Last_Failed_Buffer_Status"] = 0
    if "Statistic_Last_Failed_Buffer_Status_Text" in sdk:
        sdk["Statistic_Last_Failed_Buffer_Status_Text"] = ""

    ctx["status"] = "closed"
    ctx["cleanClose"] = True
    ctx["failureReason"] = ""
    ctx["durationMs"] = dur_ms
    if t0 is not None:
        ctx["stopEpochMs"] = t0 + last_ms

    prov["gaps"] = []
    for k in ("invalidIdEvents", "wrapEvents", "resetEvents"):
        prov[k] = []
    prov["eventOverflowCount"] = 0
    le = dict(prov.get("lastEncodedFrame") or {})
    le["blockId"] = K - 1
    le["fileFrameIndex"] = K - 1
    if t0 is not None:
        le["hostEpochMs"] = t0 + last_ms
    prov["lastEncodedFrame"] = le

    health["state"] = "healthy"
    health["events"] = []
    health["eventOverflowCount"] = 0
    health.pop("firstDegradedEpochMs", None)

    tel = sc.get("telemetry") or {}
    samples = tel.get("samples") or []
    if t0 is not None and samples:
        cut = t0 + dur_ms
        kept = [s for s in samples if s.get("epochMs", 0) <= cut]
        for s in kept:
            ssd = s.get("sdk")
            if isinstance(ssd, dict):
                ssd["Statistic_Missed_Frame_Count"] = 0
                ssd["Statistic_Failed_Buffer_Count"] = 0
        tel["samples"] = kept
        tel["discardedSamples"] = 0

    sc["derived"] = {
        "tool": TOOL, "from": block_name, "segment": "head (-1)",
        "keptFrames": K, "splitAtFrame": K,
        "note": "clean aligned head; trimmed at the keyframe before the first "
                "cross-camera dropped pulse. Frames [0,K) are drop-free and "
                "frame==pulse aligned across every camera in the block.",
    }
    return sc


def tail_sidecar(orig, K, block_name):
    """Derived sidecar for the disposable tail [K, end)."""
    sc = copy.deepcopy(orig)
    cap = sc.setdefault("capture", {})
    rec = sc.setdefault("recorder", {})
    total = rec.get("framesEncoded") or cap.get("framesEmitted") or 0
    tail_n = max(0, int(total) - K)
    cap["framesEmitted"] = tail_n
    rec["framesEncoded"] = tail_n
    sc["derived"] = {
        "tool": TOOL, "from": block_name, "segment": "tail (-2)",
        "frameOffset": K, "tailFrames": tail_n,
        "note": "DISPOSABLE tail (frames [K, end) of the original). Contains the "
                "PC2 frame drops and the early stop. provenance.gaps/telemetry "
                "still use ORIGINAL file-frame indices (subtract frameOffset for "
                "tail-local). Kept only so nothing is silently discarded.",
    }
    return sc


def write_sidecar(mkv_out, sc):
    with open(sidecar_for(mkv_out), "w", encoding="utf-8") as fh:
        json.dump(sc, fh, indent=1)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("block", help="block directory (e.g. .../20260716/block03)")
    ap.add_argument("--split-frame", type=int, default=None,
                    help="force cut frame K (default: auto from sidecar gaps + GOP)")
    ap.add_argument("--gop", type=int, default=None,
                    help="keyframe interval override (default: probe)")
    ap.add_argument("--jobs", type=int, default=2, help="parallel cameras (default 2)")
    ap.add_argument("--dry-run", action="store_true", help="plan only, do nothing")
    ap.add_argument("--verify", action="store_true",
                    help="decode-count head to confirm exactly K frames (slow)")
    ap.add_argument("--force", action="store_true",
                    help="redo cameras whose outputs already exist")
    args = ap.parse_args()

    block = os.path.abspath(args.block.rstrip("/\\"))
    if not os.path.isdir(block):
        sys.exit(f"not a directory: {block}")
    block_name = os.path.basename(block)
    parent = os.path.dirname(block)
    head_dir = os.path.join(parent, block_name + "-1")
    tail_dir = os.path.join(parent, block_name + "-2")

    global _LOG_FH
    logdir = os.path.join(block, "_split_logs")
    os.makedirs(logdir, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    _LOG_FH = open(os.path.join(logdir, f"split_{stamp}.log"), "w", encoding="utf-8")

    mkvs = sorted(glob.glob(os.path.join(block, "*.mkv")))
    if not mkvs:
        sys.exit(f"no .mkv in {block}")
    log(f"block={block}  ({len(mkvs)} videos)")

    # -- split point --------------------------------------------------------- #
    fps = DEFAULT_FPS
    drop_pulses = {}
    for m in mkvs:
        sc = load_sidecar(m)
        if sc:
            fps = _fps_of(sc)
            fd = first_drop_pulse(sc)
            if fd is not None:
                drop_pulses[os.path.basename(m)] = fd
    if args.split_frame is not None:
        K = args.split_frame
        gop = args.gop or DEFAULT_GOP
        log(f"using forced split frame K={K:,}")
    else:
        if not drop_pulses:
            sys.exit("no gaps found in any sidecar; pass --split-frame to force")
        first_drop = min(drop_pulses.values())
        first_cam = min(drop_pulses, key=drop_pulses.get)
        gop = args.gop or probe_gop(mkvs[0]) or DEFAULT_GOP
        K = (first_drop // gop) * gop            # last keyframe <= first drop
        log(f"first dropped pulse = {first_drop:,} (earliest: {first_cam}); "
            f"GOP={gop}; cut at keyframe K={K:,}")
    dur_h = K / fps / 3600
    log(f"head = frames [0,{K:,}) = {K:,} frames (~{dur_h:.2f} h @ {fps:g}fps); "
        f"tail = [{K:,}, end)")
    log(f"outputs: {head_dir}  +  {tail_dir}")

    if args.dry_run:
        log("DRY RUN -- planned per-camera actions:")
        for m in mkvs:
            sc = load_sidecar(m)
            total = ((sc or {}).get("recorder") or {}).get("framesEncoded") \
                or ((sc or {}).get("capture") or {}).get("framesEmitted")
            fd = drop_pulses.get(os.path.basename(m))
            tail_n = (int(total) - K) if total else "?"
            log(f"  {os.path.basename(m):45} total={str(total):>10} "
                f"head={K:,} tail={tail_n:>8} "
                f"{'(DROPS@'+format(fd, ',')+')' if fd else ''}")
        log("dry run complete; nothing written.")
        return

    os.makedirs(head_dir, exist_ok=True)
    os.makedirs(tail_dir, exist_ok=True)
    staging_root = os.path.join(block, "_split_staging")
    os.makedirs(staging_root, exist_ok=True)

    # session log + split manifest into both output dirs
    for sess in glob.glob(os.path.join(block, "sess_*.txt")):
        for d in (head_dir, tail_dir):
            shutil.copy2(sess, os.path.join(d, os.path.basename(sess)))
    split_info = {
        "tool": TOOL, "source_block": block, "split_frame": K, "gop": gop,
        "fps": fps, "first_dropped_pulse": (min(drop_pulses.values())
                                            if drop_pulses else None),
        "drop_pulses_per_cam": drop_pulses,
        "generated": stamp,
        "note": "head (-1) = clean aligned frames [0,K); tail (-2) = disposable "
                "remainder. Originals under source_block are untouched.",
    }
    for d in (head_dir, tail_dir):
        with open(os.path.join(d, "SPLIT_INFO.json"), "w", encoding="utf-8") as fh:
            json.dump(split_info, fh, indent=2)

    # -- per-camera work ----------------------------------------------------- #
    todo = []
    for m in mkvs:
        name = os.path.basename(m)
        done_marker = os.path.join(staging_root, name + ".ok")
        if not args.force and os.path.exists(os.path.join(head_dir, name)) \
                and os.path.exists(done_marker):
            log(f"skip (done): {name}")
            continue
        todo.append(m)
    log(f"{len(todo)} / {len(mkvs)} cameras to process (jobs={args.jobs})")

    results = []

    def work(m):
        name = os.path.basename(m)
        log(f"START {name}  ({os.path.getsize(m)/1e9:.1f} GB)")
        r = split_one(m, K, head_dir, tail_dir, staging_root, args.verify)
        if not r["ok"]:
            log(f"FAIL  {name}: {r['err']}")
            return r
        orig = load_sidecar(m) or {}
        write_sidecar(os.path.join(head_dir, name),
                      clean_head_sidecar(orig, K, block_name))
        if r["tail_present"]:
            write_sidecar(os.path.join(tail_dir, name),
                          tail_sidecar(orig, K, block_name))
        open(os.path.join(staging_root, name + ".ok"), "w").write(stamp)
        try:
            shutil.rmtree(r["stage"])
        except OSError:
            pass
        log(f"DONE  {name}  in {r['sec']/60:.1f} min "
            f"(head {r.get('head_frames') or K} frames)")
        return r

    if args.jobs <= 1:
        for m in todo:
            results.append(work(m))
    else:
        with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
            for r in ex.map(work, todo):
                results.append(r)

    ok = sum(1 for r in results if r and r.get("ok"))
    bad = [r for r in results if r and not r.get("ok")]
    log(f"==== finished: {ok} ok, {len(bad)} failed ====")
    for r in bad:
        log(f"  FAILED {r['name']}: {r['err']}")
    if not bad and ok == len(todo):
        log("all cameras split successfully. Originals untouched. "
            "block-1 = clean head, block-2 = disposable tail.")


if __name__ == "__main__":
    main()
