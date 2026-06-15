#!/usr/bin/env python3
"""Discover grid videos in an experiment dir and emit manifest.csv.

Source of truth: sidecar JSON next to each video (fast). ffprobe is only used
as a fallback when no sidecar is present, since bucket-stored AVI index reads
are slow.

Output: manifest.csv with columns
  vname,source_path,ext,fps,frame_count,duration_sec,n_chunks
"""
import argparse
import csv
import json
import math
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Accept both `cam##_` (current) and `grid##_` (future per VIDEO_AI_HANDOFF.md).
GRID_PREFIX_RE = re.compile(r"^(?:cam|grid)\d{2}_")
GLOBAL_PREFIX_RE = re.compile(r"^global_")
VIDEO_EXTS = (".mkv", ".mp4", ".avi")

# Flat-key fallback (only used when the nested pylonrecorder2 schema isn't found).
SIDECAR_FPS_KEYS = ("fps", "framerate", "FPS", "frame_rate")
SIDECAR_FRAMES_KEYS = ("frames_encoded", "frame_count", "frames", "frames_emitted", "n_frames")


def log(msg):
	sys.stderr.write(msg + "\n")
	sys.stderr.flush()


def find_sidecar(video_path):
	"""Return the path of the first sidecar candidate that exists, else None.

	pylonrecorder2 writes `<full_filename>.diag.json` (i.e. keeps the .avi/.mkv
	suffix then appends .diag.json). Other patterns are kept as fallbacks.
	"""
	candidates = [
		video_path.parent / (video_path.name + ".diag.json"),       # pylonrecorder2
		video_path.parent / (video_path.name + ".json"),            # alt: vname.avi.json
		video_path.with_suffix(".json"),                            # alt: vname.json
		video_path.parent / (video_path.stem + "_diagnostics.json"),
		video_path.parent / (video_path.stem + ".diagnostics.json"),
	]
	for cand in candidates:
		if cand.is_file():
			return cand
	return None


def read_sidecar(sidecar_path):
	"""Load JSON. Return dict or None on failure."""
	try:
		return json.loads(sidecar_path.read_text())
	except Exception as e:
		log("[WARN] failed to parse %s: %s" % (sidecar_path, e))
		return None


def pluck_number(d, keys, cast):
	for key in keys:
		if key in d:
			try:
				return cast(d[key])
			except (TypeError, ValueError):
				pass
	return None


def from_sidecar(sidecar):
	"""Extract (fps, frames) from sidecar dict. Returns (None, None) if either missing.

	Prefers the pylonrecorder2 schema (nested context.fps + recorder.framesEncoded
	or capture.framesEmitted). Falls back to flat top-level keys.
	"""
	ctx = sidecar.get("context") or {}
	rec = sidecar.get("recorder") or {}
	cap = sidecar.get("capture") or {}

	fps = None
	frames = None
	try:
		if "fps" in ctx:
			fps = float(ctx["fps"])
	except (TypeError, ValueError):
		fps = None
	for src in (rec, cap):
		try:
			if "framesEncoded" in src:
				frames = int(src["framesEncoded"]); break
			if "framesEmitted" in src:
				frames = int(src["framesEmitted"]); break
		except (TypeError, ValueError):
			pass

	if fps and frames:
		return fps, frames

	# Flat fallback.
	if fps is None:
		fps = pluck_number(sidecar, SIDECAR_FPS_KEYS, float)
	if frames is None:
		frames = pluck_number(sidecar, SIDECAR_FRAMES_KEYS, int)
	return fps, frames


def ffprobe_fps_and_frames(video_path, max_sec, timeout_sec):
	"""Slow fallback: ffprobe-decode first max_sec to derive fps + total estimate.

	Two-step: (1) get container duration with a short probe, (2) get fps + counted
	frames over the first max_sec window. Falls back gracefully when fields are
	missing.
	"""
	# Step 1: container duration (short call; should be fast even on bucket because
	# we don't ask for stream metadata).
	dur = 0.0
	try:
		out = subprocess.check_output(
			["ffprobe", "-v", "error",
			 "-show_entries", "format=duration",
			 "-of", "csv=p=0", str(video_path)],
			timeout=timeout_sec,
		).decode().strip()
		if out:
			dur = float(out)
	except Exception as e:
		log("[WARN] ffprobe duration failed for %s: %s" % (video_path.name, e))

	# Step 2: stream metadata + count over first max_sec window.
	fps = 0.0
	nb_read = 0
	try:
		out = subprocess.check_output(
			["ffprobe", "-v", "error",
			 "-select_streams", "v:0",
			 "-read_intervals", "%%+%d" % max_sec,
			 "-count_frames",
			 "-show_entries", "stream=avg_frame_rate,r_frame_rate,nb_read_frames",
			 "-of", "json", str(video_path)],
			timeout=timeout_sec * 4,  # decode is slower than format probe
		).decode()
		info = json.loads(out)["streams"][0]
		fr = info.get("avg_frame_rate") or info.get("r_frame_rate") or "0/1"
		num, den = fr.split("/")
		if int(den) > 0:
			fps = int(num) / int(den)
		nb_read = int(info.get("nb_read_frames", 0))
	except Exception as e:
		log("[WARN] ffprobe stream/count failed for %s: %s" % (video_path.name, e))

	if dur > 0 and fps > 0:
		frames = int(round(fps * dur))
	elif nb_read > 0 and dur > 0:
		frames = int(round(nb_read * dur / max_sec))
	else:
		frames = nb_read  # last-resort lower bound
	return fps, frames, dur


def probe_one(video_path, max_sec, ffprobe_timeout):
	"""Return dict with source/fps/frames/duration or None on hard failure."""
	sidecar_path = find_sidecar(video_path)
	if sidecar_path:
		sc = read_sidecar(sidecar_path)
		if sc:
			sc_fps, sc_frames = from_sidecar(sc)
			if sc_fps and sc_frames:
				duration = sc_frames / sc_fps if sc_fps > 0 else 0.0
				return {
					"source": "sidecar",
					"sidecar_path": str(sidecar_path),
					"fps": sc_fps,
					"frames": sc_frames,
					"duration": duration,
				}
			log("[WARN] sidecar %s exists but missing fps/frames; falling back to ffprobe" % sidecar_path.name)

	# Fallback: ffprobe (slow on /bucket).
	fps, frames, duration = ffprobe_fps_and_frames(video_path, max_sec, ffprobe_timeout)
	return {
		"source": "ffprobe",
		"sidecar_path": "",
		"fps": fps,
		"frames": frames,
		"duration": duration,
	}


def main():
	ap = argparse.ArgumentParser(description=__doc__)
	ap.add_argument("--dir", required=True, type=Path)
	ap.add_argument("--out", required=True, type=Path)
	ap.add_argument("--chunk-sec", type=int, default=7200)
	ap.add_argument("--max-probe-sec", type=int, default=180)
	ap.add_argument("--ffprobe-timeout", type=int, default=120,
	                help="ffprobe duration-probe timeout (s). bucket reads can be slow; default 120.")
	ap.add_argument("--workers", type=int, default=8,
	                help="parallel probe workers (sidecar reads are cheap; default 8)")
	ap.add_argument("--sidecar-only", action="store_true",
	                help="error out instead of running ffprobe when sidecar is missing")
	ap.add_argument("--no-probe", action="store_true",
	                help="Skip sidecar/ffprobe entirely; emit source_path only (for --only-backup). fps/frame_count left 0.")
	args = ap.parse_args()

	if not args.dir.is_dir():
		log("[ERR] not a directory: %s" % args.dir)
		return 2

	# Discover + filter (cheap, sequential).
	all_videos = []
	for ext in VIDEO_EXTS:
		all_videos.extend(sorted(args.dir.glob("*" + ext)))

	candidates = []
	for v in all_videos:
		name = v.name
		if GLOBAL_PREFIX_RE.match(name):
			log("[SKIP] %s (global_*)" % name)
			continue
		if not GRID_PREFIX_RE.match(name):
			log("[SKIP] %s (unrecognized prefix; expected cam##_ or grid##_)" % name)
			continue
		if v.stem.endswith(("_renc", "_nvenc")):
			log("[SKIP] %s (re-encoded artifact)" % name)
			continue
		# Skip broken symlinks / non-regular files: is_file() follows the link and
		# returns False for a dangling target. Without this, dead links (e.g. a
		# cross-block symlink whose target dir does not exist) are ingested as
		# 0-frame "grid videos", which then fail chunking and wedge the afterok DAG.
		if not v.is_file():
			log("[SKIP] %s (broken symlink or not a regular file)" % name)
			continue
		candidates.append(v)

	n = len(candidates)
	# Always write header.
	args.out.parent.mkdir(parents=True, exist_ok=True)
	if n == 0:
		log("[ERR] no grid videos found under %s" % args.dir)
		with args.out.open("w", newline="") as f:
			f.write("vname,source_path,ext,fps,frame_count,duration_sec,n_chunks\n")
		return 0

	if args.no_probe:
		# --only-backup needs only source_path; skip the slow sidecar/ffprobe probe.
		rows = [{
			"vname": v.stem,
			"source_path": str(v.resolve()),
			"ext": v.suffix.lstrip("."),
			"fps": "0.000",
			"frame_count": 0,
			"duration_sec": "0.0",
			"n_chunks": 0,
		} for v in candidates]
		with args.out.open("w", newline="") as f:
			w = csv.DictWriter(f, fieldnames=[
				"vname", "source_path", "ext", "fps",
				"frame_count", "duration_sec", "n_chunks",
			])
			w.writeheader()
			w.writerows(rows)
		log("[INFO] --no-probe: listed %d grid videos (no fps/frames) -> %s" % (len(rows), args.out))
		return 0

	log("[INFO] scanning %d videos with %d workers (sidecar=truth, ffprobe=fallback)" %
	    (n, args.workers))

	# Pre-flight: if --sidecar-only, verify all candidates have sidecars.
	if args.sidecar_only:
		missing = [v for v in candidates if find_sidecar(v) is None]
		if missing:
			log("[ERR] --sidecar-only: %d videos lack sidecar JSON:" % len(missing))
			for v in missing[:10]:
				log("       %s" % v.name)
			return 3

	# Probe in parallel.
	results = [None] * n
	with ThreadPoolExecutor(max_workers=args.workers) as ex:
		futures = {
			ex.submit(probe_one, v, args.max_probe_sec, args.ffprobe_timeout): (i, v)
			for i, v in enumerate(candidates)
		}
		done = 0
		for fut in as_completed(futures):
			i, v = futures[fut]
			r = fut.result()
			fps = r["fps"]
			frames = r["frames"]
			duration = r["duration"] or (frames / fps if fps > 0 else 0.0)
			n_chunks = max(1, math.ceil(duration / args.chunk_sec)) if duration > 0 else 1
			results[i] = {
				"vname": v.stem,
				"source_path": str(v.resolve()),
				"ext": v.suffix.lstrip("."),
				"fps": "%.3f" % fps,
				"frame_count": int(frames),
				"duration_sec": "%.1f" % duration,
				"n_chunks": n_chunks,
			}
			done += 1
			tag = "sidecar" if r["source"] == "sidecar" else "ffprobe"
			log("[scan %d/%d] %s (%s) fps=%.2f frames=%d dur=%.0fs chunks=%d" %
			    (done, n, v.stem, tag, fps, int(frames), duration, n_chunks))

	with args.out.open("w", newline="") as f:
		w = csv.DictWriter(f, fieldnames=[
			"vname", "source_path", "ext", "fps",
			"frame_count", "duration_sec", "n_chunks",
		])
		w.writeheader()
		w.writerows(results)

	log("[INFO] manifest: %d grid videos -> %s" % (len(results), args.out))
	return 0


if __name__ == "__main__":
	sys.exit(main())
