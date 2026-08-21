#!/usr/bin/env python3
"""Build a chunk-ordered worklist from manifest.csv.

Output: one line per chunk, tab-separated `<vname>\\t<NNN>\\t<expected_frames>`,
sorted by (chunk_idx ASC, vname ASC). Slurm processes array tasks in task-id
order, so this layout means earlier chunks fill concurrency slots first.

expected_frames caps inference at the chunk's design frame count via
`sleap-nn predict --n-frames`. Required because ffmpeg `-c copy` segment
propagates the source video's full-stream frame count into the `_000`
segment's container metadata, so sleap-nn 0.2 walks past the actual
decodable end and crashes with IndexError. For all-but-last chunks:
int(fps * chunk_sec). For the last (possibly short) chunk: residual frames.
"""
import argparse
import csv
import sys
from pathlib import Path

# Same dir; sys.path[0] is lib/ when chunk_finalize runs this as a script.
import pipeline_state


def main():
	ap = argparse.ArgumentParser(description=__doc__)
	ap.add_argument("--manifest", required=True, type=Path)
	ap.add_argument("--out", required=True, type=Path)
	ap.add_argument("--chunk-sec", type=int, default=7200,
	                help="ffmpeg segment duration in seconds; used to derive per-chunk frame cap")
	ap.add_argument("--chunk-range", default="",
	                help="inclusive chunk-index window 'A-B' (or 'A') to emit; "
	                     "empty = the whole block. Chunk indices are stable for a "
	                     "given --chunk-sec, so a window always names the same span "
	                     "of wall-clock across runs.")
	args = ap.parse_args()

	try:
		rng = pipeline_state.parse_range(args.chunk_range)
	except ValueError as e:
		print(f"[ERR] --chunk-range: {e}", file=sys.stderr)
		return 2

	rows = []
	with args.manifest.open() as f:
		for r in csv.DictReader(f):
			try:
				n = int(r["n_chunks"])
				fps = float(r["fps"])
				total_frames = int(r["frame_count"])
			except (KeyError, ValueError):
				print(f"[ERR] bad manifest row {r!r}", file=sys.stderr)
				return 2
			vname = r["vname"]
			per_chunk = int(round(fps * args.chunk_sec)) if fps > 0 else 0
			for i in range(n):
				if i == n - 1 and total_frames > 0 and per_chunk > 0:
					# Last chunk: residual after the n-1 full chunks.
					# Clamp to >=1 so we don't emit 0 (predict step treats 0 as "no cap").
					expected = max(1, total_frames - (n - 1) * per_chunk)
				else:
					expected = per_chunk
				rows.append((i, vname, expected))

	rows.sort()

	# Wave processing: keep only the requested chunk-index window. Filtering here
	# rather than splitting the source videos is what keeps waves safe -- the
	# index of a chunk is a pure function of (video, --chunk-sec), so `_042`
	# names the same span of wall-clock in every wave, and re-running a window
	# overwrites its own outputs instead of colliding with a neighbour's.
	# A window is clamped per video, so a short camera simply contributes fewer
	# rows instead of erroring.
	total = len(rows)
	max_idx = max((r[0] for r in rows), default=-1)
	if rng is not None:
		lo, hi = rng
		rows = [r for r in rows if lo <= r[0] <= hi]
		if not rows:
			print(f"[ERR] --chunk-range {args.chunk_range} selects no chunks "
			      f"(block has {total} chunks, highest index {max_idx})",
			      file=sys.stderr)
			return 2

	args.out.parent.mkdir(parents=True, exist_ok=True)
	with args.out.open("w") as f:
		for idx, vname, expected in rows:
			f.write(f"{vname}\t{idx:03d}\t{expected}\n")

	if rng is None:
		print(f"[INFO] worklist: {len(rows)} chunks -> {args.out}", file=sys.stderr)
	else:
		print(f"[INFO] worklist: {len(rows)} of {total} chunks "
		      f"(chunk-range {rng[0]}-{rng[1]}) -> {args.out}", file=sys.stderr)
	return 0


if __name__ == "__main__":
	sys.exit(main())
