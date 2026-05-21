#!/usr/bin/env python3
"""Build a chunk-ordered worklist from manifest.csv.

Output: one line per chunk, tab-separated `<vname>\\t<NNN>`, sorted by
(chunk_idx ASC, vname ASC). Slurm processes array tasks in task-id order,
so this layout means earlier chunks fill concurrency slots first.
"""
import argparse
import csv
import sys
from pathlib import Path


def main():
	ap = argparse.ArgumentParser(description=__doc__)
	ap.add_argument("--manifest", required=True, type=Path)
	ap.add_argument("--out", required=True, type=Path)
	args = ap.parse_args()

	rows = []
	with args.manifest.open() as f:
		for r in csv.DictReader(f):
			try:
				n = int(r["n_chunks"])
			except (KeyError, ValueError):
				print(f"[ERR] bad n_chunks for row {r!r}", file=sys.stderr)
				return 2
			vname = r["vname"]
			for i in range(n):
				rows.append((i, vname))

	rows.sort()

	args.out.parent.mkdir(parents=True, exist_ok=True)
	with args.out.open("w") as f:
		for idx, vname in rows:
			f.write(f"{vname}\t{idx:03d}\n")

	print(f"[INFO] worklist: {len(rows)} chunks -> {args.out}", file=sys.stderr)
	return 0


if __name__ == "__main__":
	sys.exit(main())
