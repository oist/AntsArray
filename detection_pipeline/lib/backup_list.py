#!/usr/bin/env python3
"""Build the Bucket Backup archive file list for a pipeline run.

The list is newline-delimited and relative to the unit bucket root, which lets
the backup job run `zip` from `/bucket/<unit>` and produce a clean archive tree.
"""
import argparse
import csv
import sys
from pathlib import Path


METADATA_SUFFIXES = {".json", ".txt"}


def log(msg):
	sys.stderr.write(msg + "\n")
	sys.stderr.flush()


def is_relative_to(path, root):
	try:
		path.relative_to(root)
		return True
	except ValueError:
		return False


def add_path(paths, path, unit_root):
	resolved = path.resolve()
	if not resolved.is_file():
		log(f"[ERR] backup input missing or not a file: {path}")
		return False
	if not is_relative_to(resolved, unit_root):
		log(f"[ERR] refusing to archive outside unit bucket root: {resolved}")
		log(f"      unit root: {unit_root}")
		return False
	paths.add(resolved)
	return True


def main():
	ap = argparse.ArgumentParser(description=__doc__)
	ap.add_argument("--manifest", required=True, type=Path)
	ap.add_argument("--experiment-dir", required=True, type=Path)
	ap.add_argument("--unit-root", required=True, type=Path)
	ap.add_argument("--out", required=True, type=Path)
	args = ap.parse_args()

	unit_root = args.unit_root.resolve()
	exp_dir = args.experiment_dir.resolve()
	if not exp_dir.is_dir():
		log(f"[ERR] not a directory: {exp_dir}")
		return 2
	if not is_relative_to(exp_dir, unit_root):
		log(f"[ERR] experiment dir is outside unit bucket root: {exp_dir}")
		log(f"      unit root: {unit_root}")
		return 2

	paths = set()
	ok = True
	try:
		with args.manifest.open(newline="") as f:
			for row in csv.DictReader(f):
				src = row.get("source_path")
				if not src:
					log(f"[ERR] manifest row lacks source_path: {row!r}")
					ok = False
					continue
				ok = add_path(paths, Path(src), unit_root) and ok
	except FileNotFoundError:
		log(f"[ERR] manifest not found: {args.manifest}")
		return 2

	for child in exp_dir.iterdir():
		if child.is_file() and child.suffix.lower() in METADATA_SUFFIXES:
			ok = add_path(paths, child, unit_root) and ok

	if not ok:
		return 2
	if not paths:
		log("[ERR] backup list is empty")
		return 2

	rel_paths = sorted(p.relative_to(unit_root).as_posix() for p in paths)
	args.out.parent.mkdir(parents=True, exist_ok=True)
	args.out.write_text("\n".join(rel_paths) + "\n")
	log(f"[INFO] backup file list: {len(rel_paths)} files -> {args.out}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
