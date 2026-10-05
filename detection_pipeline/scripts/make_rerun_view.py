#!/usr/bin/env python3
"""Build a rerun view next to a block, to SLEAP-re-process selected (camera, chunk) pairs.

<block>-<suffix>/ holds relative links to the target cameras' videos (+ sidecars) and the session
log; its data/ holds links to the outputs of every chunk of those cameras that is NOT listed.
Running `pipeline.sh --only-sleap --dir <view>` then chunks the target cameras, and the
bucket-aware skip (filter_done_chunks.py) sends only the listed chunks to the GPU. Move the results
into the block with promote_rerun.py.

  python3 make_rerun_view.py --block <basler>/<date>/<block> --rerun <tsv> --suffix k96 [--dry-run]

<tsv> rows: block<TAB>camera<TAB>chunk, block relative to the basler root (20260928/block01).
Rows for other blocks are ignored. Python 3.6 (deigo's system python), standard library only.
"""
import argparse
import datetime as dt
import json
import os
import re
import sys

TOOL = "detection_pipeline/scripts/make_rerun_view.py"
VIDEO_RE = re.compile(r"^((cam\d+)|global)_.+\.mkv$")


def read_rerun(path, block):
    """{camera: sorted chunk indices} for the rows naming this block."""
    norm = os.path.normpath(block).replace(os.sep, "/")
    out = {}
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3 or not parts[0]:
                continue
            if not norm.endswith("/" + parts[0].strip("/")):
                continue
            out.setdefault(parts[1], set()).add(int(parts[2]))
    return {cam: sorted(chunks) for cam, chunks in out.items()}


def find_videos(block):
    """{camera: video stem} for the grid videos directly in the block."""
    vids = {}
    for name in sorted(os.listdir(block)):
        m = VIDEO_RE.match(name)
        if m:
            if m.group(1) in vids:
                sys.exit("[ERR] two videos for %s in %s" % (m.group(1), block))
            vids[m.group(1)] = name[:-len(".mkv")]
    return vids


def done_chunks(data, vname):
    """Chunk indices whose .slp and _sleap_data.h5 both exist in the block's data/."""
    pat = re.compile(r"^%s_(\d{3})\.slp$" % re.escape(vname))
    out = []
    for name in os.listdir(data):
        m = pat.match(name)
        if m and os.path.exists(os.path.join(data, "%s_%s_sleap_data.h5" % (vname, m.group(1)))):
            out.append(int(m.group(1)))
    return sorted(out)


def make_dir(path, gid):
    os.makedirs(path)
    try:
        os.chown(path, -1, gid)
        os.chmod(path, 0o2775)     # setgid + group-writable: a second user runs the other waves
    except (OSError, AttributeError):
        pass


def link(target, at):
    os.symlink(os.path.relpath(target, os.path.dirname(at)), at)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--block")
    ap.add_argument("--rerun")
    ap.add_argument("--suffix", default="k96")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if not a.block or not a.rerun:
        ap.error("--block and --rerun are required")
    block = os.path.abspath(a.block.rstrip("/"))
    data = os.path.join(block, "data")
    view = "%s-%s" % (block, a.suffix)
    if os.path.lexists(view):
        sys.exit("[ERR] %s exists; remove it or pick another --suffix" % view)
    rerun = read_rerun(a.rerun, block)
    if not rerun:
        sys.exit("[ERR] %s lists no chunk of %s" % (a.rerun, block))
    vids = find_videos(block)
    missing = sorted(set(rerun) - set(vids))
    if missing:
        sys.exit("[ERR] no video in %s for %s" % (block, ", ".join(missing)))

    plan = {}
    for cam, chunks in sorted(rerun.items()):
        linked = [c for c in done_chunks(data, vids[cam]) if c not in set(chunks)]
        plan[cam] = {"vname": vids[cam], "rerun_chunks": chunks, "linked_chunks": linked}
        print("%-6s rerun %3d chunks, link %3d done chunks" % (cam, len(chunks), len(linked)))
    if a.dry_run:
        print("[DRY RUN] would create %s" % view)
        return 0

    gid = os.stat(block).st_gid
    make_dir(view, gid)
    make_dir(os.path.join(view, "data"), gid)
    make_dir(os.path.join(view, "hpc_logs"), gid)
    for name in sorted(os.listdir(block)):
        if name.startswith("sess_") and name.endswith(".txt"):
            link(os.path.join(block, name), os.path.join(view, name))
    for cam, p in plan.items():
        for name in (p["vname"] + ".mkv", p["vname"] + ".mkv.diag.json"):
            if os.path.exists(os.path.join(block, name)):
                link(os.path.join(block, name), os.path.join(view, name))
        for c in p["linked_chunks"]:
            for name in ("%s_%03d.slp" % (p["vname"], c), "%s_%03d_sleap_data.h5" % (p["vname"], c)):
                link(os.path.join(data, name), os.path.join(view, "data", name))
        p["linked_chunks"] = len(p["linked_chunks"])
    info = {"tool": TOOL, "source_block": block, "view": view, "suffix": a.suffix,
            "rerun_tsv": os.path.abspath(a.rerun),
            "created": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "cams": plan,
            "note": "SLEAP re-run of the listed chunks; data/ links are the chunks NOT re-run "
                    "(the bucket-aware skip treats them as done). Promote with promote_rerun.py."}
    with open(os.path.join(view, "VIEW_INFO.json"), "w") as fh:
        json.dump(info, fh, indent=2)
    print("[OK] %s: %d cameras, %d chunks to re-run" % (view, len(plan), sum(len(p["rerun_chunks"]) for p in plan.values())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
