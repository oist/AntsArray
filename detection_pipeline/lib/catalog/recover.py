"""Targeted recovery for partially-processed blocks.

Given a block's footprint (per-video chunk sets) it computes exactly which
(video, chunk) outputs are missing per stage, writes a pipeline-format
sub-worklist of only those chunks, and prints the resubmit command(s). The
pipeline's array jobs are worklist-driven and skip existing outputs, so feeding
a stage a sub-worklist recovers "the necessary files only".

Recovery types (cheapest first):
  upload   SILENT_PARTIAL: outputs computed but not uploaded -> rescue-copy
  slp2h5   .slp present but _sleap_data.h5 missing -> slp2h5 over the bucket .slp
  aruco    aruco outputs behind -> aruco array over a sub-worklist
  sleap    .slp missing -> re-chunk + SLEAP predict (+slp2h5) for those chunks
"""
import csv
import math
import os

from . import const, footprint as fp_mod, provenance
from .model import Unit

_STD_CHUNK_SEC = (7200, 3600, 1800, 900, 600, 300)


def infer_chunk_sec(fps, frames, n):
    """Recover the pipeline's --chunk-sec from fps/frames and the chunk count."""
    if not fps or not frames or n <= 1:
        return None
    lo, hi = frames / n, frames / (n - 1)
    for cs in _STD_CHUNK_SEC:
        if lo < fps * cs <= hi:
            return cs
    return None


def _universe(fp, v):
    """The 0..maxchunk set the deepest stage reached for video v."""
    mx = -1
    for d in fp.chunk_sets.values():
        s = d.get(v)
        if s:
            mx = max(mx, max(s))
    return set(range(mx + 1)) if mx >= 0 else set()


def missing_sets(fp):
    """stage -> {vname: sorted [missing chunk idx]} for sleap/slp2h5/aruco_det/aruco_trk."""
    cs = fp.chunk_sets or {}
    vids = set()
    for d in cs.values():
        vids |= set(d)
    out = {"sleap": {}, "slp2h5": {}, "aruco_det": {}, "aruco_trk": {}}
    for v in vids:
        uni = _universe(fp, v)
        slp = cs.get("slp", {}).get(v, set())
        det = cs.get("det", {}).get(v, set())
        trk = cs.get("trk", {}).get(v, set())
        sdat = cs.get("sdat", {}).get(v, set())
        m = sorted(uni - slp)
        if m:
            out["sleap"][v] = m
        m = sorted((slp & uni) - sdat)      # slp exists, h5 doesn't -> slp2h5 only
        if m:
            out["slp2h5"][v] = m
        m = sorted(uni - det)
        if m:
            out["aruco_det"][v] = m
        m = sorted(det - trk)               # detections exist, tracks don't
        if m:
            out["aruco_trk"][v] = m
    return out


def _count(m):
    return sum(len(x) for x in m.values())


def classify(fp, has_silent_partial):
    """Return (recover_type, missing_str, missing_sets_dict)."""
    m = missing_sets(fp)
    n_sleap, n_h5 = _count(m["sleap"]), _count(m["slp2h5"])
    n_det, n_trk = _count(m["aruco_det"]), _count(m["aruco_trk"])
    parts = []
    for name, n in (("sleap", n_sleap), ("slp2h5", n_h5),
                    ("aruco_det", n_det), ("aruco_trk", n_trk)):
        if n:
            parts.append("%s:%d" % (name, n))
    # Recovery is only defined for the modern h5 4-tuple format; and if nothing is
    # actually missing there is nothing to reprocess (a SILENT_PARTIAL flag on a
    # complete block is a logs-only issue, not a recovery case).
    if fp.pipeline_format != "h5_4tuple" or not parts:
        return "none", "", m
    if has_silent_partial:
        typ = "upload"          # outputs may sit on flash/work; try upload first
    elif n_sleap:
        typ = "sleap"
    elif n_h5:
        typ = "slp2h5"
    else:
        typ = "aruco"
    return typ, "|".join(parts), m


# --------------------------------------------------------------------------
# worklist rows
# --------------------------------------------------------------------------
def _expected(fps, frames, n, idx, chunk_sec):
    if not chunk_sec or not fps:
        return 0
    per = int(round(fps * chunk_sec))
    if idx == n - 1 and frames:
        return max(1, frames - (n - 1) * per)
    return per


def chunk_worklist_rows(missing_map, fp, vinfo, prov=None):
    """pipeline worklist rows 'vname\\tNNN\\texpected_frames', pipeline sort order.

    Uses the authoritative expected_frames from the block's real worklist
    (prov['worklist']) when available; otherwise derives it from fps/chunk_sec.
    """
    wl = (prov or {}).get("worklist") or {}
    rows = []
    for v, idxs in missing_map.items():
        fps, frames = vinfo.get(v, (0, 0))
        uni = _universe(fp, v)
        n = (max(uni) + 1) if uni else 0
        cs = infer_chunk_sec(fps, frames, n)
        vwl = wl.get(v, {})
        for idx in idxs:
            exp = vwl.get(idx)
            if exp is None:
                exp = _expected(fps, frames, n, idx, cs)
            rows.append((idx, v, exp))
    rows.sort()   # (chunk_idx, vname) — matches worklist.py
    return ["%s\t%03d\t%d" % (v, idx, exp) for idx, v, exp in rows]


def slp_worklist_rows(missing_map, data_dir):
    """Absolute bucket .slp paths for chunks needing slp2h5 (what slp2h5_array.sh reads)."""
    rows = []
    for v in sorted(missing_map):
        for idx in missing_map[v]:
            rows.append(os.path.join(data_dir, "%s_%03d.slp" % (v, idx)))
    return rows


# --------------------------------------------------------------------------
# paths / ids
# --------------------------------------------------------------------------
def block_dir(root, session_id, block):
    parts = [root] + session_id.split("/")
    if block:
        parts += block.split("/")
    return os.path.join(*parts)


def _slug(session_id, block):
    s = session_id.replace("/", "_")
    return s + ("_" + block.replace("/", "_") if block else "")


def parse_block_id(bid):
    """'session::block' | 'session/blockNN' | 'session' | 'session/-' -> (session_id, block)."""
    bid = bid.strip()
    if "::" in bid:
        s, b = bid.split("::", 1)
        return s, b
    if bid.endswith("/-"):
        return bid[:-2], ""
    parts = bid.split("/")
    if len(parts) >= 2 and const.BLOCK_DIR_RE.match(parts[-1]):
        return "/".join(parts[:-1]), parts[-1]
    return bid, ""


# --------------------------------------------------------------------------
# command / steps text
# --------------------------------------------------------------------------
def build_steps(root, session_id, block, fp, vinfo, has_silent_partial, outdir,
                prov=None, config_default=None):
    """Return (recover_type, missing_str, short_cmd, steps_text)."""
    typ, missing, m = classify(fp, has_silent_partial)
    # Display paths use forward slashes: these commands are copy-pasted into a
    # Linux/HPC shell (on saion the paths are already POSIX).
    bd = block_dir(root, session_id, block).replace(os.sep, "/").replace("\\", "/")
    data_dir = bd + "/data"
    slug = _slug(session_id, block)
    rec_dir = os.path.join(outdir, "recover").replace(os.sep, "/").replace("\\", "/")
    cs = None
    for v in (fp.chunk_sets.get("det") or {}):
        fps, frames = vinfo.get(v, (0, 0))
        uni = _universe(fp, v)
        cs = infer_chunk_sec(fps, frames, (max(uni) + 1) if uni else 0)
        if cs:
            break

    gen = "python detection_pipeline/catalog.py recover %s::%s" % (session_id, block)
    if typ == "none":
        return typ, missing, "", "Nothing to recover."
    if typ == "upload":
        short = "# rescue-copy outputs (SILENT_PARTIAL)"
        steps = [
            "This block has outputs but no HPC logs (SILENT_PARTIAL): compute likely",
            "succeeded but the upload was lost. Rescue from the saion login side:",
            "  ssh saion  # login node has bucket write; compute nodes do not",
            "  # locate the /work or /flash output dir from the last run, then:",
            "  cp -n <workdir>/*.slp <workdir>/*_sleap_data.h5 '%s/'" % data_dir,
            "If the outputs are truly gone, treat as the 'sleap' recovery.",
        ]
        return typ, missing, short, "\n".join(steps)
    if typ == "slp2h5":
        n = _count(m["slp2h5"])
        wl = rec_dir + "/" + slug + ".slp2h5.worklist.txt"
        upper = max(0, math.ceil(n / 8) - 1)
        short = ("WORKLIST=%s BATCH=8 sbatch --array=0-%d%%8 "
                 "~/detection_pipeline/scripts/slp2h5_array.sh" % (wl, upper))
        steps = [
            "%d chunks have a .slp but no _sleap_data.h5 (slp2h5 gotcha). CPU-only," % n,
            "no re-chunk — the .slp are already on the bucket.",
            "  1) %s   # writes %s" % (gen, wl),
            "  2) " + short,
            "     (.slp read from the bucket; _sleap_data.h5 written back beside them)",
        ]
        return typ, missing, short, "\n".join(steps)
    if typ == "aruco":
        n = _count(m["aruco_det"]) + _count(m["aruco_trk"])
        wl = rec_dir + "/" + slug + ".aruco.worklist.txt"
        short = ("sbatch --array=0-%d%%16 ~/detection_pipeline/templates/aruco_array.sbatch"
                 "  # WORKLIST=%s" % (max(0, n - 1), wl))
        steps = [
            "%d aruco chunk-outputs missing. aruco reads chunk videos from /flash;" % n,
            "if they were cleaned up, re-chunk those chunks first (see 'sleap' notes).",
            "  1) %s   # writes %s" % (gen, wl),
            "  2) stage %s as $JOBS_ROOT/aruco_worklist.txt, then:" % os.path.basename(wl),
            "     " + short,
        ]
        return typ, missing, short, "\n".join(steps)
    # typ == "sleap"
    n = _count(m["sleap"])
    wl = rec_dir + "/" + slug + ".sleap.worklist.txt"
    cs_txt = str(cs) if cs else "<chunk_sec>"
    centroid, instance, _rt, msrc = provenance.resolve_models(prov or {}, config_default or {})
    if centroid and instance:
        mflags = (" \\\n    --sleap-model-centroid %s \\\n    --sleap-model-instance %s"
                  % (centroid, instance))
        mnote = "   # models + partition auto-filled from %s" % msrc
    else:
        mflags = " \\\n    --sleap-model-centroid <CENTROID_DIR> --sleap-model-instance <INSTANCE_DIR>"
        mnote = "   # models not in hpc_logs; set _catalog/recover.config.json to auto-fill"

    part = (prov or {}).get("saion_partition", "")
    parts_all = (prov or {}).get("partitions") or []
    mixed = [p for p in parts_all if p != part]
    pflag = (" \\\n    --saion-partition %s" % part) if part else ""
    b_part = part or "short-a100"
    b_conc = provenance.partition_conc(b_part)
    b_wall = provenance.partition_wall(b_part)

    simple = "pipeline.sh --dir %s --only-sleap --chunk-sec %s%s%s" % (bd, cs_txt, mflags, pflag)
    steps = [
        "%d chunks are missing their .slp (SLEAP predict never completed -" % n,
        "the scattered-across-cameras pattern is the array-wall timeout).",
        "",
        "Option A - simple (re-chunks the whole block; skip-logic keeps it correct,",
        "but recomputes all chunks, not just the %d):%s" % (n, mnote),
        "  " + simple,
    ]
    if part:
        line = "  # partition = %s (same as the original run)" % part
        if mixed:
            line += "; block also ran on: %s" % ", ".join(mixed)
        steps.append(line)
        if part == "short-a100":
            steps.append("  # note: short-a100 has the 1h wall that caused the original scatter -")
            steps.append("  #       use --saion-partition largegpu for a 12h wall if timeouts recur.")
    steps += [
        "",
        "Option B - minimal (only the %d missing chunks):" % n,
        "  1) %s   # writes the %d-row sub-worklist:" % (gen, n),
        "     %s" % wl,
        "  2) re-chunk ONLY those chunks onto /flash (chunk videos are removed by",
        "     cleanup after a run; a %ss chunk of each listed video is needed)," % cs_txt,
        "  3) copy the sub-worklist to $JOBS_ROOT/aruco_worklist.txt on deigo, then:",
        "     sbatch --partition=%s --array=0-%d%%%d ~/detection_pipeline/templates/sleap_predict_array.sh"
        % (b_part, max(0, n - 1), b_conc),
        "     (%s = %d GPUs, %s wall; BATCH_SIZE=1 -> one chunk per task)" % (b_part, b_conc, b_wall),
        "  4) slp2h5 + upload run inline in the SLEAP task; re-run the catalog to confirm.",
    ]
    return typ, missing, simple, "\n".join(steps)


# --------------------------------------------------------------------------
# dashboard summary (called from build.py, no extra I/O)
# --------------------------------------------------------------------------
def recovery_summary(root, session_id, block, fp, vinfo, has_silent_partial, outdir,
                     prov=None, config_default=None):
    typ, missing, short, steps = build_steps(
        root, session_id, block, fp, vinfo, has_silent_partial, outdir, prov, config_default)
    return {
        "recover_type": typ if typ != "none" else "",
        "recover_missing": missing,
        "recover_cmd": short,
        "recover_steps": steps if typ != "none" else "",
    }


# --------------------------------------------------------------------------
# standalone subcommand
# --------------------------------------------------------------------------
def _vinfo_from_csv(path, session_id, block):
    vinfo = {}
    if not os.path.isfile(path):
        return vinfo
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["session_id"] == session_id and r["block"] == block:
                try:
                    vinfo[r["vname"]] = (float(r["fps"] or 0), int(r["frame_count"] or 0))
                except (ValueError, KeyError):
                    vinfo[r["vname"]] = (0, 0)
    return vinfo


def _write(path, rows):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + ("\n" if rows else ""))
    return path, len(rows)


def write_worklists(rec_dir, session_id, block, fp, vinfo, prov=None):
    """Write the applicable sub-worklists; return {stage: (path, nrows)}."""
    m = missing_sets(fp)
    slug = _slug(session_id, block)
    written = {}
    if m["sleap"]:
        written["sleap"] = _write(os.path.join(rec_dir, slug + ".sleap.worklist.txt"),
                                  chunk_worklist_rows(m["sleap"], fp, vinfo, prov))
    if m["slp2h5"]:
        dd = vinfo.get("__data_dir__", "")
        written["slp2h5"] = _write(os.path.join(rec_dir, slug + ".slp2h5.worklist.txt"),
                                   slp_worklist_rows(m["slp2h5"], dd))
    aruco = {}
    for v, idxs in m["aruco_det"].items():
        aruco[v] = list(idxs)
    for v, idxs in m["aruco_trk"].items():
        aruco[v] = sorted(set(aruco.get(v, [])) | set(idxs))
    if aruco:
        written["aruco"] = _write(os.path.join(rec_dir, slug + ".aruco.worklist.txt"),
                                  chunk_worklist_rows(aruco, fp, vinfo, prov))
    return written


def run_recover(root, outdir, block_id, log):
    session_id, block = parse_block_id(block_id)
    bd = block_dir(root, session_id, block)
    if not os.path.isdir(bd):
        log("[recover][ERR] not a directory: %s" % bd)
        return 2

    u = Unit(session_id=session_id, block=block, path=bd)
    subs = []
    try:
        for e in os.scandir(bd):
            if e.is_dir():
                subs.append(e.name)
                if e.name.lower() == "data":
                    u.has_data_dir, u.data_dir = True, e.path
    except OSError as ex:
        log("[recover][ERR] %s" % ex)
        return 2
    u.subdir_names = subs
    fp = fp_mod.scan_footprint(u)

    has_sp = bool(fp.has_data_dir and (fp.n_slp or fp.n_aruco_det) and not fp.has_hpc_logs)
    vinfo = _vinfo_from_csv(os.path.join(outdir, "videos.csv"), session_id, block)
    vinfo["__data_dir__"] = u.data_dir

    prov = provenance.read_provenance(bd)
    config_default = provenance.load_config(outdir)
    typ, missing, short, steps = build_steps(root, session_id, block, fp, vinfo, has_sp,
                                             outdir, prov, config_default)
    written = write_worklists(os.path.join(outdir, "recover"), session_id, block,
                              fp, vinfo, prov)

    centroid, instance, _rt, msrc = provenance.resolve_models(prov, config_default)
    log("== recover %s/%s ==" % (session_id, block or "-"))
    log("counts: slp=%d aruco_det=%d aruco_tracks=%d sleap_data=%d"
        % (fp.n_slp, fp.n_aruco_det, fp.n_aruco_tracks, fp.n_sleap_data))
    if centroid:
        log("models (%s): %s | %s" % (msrc, centroid, instance))
    part = prov.get("saion_partition", "")
    if part:
        others = [p for p in prov.get("partitions", []) if p != part]
        log("partition: %s%s" % (part, " (also ran on: %s)" % ", ".join(others) if others else ""))
    if prov.get("worklist_path"):
        log("worklist source: %s" % prov["worklist_path"])
    log("recovery type: %s   missing: %s" % (typ, missing or "none"))
    for stage, (path, nrows) in written.items():
        log("worklist[%s]: %s  (%d rows)" % (stage, path, nrows))
    log("")
    for line in steps.split("\n"):
        log(line)
    return 0
