"""Recover per-block processing provenance.

Two source trees, in priority order.

1. ``data/PIPELINE_STATE.json`` -- the block's declared processing contract,
   written by pipeline.sh (see lib/pipeline_state.py). Authoritative: chunking,
   model paths and per-video chunk counts are *recorded* rather than inferred,
   and the wave ledger says which parts of the block were ever claimed.

2. ``hpc_logs/`` -- the legacy path, and still the only one for every block
   processed before the contract existed. The bridge job echoes the SLEAP model
   paths it used and the real worklist is uploaded to ``hpc_logs/pipeline/``, so
   both can be scraped back out. This is genuinely a scrape: model paths come
   from a regex over log text, and ``--chunk-sec`` cannot be read at all, only
   guessed from frame counts (see recover.infer_chunk_sec).

   ``hpc_logs/pipeline/pipeline.env`` sits alongside: pipeline.sh has always
   written its full configuration there as ``export KEY="value"`` lines, and it
   is archived with the logs. Bridge jobs only started echoing their models in
   summer 2026, so for most contract-less blocks this file is the *only* record
   of the model set (and of the saion partition). It is read after the bridge
   logs and fills only what they left blank. It also records ``CHUNK_SEC``, but
   that is deliberately not consumed here: footprint/recover derive chunk_sec
   from the contract or from frame counts, and switching them to an archived
   value needs the manifest cross-check that ``state-init`` performs.

If a block has neither, model paths fall back to an optional
`_catalog/recover.config.json` default so the recovery command is still complete.
"""
import glob
import json
import math
import os
import re

# lib/ is on sys.path when catalog.py is the entry point; add it ourselves when
# this package is imported some other way (a REPL, another tool).
try:
    import pipeline_state
except ImportError:  # pragma: no cover - import-path fallback
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import pipeline_state

_CENTROID_RE = re.compile(r"centroid:\s*(\S+)")
_INSTANCE_RE = re.compile(r"instance:\s*(\S+)")
_RUNTIME_RE = re.compile(r"runtime:\s*(\S+)")
# saion sleap partition, taken from the TRT engine dir suffix ..__<partition>/model
_ENGINE_PART_RE = re.compile(r"__([a-z0-9-]+)/model")
# one pipeline.env assignment: `export KEY="value"`, `KEY='value'` or bare `KEY=value`
_ENV_LINE_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")

# saion partition -> (sleap GPU concurrency cap, per-task walltime label).
# Must match saion_caps() in pipeline.sh; short-a100's wall is 2h, not 1h.
PARTITION_CONC = {"largegpu": 8, "short-a100": 32, "gpu-a100": 8}
PARTITION_WALL = {"largegpu": "12h", "short-a100": "2h", "gpu-a100": "8h"}


def partition_conc(p):
    return PARTITION_CONC.get(p, 8)


def partition_wall(p):
    return PARTITION_WALL.get(p, "the partition wall")


def _hpc_logs_dir(blockdir):
    for name in ("hpc_logs", "hpc_log"):
        p = os.path.join(blockdir, name)
        if os.path.isdir(p):
            return p
    return None


def _read_pipeline_env(pdir):
    """{KEY: value} from <pdir>/pipeline.env, or {} when absent or unreadable.

    Only flat assignments are parsed (one per line, optional ``export``, one
    layer of matching quotes stripped). Anything else -- comments, blank lines,
    shell constructs -- is skipped rather than interpreted: this is a record to
    read back, not a script to run.
    """
    path = os.path.join(pdir, "pipeline.env")
    if not os.path.isfile(path):
        return {}
    env = {}
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = _ENV_LINE_RE.match(line.rstrip("\n"))
                if not m:
                    continue
                val = m.group(2).strip()
                if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
                    val = val[1:-1]
                env[m.group(1)] = val
    except OSError:
        return {}
    return env


def _fill_from_env(prov, env):
    """Fill the model/partition gaps pipeline.env can answer; never override.

    Models are all-or-nothing: a centroid without an instance is not a usable
    model set, and a recovery command built from half of one would be worse than
    an empty cell. ``source`` names where the *models* came from, exactly as the
    bridge-log branch sets it. Returns the (mutated) prov for convenience.
    """
    if not prov["sleap_model_centroid"]:
        c = env.get("SLEAP_MODEL_CENTROID", "")
        i = env.get("SLEAP_MODEL_INSTANCE", "")
        if c and i:
            prov["sleap_model_centroid"] = c
            prov["sleap_model_instance"] = i
            prov["sleap_runtime"] = env.get("SLEAP_RUNTIME", "")
            prov["source"] = "hpc_logs"
    if not prov["saion_partition"] and env.get("SAION_PARTITION"):
        prov["saion_partition"] = env["SAION_PARTITION"]
        prov["partitions"] = [env["SAION_PARTITION"]]
    return prov


def read_state(blockdir):
    """The block's declared processing contract, or None.

    A malformed contract is treated as absent here -- the catalog is a reporting
    tool and must survive a bad file on one block out of hundreds. pipeline.sh
    makes the opposite choice and refuses to run, which is where it matters.
    """
    try:
        return pipeline_state.load(os.path.join(blockdir, "data"))
    except (ValueError, OSError):
        return None


def worklist_from_state(state):
    """{vname: {idx: expected_frames}} derived from the contract.

    Same arithmetic as lib/worklist.py, so recovery reproduces the exact frame
    caps the original run used without needing hpc_logs to have survived.
    """
    ch = (state or {}).get("chunking", {})
    chunk_sec = ch.get("chunk_sec")
    out = {}
    if not chunk_sec:
        return out
    for vname, meta in ch.get("videos", {}).items():
        fps = float(meta.get("fps") or 0)
        n = int(meta.get("n_chunks") or 0)
        frames = int(meta.get("frame_count") or 0)
        per = int(round(fps * chunk_sec)) if fps > 0 else 0
        if n <= 0 or per <= 0:
            continue
        rows = {}
        for i in range(n):
            if i == n - 1 and frames > 0:
                rows[i] = max(1, frames - (n - 1) * per)
            else:
                rows[i] = per
        out[vname] = rows
    return out


def read_provenance(blockdir):
    """Return {sleap_model_centroid, sleap_model_instance, sleap_runtime,
    worklist{vname:{idx:expected}}, worklist_path, source, chunk_sec, declared,
    waves, unclaimed, state_path}."""
    prov = {"sleap_model_centroid": "", "sleap_model_instance": "",
            "sleap_runtime": "", "saion_partition": "", "partitions": [],
            "worklist": {}, "worklist_path": "", "source": "",
            "chunk_sec": None, "declared": {}, "waves": [], "unclaimed": {},
            "state_path": ""}

    # --- 1. the declared contract, when the block has one ---------------------
    state = read_state(blockdir)
    if state:
        det = state.get("detection", {})
        ch = state.get("chunking", {})
        prov["source"] = "state_file"
        prov["state_path"] = pipeline_state.state_path(os.path.join(blockdir, "data"))
        prov["sleap_model_centroid"] = det.get("sleap_model_centroid", "")
        prov["sleap_model_instance"] = det.get("sleap_model_instance", "")
        prov["sleap_runtime"] = det.get("sleap_runtime", "")
        prov["saion_partition"] = det.get("saion_partition", "")
        prov["partitions"] = [det["saion_partition"]] if det.get("saion_partition") else []
        prov["chunk_sec"] = ch.get("chunk_sec")
        prov["declared"] = dict((v, int(m.get("n_chunks", 0)))
                                for v, m in ch.get("videos", {}).items())
        prov["waves"] = list(state.get("waves", []))
        prov["unclaimed"] = pipeline_state.unclaimed_ranges(state)
        prov["worklist"] = worklist_from_state(state)

    # --- 2. hpc_logs, for blocks with no contract -----------------------------
    # Still scanned when a contract exists, but only to fill gaps it left: the
    # contract always wins on anything both can answer.
    hl = _hpc_logs_dir(blockdir)
    if not hl:
        return prov
    pdir = os.path.join(hl, "pipeline")

    # Model paths from the newest bridge_*.out that records them; partition from
    # the TRT engine suffix across ALL bridge runs (a block may have been
    # resubmitted to more than one partition -> keep the most-used as primary).
    part_counts = {}
    for bo in sorted(glob.glob(os.path.join(pdir, "bridge_*.out")), reverse=True):
        try:
            txt = open(bo, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        if not prov["sleap_model_centroid"]:
            c, i, r = _CENTROID_RE.search(txt), _INSTANCE_RE.search(txt), _RUNTIME_RE.search(txt)
            if c and i:
                prov["sleap_model_centroid"] = c.group(1)
                prov["sleap_model_instance"] = i.group(1)
                prov["sleap_runtime"] = r.group(1) if r else ""
                prov["source"] = "hpc_logs"
        for p in _ENGINE_PART_RE.findall(txt):
            part_counts[p] = part_counts.get(p, 0) + 1
    if part_counts and not prov["saion_partition"]:
        prov["saion_partition"] = max(part_counts, key=lambda k: (part_counts[k], k))
        prov["partitions"] = sorted(part_counts)

    # pipeline.env: the configuration the run was submitted with. Bridge jobs
    # before the `centroid:` echo left nothing else to read the models from.
    env = _read_pipeline_env(pdir)
    if env:
        _fill_from_env(prov, env)

    # Worklist (vname -> {chunk_idx -> expected_frames}) from the archived copy.
    # Only when the contract did not already supply one: on a wave-processed
    # block the archived aruco_worklist.txt is just the LAST wave's window, so
    # trusting it would understate the block by every earlier wave.
    wl = os.path.join(pdir, "aruco_worklist.txt")
    if os.path.isfile(wl) and not prov["worklist"]:
        prov["worklist_path"] = wl
        m = {}
        try:
            with open(wl, encoding="utf-8", errors="replace") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) >= 2 and parts[1].isdigit():
                        exp = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else 0
                        m.setdefault(parts[0], {})[int(parts[1])] = exp
        except OSError:
            pass
        prov["worklist"] = m
    return prov


def _videos_from_worklist(wl_map):
    """Reconstruct per-video shape from an archived worklist.

    Fallback for blocks whose manifest.csv did not survive. n_chunks is the row
    count and frame_count the sum of the per-chunk caps; fps is unknowable from
    a worklist, and is left at 0 rather than invented -- worklist_from_state
    skips any video without it instead of deriving wrong frame caps.
    """
    out = {}
    for vname, rows in wl_map.items():
        if not rows:
            continue
        out[vname] = {"n_chunks": max(rows) + 1,
                      "frame_count": sum(rows.values()),
                      "fps": 0.0}
    return out


def backfill_state(blockdir, chunk_sec, chunk_ext="mkv", overrides=None,
                   dry_run=False, log=print):
    """Write a PIPELINE_STATE.json for a block processed before contracts existed.

    chunk_sec must be supplied, not guessed. Guessing it from frame counts is the
    exact weakness the contract exists to remove, and a wrong value here would be
    worse than no contract at all: it would look authoritative while licensing a
    future run to re-chunk the block incompatibly.

    The whole block is recorded as one already-submitted wave, so a legacy block
    does not read as "nothing claimed yet". Marked source=backfilled so it is
    never mistaken for a contract that was declared up front.
    """
    data_dir = os.path.join(blockdir, "data")
    if not os.path.isdir(data_dir):
        raise ValueError("no data/ dir under %s; nothing to backfill" % blockdir)
    if pipeline_state.load(data_dir) is not None:
        raise ValueError("%s already has a contract; refusing to overwrite"
                         % pipeline_state.state_path(data_dir))

    prov = read_provenance(blockdir)
    hl = _hpc_logs_dir(blockdir)
    manifest = os.path.join(hl, "pipeline", "manifest.csv") if hl else ""

    if manifest and os.path.isfile(manifest):
        videos = pipeline_state.read_manifest(manifest)
        src = "manifest.csv"
        # Cross-check the operator's chunk_sec against the archived manifest
        # before trusting it: n_chunks there was computed as ceil(dur/chunk_sec).
        for vname, meta in videos.items():
            fps, frames, n = meta["fps"], meta["frame_count"], meta["n_chunks"]
            if fps > 0 and n > 0:
                implied = int(math.ceil((frames / fps) / float(chunk_sec)))
                if implied != n:
                    raise ValueError(
                        "--chunk-sec %s disagrees with the archived manifest: %s has "
                        "%d chunks, but %ss chunking implies %d. Pass the value the "
                        "original run used." % (chunk_sec, vname, n, chunk_sec, implied))
                break
    elif prov.get("worklist"):
        videos = _videos_from_worklist(prov["worklist"])
        src = "aruco_worklist.txt"
        log("[WARN] no archived manifest.csv; reconstructing per-video shape from "
            "the worklist (fps unavailable, frame caps preserved)")
    else:
        raise ValueError(
            "no archived manifest.csv or aruco_worklist.txt under %s/hpc_logs; "
            "cannot establish what this block contains" % blockdir)

    detection = {}
    for key, val in (("sleap_model_centroid", prov.get("sleap_model_centroid")),
                     ("sleap_model_instance", prov.get("sleap_model_instance")),
                     ("sleap_runtime", prov.get("sleap_runtime")),
                     ("saion_partition", prov.get("saion_partition"))):
        if val:
            detection[key] = val
    detection.update(overrides or {})

    state = pipeline_state.new_state(blockdir, chunk_sec, chunk_ext, videos, detection)
    state["source"] = "backfilled"
    state["backfilled_from"] = src
    pipeline_state.add_wave(state, None, pipeline_state.total_rows(videos),
                            {"note": "backfilled: pre-contract run, whole block"})

    log("[INFO] %s: %d videos, %d chunks declared, chunk_sec=%s (from %s)"
        % (blockdir, len(videos), pipeline_state.total_rows(videos), chunk_sec, src))
    for key in sorted(detection):
        log("       %-24s %s" % (key, detection[key]))
    if dry_run:
        log("[INFO] --dry-run: not written")
        return state
    pipeline_state.write(data_dir, state)
    log("[OK] wrote %s" % pipeline_state.state_path(data_dir))
    return state


def load_config(outdir):
    """Optional default model paths: _catalog/recover.config.json."""
    path = os.path.join(outdir, "recover.config.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        return {
            "sleap_model_centroid": cfg.get("sleap_model_centroid", ""),
            "sleap_model_instance": cfg.get("sleap_model_instance", ""),
            "sleap_runtime": cfg.get("sleap_runtime", ""),
        }
    except (OSError, ValueError):
        return {}


def resolve_models(prov, config_default):
    """(centroid, instance, runtime, source): the block wins, else config default.

    Source is reported as recorded ("state_file") vs scraped ("hpc_logs") so a
    recovery command makes clear which it is standing on.
    """
    if prov.get("sleap_model_centroid") and prov.get("sleap_model_instance"):
        return (prov["sleap_model_centroid"], prov["sleap_model_instance"],
                prov.get("sleap_runtime", ""), prov.get("source") or "hpc_logs")
    cd = config_default or {}
    if cd.get("sleap_model_centroid") and cd.get("sleap_model_instance"):
        return (cd["sleap_model_centroid"], cd["sleap_model_instance"],
                cd.get("sleap_runtime", ""), "config")
    return "", "", "", ""


def model_label(prov):
    """Short label for a table cell: the model set's directory name."""
    c = prov.get("sleap_model_centroid") or ""
    if not c:
        return ""
    parent = os.path.basename(os.path.dirname(c.replace("\\", "/")))
    return parent or os.path.basename(c.replace("\\", "/")).split(".")[0]
