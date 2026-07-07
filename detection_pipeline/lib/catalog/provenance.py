"""Recover per-block processing provenance from hpc_logs on the bucket.

The pipeline's bridge job echoes the SLEAP model paths it used, and the real
worklist (authoritative chunk list + expected_frames) is uploaded to
hpc_logs/pipeline/. Reading these lets recovery reuse the exact models and the
exact per-chunk frame caps, so a resubmit is consistent with the original run.

If a block has no such logs, model paths fall back to an optional
`_catalog/recover.config.json` default so the recovery command is still complete.
"""
import glob
import json
import os
import re

_CENTROID_RE = re.compile(r"centroid:\s*(\S+)")
_INSTANCE_RE = re.compile(r"instance:\s*(\S+)")
_RUNTIME_RE = re.compile(r"runtime:\s*(\S+)")
# saion sleap partition, taken from the TRT engine dir suffix ..__<partition>/model
_ENGINE_PART_RE = re.compile(r"__([a-z0-9-]+)/model")

# saion partition -> (sleap GPU concurrency cap, per-task walltime label)
PARTITION_CONC = {"largegpu": 8, "short-a100": 32, "gpu-a100": 8}
PARTITION_WALL = {"largegpu": "12h", "short-a100": "1h", "gpu-a100": "2h"}


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


def read_provenance(blockdir):
    """Return {sleap_model_centroid, sleap_model_instance, sleap_runtime,
    worklist{vname:{idx:expected}}, worklist_path, source}."""
    prov = {"sleap_model_centroid": "", "sleap_model_instance": "",
            "sleap_runtime": "", "saion_partition": "", "partitions": [],
            "worklist": {}, "worklist_path": "", "source": ""}
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
    if part_counts:
        prov["saion_partition"] = max(part_counts, key=lambda k: (part_counts[k], k))
        prov["partitions"] = sorted(part_counts)

    # Authoritative worklist (vname -> {chunk_idx -> expected_frames}).
    wl = os.path.join(pdir, "aruco_worklist.txt")
    if os.path.isfile(wl):
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
    """(centroid, instance, runtime, source): block logs win, else config default."""
    if prov.get("sleap_model_centroid") and prov.get("sleap_model_instance"):
        return (prov["sleap_model_centroid"], prov["sleap_model_instance"],
                prov.get("sleap_runtime", ""), "hpc_logs")
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
