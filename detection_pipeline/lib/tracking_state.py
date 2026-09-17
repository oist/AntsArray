#!/usr/bin/env python3
"""Per-block tracking provenance: ``tracks/TRACKING_STATE.json``.

Why this file exists
--------------------
Detection records what it ran with: ``hpc_logs/pipeline/pipeline.env`` and the
``data/PIPELINE_STATE.json`` contract. Tracking did not. The homography stack
(``--hmats``) that maps every camera into the panorama, and the left/right split
(``--x_threshold``) that depends on it, were visible only in the rendered sbatch
under ``/flash`` -- which is purged. A block tracked against a stale calibration
is byte-for-byte indistinguishable on disk from one tracked against the right
one, so once ``/flash`` is gone the question "which hmats made these tracks?"
has no answer. The mapper therefore writes the answer next to its outputs.

It lives in ``tracks/`` because that directory already travels to the bucket
with the tracking transfer manifest, and because every window view of a block
has its own ``tracks/`` -- so each view carries its own record.

Structure
---------
``map``     the mapping run whose panorama_pkls/tracks are on disk. Replaced by
            every new mapping run; when the displaced record differs materially
            (hmats content or x_threshold) it is appended to ``history``.
``stitch``  the last stitching run (fps, frames per chunk, side).
``history`` displaced ``map`` records, oldest first. Never pruned.

``source`` on a map record is ``pipeline`` (the mapper wrote it while running)
or ``backfill`` (``catalog.py track-init`` wrote it from a human's evidence). A
backfill never silently replaces a pipeline record.

Kept python 3.6-compatible (no dataclasses, no f-strings, no 3.7+ syntax): the
catalog imports this on the deigo login node, which runs 3.6.8.
"""
import getpass
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

STATE_BASENAME = "TRACKING_STATE.json"
SCHEMA = 1

# '.../cameraArray_calib/<calib_id>/frame0/aruco_stitch/aruco_H_mats.npz'
_CALIB_ROOT_RE = re.compile(r"cameraArray_calib[\\/]+([^\\/]+)", re.IGNORECASE)
_RELEASE_RE = re.compile(r"/releases/([^/]+)")


def _utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def state_path(tracks_dir):
    return os.path.join(str(tracks_dir), STATE_BASENAME)


def calib_id_from_hmats(path):
    """Label a homography file by the calibration it came from.

    The calibration directory name carries the board recording date
    (``20260623_calib_elevated_by_2mm_from_arenafloor``); the npz itself stores
    no date. A homography kept elsewhere falls back to its parent directory
    name so the cell is never blank when a path is known.
    """
    s = str(path).replace("\\", "/")
    m = _CALIB_ROOT_RE.search(s)
    if m:
        return m.group(1)
    parent = os.path.basename(os.path.dirname(s))
    return parent or os.path.basename(s)


def file_fingerprint(path):
    """{sha256, size, mtime} of the homography file, or {} when unreadable.

    The hash is what makes two records comparable: the same path can hold
    different matrices after a recalibration overwrote it in place.
    """
    try:
        st = os.stat(str(path))
        h = hashlib.sha256()
        with open(str(path), "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        mtime = datetime.fromtimestamp(st.st_mtime, timezone.utc)
        return {"sha256": h.hexdigest(), "size": st.st_size,
                "mtime": mtime.strftime("%Y-%m-%dT%H:%M:%SZ")}
    except OSError:
        return {}


def code_identity(code_dir):
    """Which code ran: the deployed release name when the path has one, else a
    git hash (best effort, never fatal)."""
    if not code_dir:
        return {}
    s = str(code_dir).replace("\\", "/")
    out = {"code_dir": str(code_dir)}
    m = _RELEASE_RE.search(s)
    if m:
        out["release"] = m.group(1)
        return out
    try:
        r = subprocess.run(["git", "-C", str(code_dir), "rev-parse", "--short", "HEAD"],
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                           timeout=5, universal_newlines=True)
        if r.returncode == 0 and r.stdout.strip():
            out["git"] = r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return out


def _runtime():
    try:
        user = getpass.getuser()
    except Exception:  # no passwd entry inside some containers
        user = os.environ.get("USER", "")
    d = {"user": user, "host": socket.gethostname(), "python": sys.executable}
    jid = os.environ.get("SLURM_JOB_ID")
    if jid:
        d["slurm_job_id"] = jid
    return d


def _atomic_write(path, obj):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tracking_state.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
        # mkstemp creates 0600. Best-effort group share, as pipeline_state does:
        # the mapper and the catalog (or a later backfill) run as different
        # members of the unit group on the same /bucket tree.
        try:
            os.chmod(path, 0o664)
        except OSError:
            pass
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _empty():
    return {"schema": SCHEMA, "map": None, "stitch": None, "history": []}


def load(tracks_dir):
    """The block's tracking record, or None when absent. Malformed -> ValueError."""
    p = state_path(tracks_dir)
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        try:
            state = json.load(f)
        except ValueError as e:
            raise ValueError("%s is not valid JSON: %s" % (p, e))
    if not isinstance(state, dict) or "map" not in state:
        raise ValueError("%s has no 'map' record" % p)
    got = state.get("schema")
    if got != SCHEMA:
        raise ValueError("%s has schema %r, this code speaks %d." % (p, got, SCHEMA))
    return state


def _load_or_set_aside(tracks_dir):
    """Load, or move a corrupt record out of the way and start fresh.

    The mapper must not be blocked by a damaged record -- but the bytes are kept
    (renamed with a timestamp) rather than overwritten, so nothing is lost.
    """
    try:
        state = load(tracks_dir)
    except ValueError:
        p = state_path(tracks_dir)
        try:
            os.replace(p, "%s.corrupt.%s.%d" % (p, _utcnow().replace(":", ""), os.getpid()))
        except OSError:
            pass  # a concurrent writer already moved (or replaced) it; proceed
        state = None
    return state if state is not None else _empty()


def _differs(a, b):
    """Two map records are the 'same run' if they used the same hmats content
    (or, lacking a hash, the same path) at the same x_threshold."""
    if not a or not b:
        return True
    ka = (a.get("hmats_sha256") or a.get("hmats"), a.get("x_threshold"))
    kb = (b.get("hmats_sha256") or b.get("hmats"), b.get("x_threshold"))
    return ka != kb


def _chunk_span(chunks):
    if not chunks:
        return None
    cs = sorted(str(c) for c in chunks)
    return {"first": cs[0], "last": cs[-1], "count": len(cs)}


def _map_record(hmats, x_threshold, source, **extra):
    hm = str(hmats)
    rec = {"recorded_at": _utcnow(), "source": source, "hmats": hm,
           "hmats_calib_id": calib_id_from_hmats(hm), "x_threshold": x_threshold}
    fp = file_fingerprint(hm)
    if fp:
        rec["hmats_sha256"] = fp["sha256"]
        rec["hmats_size"] = fp["size"]
        rec["hmats_mtime"] = fp["mtime"]
    for k, v in extra.items():
        if v is not None:
            rec[k] = v
    return rec


def record_map(tracks_dir, hmats, x_threshold, map_mode=None,
               min_instance_frame_frac=None, data_dir=None, chunks=None,
               code_dir=None, argv=None):
    """Record the mapping run about to produce ``tracks_dir``'s contents.

    Called by the pipeline *before* mapping: if the run dies half-way, the
    outputs that did land were still made with these settings. Returns the
    written state.
    """
    state = _load_or_set_aside(tracks_dir)
    rec = _map_record(hmats, x_threshold, "pipeline",
                      map_mode=map_mode,
                      min_instance_frame_frac=min_instance_frame_frac,
                      data_dir=str(data_dir) if data_dir else None,
                      chunks=_chunk_span(chunks),
                      argv=list(argv) if argv else None)
    rec.update(code_identity(code_dir))
    rec.update(_runtime())
    prev = state.get("map")
    if prev and _differs(prev, rec):
        state.setdefault("history", []).append(prev)
    state["map"] = rec
    _atomic_write(state_path(tracks_dir), state)
    return state


def record_stitch(tracks_dir, fps=None, chunk_frames=None, side=None, code_dir=None):
    """Record the stitching run. Leaves ``map`` untouched: the stitch job is
    handed ``--hmats`` too, but does not map with it."""
    state = _load_or_set_aside(tracks_dir)
    rec = {"recorded_at": _utcnow()}
    for k, v in (("fps", fps), ("chunk_frames", chunk_frames), ("side", side)):
        if v is not None:
            rec[k] = v
    rec.update(code_identity(code_dir))
    rec.update(_runtime())
    state["stitch"] = rec
    _atomic_write(state_path(tracks_dir), state)
    return state


def backfill(tracks_dir, hmats, x_threshold=None, tracked_at=None, note=None,
             force=False, dry_run=False):
    """Write a map record for a block tracked before records existed
    (``catalog.py track-init``).

    Refuses to replace a record the pipeline wrote itself unless ``force``: a
    backfill is a human's evidence (a rendered sbatch, a lab note), the
    pipeline's record is a measurement. A malformed existing file is an error
    here, not something to set aside -- the operator is present to look at it.
    """
    state = load(tracks_dir)
    if state is None:
        state = _empty()
    prev = state.get("map")
    if prev and prev.get("source") == "pipeline" and not force:
        raise ValueError("%s already holds a record written by the pipeline "
                         "(hmats %s); pass --force to replace it"
                         % (state_path(tracks_dir), prev.get("hmats_calib_id")))
    rec = _map_record(hmats, x_threshold, "backfill",
                      tracked_at=tracked_at, note=note)
    rec["recorded_by"] = _runtime()["user"]
    # A forced backfill over the pipeline's own record is an operator
    # overruling a measurement: keep the measurement even if the values agree.
    if prev and (_differs(prev, rec) or prev.get("source") == "pipeline"):
        state.setdefault("history", []).append(prev)
    state["map"] = rec
    if not dry_run:
        _atomic_write(state_path(tracks_dir), state)
    return state


def summary(state):
    """Flat cells for a catalog row; empty strings when nothing is recorded."""
    out = {"tracking_hmats": "", "tracking_hmats_path": "",
           "tracking_x_threshold": "", "tracked_at": "", "tracking_source": ""}
    m = (state or {}).get("map") or {}
    if not m:
        return out
    out["tracking_hmats"] = m.get("hmats_calib_id", "")
    out["tracking_hmats_path"] = m.get("hmats", "")
    xt = m.get("x_threshold")
    if isinstance(xt, (int, float)):
        out["tracking_x_threshold"] = "%g" % xt
    elif xt is not None:
        out["tracking_x_threshold"] = str(xt)
    # A backfill says when tracking happened (tracked_at); a pipeline record IS
    # the tracking run, so its own timestamp is the answer.
    out["tracked_at"] = m.get("tracked_at") or m.get("recorded_at", "")
    out["tracking_source"] = m.get("source", "")
    return out
