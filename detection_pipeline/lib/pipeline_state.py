#!/usr/bin/env python3
"""The per-block processing contract and wave ledger: ``data/PIPELINE_STATE.json``.

Why this file exists
--------------------
Chunk identity is *positional*: ``<vname>_042`` means "the 43rd chunk of this
video **under this --chunk-sec**". Change ``--chunk-sec`` and the very same
filename denotes a different span of wall-clock, so a later run overwrites the
earlier outputs with content that no longer lines up with the ones it did not
overwrite. Nothing downstream can detect that afterwards: the names are
identical and every file is individually valid. ``filter_done_chunks.py`` guards
the SLEAP leg through the h5 ``expected_frames`` attr, but the ArUco leg has no
equivalent, and neither guards a model swap -- and detection counts move ~4x
between model generations, so a half-and-half block is quietly worthless.

The first run therefore *declares* its chunking and detection settings here, and
every later run must agree with the declaration or be refused.

The declaration also gives the catalog a real denominator. Without it,
completeness is inferred as ``0..max(chunk_idx seen)`` (``footprint.py`` and
``recover.py`` both do this), which reports a block processed in waves as
**complete** the moment the first wave lands.

Structure
---------
``chunking`` + ``detection`` are the CONTRACT -- frozen once written.
``waves`` is an append-only LEDGER of the ranges that were submitted.

Completion is deliberately NOT read from the ledger. The ledger records what was
*submitted*; the filesystem records what *exists*. Deriving coverage from
``data/`` means a killed job, a lost login-side poller or a hand-run rescue can
never leave this file claiming work that is not there.

Kept python 3.6-compatible (no dataclasses, no f-strings, no 3.7+ syntax):
deigo login and compute nodes run 3.6.8, and this module has to import there or
the contract silently stops being enforced.
"""
import argparse
import csv
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone

STATE_BASENAME = "PIPELINE_STATE.json"
SCHEMA = 1

# Sentinel: "this key was never recorded", distinct from a recorded empty
# string. An empty --aruco-params IS a real setting (detector defaults), so ""
# must compare as a value, not as absence.
MISSING = object()

# Keys whose value changes the CONTENT of an output that keeps the same name.
# A mismatch is refused outright.
HARD_KEYS = ("chunk_sec", "chunk_ext", "aruco_dict", "aruco_params",
             "sleap_model_centroid", "sleap_model_instance")

# Keys that describe how the work was scheduled or executed rather than what it
# produced. Recorded for provenance; a change only warns. sleap_runtime belongs
# here deliberately -- pipeline.sh auto-falls back to the pytorch path for
# legacy model dirs, so refusing a runtime change would refuse a legitimate
# rerun that the pipeline itself chose.
SOFT_KEYS = ("sleap_module", "sleap_runtime", "saion_partition", "aruco_script")

# Which contract keys belong to which leg. An --only-aruco run must not be
# judged against sleap keys it never supplied, and vice versa.
ALWAYS_KEYS = ("chunk_sec", "chunk_ext")
LEG_KEYS = {
    "aruco": ("aruco_dict", "aruco_params", "aruco_script"),
    "sleap": ("sleap_model_centroid", "sleap_model_instance", "sleap_module",
              "sleap_runtime", "saion_partition"),
}

# Output basenames in data/. Mirrors catalog/const.py, kept local so this module
# imports with no dependencies beyond deigo's system python.
_OUT_RE = {
    "slp": re.compile(r"^(?P<v>.+)_(?P<i>\d{3})\.slp$"),
    "sdat": re.compile(r"^(?P<v>.+)_(?P<i>\d{3})_sleap_data\.h5$"),
    "det": re.compile(r"^(?P<v>.+)_(?P<i>\d{3})_aruco_detections\.h5$"),
    "trk": re.compile(r"^(?P<v>.+)_(?P<i>\d{3})_aruco_tracks\.h5$"),
}
STAGES = ("det", "trk", "slp", "sdat")
STAGE_LABEL = {"det": "aruco_detections", "trk": "aruco_tracks",
               "slp": "sleap_slp", "sdat": "sleap_data"}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def state_path(data_dir):
    return os.path.join(data_dir, STATE_BASENAME)


def parse_range(text):
    """'A-B' or 'A' -> (A, B) inclusive. '' / None -> None (whole block)."""
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None
    if "-" in text:
        a, _, b = text.partition("-")
    else:
        a = b = text
    try:
        lo, hi = int(a), int(b)
    except ValueError:
        raise ValueError("chunk range must be 'A-B' or 'A' (integers), got %r" % text)
    if lo < 0 or hi < lo:
        raise ValueError("chunk range must satisfy 0 <= A <= B, got %r" % text)
    return (lo, hi)


def fmt_ranges(idxs):
    """[0,1,2,4,7,8] -> '0-2,4,7-8'. Empty -> '-'."""
    idxs = sorted(set(int(i) for i in idxs))
    if not idxs:
        return "-"
    out = []
    start = prev = idxs[0]
    for i in idxs[1:]:
        if i == prev + 1:
            prev = i
            continue
        out.append((start, prev))
        start = prev = i
    out.append((start, prev))
    return ",".join(str(a) if a == b else "%d-%d" % (a, b) for a, b in out)


def _atomic_write(path, obj):
    """Write JSON via temp+rename: a crash must never leave a half-written contract."""
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".pipeline_state.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
            f.write("\n")
        # os.replace, not os.rename: rename refuses an existing destination on
        # Windows, so every contract update after the first would fail there.
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    # Best-effort group share; the pipeline runs under a shared unit group.
    try:
        os.chmod(path, 0o664)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# load / manifest
# ---------------------------------------------------------------------------
def load(data_dir):
    """Return the state dict, or None when the block has no contract yet.

    A corrupt file raises: it must never degrade to "no contract", which would
    silently re-open the block to a conflicting chunking.
    """
    p = state_path(data_dir)
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        try:
            state = json.load(f)
        except ValueError as e:
            raise ValueError(
                "%s is not valid JSON (%s). Refusing to proceed: fix it, or move it "
                "aside with --new-processing-run." % (p, e))
    if not isinstance(state, dict) or "chunking" not in state:
        raise ValueError("%s is not a pipeline state file (no 'chunking' key)." % p)
    got = state.get("schema")
    if got != SCHEMA:
        raise ValueError("%s has schema %r, this code speaks %d." % (p, got, SCHEMA))
    return state


def write(data_dir, state):
    """Persist a state dict for a block (atomically)."""
    _atomic_write(state_path(data_dir), state)


def read_manifest(manifest_path):
    """manifest.csv -> {vname: {n_chunks, fps, frame_count}}."""
    videos = {}
    with open(manifest_path) as f:
        for r in csv.DictReader(f):
            try:
                videos[r["vname"]] = {
                    "n_chunks": int(r["n_chunks"]),
                    "fps": float(r["fps"]),
                    "frame_count": int(r["frame_count"]),
                }
            except (KeyError, ValueError) as e:
                raise ValueError("bad manifest row %r (%s)" % (r, e))
    if not videos:
        raise ValueError("manifest %s has no rows" % manifest_path)
    return videos


def total_rows(videos):
    return sum(v["n_chunks"] for v in videos.values())


# ---------------------------------------------------------------------------
# contract
# ---------------------------------------------------------------------------
def relevant_keys(legs):
    keys = list(ALWAYS_KEYS)
    for leg in legs:
        keys.extend(LEG_KEYS.get(leg, ()))
    return tuple(keys)


def _norm(v):
    """Compare ints as ints and everything else as stripped text."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return v
    return str(v).strip()


def check_contract(state, proposed, legs):
    """Compare a proposed run against the recorded contract.

    Returns (violations, warnings, fills); each entry is (key, recorded, proposed).
    A key the caller did not supply is skipped -- absence means "this run does not
    exercise that setting", not "this run wants it cleared".
    """
    violations, warnings, fills = [], [], []
    recorded = {}
    recorded.update(state.get("chunking", {}))
    recorded.update(state.get("detection", {}))
    for key in relevant_keys(legs):
        if key not in proposed:
            continue
        have = recorded.get(key, MISSING)
        want = proposed[key]
        if have is MISSING:
            fills.append((key, None, want))
        elif _norm(have) != _norm(want):
            if key in HARD_KEYS:
                violations.append((key, have, want))
            else:
                warnings.append((key, have, want))
    return violations, warnings, fills


def check_videos(state, videos):
    """Source videos must not change shape under a live contract.

    A differing n_chunks/frame_count means the underlying recording was replaced
    or repaired, which invalidates every chunk index already on the bucket.
    """
    violations, added = [], []
    recorded = state.get("chunking", {}).get("videos", {})
    for vname in sorted(videos):
        have = recorded.get(vname)
        want = videos[vname]
        if have is None:
            added.append(vname)
            continue
        for field in ("n_chunks", "frame_count"):
            if int(have.get(field, -1)) != int(want[field]):
                violations.append(("videos.%s.%s" % (vname, field),
                                   have.get(field), want[field]))
    dropped = sorted(set(recorded) - set(videos))
    return violations, added, dropped


def new_state(block_dir, chunk_sec, chunk_ext, videos, detection):
    return {
        "schema": SCHEMA,
        "block": block_dir,
        "created": _utcnow(),
        "chunking": {
            "chunk_sec": int(chunk_sec),
            "chunk_ext": chunk_ext,
            "total_rows": total_rows(videos),
            "videos": videos,
        },
        "detection": dict(detection),
        "waves": [],
    }


def archive(data_dir):
    """Move an existing state aside; returns the archive path or None."""
    p = state_path(data_dir)
    if not os.path.isfile(p):
        return None
    dest = os.path.join(
        data_dir,
        "PIPELINE_STATE.%s.json" % datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    os.rename(p, dest)
    return dest


# ---------------------------------------------------------------------------
# wave ledger
# ---------------------------------------------------------------------------
def add_wave(state, chunk_range, rows, extra=None):
    wave = {
        "wave": len(state.get("waves", [])) + 1,
        "chunk_range": list(chunk_range) if chunk_range else None,
        "rows": int(rows),
        "submitted": _utcnow(),
        "completed": None,
    }
    if extra:
        wave.update(extra)
    state.setdefault("waves", []).append(wave)
    return wave


def wave_indices(state, wave):
    """The chunk indices a wave covers, per video, honouring each video's length.

    A range is clamped per video, so a short camera in a block of long ones
    contributes only the chunks it actually has instead of inflating the
    denominator with indices that can never exist.
    """
    rng = wave.get("chunk_range")
    videos = state.get("chunking", {}).get("videos", {})
    out = {}
    for vname, meta in videos.items():
        n = int(meta.get("n_chunks", 0))
        if not rng:
            lo, hi = 0, n - 1
        else:
            lo, hi = rng[0], min(rng[1], n - 1)
        out[vname] = set(range(lo, hi + 1)) if hi >= lo else set()
    return out


# ---------------------------------------------------------------------------
# coverage (observed, from data/)
# ---------------------------------------------------------------------------
def scan_outputs(data_dir):
    """{stage: {vname: set(chunk_idx)}} from data/ filenames only."""
    found = dict((s, {}) for s in STAGES)
    try:
        names = os.listdir(data_dir)
    except OSError:
        return found
    for nm in names:
        for stage in STAGES:
            m = _OUT_RE[stage].match(nm)
            if m:
                found[stage].setdefault(m.group("v"), set()).add(int(m.group("i")))
                break
    return found


def coverage(state, data_dir):
    """Declared vs observed, per stage. The catalog's honest denominator."""
    videos = state.get("chunking", {}).get("videos", {})
    declared = dict((v, set(range(int(m.get("n_chunks", 0)))))
                    for v, m in videos.items())
    declared_total = sum(len(s) for s in declared.values())
    found = scan_outputs(data_dir)

    stages = {}
    for stage in STAGES:
        obs = found.get(stage, {})
        present = 0
        missing = {}
        for vname, want in declared.items():
            have = obs.get(vname, set()) & want
            present += len(have)
            gap = want - have
            if gap:
                missing[vname] = sorted(gap)
        stages[stage] = {
            "present": present,
            "declared": declared_total,
            "pct": (round(present / float(declared_total), 4)
                    if declared_total else None),
            "missing": missing,
            "extra": sorted(set(obs) - set(declared)),
        }

    waves = []
    for w in state.get("waves", []):
        want_map = wave_indices(state, w)
        want_total = sum(len(s) for s in want_map.values())
        per_stage = {}
        for stage in STAGES:
            obs = found.get(stage, {})
            per_stage[stage] = sum(len(obs.get(v, set()) & s)
                                   for v, s in want_map.items())
        waves.append({
            "wave": w.get("wave"), "chunk_range": w.get("chunk_range"),
            "declared": want_total, "stages": per_stage,
            "submitted": w.get("submitted"), "completed": w.get("completed"),
        })

    return {"declared_total": declared_total, "stages": stages, "waves": waves,
            "unclaimed": unclaimed_ranges(state)}


def interior_gaps(state):
    """Unclaimed chunk indices that are NOT a trailing tail.

    The distinction is the whole point of the ledger. A tail simply means the
    next wave has not been submitted yet -- expected, and not a problem. A hole
    *between* two claimed waves means a window was skipped, and nothing else in
    the system will ever notice: the outputs around it are all present and valid,
    so every count and every spot check looks healthy.
    """
    videos = state.get("chunking", {}).get("videos", {})
    claimed = {}
    for w in state.get("waves", []):
        for vname, idxs in wave_indices(state, w).items():
            claimed.setdefault(vname, set()).update(idxs)
    out = {}
    for vname in videos:
        got = claimed.get(vname, set())
        if not got:
            continue  # never started; that is a tail, not a hole
        gap = set(range(max(got))) - got
        if gap:
            out[vname] = fmt_ranges(gap)
    return out


def unclaimed_ranges(state):
    """Chunk indices no wave has ever claimed -- the gap a wave workflow can leave.

    Reported per video so a 198-chunk and a 40-chunk camera in one block do not
    smear into a single misleading range.
    """
    videos = state.get("chunking", {}).get("videos", {})
    waves = state.get("waves", [])
    if not videos:
        return {}
    if not waves:
        return dict((v, fmt_ranges(range(int(m.get("n_chunks", 0)))))
                    for v, m in videos.items())
    claimed = {}
    for w in waves:
        for vname, idxs in wave_indices(state, w).items():
            claimed.setdefault(vname, set()).update(idxs)
    out = {}
    for vname, meta in videos.items():
        gap = set(range(int(meta.get("n_chunks", 0)))) - claimed.get(vname, set())
        if gap:
            out[vname] = fmt_ranges(gap)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _kv(pairs):
    out = {}
    for item in pairs or ():
        if "=" not in item:
            raise ValueError("--set expects key=value, got %r" % item)
        k, _, v = item.partition("=")
        k = k.strip()
        if k not in HARD_KEYS and k not in SOFT_KEYS:
            raise ValueError("unknown contract key %r" % k)
        out[k] = int(v) if k == "chunk_sec" else v
    return out


def _print_diff(title, entries, stream=sys.stderr):
    stream.write("%s\n" % title)
    for key, have, want in entries:
        stream.write("    %-30s recorded=%-30r requested=%r\n" % (key, have, want))


def cmd_sync(args):
    proposed = _kv(args.set)
    legs = [s for s in (args.legs or "aruco,sleap").split(",") if s]
    videos = read_manifest(args.manifest)

    try:
        state = load(args.data_dir)
    except ValueError as e:
        if not args.new_run:
            sys.stderr.write("[ERR] %s\n" % e)
            return 3
        state = None

    if state is not None and args.new_run:
        dest = archive(args.data_dir)
        sys.stderr.write("[INFO] archived previous contract -> %s\n" % dest)
        state = None

    if state is None:
        for key in ALWAYS_KEYS:
            if key not in proposed:
                sys.stderr.write("[ERR] cannot create a contract without %s\n" % key)
                return 2
        detection = dict((k, v) for k, v in proposed.items() if k not in ALWAYS_KEYS)
        state = new_state(args.block_dir or "", proposed["chunk_sec"],
                          proposed["chunk_ext"], videos, detection)
        if not args.dry_run:
            if not os.path.isdir(args.data_dir):
                os.makedirs(args.data_dir)
            _atomic_write(state_path(args.data_dir), state)
        sys.stderr.write(
            "[INFO] processing contract created: %s (chunk_sec=%s, %d videos, "
            "%d chunks declared)\n"
            % (state_path(args.data_dir), proposed["chunk_sec"], len(videos),
               total_rows(videos)))
        return 0

    violations, warnings, fills = check_contract(state, proposed, legs)
    vio_v, added, dropped = check_videos(state, videos)
    violations.extend(vio_v)

    if violations:
        sys.stderr.write(
            "[ERR] this run disagrees with the processing contract already recorded\n"
            "      for this block. Re-processing under different settings would\n"
            "      overwrite existing outputs with content that no longer lines up\n"
            "      with the outputs it does not overwrite.\n\n")
        _print_diff("      conflicting settings:", violations)
        sys.stderr.write(
            "\n      contract: %s\n"
            "      Either match the recorded settings, or start a separate\n"
            "      processing run with --new-processing-run (archives the contract;\n"
            "      existing outputs in data/ are NOT deleted -- move them yourself).\n"
            % state_path(args.data_dir))
        return 3

    for key, have, want in warnings:
        sys.stderr.write("[WARN] %s changed since the first run: %r -> %r "
                         "(execution setting, not output content; allowed)\n"
                         % (key, have, want))
    for vname in dropped:
        sys.stderr.write("[WARN] %s is in the contract but not in this manifest\n"
                         % vname)

    if fills or added:
        for key, _, want in fills:
            state.setdefault("detection", {})[key] = want
            sys.stderr.write("[INFO] contract gained %s=%r (not recorded before)\n"
                             % (key, want))
        for vname in added:
            state["chunking"]["videos"][vname] = videos[vname]
            sys.stderr.write("[INFO] contract gained video %s\n" % vname)
        state["chunking"]["total_rows"] = total_rows(state["chunking"]["videos"])
        if not args.dry_run:
            _atomic_write(state_path(args.data_dir), state)

    sys.stderr.write("[OK] run agrees with the processing contract "
                     "(chunk_sec=%s, %d chunks declared)\n"
                     % (state["chunking"]["chunk_sec"],
                        state["chunking"]["total_rows"]))
    return 0


def cmd_add_wave(args):
    state = load(args.data_dir)
    if state is None:
        sys.stderr.write("[ERR] no contract at %s; run sync first\n"
                         % state_path(args.data_dir))
        return 2
    rng = parse_range(args.range)
    extra = {}
    if args.jids:
        extra["jids"] = args.jids
    if args.batch_size:
        extra["batch_size"] = int(args.batch_size)
    if args.version:
        extra["pipeline_version"] = args.version

    want = list(rng) if rng else None
    prev = [w for w in state.get("waves", []) if w.get("chunk_range") == want]
    if prev:
        sys.stderr.write("[INFO] wave %s already covers %s; recording a re-run\n"
                         % (prev[-1].get("wave"), args.range or "the whole block"))
    wave = add_wave(state, rng, args.rows, extra)
    _atomic_write(state_path(args.data_dir), state)
    sys.stderr.write("[INFO] wave %d recorded: chunks %s, %d rows\n"
                     % (wave["wave"], args.range or "all", int(args.rows)))
    print(wave["wave"])
    return 0


def cmd_complete_wave(args):
    state = load(args.data_dir)
    if state is None:
        return 2
    for w in state.get("waves", []):
        if int(w.get("wave", 0)) == int(args.wave):
            w["completed"] = _utcnow()
            _atomic_write(state_path(args.data_dir), state)
            sys.stderr.write("[INFO] wave %s marked complete\n" % args.wave)
            return 0
    sys.stderr.write("[ERR] no wave %s\n" % args.wave)
    return 2


def cmd_show(args):
    state = load(args.data_dir)
    if state is None:
        sys.stderr.write("[INFO] no contract at %s\n" % state_path(args.data_dir))
        return 1
    cov = coverage(state, args.data_dir)
    if args.json:
        json.dump({"state": state, "coverage": cov}, sys.stdout,
                  indent=2, sort_keys=True, default=sorted)
        sys.stdout.write("\n")
        return 0

    ch = state["chunking"]
    print("block          %s" % state.get("block", ""))
    print("created        %s" % state.get("created", ""))
    print("chunk_sec      %s   chunk_ext %s" % (ch["chunk_sec"], ch["chunk_ext"]))
    print("declared       %d chunks over %d videos"
          % (ch["total_rows"], len(ch["videos"])))
    print("")
    for key in sorted(state.get("detection", {})):
        print("  %-24s %s" % (key, state["detection"][key]))
    print("")
    print("stage coverage (observed in data/):")
    for stage in STAGES:
        s = cov["stages"][stage]
        pct = "%5.1f%%" % (100 * s["pct"]) if s["pct"] is not None else "    -"
        print("  %-20s %6d / %-6d %s"
              % (STAGE_LABEL[stage], s["present"], s["declared"], pct))
    print("")
    print("waves:")
    if not cov["waves"]:
        print("  (none recorded)")
    for w in cov["waves"]:
        rng = "%d-%d" % tuple(w["chunk_range"]) if w["chunk_range"] else "all"
        done = w["completed"] or "-"
        print("  #%-3s chunks %-11s declared %-6d det/trk/slp/sdat %d/%d/%d/%d  "
              "submitted %s  completed %s"
              % (w["wave"], rng, w["declared"], w["stages"]["det"],
                 w["stages"]["trk"], w["stages"]["slp"], w["stages"]["sdat"],
                 w["submitted"], done))
    if cov["unclaimed"]:
        print("")
        print("NOT YET CLAIMED BY ANY WAVE:")
        for vname in sorted(cov["unclaimed"]):
            print("  %-46s %s" % (vname, cov["unclaimed"][vname]))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("sync", help="create or validate the contract for a run")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--block-dir", default="")
    p.add_argument("--set", action="append", metavar="KEY=VALUE")
    p.add_argument("--legs", default="aruco,sleap")
    p.add_argument("--new-run", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_sync)

    p = sub.add_parser("add-wave", help="record a submitted chunk range")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--range", default="")
    p.add_argument("--rows", required=True, type=int)
    p.add_argument("--jids", default="")
    p.add_argument("--batch-size", default="")
    p.add_argument("--version", default="")
    p.set_defaults(func=cmd_add_wave)

    p = sub.add_parser("complete-wave", help="stamp a wave as finished")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--wave", required=True)
    p.set_defaults(func=cmd_complete_wave)

    p = sub.add_parser("show", help="print the contract and observed coverage")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_show)

    args = ap.parse_args(argv)
    if not getattr(args, "func", None):
        ap.print_help()
        return 2
    try:
        return args.func(args)
    except (ValueError, OSError) as e:
        sys.stderr.write("[ERR] %s\n" % e)
        return 2


if __name__ == "__main__":
    sys.exit(main())
