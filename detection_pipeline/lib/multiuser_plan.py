#!/usr/bin/env python3
"""Multi-user wave plan: ``data/MULTIUSER_PLAN.json``.

One block can be processed by 2-3 unit members in parallel because every Slurm
limit that matters (GrpSubmit, cpu, GPU) is per user and every mutable control
path is namespaced by ``$USER``. What the users must share is *policy*: the
same detection settings (or the processing contract refuses the run), a
non-overlapping wave split, exactly one backup writer and exactly one tracking
poller. This module is where that policy lives, so it cannot drift between
people's shell histories.

The plan file sits next to PIPELINE_STATE.json in ``<exp>/data/`` -- the block
it governs is derived from its location, never written inside it::

    {
      "settings": { "chunk_sec": 1800, "aruco_dict": "A",
                    "sleap_model_centroid": "/bucket/.../x.centroid",
                    "sleap_model_instance": "/bucket/.../x.centered_instance",
                    "saion_partition": "largegpu" },
      "slots": {
        "makoto-hiroi": { "waves": ["0-499", "500-999"],
                          "backup": true, "tracking": true },
        "user2":        { "waves": ["1000-1499", "1500-1999"] }
      }
    }

``settings`` keys are pipeline.sh long options with underscores
(``chunk_sec`` -> ``--chunk-sec``); ``true`` emits a bare flag. Keys the plan
must own itself (``chunk_range``, ``no_backup``, ``run_tracking``, ...) are
refused in ``settings`` -- they are exactly the per-slot policy this file
exists to centralize.

Used by pipeline_multi.sh; the CLI prints machine-readable output (one token
or one TSV row per line) so the shell side never parses JSON.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_state as ps  # noqa: E402

PLAN_BASENAME = "MULTIUSER_PLAN.json"

# Keys pipeline_multi.sh derives or per-slot policy owns. Accepting them in
# settings would let a plan silently double-submit backup or tracking -- the
# two things a multi-user run must keep unique.
FORBIDDEN_SETTINGS = (
    "dir", "chunk_range", "backup", "no_backup", "only_backup",
    "run_tracking", "no_run_tracking", "force_submit",
    "jobs_root", "flash_root", "new_processing_run",
)

# Slot keys become `squeue -u`/ssh arguments in pipeline_multi.sh, and the
# plan file lives in a group-writable directory — so a slot name is untrusted
# input to a remote shell unless it is shaped like a username. Refuse anything
# else outright.
_USERNAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

# Widest wave one plan entry may claim. Real blocks are a few thousand chunks;
# the overlap check materializes every claimed index, so an unbounded range in
# a shared, group-writable file would be an accidental-DoS on every caller.
MAX_WAVE_WIDTH = 100_000


# ---------------------------------------------------------------------------
# load / validate
# ---------------------------------------------------------------------------
def load_plan(path):
    """Parse + structurally validate a plan file. Raises ValueError on any
    problem: a half-valid plan must never submit anything."""
    with open(path) as f:
        try:
            plan = json.load(f)
        except ValueError as e:
            raise ValueError("%s is not valid JSON (%s)" % (path, e))
    if not isinstance(plan, dict):
        raise ValueError("plan root must be an object")

    settings = plan.get("settings")
    if not isinstance(settings, dict) or not settings:
        raise ValueError("plan needs a non-empty 'settings' object")
    for key, val in settings.items():
        if key in FORBIDDEN_SETTINGS:
            raise ValueError(
                "settings key %r is owned by per-slot policy / derivation; "
                "remove it (backup/tracking go on the slot)" % key)
        if not isinstance(key, str) or not key.replace("_", "").isalnum():
            raise ValueError("settings key %r is not a flag-shaped name" % key)
        if isinstance(val, str) and "\n" in val:
            raise ValueError("settings value for %r contains a newline" % key)

    slots = plan.get("slots")
    if not isinstance(slots, dict) or not slots:
        raise ValueError("plan needs a non-empty 'slots' object")

    n_backup = n_tracking = 0
    for user, slot in slots.items():
        if not isinstance(user, str) or not _USERNAME_RE.match(user):
            raise ValueError("slot name %r is not a valid username" % user)
        if not isinstance(slot, dict):
            raise ValueError("slot %r must be an object" % user)
        waves = slot.get("waves")
        if not isinstance(waves, list) or not waves:
            raise ValueError("slot %r needs a non-empty 'waves' list" % user)
        for w in waves:
            lo, hi = ps.parse_range(str(w))  # raises on malformed
            if hi - lo + 1 > MAX_WAVE_WIDTH:
                raise ValueError("slot %r wave %s is implausibly wide (> %d chunks)"
                                 % (user, w, MAX_WAVE_WIDTH))
        n_backup += 1 if slot.get("backup") else 0
        n_tracking += 1 if slot.get("tracking") else 0

    if n_backup > 1:
        raise ValueError(
            "more than one slot has backup=true; concurrent zip -FS updates "
            "of the same archive can corrupt it -- pick one backup writer")
    if n_tracking > 1:
        raise ValueError(
            "more than one slot has tracking=true; two pollers would submit "
            "colony tracking twice -- pick one")

    _check_wave_overlap(slots)
    return plan


def _check_wave_overlap(slots):
    """Same-index chunks from two users produce identical outputs, but both
    would spend GPU time and could interleave partial writes -- refuse."""
    claimed = {}  # chunk_idx -> "user wave"
    for user in sorted(slots):
        for w in slots[user]["waves"]:
            lo, hi = ps.parse_range(str(w))
            for i in range(lo, hi + 1):
                owner = claimed.get(i)
                if owner:
                    raise ValueError(
                        "wave overlap: chunk %d claimed by both %s and %s %s"
                        % (i, owner, user, w))
            for i in range(lo, hi + 1):
                claimed[i] = "%s %s" % (user, w)


def slot_for(plan, user):
    slot = plan["slots"].get(user)
    if slot is None:
        raise ValueError(
            "user %r has no slot in this plan (slots: %s)"
            % (user, ", ".join(sorted(plan["slots"]))))
    return slot


# ---------------------------------------------------------------------------
# flag emission
# ---------------------------------------------------------------------------
def settings_flags(settings):
    """settings dict -> flat list of pipeline.sh argv tokens (sorted by key,
    so the emitted command is stable and diffable across runs)."""
    out = []
    for key in sorted(settings):
        val = settings[key]
        flag = "--" + key.replace("_", "-")
        if isinstance(val, bool):
            if val:
                out.append(flag)
        else:
            out.extend([flag, str(val)])
    return out


def wave_flags(plan, user, wave, exp_dir):
    """The full pipeline.sh argv for one user's one wave, policy applied."""
    slot = slot_for(plan, user)
    waves = [str(w) for w in slot["waves"]]
    if wave not in waves:
        raise ValueError("wave %r is not in %s's slot (%s)"
                         % (wave, user, ", ".join(waves)))

    args = ["--dir", exp_dir, "--chunk-range", wave]
    args.extend(settings_flags(plan["settings"]))

    # Backup: pipeline.sh submits one backup job per run, all updating the
    # same zip via -FS. One writer, once -- the slot's FIRST wave -- and
    # --no-backup everywhere else, so overlapping waves cannot race the archive.
    if not (slot.get("backup") and wave == waves[0]):
        args.append("--no-backup")

    # Tracking: one poller for the whole block, launched from the tracking
    # slot's LAST wave (the poller gates on the block's declared total, so a
    # late launch minimizes the window its --tracking-timeout has to cover).
    if slot.get("tracking") and wave == waves[-1]:
        args.append("--run-tracking")
    return args


# ---------------------------------------------------------------------------
# wave completion (observed from data/, mirroring track_trigger's gate)
# ---------------------------------------------------------------------------
def _gate_stages(settings):
    """Which output stages prove a wave done. Mirrors track_trigger.sh: aruco
    gates on _aruco_tracks.h5, sleap on _sleap_data.h5 (never .slp -- the
    slp->h5 conversion is best-effort and tracking reads only the h5)."""
    if settings.get("only_aruco"):
        return ("trk",)
    if settings.get("only_sleap"):
        return ("sdat",)
    return ("trk", "sdat")


def wave_status(plan, user, data_dir):
    """Per-wave completion for one slot.

    Returns [(range_str, expected, {stage: n_done}, state)] with state one of
    'done' | 'pending' | 'unknown'. Expected counts come from the processing
    contract (range clamped per video, like wave_indices); before the first
    run declares a contract there is no denominator, so state is 'unknown' --
    which callers must treat as pending, never as done.
    """
    slot = slot_for(plan, user)
    stages = _gate_stages(plan["settings"])
    state = None
    try:
        state = ps.load(data_dir)
    except ValueError:
        pass  # corrupt contract: report unknown; pipeline.sh will refuse anyway
    found = ps.scan_outputs(data_dir)

    rows = []
    for w in (str(x) for x in slot["waves"]):
        lo, hi = ps.parse_range(w)
        if state is None:
            rows.append((w, None, dict((s, 0) for s in stages), "unknown"))
            continue
        # Per-video clamping is pipeline_state's logic; reuse it via a
        # synthetic wave entry rather than re-deriving the formula here.
        want = ps.wave_indices(state, {"chunk_range": [lo, hi]})
        expected = sum(len(s) for s in want.values())
        done = {}
        for stage in stages:
            obs = found.get(stage, {})
            done[stage] = sum(len(obs.get(v, set()) & idx)
                              for v, idx in want.items())
        complete = expected > 0 and all(done[s] >= expected for s in stages)
        rows.append((w, expected, done, "done" if complete else "pending"))
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _exp_dir_of(plan_path):
    """<exp>/data/MULTIUSER_PLAN.json -> <exp>. Location IS the binding to a
    block; storing the path inside the plan too would let the two disagree."""
    data_dir = os.path.dirname(os.path.abspath(plan_path))
    if os.path.basename(data_dir) != "data":
        raise ValueError(
            "plan must live in a block's data/ directory, got %s" % data_dir)
    return os.path.dirname(data_dir)


def cmd_validate(args):
    plan = load_plan(args.plan)
    exp_dir = _exp_dir_of(args.plan)
    total = sum(
        ps.parse_range(str(w))[1] - ps.parse_range(str(w))[0] + 1
        for s in plan["slots"].values() for w in s["waves"])
    sys.stderr.write("[OK] %s: %d slots, %d chunk indices claimed, block %s\n"
                     % (args.plan, len(plan["slots"]), total, exp_dir))
    for user in sorted(plan["slots"]):
        slot = plan["slots"][user]
        marks = [k for k in ("backup", "tracking") if slot.get(k)]
        sys.stderr.write("     %-16s waves=%s%s\n" % (
            user, ",".join(str(w) for w in slot["waves"]),
            (" [" + ",".join(marks) + "]") if marks else ""))
    return 0


def cmd_waves(args):
    for w in slot_for(load_plan(args.plan), args.user)["waves"]:
        print(w)
    return 0


def cmd_flags(args):
    plan = load_plan(args.plan)
    for tok in wave_flags(plan, args.user, args.wave, _exp_dir_of(args.plan)):
        print(tok)
    return 0


def cmd_slot_status(args):
    plan = load_plan(args.plan)
    data_dir = os.path.join(_exp_dir_of(args.plan), "data")
    for w, expected, done, state in wave_status(plan, args.user, data_dir):
        done_txt = ",".join("%s=%d" % (s, done[s]) for s in sorted(done))
        print("%s\t%s\t%s\t%s"
              % (w, "?" if expected is None else expected, done_txt, state))
    return 0


def cmd_status(args):
    plan = load_plan(args.plan)
    data_dir = os.path.join(_exp_dir_of(args.plan), "data")
    for user in sorted(plan["slots"]):
        rows = wave_status(plan, user, data_dir)
        n_done = sum(1 for r in rows if r[3] == "done")
        slot = plan["slots"][user]
        marks = [k for k in ("backup", "tracking") if slot.get(k)]
        print("slot %-16s %d/%d waves done%s"
              % (user, n_done, len(rows),
                 (" [" + ",".join(marks) + "]") if marks else ""))
        for w, expected, done, state in rows:
            done_txt = " ".join("%s %d/%s" % (s, done[s],
                                              "?" if expected is None else expected)
                                for s in sorted(done))
            print("  wave %-12s %-8s %s" % (w, state, done_txt))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, **extra_args):
        p = sub.add_parser(name)
        p.add_argument("--plan", required=True)
        for arg, kw in extra_args.items():
            p.add_argument("--" + arg, **kw)
        p.set_defaults(fn=fn)
        return p

    add("validate", cmd_validate)
    add("status", cmd_status)
    add("waves", cmd_waves, user={"required": True})
    add("slot-status", cmd_slot_status, user={"required": True})
    add("flags", cmd_flags, user={"required": True}, wave={"required": True})

    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except (ValueError, OSError) as e:
        sys.stderr.write("[ERR] %s\n" % e)
        return 2


if __name__ == "__main__":
    sys.exit(main())
