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
import datetime
import json
import os
import re
import subprocess
import sys
import tempfile

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
    # A plan is group-writable data, and these two name a script and an
    # interpreter that track_trigger.sh EXECUTES as the submitting user. Left
    # settable, they would turn the forced-command SSH key (which can only run
    # this wrapper) into an arbitrary-execution primitive for anyone able to
    # write a plan file. Override them on the pipeline.sh command line instead.
    "tracking_submit", "tracking_python_bin",
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
# plan generation (`pipeline_multi.sh plan`)
# ---------------------------------------------------------------------------
# What people get wrong in a hand-written plan is never the syntax, it is the
# values: settings that disagree with the block's recorded contract (refused at
# submit), a total chunk count nobody looked up, and wave widths that overrun
# the per-user submit cap. So the generator takes settings FROM the contract
# when one exists, counts chunks itself, and sizes waves from the cap.
DEFAULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "templates", "multiuser_defaults.json")

# deigo compute GrpSubmit is 2016 jobs per user. A wave costs about
# rows/batch_size aruco tasks plus ~30 fixed jobs; the same user's OTHER work
# (tracking DAGs, sleep prediction) shares the cap, so leave real headroom.
DEFAULT_WAVE_ROWS = 1500

# Contract keys that are not pipeline.sh flags.
_CONTRACT_ONLY = ("aruco_script",)


def load_defaults(path):
    with open(path) as f:
        d = json.load(f)
    if not isinstance(d, dict):
        raise ValueError("%s: defaults must be an object" % path)
    return dict((k, v) for k, v in d.items() if not k.startswith("_"))


def parse_override(item):
    """--set key=value -> (key, typed value). true/false -> bool, digits -> int."""
    if "=" not in item:
        raise ValueError("--set expects key=value, got %r" % item)
    k, _, v = item.partition("=")
    k = k.strip()
    if v in ("true", "false"):
        return k, v == "true"
    if v.isdigit():
        return k, int(v)
    return k, v


def resolve_settings(defaults, state, overrides):
    """defaults < recorded contract < --set overrides; empty strings dropped.

    The contract wins over defaults because pipeline.sh will refuse anything
    else; an override that contradicts a HARD contract key is reported so the
    author learns it before the submit does.
    """
    settings = dict(defaults)
    if state is not None:
        ch = state.get("chunking", {})
        settings["chunk_sec"] = int(ch.get("chunk_sec"))
        settings["chunk_ext"] = ch.get("chunk_ext")
        for k, v in state.get("detection", {}).items():
            if k in _CONTRACT_ONLY:
                continue
            settings[k] = v
    conflicts = []
    for k, v in overrides.items():
        if state is not None and k in ps.HARD_KEYS and k in settings \
                and str(settings[k]) != str(v):
            conflicts.append((k, settings[k], v))
        settings[k] = v
    settings = dict((k, v) for k, v in settings.items() if v not in ("", None))
    return settings, conflicts


def split_waves(total, users, n_videos, wave_rows=DEFAULT_WAVE_ROWS, max_live=1,
                first=0):
    """Contiguous, near-equal split of chunk indices first..total-1 across users,
    each share cut into equal-width waves no wider than the submit cap allows.

    Width is derived from ROWS (chunk x camera), the unit the cap actually
    counts, so a 19-camera and a 25-camera block size themselves correctly.
    Returns ({user: [range_str, ...]}, width).
    """
    if total <= 0:
        raise ValueError("block declares no chunks")
    if n_videos <= 0:
        raise ValueError("block declares no videos")
    count = total - first
    if count <= 0:
        raise ValueError("chunk range starts at %d but the block declares only %d"
                         % (first, total))
    if len(users) > count:
        raise ValueError("%d users but only %d chunk indices" % (len(users), count))
    width = max(1, wave_rows // (n_videos * max(1, max_live)))
    base, rem = divmod(count, len(users))
    slots, lo = {}, first
    for i, user in enumerate(users):
        share = base + (1 if i < rem else 0)
        n_waves = -(-share // width)
        wb, wr = divmod(share, n_waves)
        waves, cur = [], lo
        for j in range(n_waves):
            w = wb + (1 if j < wr else 0)
            waves.append("%d-%d" % (cur, cur + w - 1))
            cur += w
        slots[user] = waves
        lo += share
    return slots, width


def _videos_via_manifest(exp_dir, chunk_sec):
    """No contract yet: probe the block the way pipeline.sh does."""
    manifest_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifest.py")
    fd, tmp = tempfile.mkstemp(prefix="muplan_manifest_", suffix=".csv")
    os.close(fd)
    try:
        subprocess.check_call([sys.executable, manifest_py, "--dir", exp_dir,
                               "--out", tmp, "--chunk-sec", str(chunk_sec)])
        return ps.read_manifest(tmp)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def build_plan(exp_dir, users, defaults, overrides, wave_rows, max_live,
               backup_user=None, tracking_user=None, videos=None, chunks=None,
               want_backup=True, want_tracking=True):
    data_dir = os.path.join(exp_dir, "data")
    state = ps.load(data_dir) if os.path.isfile(ps.state_path(data_dir)) else None
    settings, conflicts = resolve_settings(defaults, state, overrides)

    if not settings.get("only_aruco"):
        for k in ("sleap_model_centroid", "sleap_model_instance"):
            if k not in settings:
                raise ValueError(
                    "settings lack %s: pass --set %s=<dir> (or run after a first "
                    "contract exists, or --set only_aruco=true)" % (k, k))

    if videos is None:
        videos = (state["chunking"]["videos"] if state is not None
                  else _videos_via_manifest(exp_dir, settings["chunk_sec"]))
    total = max(int(v.get("n_chunks", 0)) for v in videos.values())
    # A plan may cover a SLICE of the block (a short trial run, or finishing a
    # block someone started by hand). Indices keep their absolute value, so a
    # later plan for the rest of the block lines up with the same outputs.
    first, last = 0, total - 1
    if chunks:
        first, last = ps.parse_range(str(chunks))
        if last >= total:
            raise ValueError("--chunks %s runs past the block's last index %d"
                             % (chunks, total - 1))
    slots, width = split_waves(last + 1, users, len(videos), wave_rows, max_live,
                               first=first)

    # None = nobody does it. Both are per-block singletons, so a plan covering
    # only part of a block usually wants neither: the backup archives raw
    # videos (already done for a reprocessed block) and the tracking poller
    # gates on the block total, which a slice can never reach.
    backup_user = (backup_user or users[0]) if want_backup else None
    tracking_user = (tracking_user or users[0]) if want_tracking else None
    for role, u in (("backup", backup_user), ("tracking", tracking_user)):
        if u is not None and u not in slots:
            raise ValueError("--%s-user %r is not one of the users" % (role, u))
    full_slots = {}
    for u, w in slots.items():
        slot = {"waves": w}
        if u == backup_user:
            slot["backup"] = True
        if u == tracking_user:
            slot["tracking"] = True
        full_slots[u] = slot

    plan = {
        "meta": {
            "generated": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "settings_source": "contract" if state is not None else "defaults",
            "total_chunk_indices": total,
            "planned_indices": "%d-%d" % (first, last),
            "n_videos": len(videos),
            "wave_rows": wave_rows,
            "max_live": max_live,
            "wave_width": width,
        },
        "settings": settings,
        "slots": full_slots,
    }
    return plan, conflicts


def cmd_init(args):
    exp_dir = os.path.abspath(args.dir)
    if not os.path.isdir(exp_dir):
        raise ValueError("--dir not found: %s" % exp_dir)
    users = [u for u in args.users.split(",") if u]
    if not users:
        raise ValueError("--users needs at least one username")
    out = args.out or os.path.join(exp_dir, "data", PLAN_BASENAME)
    if os.path.exists(out) and not args.force:
        raise ValueError("%s exists; pass --force to overwrite" % out)

    overrides = dict(parse_override(x) for x in (args.set or ()))
    plan, conflicts = build_plan(
        exp_dir, users, load_defaults(args.defaults or DEFAULTS_PATH), overrides,
        args.wave_rows, args.max_live, args.backup_user, args.tracking_user,
        chunks=args.chunks, want_backup=not args.no_backup,
        want_tracking=not args.no_tracking)
    # pipeline.sh refuses --run-tracking without --tracking-hmats, and that
    # flag rides the tracking slot's LAST wave -- so without this warning the
    # plan looks fine and then fails days later, on the final submission.
    if not args.no_tracking and "tracking_hmats" not in plan["settings"]:
        sys.stderr.write(
            "[WARN] a slot has tracking=true but settings lack tracking_hmats; "
            "pipeline.sh will refuse that slot's LAST wave. Add "
            "--set tracking_hmats=<npz>, or --no-tracking.\n")
    for k, have, want in conflicts:
        sys.stderr.write("[WARN] --set %s=%r contradicts the recorded contract (%r); "
                         "pipeline.sh will refuse this plan\n" % (k, want, have))

    if not os.path.isdir(os.path.dirname(out)):
        os.makedirs(os.path.dirname(out))
    with open(out, "w") as f:
        json.dump(plan, f, indent=2)
        f.write("\n")
    try:
        os.chmod(out, 0o664)  # group-editable: the plan is the team's, not the author's
    except OSError:
        pass
    load_plan(out)  # the generator must never emit a plan it would refuse
    m = plan["meta"]
    sys.stderr.write("[OK] wrote %s (settings from %s; indices %s of %d x %d "
                     "videos; wave width %d = %d rows / %d videos / max_live %d)\n"
                     % (out, m["settings_source"], m["planned_indices"],
                        m["total_chunk_indices"], m["n_videos"], m["wave_width"],
                        m["wave_rows"], m["n_videos"], m["max_live"]))
    args.plan = out
    return cmd_validate(args)


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
    # No required=True: that kwarg is 3.7+, and deigo/saion logins run the
    # system python 3.6. Enforced manually after parsing instead.
    sub = ap.add_subparsers(dest="cmd")

    def add(name, fn, **extra_args):
        p = sub.add_parser(name)
        p.add_argument("--plan", required=True)
        for arg, kw in extra_args.items():
            p.add_argument("--" + arg, **kw)
        p.set_defaults(fn=fn)
        return p

    add("validate", cmd_validate)
    add("status", cmd_status)
    p_init = sub.add_parser("init")
    p_init.add_argument("--dir", required=True)
    p_init.add_argument("--users", required=True, help="comma-separated usernames")
    p_init.add_argument("--wave-rows", type=int, default=DEFAULT_WAVE_ROWS)
    p_init.add_argument("--max-live", type=int, default=1)
    p_init.add_argument("--chunks", help="only this A-B slice of the block")
    p_init.add_argument("--no-backup", action="store_true")
    p_init.add_argument("--no-tracking", action="store_true")
    p_init.add_argument("--backup-user")
    p_init.add_argument("--tracking-user")
    p_init.add_argument("--defaults")
    p_init.add_argument("--set", action="append")
    p_init.add_argument("--out")
    p_init.add_argument("--force", action="store_true")
    p_init.set_defaults(fn=cmd_init)
    add("waves", cmd_waves, user={"required": True})
    add("slot-status", cmd_slot_status, user={"required": True})
    add("flags", cmd_flags, user={"required": True}, wave={"required": True})

    args = ap.parse_args(argv)
    if not getattr(args, "fn", None):
        ap.error("a subcommand is required (init|validate|status|waves|slot-status|flags)")
    try:
        return args.fn(args)
    except (ValueError, OSError) as e:
        sys.stderr.write("[ERR] %s\n" % e)
        return 2


if __name__ == "__main__":
    sys.exit(main())
