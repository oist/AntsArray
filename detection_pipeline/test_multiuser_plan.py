#!/usr/bin/env python3
"""Regression tests for the multi-user wave plan (lib/multiuser_plan.py).

Run with pytest, or directly (deigo's system python has no pytest)::

    python3 detection_pipeline/test_multiuser_plan.py

The tests that matter most pin the policy invariants a multi-user run depends
on: exactly one backup writer emitting backup exactly once, exactly one
tracking poller launched from the LAST wave of its slot, no chunk index owned
by two slots, and "no contract yet" reported as unknown — never as done.
"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
import multiuser_plan as mp  # noqa: E402
import pipeline_state as ps  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------
VID_A = "cam01_cam0_2099-01-01-00-00-00"
VID_B = "cam02_cam1_2099-01-01-00-00-00"

SETTINGS = {
    "chunk_sec": 1800,
    "aruco_dict": "A",
    "sleap_model_centroid": "/models/x.centroid",
    "sleap_model_instance": "/models/x.centered_instance",
    "saion_partition": "largegpu",
}


def make_plan(slots, settings=None):
    return {"settings": dict(settings or SETTINGS), "slots": slots}


class Block:
    """A temp <exp>/ with data/ holding a plan and (optionally) a contract."""

    def __init__(self, plan, videos=None):
        self.root = tempfile.mkdtemp(prefix="muptest_")
        self.data_dir = os.path.join(self.root, "data")
        os.makedirs(self.data_dir)
        self.plan_path = os.path.join(self.data_dir, mp.PLAN_BASENAME)
        with open(self.plan_path, "w") as f:
            json.dump(plan, f)
        if videos is not None:
            state = ps.new_state(self.root, 1800, "mkv", videos, {})
            ps.write(self.data_dir, state)

    def touch_outputs(self, vname, indices, suffixes):
        for i in indices:
            for sfx in suffixes:
                open(os.path.join(self.data_dir,
                                  "%s_%03d%s" % (vname, i, sfx)), "w").close()

    def close(self):
        shutil.rmtree(self.root, ignore_errors=True)


def raises_value_error(fn, needle):
    try:
        fn()
    except ValueError as e:
        assert needle in str(e), "expected %r in %r" % (needle, str(e))
        return
    raise AssertionError("expected ValueError mentioning %r" % needle)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def test_wave_overlap_across_slots_is_refused():
    plan = make_plan({
        "usera": {"waves": ["0-4"]},
        "userb": {"waves": ["4-9"]},   # chunk 4 claimed twice
    })
    b = Block(plan)
    try:
        raises_value_error(lambda: mp.load_plan(b.plan_path), "overlap")
    finally:
        b.close()


def test_two_backup_or_tracking_slots_are_refused():
    for key in ("backup", "tracking"):
        plan = make_plan({
            "usera": {"waves": ["0-4"], key: True},
            "userb": {"waves": ["5-9"], key: True},
        })
        b = Block(plan)
        try:
            raises_value_error(lambda: mp.load_plan(b.plan_path), key)
        finally:
            b.close()


def test_policy_keys_in_settings_are_refused():
    # chunk_range / no_backup / run_tracking in settings would bypass the very
    # policy the plan centralizes.
    for bad in ("chunk_range", "no_backup", "run_tracking", "dir"):
        settings = dict(SETTINGS)
        settings[bad] = "x"
        b = Block(make_plan({"usera": {"waves": ["0-4"]}}, settings))
        try:
            raises_value_error(lambda: mp.load_plan(b.plan_path), bad)
        finally:
            b.close()


def test_shell_unsafe_slot_names_are_refused():
    # Slot names reach squeue/ssh command lines in pipeline_multi.sh; anything
    # that is not username-shaped must be rejected at load time.
    for bad in ("a'; rm -rf /; '", "user name", "us$er", "-leadingdash", ""):
        b = Block(make_plan({bad: {"waves": ["0-4"]}}))
        try:
            raises_value_error(lambda: mp.load_plan(b.plan_path), "username")
        finally:
            b.close()


def test_implausibly_wide_wave_is_refused():
    b = Block(make_plan({"usera": {"waves": ["0-999999999"]}}))
    try:
        raises_value_error(lambda: mp.load_plan(b.plan_path), "wide")
    finally:
        b.close()


def test_plan_must_live_in_a_data_dir():
    d = tempfile.mkdtemp(prefix="muptest_")
    try:
        p = os.path.join(d, mp.PLAN_BASENAME)  # not under .../data/
        with open(p, "w") as f:
            json.dump(make_plan({"usera": {"waves": ["0-4"]}}), f)
        raises_value_error(lambda: mp._exp_dir_of(p), "data/")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# flag emission
# ---------------------------------------------------------------------------
def test_flags_backup_once_tracking_on_last_wave():
    plan = make_plan({
        "usera": {"waves": ["0-4", "5-9", "10-14"],
                  "backup": True, "tracking": True},
        "userb": {"waves": ["15-19"]},
    })
    b = Block(plan)
    try:
        loaded = mp.load_plan(b.plan_path)
        exp = mp._exp_dir_of(b.plan_path)

        first = mp.wave_flags(loaded, "usera", "0-4", exp)
        mid = mp.wave_flags(loaded, "usera", "5-9", exp)
        last = mp.wave_flags(loaded, "usera", "10-14", exp)
        other = mp.wave_flags(loaded, "userb", "15-19", exp)

        # Backup: only the backup slot's FIRST wave keeps it.
        assert "--no-backup" not in first
        assert "--no-backup" in mid and "--no-backup" in last
        assert "--no-backup" in other
        # Tracking: only the tracking slot's LAST wave launches the poller.
        assert "--run-tracking" not in first and "--run-tracking" not in mid
        assert "--run-tracking" in last
        assert "--run-tracking" not in other
        # Settings mapping + derivations.
        assert ["--dir", exp] == first[:2]
        assert "--chunk-range" in first and "0-4" in first
        i = first.index("--chunk-sec")
        assert first[i + 1] == "1800"
        assert "--sleap-model-centroid" in first
    finally:
        b.close()


def test_flags_bool_setting_emits_bare_flag():
    settings = dict(SETTINGS)
    settings["skip_trt_export"] = True
    settings["only_sleap"] = False
    b = Block(make_plan({"usera": {"waves": ["0-4"]}}, settings))
    try:
        loaded = mp.load_plan(b.plan_path)
        flags = mp.wave_flags(loaded, "usera", "0-4", mp._exp_dir_of(b.plan_path))
        assert "--skip-trt-export" in flags
        assert "--only-sleap" not in flags   # false => omitted entirely
    finally:
        b.close()


def test_flags_refuse_foreign_wave_and_unknown_user():
    b = Block(make_plan({"usera": {"waves": ["0-4"]}}))
    try:
        loaded = mp.load_plan(b.plan_path)
        exp = mp._exp_dir_of(b.plan_path)
        raises_value_error(lambda: mp.wave_flags(loaded, "usera", "5-9", exp),
                           "not in")
        raises_value_error(lambda: mp.wave_flags(loaded, "ghost", "0-4", exp),
                           "no slot")
    finally:
        b.close()


# ---------------------------------------------------------------------------
# wave completion
# ---------------------------------------------------------------------------
def test_wave_status_done_requires_both_modalities_and_clamps():
    # VID_A has 10 chunks, VID_B only 3: wave 0-4 expects 5 + 3 = 8 per stage.
    videos = {VID_A: {"n_chunks": 10, "fps": 24.0, "frame_count": 432000},
              VID_B: {"n_chunks": 3, "fps": 24.0, "frame_count": 129600}}
    plan = make_plan({"usera": {"waves": ["0-4", "5-9"]}})
    b = Block(plan, videos=videos)
    try:
        loaded = mp.load_plan(b.plan_path)
        b.touch_outputs(VID_A, range(5), ["_aruco_tracks.h5", "_sleap_data.h5"])
        b.touch_outputs(VID_B, range(3), ["_aruco_tracks.h5"])  # sleap missing

        rows = mp.wave_status(loaded, "usera", b.data_dir)
        (w1, exp1, done1, state1), (w2, exp2, done2, state2) = rows
        assert (w1, exp1) == ("0-4", 8)
        assert done1 == {"trk": 8, "sdat": 5}
        assert state1 == "pending"          # one modality short of done

        b.touch_outputs(VID_B, range(3), ["_sleap_data.h5"])
        rows = mp.wave_status(loaded, "usera", b.data_dir)
        assert rows[0][3] == "done"
        # Wave 5-9 is beyond VID_B's 3 chunks: expected clamps to VID_A's 5.
        assert rows[1][1] == 5
        assert rows[1][3] == "pending"
    finally:
        b.close()


def test_no_contract_reports_unknown_not_done():
    # Before the first run declares a contract there is no denominator; a
    # driver that read this as "done" would never submit the first wave.
    b = Block(make_plan({"usera": {"waves": ["0-4"]}}))
    try:
        loaded = mp.load_plan(b.plan_path)
        rows = mp.wave_status(loaded, "usera", b.data_dir)
        assert rows[0][1] is None
        assert rows[0][3] == "unknown"
    finally:
        b.close()


def test_only_aruco_gates_on_tracks_alone():
    videos = {VID_A: {"n_chunks": 5, "fps": 24.0, "frame_count": 216000}}
    settings = dict(SETTINGS)
    settings["only_aruco"] = True
    b = Block(make_plan({"usera": {"waves": ["0-4"]}}, settings), videos=videos)
    try:
        loaded = mp.load_plan(b.plan_path)
        b.touch_outputs(VID_A, range(5), ["_aruco_tracks.h5"])
        rows = mp.wave_status(loaded, "usera", b.data_dir)
        assert rows[0][3] == "done"         # no _sleap_data.h5 required
    finally:
        b.close()


# ---------------------------------------------------------------------------
# CLI surface (what pipeline_multi.sh actually parses)
# ---------------------------------------------------------------------------
def test_cli_slot_status_tsv_and_flags_tokens():
    videos = {VID_A: {"n_chunks": 5, "fps": 24.0, "frame_count": 216000}}
    b = Block(make_plan({"usera": {"waves": ["0-4"]}}), videos=videos)
    try:
        import io
        buf, real = io.StringIO(), sys.stdout
        sys.stdout = buf
        try:
            rc = mp.main(["slot-status", "--plan", b.plan_path, "--user", "usera"])
        finally:
            sys.stdout = real
        assert rc == 0
        cols = buf.getvalue().strip().split("\t")
        assert cols[0] == "0-4" and cols[1] == "5" and cols[3] == "pending"

        buf, real = io.StringIO(), sys.stdout
        sys.stdout = buf
        try:
            rc = mp.main(["flags", "--plan", b.plan_path,
                          "--user", "usera", "--wave", "0-4"])
        finally:
            sys.stdout = real
        assert rc == 0
        toks = buf.getvalue().splitlines()
        assert toks[0] == "--dir"
        assert "--no-backup" in toks
    finally:
        b.close()


# ---------------------------------------------------------------------------
# plan generation
# ---------------------------------------------------------------------------
def test_split_waves_is_contiguous_disjoint_and_cap_sized():
    # 196 indices (98 h at 1800 s), 25 cameras, 3 users, 1500 rows/wave:
    # width = 1500 // 25 = 60; shares 66/65/65 -> 2 waves each, equal widths.
    slots, width = mp.split_waves(196, ["a", "b", "c"], 25, wave_rows=1500)
    assert width == 60
    assert slots["a"] == ["0-32", "33-65"]
    assert slots["b"] == ["66-98", "99-130"]
    assert slots["c"] == ["131-163", "164-195"]
    # Every index exactly once, no wave wider than the cap.
    seen = []
    for waves in slots.values():
        for w in waves:
            lo, hi = ps.parse_range(w)
            assert hi - lo + 1 <= width
            seen.extend(range(lo, hi + 1))
    assert sorted(seen) == list(range(196))


def test_split_waves_halves_width_for_overlapping_waves_and_scales_by_cameras():
    _, w1 = mp.split_waves(196, ["a"], 25, wave_rows=1500, max_live=1)
    _, w2 = mp.split_waves(196, ["a"], 25, wave_rows=1500, max_live=2)
    _, w19 = mp.split_waves(196, ["a"], 19, wave_rows=1500)
    assert (w1, w2, w19) == (60, 30, 78)
    raises_value_error(lambda: mp.split_waves(2, ["a", "b", "c"], 25), "users")


def test_resolve_settings_contract_beats_defaults_and_flags_conflicts():
    defaults = {"chunk_sec": 7200, "aruco_dict": "A", "saion_partition": "largegpu"}
    state = ps.new_state("/blk", 1800, "mkv",
                         {VID_A: {"n_chunks": 5, "fps": 24.0, "frame_count": 216000}},
                         {"aruco_dict": "/dicts/custom_4x4_A100.npz", "aruco_params": "",
                          "aruco_script": "run_aruco_mp.py",
                          "sleap_model_centroid": "/models/x.centroid",
                          "sleap_model_instance": "/models/x.centered_instance",
                          "sleap_module": "sleap-nn/0.3.3", "sleap_runtime": "tensorrt",
                          "saion_partition": "short-a100"})
    settings, conflicts = mp.resolve_settings(defaults, state, {"saion_partition": "largegpu"})
    assert settings["chunk_sec"] == 1800                      # contract, not default
    assert settings["aruco_dict"] == "/dicts/custom_4x4_A100.npz"
    assert "aruco_script" not in settings                     # contract-only key
    assert "aruco_params" not in settings                     # empty dropped
    assert settings["saion_partition"] == "largegpu"          # soft override, no conflict
    assert conflicts == []
    _, conflicts = mp.resolve_settings(defaults, state, {"sleap_model_centroid": "/models/other"})
    assert conflicts and conflicts[0][0] == "sleap_model_centroid"


def _defaults_file(block):
    p = os.path.join(block.root, "defaults.json")
    with open(p, "w") as f:
        json.dump({"_comment": "x", "chunk_sec": 7200, "aruco_dict": "A",
                   "saion_partition": "largegpu"}, f)
    return p


def test_init_writes_a_plan_the_loader_accepts():
    videos = {VID_A: {"n_chunks": 10, "fps": 24.0, "frame_count": 432000},
              VID_B: {"n_chunks": 10, "fps": 24.0, "frame_count": 432000}}
    b = Block(make_plan({"usera": {"waves": ["0-4"]}}), videos=videos)
    try:
        os.remove(b.plan_path)  # init must create it, not find it
        # Contract lacks sleap models: init must demand them.
        rc = mp.main(["init", "--dir", b.root, "--users", "usera,userb",
                      "--defaults", _defaults_file(b)])
        assert rc == 2
        rc = mp.main(["init", "--dir", b.root, "--users", "usera,userb",
                      "--defaults", _defaults_file(b),
                      "--set", "sleap_model_centroid=/models/x.centroid",
                      "--set", "sleap_model_instance=/models/x.centered_instance",
                      "--wave-rows", "6"])
        assert rc == 0
        plan = mp.load_plan(b.plan_path)
        assert plan["settings"]["chunk_sec"] == 1800          # from the contract
        assert plan["meta"]["settings_source"] == "contract"
        assert plan["meta"]["wave_width"] == 3                # 6 rows / 2 videos
        assert plan["slots"]["usera"] == {"waves": ["0-2", "3-4"], "backup": True, "tracking": True}
        assert plan["slots"]["userb"] == {"waves": ["5-7", "8-9"]}
        # Refuses to clobber without --force.
        rc = mp.main(["init", "--dir", b.root, "--users", "usera",
                      "--defaults", _defaults_file(b)])
        assert rc == 2
    finally:
        b.close()


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------
def _run_all():
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            sys.stdout.write(".")
        except Exception as e:  # a test runner reports everything
            failed.append((name, e))
            sys.stdout.write("F")
        sys.stdout.flush()
    sys.stdout.write("\n")
    for name, e in failed:
        sys.stdout.write("FAIL %s: %s: %s\n" % (name, type(e).__name__, e))
    sys.stdout.write("%d passed, %d failed (of %d)\n"
                     % (len(tests) - len(failed), len(failed), len(tests)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
