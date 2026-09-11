#!/usr/bin/env python3
"""Regression tests for the per-block processing contract (lib/pipeline_state.py).

Run with pytest, or directly -- deigo's system python has no pytest and this file
has to be runnable there, so it carries its own tiny runner::

    python3 detection_pipeline/test_pipeline_state.py

The two tests that matter most pin the failure modes the contract exists to stop:

``test_chunk_sec_change_is_refused`` -- a second run at a different --chunk-sec.
The filenames collide but each denotes a different span of wall-clock, and every
file stays individually valid, so nothing downstream can notice afterwards.

``test_coverage_denominator_is_declared_not_observed`` -- the wave-processing
trap. ``footprint.py`` and ``recover.py`` both derive "expected" as
``0..max(chunk_idx seen)``, which calls a block complete as soon as wave 1
lands. Coverage here must measure against the *declared* total instead.
"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
import pipeline_state as ps  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------
VID_A = "cam01_cam0_2099-01-01-00-00-00"
VID_B = "cam02_cam1_2099-01-01-00-00-00"

CONTRACT = {
    "chunk_sec": 1800,
    "chunk_ext": "mkv",
    "aruco_dict": "/dicts/custom_4x4_A100.npz",
    "aruco_params": "",
    "sleap_model_centroid": "/models/x.centroid",
    "sleap_model_instance": "/models/x.centered_instance",
    "sleap_module": "sleap-nn/0.3.3",
    "sleap_runtime": "tensorrt",
    "saion_partition": "short-a100",
}


def _capture(argv):
    """Run the CLI and return (stdout, exit_code).

    check-range's contract IS its stdout -- pipeline.sh's guard tests for empty
    output -- so asserting on the exit code alone would not pin the behaviour
    that matters.
    """
    import io
    buf, real = io.StringIO(), sys.stdout
    sys.stdout = buf
    try:
        rc = ps.main(argv)
    finally:
        sys.stdout = real
    return buf.getvalue(), rc


def write_manifest(path, videos):
    """videos: {vname: (n_chunks, frame_count)} -> a manifest.csv the module reads."""
    with open(path, "w") as f:
        f.write("vname,source_path,ext,fps,frame_count,duration_sec,n_chunks\n")
        for vname in sorted(videos):
            n_chunks, frames = videos[vname]
            f.write("%s,/src/%s.mkv,mkv,24.000,%d,%.1f,%d\n"
                    % (vname, vname, frames, frames / 24.0, n_chunks))


class Block(object):
    """A throwaway block dir with data/ and manifest.csv."""

    def __init__(self, videos=None):
        self.root = tempfile.mkdtemp(prefix="pstest_")
        self.data_dir = os.path.join(self.root, "data")
        os.makedirs(self.data_dir)
        self.manifest = os.path.join(self.root, "manifest.csv")
        write_manifest(self.manifest,
                       videos or {VID_A: (10, 432000), VID_B: (10, 432000)})

    def sync(self, overrides=None, legs="aruco,sleap", new_run=False, drop=()):
        """Run the sync CLI the way pipeline.sh does. Returns the exit code."""
        merged = dict(CONTRACT)
        merged.update(overrides or {})
        argv = ["sync", "--data-dir", self.data_dir, "--manifest", self.manifest,
                "--block-dir", self.root, "--legs", legs]
        for k in sorted(merged):
            if k in drop:
                continue
            argv += ["--set", "%s=%s" % (k, merged[k])]
        if new_run:
            argv.append("--new-run")
        return ps.main(argv)

    def touch_outputs(self, vname, idxs, stages=ps.STAGES):
        suffix = {"slp": ".slp", "sdat": "_sleap_data.h5",
                  "det": "_aruco_detections.h5", "trk": "_aruco_tracks.h5"}
        for i in idxs:
            for st in stages:
                p = os.path.join(self.data_dir, "%s_%03d%s" % (vname, i, suffix[st]))
                with open(p, "w") as f:
                    f.write("x")

    def state(self):
        return ps.load(self.data_dir)

    def close(self):
        shutil.rmtree(self.root, ignore_errors=True)


# ---------------------------------------------------------------------------
# range parsing / formatting
# ---------------------------------------------------------------------------
def test_parse_range_forms():
    assert ps.parse_range("0-4") == (0, 4)
    assert ps.parse_range("5") == (5, 5)
    assert ps.parse_range("") is None
    assert ps.parse_range(None) is None
    assert ps.parse_range(" 5-6 ") == (5, 6)


def test_parse_range_rejects_inverted_and_negative():
    for bad in ("4-0", "-1-3", "a-b", "3-"):
        try:
            ps.parse_range(bad)
        except ValueError:
            continue
        raise AssertionError("parse_range accepted %r" % bad)


def test_fmt_ranges_compresses_runs():
    assert ps.fmt_ranges([0, 1, 2, 4, 7, 8]) == "0-2,4,7-8"
    assert ps.fmt_ranges([]) == "-"
    assert ps.fmt_ranges([3]) == "3"


# ---------------------------------------------------------------------------
# contract creation
# ---------------------------------------------------------------------------
def test_first_run_creates_contract():
    b = Block()
    try:
        assert b.sync() == 0
        st = b.state()
        assert st["schema"] == ps.SCHEMA
        assert st["chunking"]["chunk_sec"] == 1800
        assert st["chunking"]["total_rows"] == 20
        assert st["detection"]["sleap_module"] == "sleap-nn/0.3.3"
        assert st["waves"] == []
    finally:
        b.close()


def test_second_identical_run_is_accepted():
    b = Block()
    try:
        assert b.sync() == 0
        assert b.sync() == 0
    finally:
        b.close()


# ---------------------------------------------------------------------------
# contract enforcement -- the reason this module exists
# ---------------------------------------------------------------------------
def test_chunk_sec_change_is_refused():
    b = Block()
    try:
        assert b.sync() == 0
        # Same block, same filenames, different meaning per filename.
        assert b.sync({"chunk_sec": 3600}) == 3
        # ... and the recorded contract is untouched by the refusal.
        assert b.state()["chunking"]["chunk_sec"] == 1800
    finally:
        b.close()


def test_model_swap_is_refused():
    b = Block()
    try:
        assert b.sync() == 0
        assert b.sync({"sleap_model_centroid": "/models/other.centroid"}) == 3
    finally:
        b.close()


def test_aruco_dict_change_is_refused():
    b = Block()
    try:
        assert b.sync() == 0
        assert b.sync({"aruco_dict": "/dicts/custom_4x4_B300.npz"}) == 3
    finally:
        b.close()


def test_execution_setting_change_only_warns():
    """runtime/module/partition change how work ran, not what it produced."""
    b = Block()
    try:
        assert b.sync() == 0
        assert b.sync({"saion_partition": "largegpu"}) == 0
        assert b.sync({"sleap_runtime": "pytorch"}) == 0
    finally:
        b.close()


def test_empty_aruco_params_is_a_value_not_absence():
    """'' means detector defaults. A later run must not be free to change it."""
    b = Block()
    try:
        assert b.sync() == 0
        assert b.state()["detection"]["aruco_params"] == ""
        assert b.sync({"aruco_params": "--error-correction-rate 0.0"}) == 3
    finally:
        b.close()


def test_only_sleap_run_is_not_judged_on_aruco_keys():
    b = Block()
    try:
        assert b.sync() == 0
        # A --only-sleap rerun supplies no aruco keys at all; not a conflict.
        assert b.sync(legs="sleap", drop=("aruco_dict", "aruco_params")) == 0
    finally:
        b.close()


def test_key_absent_from_contract_is_filled_not_refused():
    """An --only-aruco first run leaves sleap keys unrecorded; a later run fills them."""
    b = Block()
    try:
        assert b.sync(legs="aruco",
                      drop=("sleap_model_centroid", "sleap_model_instance",
                            "sleap_module", "sleap_runtime", "saion_partition")) == 0
        assert "sleap_model_centroid" not in b.state()["detection"]
        assert b.sync() == 0
        assert b.state()["detection"]["sleap_model_centroid"] == "/models/x.centroid"
    finally:
        b.close()


def test_source_video_reshape_is_refused():
    """A repaired/replaced source invalidates every chunk index already on the bucket."""
    b = Block()
    try:
        assert b.sync() == 0
        write_manifest(b.manifest, {VID_A: (12, 518400), VID_B: (10, 432000)})
        assert b.sync() == 3
    finally:
        b.close()


def test_new_processing_run_archives_and_restarts():
    b = Block()
    try:
        assert b.sync() == 0
        assert b.sync({"chunk_sec": 3600}, new_run=True) == 0
        assert b.state()["chunking"]["chunk_sec"] == 3600
        archived = [n for n in os.listdir(b.data_dir)
                    if n.startswith("PIPELINE_STATE.") and n != ps.STATE_BASENAME]
        assert len(archived) == 1, archived
        with open(os.path.join(b.data_dir, archived[0])) as f:
            assert json.load(f)["chunking"]["chunk_sec"] == 1800
    finally:
        b.close()


def test_corrupt_state_refuses_rather_than_degrading():
    """'Unreadable' must never become 'no contract' -- that reopens the block."""
    b = Block()
    try:
        assert b.sync() == 0
        with open(ps.state_path(b.data_dir), "w") as f:
            f.write("{not json")
        assert b.sync() == 3
    finally:
        b.close()


# ---------------------------------------------------------------------------
# wave ledger
# ---------------------------------------------------------------------------
def test_add_wave_numbers_and_records_range():
    b = Block()
    try:
        b.sync()
        assert ps.main(["add-wave", "--data-dir", b.data_dir,
                        "--range", "0-4", "--rows", "10"]) == 0
        assert ps.main(["add-wave", "--data-dir", b.data_dir,
                        "--range", "5-6", "--rows", "4"]) == 0
        waves = b.state()["waves"]
        assert [w["wave"] for w in waves] == [1, 2]
        assert waves[0]["chunk_range"] == [0, 4]
        assert waves[1]["rows"] == 4
        assert waves[0]["completed"] is None
    finally:
        b.close()


def test_ranges_overlap_treats_none_as_the_whole_block():
    assert ps.ranges_overlap((0, 4), (5, 6)) is False
    assert ps.ranges_overlap((0, 4), (4, 6)) is True     # touching at one index
    assert ps.ranges_overlap((5, 6), (0, 4)) is False    # order must not matter
    assert ps.ranges_overlap((2, 3), (0, 9)) is True     # fully contained
    assert ps.ranges_overlap(None, (5, 6)) is True       # whole block vs a wave
    assert ps.ranges_overlap((5, 6), None) is True
    assert ps.ranges_overlap(None, None) is True


def test_check_range_reports_only_real_collisions():
    """A free range must print NOTHING (the guard tests for empty output), and an
    overlap must still exit 0 -- recomputing a claimed window, for a new model or
    to rescue half-landed chunks, has to stay possible."""
    b = Block()
    try:
        b.sync()
        ps.main(["add-wave", "--data-dir", b.data_dir, "--range", "0-4", "--rows", "10"])
        ps.main(["add-wave", "--data-dir", b.data_dir, "--range", "5-6", "--rows", "4"])

        assert _capture(["check-range", "--data-dir", b.data_dir,
                         "--range", "7-8"]) == ("", 0)

        out, rc = _capture(["check-range", "--data-dir", b.data_dir, "--range", "4-5"])
        assert rc == 0
        assert "wave #1" in out and "wave #2" in out, out

        # No range at all means the whole block, which collides with every wave.
        out, rc = _capture(["check-range", "--data-dir", b.data_dir, "--range", ""])
        assert rc == 0
        assert "wave #1" in out and "wave #2" in out, out
    finally:
        b.close()


def test_check_range_is_silent_without_a_contract_or_on_a_bad_range():
    """It runs before anything else is validated, so it must never be the thing
    that stops a submission."""
    b = Block()
    try:
        assert _capture(["check-range", "--data-dir", b.data_dir,
                         "--range", "0-4"]) == ("", 0)
        b.sync()
        ps.main(["add-wave", "--data-dir", b.data_dir, "--range", "0-4", "--rows", "10"])
        assert _capture(["check-range", "--data-dir", b.data_dir,
                         "--range", "9-2"]) == ("", 0)
        assert _capture(["check-range", "--data-dir", b.data_dir,
                         "--range", "notarange"]) == ("", 0)
    finally:
        b.close()


def test_wave_indices_clamp_to_each_videos_length():
    """A short camera must not inflate the denominator with impossible indices."""
    b = Block({VID_A: (10, 432000), VID_B: (3, 129600)})
    try:
        b.sync()
        idx = ps.wave_indices(b.state(), {"chunk_range": [0, 4]})
        assert idx[VID_A] == set(range(0, 5))
        assert idx[VID_B] == set(range(0, 3))
    finally:
        b.close()


def test_declared_rows_is_block_wide_not_per_wave():
    """track_trigger.sh's denominator.

    It counts output files across all of data/, so gating on a wave's worklist
    would call the block finished on wave 2's first poll -- wave 1's files
    already outnumber wave 2's row count.
    """
    b = Block({VID_A: (10, 432000)})
    try:
        b.sync()
        ps.main(["add-wave", "--data-dir", b.data_dir, "--range", "0-4", "--rows", "5"])
        assert b.state()["chunking"]["total_rows"] == 10   # not the wave's 5
        assert ps.main(["declared-rows", "--data-dir", b.data_dir]) == 0
    finally:
        b.close()


def test_declared_rows_exits_nonzero_without_a_contract():
    """Lets the shell caller fall back to the worklist count on old blocks."""
    b = Block()
    try:
        assert ps.main(["declared-rows", "--data-dir", b.data_dir]) == 1
    finally:
        b.close()


def test_complete_wave_stamps_only_that_wave():
    b = Block()
    try:
        b.sync()
        ps.main(["add-wave", "--data-dir", b.data_dir, "--range", "0-4", "--rows", "10"])
        ps.main(["add-wave", "--data-dir", b.data_dir, "--range", "5-6", "--rows", "4"])
        assert ps.main(["complete-wave", "--data-dir", b.data_dir, "--wave", "1"]) == 0
        waves = b.state()["waves"]
        assert waves[0]["completed"] is not None
        assert waves[1]["completed"] is None
    finally:
        b.close()


# ---------------------------------------------------------------------------
# coverage -- the wave-processing regression
# ---------------------------------------------------------------------------
def test_coverage_denominator_is_declared_not_observed():
    """Wave 1 lands; the block must NOT read as complete.

    This is exactly what footprint.py's `0..max(chunk_idx seen)` gets wrong: with
    only chunks 0-4 on disk it infers 5 expected and reports 100%.
    """
    b = Block()
    try:
        b.sync()
        ps.main(["add-wave", "--data-dir", b.data_dir, "--range", "0-4", "--rows", "10"])
        b.touch_outputs(VID_A, range(5))
        b.touch_outputs(VID_B, range(5))
        cov = ps.coverage(b.state(), b.data_dir)
        assert cov["declared_total"] == 20
        assert cov["stages"]["det"]["present"] == 10
        assert cov["stages"]["det"]["pct"] == 0.5
        # The wave itself, however, is fully covered.
        assert cov["waves"][0]["declared"] == 10
        assert cov["waves"][0]["stages"]["det"] == 10
    finally:
        b.close()


def test_coverage_reaches_full_only_when_every_wave_lands():
    b = Block({VID_A: (7, 302400)})
    try:
        b.sync()
        ps.main(["add-wave", "--data-dir", b.data_dir, "--range", "0-4", "--rows", "5"])
        b.touch_outputs(VID_A, range(5))
        assert ps.coverage(b.state(), b.data_dir)["stages"]["slp"]["pct"] < 1.0
        ps.main(["add-wave", "--data-dir", b.data_dir, "--range", "5-6", "--rows", "2"])
        b.touch_outputs(VID_A, [5, 6])
        cov = ps.coverage(b.state(), b.data_dir)
        assert cov["stages"]["slp"]["pct"] == 1.0
        assert cov["unclaimed"] == {}
    finally:
        b.close()


def test_partial_stage_shows_as_missing_indices():
    """A chunk with .slp but no _sleap_data.h5 is what tracking actually gates on."""
    b = Block({VID_A: (5, 216000)})
    try:
        b.sync()
        b.touch_outputs(VID_A, range(5), stages=("det", "trk", "slp"))
        b.touch_outputs(VID_A, [0, 1], stages=("sdat",))
        cov = ps.coverage(b.state(), b.data_dir)
        assert cov["stages"]["slp"]["present"] == 5
        assert cov["stages"]["sdat"]["missing"][VID_A] == [2, 3, 4]
    finally:
        b.close()


def test_unclaimed_reports_the_gap_between_waves():
    b = Block({VID_A: (10, 432000)})
    try:
        b.sync()
        ps.main(["add-wave", "--data-dir", b.data_dir, "--range", "0-2", "--rows", "3"])
        ps.main(["add-wave", "--data-dir", b.data_dir, "--range", "6-9", "--rows", "4"])
        assert ps.unclaimed_ranges(b.state()) == {VID_A: "3-5"}
    finally:
        b.close()


def test_interior_gap_is_distinguished_from_an_unstarted_tail():
    """A skipped middle window is a defect; a not-yet-run tail is just progress."""
    b = Block({VID_A: (10, 432000)})
    try:
        b.sync()
        ps.main(["add-wave", "--data-dir", b.data_dir, "--range", "0-2", "--rows", "3"])
        # Tail only: 3-9 unclaimed, but nothing was skipped.
        assert ps.interior_gaps(b.state()) == {}
        assert ps.unclaimed_ranges(b.state()) == {VID_A: "3-9"}
        # Now jump past 3-5: that IS a hole, and only the ledger can see it.
        ps.main(["add-wave", "--data-dir", b.data_dir, "--range", "6-7", "--rows", "2"])
        assert ps.interior_gaps(b.state()) == {VID_A: "3-5"}
    finally:
        b.close()


def test_no_waves_recorded_means_everything_unclaimed():
    b = Block({VID_A: (4, 172800)})
    try:
        b.sync()
        assert ps.unclaimed_ranges(b.state()) == {VID_A: "0-3"}
    finally:
        b.close()


# ---------------------------------------------------------------------------
# catalog integration -- does the reporting layer actually use the declaration?
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
from catalog import footprint as fp_mod, provenance, qc  # noqa: E402
from catalog.model import Unit  # noqa: E402


def _unit(block):
    u = Unit(session_id="2099-01-01", block="block01", path=block.root)
    u.has_data_dir = True
    u.data_dir = block.data_dir
    u.subdir_names = ["data"]
    return u


def _has_wave_gap(flags):
    from catalog import const
    return const.HZ_WAVE_GAP in flags


def test_footprint_prefers_the_declared_chunk_count():
    b = Block({VID_A: (10, 432000)})
    try:
        b.sync()
        b.touch_outputs(VID_A, range(4))
        fp = fp_mod.scan_footprint(_unit(b))
        assert fp.expected_source == "declared"
        assert fp.expected_total == 10          # declared, not the 4 observed
        assert fp.chunk_sec == 1800
        assert fp.chunk_sec_source == "state"
        assert fp.completeness_state == "declared"
        assert fp.completeness_pct == 0.4
    finally:
        b.close()


def test_status_is_partial_after_wave_one_and_complete_only_at_the_end():
    """The regression. Without a declaration this block reports 'complete'."""
    b = Block({VID_A: (10, 432000)})
    try:
        b.sync()
        ps.main(["add-wave", "--data-dir", b.data_dir, "--range", "0-4", "--rows", "5"])
        b.touch_outputs(VID_A, range(5))
        u = _unit(b)
        assert qc.pipeline_status(u, fp_mod.scan_footprint(u)) == "partial"
        ps.main(["add-wave", "--data-dir", b.data_dir, "--range", "5-9", "--rows", "5"])
        b.touch_outputs(VID_A, range(5, 10))
        assert qc.pipeline_status(u, fp_mod.scan_footprint(u)) == "complete"
    finally:
        b.close()


def test_contractless_block_keeps_the_old_observed_behaviour():
    """Pins backward compatibility, including the old blindness it implies."""
    b = Block({VID_A: (10, 432000)})
    try:
        # Deliberately no b.sync(): this block never declared a contract.
        b.touch_outputs(VID_A, range(5))
        u = _unit(b)
        fp = fp_mod.scan_footprint(u)
        assert fp.expected_source == "observed"
        assert fp.expected_total == 5           # 0..max(seen)
        assert fp.completeness_state == "internal"
        # ... and this is exactly why the contract exists: half a block, "complete".
        assert qc.pipeline_status(u, fp) == "complete"
    finally:
        b.close()


def test_wave_gap_hazard_fires_only_on_an_interior_hole():
    b = Block({VID_A: (10, 432000)})
    try:
        b.sync()
        ps.main(["add-wave", "--data-dir", b.data_dir, "--range", "0-2", "--rows", "3"])
        b.touch_outputs(VID_A, range(3))
        u = _unit(b)
        # Chunks 3-9 are unclaimed, but that is just the tail: no hazard.
        assert _has_wave_gap(qc.derive_hazards(u, fp_mod.scan_footprint(u), [])) is False
        # Jumping to 6-9 leaves 3-5 skipped between two claimed waves.
        ps.main(["add-wave", "--data-dir", b.data_dir, "--range", "6-9", "--rows", "4"])
        b.touch_outputs(VID_A, range(6, 10))
        assert _has_wave_gap(qc.derive_hazards(u, fp_mod.scan_footprint(u), [])) is True
    finally:
        b.close()


def test_provenance_reads_the_contract_and_rebuilds_frame_caps():
    b = Block({VID_A: (10, 432000)})
    try:
        b.sync()
        prov = provenance.read_provenance(b.root)
        assert prov["source"] == "state_file"
        assert prov["chunk_sec"] == 1800
        assert prov["declared"][VID_A] == 10
        assert prov["sleap_model_centroid"] == "/models/x.centroid"
        # Frame caps must match lib/worklist.py exactly: full chunks at fps*sec,
        # last chunk the residual.
        wl = prov["worklist"][VID_A]
        assert len(wl) == 10
        assert wl[0] == 43200
        assert wl[9] == 432000 - 9 * 43200
    finally:
        b.close()


# ---------------------------------------------------------------------------
# provenance fallback: hpc_logs/pipeline/pipeline.env
#
# Blocks processed before PIPELINE_STATE.json existed have no contract, and
# their bridge_*.out predate the `centroid:` echo -- but pipeline.sh always
# wrote its configuration to pipeline.env, which is archived with the logs.
# ---------------------------------------------------------------------------
ENV_CENTROID = "/bucket/SLEAP_files/Simple_skeleton_nn/250408_141245.centroid"
ENV_INSTANCE = "/bucket/SLEAP_files/Simple_skeleton_nn/250408_141245.centered_instance"


def _write_pipeline_logs(block_root, env_lines=None, bridge_text=None):
    pdir = os.path.join(block_root, "hpc_logs", "pipeline")
    os.makedirs(pdir, exist_ok=True)
    if env_lines is not None:
        with open(os.path.join(pdir, "pipeline.env"), "w") as f:
            f.write("\n".join(env_lines) + "\n")
    if bridge_text is not None:
        with open(os.path.join(pdir, "bridge_1.out"), "w") as f:
            f.write(bridge_text)
    return pdir


def _env_lines(centroid=ENV_CENTROID, instance=ENV_INSTANCE):
    return [
        "# generated by pipeline.sh",
        'export CHUNK_SEC="1800"',
        'export CHUNK_EXT="mkv"',
        'export SAION_PARTITION="short-a100"',
        'export SLEAP_MODEL_CENTROID="%s"' % centroid,
        'export SLEAP_MODEL_INSTANCE="%s"' % instance,
        "export SLEAP_RUNTIME='tensorrt'",
        "",
    ]


def test_provenance_falls_back_to_pipeline_env_when_bridge_logs_are_silent():
    b = Block()                                  # no sync(): no contract
    try:
        _write_pipeline_logs(b.root, env_lines=_env_lines(),
                             bridge_text="bridge started\nno model echo here\n")
        prov = provenance.read_provenance(b.root)
        assert prov["sleap_model_centroid"] == ENV_CENTROID
        assert prov["sleap_model_instance"] == ENV_INSTANCE
        assert prov["sleap_runtime"] == "tensorrt"     # single quotes stripped
        assert prov["saion_partition"] == "short-a100"
        assert prov["partitions"] == ["short-a100"]
        # CHUNK_SEC is in the file but is NOT taken from it: footprint/recover
        # derive chunk_sec from the contract or frame counts (see module doc).
        assert prov["chunk_sec"] is None
        assert prov["source"] == "hpc_logs"
        assert provenance.model_label(prov) == "Simple_skeleton_nn"
    finally:
        b.close()


def test_bridge_log_echo_wins_over_pipeline_env():
    b = Block()
    try:
        _write_pipeline_logs(
            b.root, env_lines=_env_lines(),
            bridge_text=("centroid: /m/bridge.centroid\n"
                         "instance: /m/bridge.centered_instance\n"
                         "engine: /e/x__largegpu/model\n"))
        prov = provenance.read_provenance(b.root)
        assert prov["sleap_model_centroid"] == "/m/bridge.centroid"
        assert prov["sleap_model_instance"] == "/m/bridge.centered_instance"
        # Partition was answered by the engine suffix; env must not override it.
        assert prov["saion_partition"] == "largegpu"
        assert prov["partitions"] == ["largegpu"]
    finally:
        b.close()


def test_contract_wins_over_pipeline_env():
    b = Block()
    try:
        b.sync()
        _write_pipeline_logs(b.root, env_lines=_env_lines())
        prov = provenance.read_provenance(b.root)
        assert prov["source"] == "state_file"
        assert prov["sleap_model_centroid"] == "/models/x.centroid"
        assert prov["saion_partition"] == "short-a100"
        assert prov["chunk_sec"] == 1800
    finally:
        b.close()


def test_pipeline_env_needs_both_model_paths_and_ignores_junk():
    b = Block()
    try:
        _write_pipeline_logs(b.root, env_lines=[
            "this line is not an assignment",
            'export SLEAP_MODEL_CENTROID="%s"' % ENV_CENTROID,   # instance missing
            "CHUNK_SEC=notanumber",
        ])
        prov = provenance.read_provenance(b.root)
        assert prov["sleap_model_centroid"] == ""
        assert prov["sleap_model_instance"] == ""
        assert prov["source"] == ""
        assert prov["chunk_sec"] is None
    finally:
        b.close()


def test_missing_pipeline_env_leaves_provenance_empty():
    b = Block()
    try:
        _write_pipeline_logs(b.root, bridge_text="nothing useful\n")
        prov = provenance.read_provenance(b.root)
        assert prov["sleap_model_centroid"] == ""
        assert prov["source"] == ""
    finally:
        b.close()


# ---------------------------------------------------------------------------
# tracking provenance: tracks/TRACKING_STATE.json (lib/tracking_state.py)
# ---------------------------------------------------------------------------
import tracking_state as ts  # noqa: E402
from catalog import const, tracking as trk_mod  # noqa: E402

CALIB_A = "cameraArray_calib/20260623_calib_elevated_by_2mm_from_arenafloor/frame0/aruco_stitch/aruco_H_mats.npz"
CALIB_B = "cameraArray_calib/20260810_calib_elevated_by_2mm/frame0/aruco_stitch/aruco_H_mats.npz"


class TrackedBlock(object):
    """A throwaway block dir with tracks/ and fake calibration files."""

    def __init__(self):
        self.root = tempfile.mkdtemp(prefix="trktest_")
        self.tracks = os.path.join(self.root, "tracks")
        os.makedirs(self.tracks)
        self.bucket = os.path.join(self.root, "bucket")

    def hmats(self, rel, content=b"H-stack-A"):
        p = os.path.join(self.bucket, *rel.split("/"))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(content)
        return p

    def close(self):
        shutil.rmtree(self.root, ignore_errors=True)


class _FP(object):
    def __init__(self, downstream):
        self.downstream = downstream


def test_record_map_captures_hmats_identity_and_settings():
    b = TrackedBlock()
    try:
        h = b.hmats(CALIB_A)
        state = ts.record_map(b.tracks, hmats=h, x_threshold=2475.0, map_mode="both",
                              min_instance_frame_frac=0.25, data_dir=os.path.join(b.root, "data"),
                              chunks=["003", "000", "001"], code_dir="/apps/x/releases/20260911_abc/tracking",
                              argv=["pipeline.py", "--hmats", h])
        m = ts.load(b.tracks)["map"]
        assert m["source"] == "pipeline"
        assert m["hmats"] == h
        assert m["hmats_calib_id"] == "20260623_calib_elevated_by_2mm_from_arenafloor"
        assert m["hmats_sha256"] == ts.file_fingerprint(h)["sha256"]
        assert m["x_threshold"] == 2475.0
        assert m["chunks"] == {"first": "000", "last": "003", "count": 3}
        assert m["release"] == "20260911_abc"
        assert m["user"] and m["host"] and m["recorded_at"].endswith("Z")
        assert state["stitch"] is None and state["history"] == []
        s = ts.summary(state)
        assert s["tracking_hmats"] == "20260623_calib_elevated_by_2mm_from_arenafloor"
        assert s["tracking_x_threshold"] == "2475"
        assert s["tracked_at"] == m["recorded_at"]
    finally:
        b.close()


def test_remapping_with_other_hmats_keeps_the_displaced_record_in_history():
    b = TrackedBlock()
    try:
        ha, hb = b.hmats(CALIB_A), b.hmats(CALIB_B, b"H-stack-B")
        ts.record_map(b.tracks, hmats=ha, x_threshold=2475.0)
        ts.record_map(b.tracks, hmats=ha, x_threshold=2475.0)       # same run again
        assert ts.load(b.tracks)["history"] == []
        ts.record_map(b.tracks, hmats=hb, x_threshold=2475.0)       # recalibrated
        state = ts.load(b.tracks)
        assert state["map"]["hmats_calib_id"] == "20260810_calib_elevated_by_2mm"
        assert [h["hmats_calib_id"] for h in state["history"]] == \
            ["20260623_calib_elevated_by_2mm_from_arenafloor"]
        ts.record_map(b.tracks, hmats=hb, x_threshold=2500.0)       # split changed
        assert len(ts.load(b.tracks)["history"]) == 2
    finally:
        b.close()


def test_record_stitch_leaves_the_map_record_alone():
    b = TrackedBlock()
    try:
        ts.record_stitch(b.tracks, fps=24.0, chunk_frames=43200, side="both")  # no map yet
        state = ts.load(b.tracks)
        assert state["map"] is None and state["stitch"]["chunk_frames"] == 43200
        h = b.hmats(CALIB_A)
        ts.record_map(b.tracks, hmats=h, x_threshold=2475.0)
        ts.record_stitch(b.tracks, fps=24.0, chunk_frames=43200, side="left")
        state = ts.load(b.tracks)
        assert state["map"]["hmats_calib_id"].startswith("20260623")
        assert state["stitch"]["side"] == "left"
    finally:
        b.close()


def test_backfill_refuses_to_overwrite_a_pipeline_record_unless_forced():
    b = TrackedBlock()
    try:
        h = b.hmats(CALIB_A)
        ts.record_map(b.tracks, hmats=h, x_threshold=2475.0)
        hb = b.hmats(CALIB_B, b"H-stack-B")
        try:
            ts.backfill(b.tracks, hb, x_threshold=2475.0)
            assert False, "backfill over a pipeline record must be refused"
        except ValueError as e:
            assert "--force" in str(e)
        assert ts.load(b.tracks)["map"]["source"] == "pipeline"
        ts.backfill(b.tracks, hb, x_threshold=2475.0, force=True,
                    tracked_at="2026-08-21", note="rendered sbatch")
        state = ts.load(b.tracks)
        assert state["map"]["source"] == "backfill"
        assert state["map"]["tracked_at"] == "2026-08-21"
        assert state["history"][0]["source"] == "pipeline"
        assert ts.summary(state)["tracked_at"] == "2026-08-21"   # tracked, not backfilled
    finally:
        b.close()


def test_backfill_dry_run_writes_nothing_and_path_only_hmats_still_labels():
    b = TrackedBlock()
    try:
        missing = "/bucket/ReiterU/Ants/basler/" + CALIB_A       # not readable here
        state = ts.backfill(b.tracks, missing, x_threshold=2475.0, dry_run=True)
        assert state["map"]["hmats_calib_id"] == "20260623_calib_elevated_by_2mm_from_arenafloor"
        assert "hmats_sha256" not in state["map"]
        assert ts.load(b.tracks) is None
    finally:
        b.close()


def test_corrupt_record_is_set_aside_by_the_pipeline_but_fatal_for_backfill():
    b = TrackedBlock()
    try:
        with open(ts.state_path(b.tracks), "w") as f:
            f.write("{not json")
        try:
            ts.backfill(b.tracks, b.hmats(CALIB_A))
            assert False, "backfill must not paper over a corrupt record"
        except ValueError:
            pass
        ts.record_map(b.tracks, hmats=b.hmats(CALIB_A), x_threshold=2475.0)
        assert ts.load(b.tracks)["map"]["source"] == "pipeline"
        kept = [n for n in os.listdir(b.tracks) if ".corrupt." in n]
        assert len(kept) == 1
    finally:
        b.close()


def test_calib_id_falls_back_to_the_parent_directory():
    assert ts.calib_id_from_hmats("C:\\data\\cameraArray_calib\\20260414_calibration_dataset\\x\\initial_H_mats.npz") \
        == "20260414_calibration_dataset"
    assert ts.calib_id_from_hmats("/home/sam/hmats/refined/refined_H_mats.npz") == "refined"


def test_catalog_reads_the_record_and_flags_unrecorded_tracks():
    b = TrackedBlock()
    try:
        trk = trk_mod.read_tracking(b.root)
        assert trk["tracking_hmats"] == "" and trk["tracking_error"] == ""
        assert qc.tracking_hazards(_FP(["tracks"]), trk) == [const.HZ_TRACKING_UNRECORDED]
        assert qc.tracking_hazards(_FP([]), trk) == []          # never tracked: no flag
        ts.record_map(b.tracks, hmats=b.hmats(CALIB_A), x_threshold=2475.0)
        trk = trk_mod.read_tracking(b.root)
        assert trk["tracking_hmats"] == "20260623_calib_elevated_by_2mm_from_arenafloor"
        assert trk["tracking_x_threshold"] == "2475"
        assert qc.tracking_hazards(_FP(["tracks"]), trk) == []
        with open(ts.state_path(b.tracks), "w") as f:
            f.write("{not json")
        trk = trk_mod.read_tracking(b.root)
        assert trk["tracking_hmats"] == "" and "not valid JSON" in trk["tracking_error"]
        # corrupt is told apart from never-recorded: the fix differs
        assert qc.tracking_hazards(_FP(["tracks"]), trk) == [const.HZ_TRACKING_STATE_CORRUPT]
    finally:
        b.close()


def _catalog_cli(argv, cwd):
    """Run detection_pipeline/catalog.py as a subprocess; returns its stderr log."""
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    r = subprocess.run([sys.executable, os.path.join(here, "catalog.py")] + argv,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       universal_newlines=True, cwd=cwd)
    return r.returncode, r.stderr


def test_track_init_cli_backfills_and_refuses_what_it_should():
    root = tempfile.mkdtemp(prefix="trkcli_")
    try:
        blk = os.path.join(root, "20260716", "block01")
        os.makedirs(os.path.join(blk, "tracks"))
        os.makedirs(os.path.join(root, "20260726", "block01"))       # never tracked
        hm = os.path.join(root, "cameraArray_calib", "20260623_calib_x", "aruco_H_mats.npz")
        os.makedirs(os.path.dirname(hm))
        with open(hm, "wb") as f:
            f.write(b"H")
        outdir = os.path.join(root, "_catalog")
        base = ["--root", root, "--outdir", outdir]

        rc, err = _catalog_cli(["track-init", "20260716/block01", "--hmats", hm,
                                "--x-threshold", "2475", "--dry-run"] + base, root)
        assert rc == 0 and "would record" in err
        assert ts.load(os.path.join(blk, "tracks")) is None

        rc, err = _catalog_cli(["track-init", "20260716/block01", "--hmats", hm,
                                "--x-threshold", "2475", "--tracked-at", "2026-07-20",
                                "--note", "test"] + base, root)
        assert rc == 0 and "recorded" in err
        state = ts.load(os.path.join(blk, "tracks"))
        assert state["map"]["source"] == "backfill"
        assert state["map"]["hmats_calib_id"] == "20260623_calib_x"
        assert state["map"]["x_threshold"] == 2475.0
        assert state["map"]["tracked_at"] == "2026-07-20"

        # a block that was never tracked is refused without --allow-missing-tracks
        rc, err = _catalog_cli(["track-init", "20260726/block01", "--hmats", hm] + base, root)
        assert rc == 0 and "no tracks/" in err
        assert not os.path.isdir(os.path.join(root, "20260726", "block01", "tracks"))
        rc, err = _catalog_cli(["track-init", "20260726/block01", "--hmats", hm,
                                "--allow-missing-tracks"] + base, root)
        assert rc == 0 and "recorded" in err
        assert ts.load(os.path.join(root, "20260726", "block01", "tracks"))["map"]["source"] == "backfill"

        # a pipeline-written record needs --overwrite, and --force is NOT it
        ts.record_map(os.path.join(blk, "tracks"), hmats=hm, x_threshold=2475.0)
        rc, err = _catalog_cli(["track-init", "20260716/block01", "--hmats", hm,
                                "--force"] + base, root)
        assert rc == 0 and "track-init failed" in err and "--force" in err
        assert ts.load(os.path.join(blk, "tracks"))["map"]["source"] == "pipeline"
        rc, err = _catalog_cli(["track-init", "20260716/block01", "--hmats", hm,
                                "--overwrite"] + base, root)
        assert rc == 0 and "recorded" in err
        state = ts.load(os.path.join(blk, "tracks"))
        assert state["map"]["source"] == "backfill"
        assert state["history"][-1]["source"] == "pipeline"
    finally:
        shutil.rmtree(root, ignore_errors=True)


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
