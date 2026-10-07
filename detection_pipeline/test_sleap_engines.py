"""The saion SLEAP array picks each chunk's engine from the bridge's per-chunk caps."""
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
ENGINES = "96:2:/work/x/m__k96b2__largegpu 20:8:/work/x/m__k20b8__largegpu"


def engine_for(tmp_path, rows, vname, chunk, engines=ENGINES):
    caps = tmp_path / "sleap_caps.tsv"
    caps.write_text("".join("%s\t%s\t%s\n" % r for r in rows))
    script = "source %s; engine_for_chunk %s %s %s %s" % tuple(
        shlex.quote(str(x)) for x in (ROOT / "lib/sleap_engines.sh", caps.as_posix(), engines, vname, chunk))
    return subprocess.run(["bash", "-c", script], text=True, capture_output=True)


ROWS = [("cam01_cam0_x", "003", "96"), ("cam11_cam2_x", "003", "20"), ("cam01_cam0_x", "004", "96")]


def test_a_nest_chunk_gets_the_large_engine_and_its_batch(tmp_path):
    r = engine_for(tmp_path, ROWS, "cam01_cam0_x", "004")

    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["96", "2", "/work/x/m__k96b2__largegpu"]


def test_another_camera_gets_the_small_engine(tmp_path):
    r = engine_for(tmp_path, ROWS, "cam11_cam2_x", "003")

    assert r.stdout.split() == ["20", "8", "/work/x/m__k20b8__largegpu"]


def test_a_chunk_without_a_cap_row_is_refused(tmp_path):
    r = engine_for(tmp_path, ROWS, "cam11_cam2_x", "004")

    assert r.returncode != 0
    assert r.stdout == ""


def test_a_cap_without_a_built_engine_is_refused(tmp_path):
    r = engine_for(tmp_path, ROWS, "cam11_cam2_x", "003", engines="96:2:/work/x/m__k96b2__largegpu")

    assert r.returncode != 0
    assert r.stdout == ""


def _bridge_caps_section():
    bridge = (ROOT / "templates/bridge.sbatch").read_text()
    fn = bridge[bridge.index("engine_dir_for() {"):]
    fn = fn[:fn.index("\n}\n") + 3]
    section = bridge[bridge.index("# --- Per-chunk instance caps"):bridge.index("# --- Render saion-side templates")]
    return fn + section


def run_bridge_caps(tmp_path, env):
    """Run bridge.sbatch's per-chunk caps section with ensure_engine stubbed."""
    from test_sleap_caps import JST_0701, HOUR, make_block, write_table, write_worklist
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    manifest = make_block(jobs, JST_0701 + 24 * HOUR)
    worklist = write_worklist(tmp_path / "worklist.txt", manifest, [0, 1])
    table = write_table(jobs / "nest_cams.tsv", [("2026-07-01", "cam01", "A")])
    calls = tmp_path / "calls.txt"
    sh = "\n".join([
        'python3() { %s "$@"; }' % shlex.quote(sys.executable),
        'ensure_engine() { echo "$1 $2" >> %s; }' % shlex.quote(calls.as_posix()),
        "SKIP_TRT_EXPORT=0 SLEAP_RUNTIME=tensorrt CHUNK_SEC=1800 EXPORT_ROOT=/work/x SAION_PARTITION=largegpu",
        "SLEAP_MODEL_CENTROID=/m/a.centroid SLEAP_MODEL_INSTANCE=/m/a.centered_instance",
        "SCRIPTS_DIR=%s JOBS_ROOT=%s WORKLIST=%s SLEAP_STAGE=%s SLEAP_NEST_CAMS=%s" % tuple(
            shlex.quote(p.as_posix()) for p in (ROOT / "scripts", jobs, worklist, tmp_path, table)),
        env,
        _bridge_caps_section(),
        'echo "ENGINES=$ENGINES"; echo "EXPORT_DIR=$EXPORT_DIR"'])
    r = subprocess.run(["bash", "-c", sh], text=True, capture_output=True, timeout=60)
    out = dict(line.split("=", 1) for line in r.stdout.splitlines() if line.startswith(("ENGINES=", "EXPORT_DIR=")))
    return r, out, (calls.read_text().split("\n")[:-1] if calls.exists() else [])


K96 = "/work/x/a.centroid__a.centered_instance__k96b2__largegpu"
K20 = "/work/x/a.centroid__a.centered_instance__k20b8__largegpu"


def test_bridge_makes_one_engine_per_cap_in_nest_mode(tmp_path):
    r, out, calls = run_bridge_caps(tmp_path, "SLEAP_CAP_MODE=nest SLEAP_NEST_CAP=96 SLEAP_OTHER_CAP=20 "
                                              "SLEAP_MAX_INSTANCES=96 SLEAP_BATCH_SIZE=2")

    assert r.returncode == 0, r.stdout + r.stderr
    assert calls == ["96 2", "20 8"]
    assert out == {"ENGINES": "96:2:%s 20:8:%s" % (K96, K20), "EXPORT_DIR": K96}
    caps = [line.split("\t") for line in (tmp_path / "sleap_caps.tsv").read_text().splitlines()]
    assert sorted((v.split("_")[0], c, k) for v, c, k in caps) == [
        ("cam01", "000", "96"), ("cam01", "001", "96"), ("cam11", "000", "20"), ("cam11", "001", "20")]


def test_bridge_keeps_one_engine_for_a_uniform_run(tmp_path):
    # An env file from before per-camera caps (no SLEAP_CAP_MODE) or --sleap-max-instances.
    r, out, calls = run_bridge_caps(tmp_path, "SLEAP_MAX_INSTANCES=96 SLEAP_BATCH_SIZE=2")

    assert r.returncode == 0, r.stdout + r.stderr
    assert calls == ["96 2"]
    assert out == {"ENGINES": "96:2:%s" % K96, "EXPORT_DIR": K96}
