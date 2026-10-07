"""export_sleap_trt.sh: the per-frame instance cap is explicit, recorded, and part of the cache.

sleap-nn bakes --max-instances into the exported centroid graph (default 20). The colony nest
cameras routinely hold 30-90 ants per frame, so a silent 20 truncated every block processed
through the exported path. These tests pin the properties that prevent a repeat.
"""
import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "scripts/export_sleap_trt.sh"

# Fake module system + a fake `sleap-nn export` that behaves like the real one where it matters:
# it records its argv and writes the export metadata carrying the max_instances it was given.
FAKES = """
module() { return 0; }
sleap-nn() {
    printf '%s\\n' "$*" >> "$FAKE_LOG"
    local out="" k=20 b=8
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -o) out="$2"; shift 2 ;;
            --max-instances) k="$2"; shift 2 ;;
            --max-batch-size) b="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    mkdir -p "$out"
    printf '{\\n  "max_instances": %s,\\n  "max_batch_size": %s\\n}\\n' "$k" "$b" > "$out/export_metadata.json"
    cp "$out/export_metadata.json" "$out/model.trt.metadata.json"
    : > "$out/model.trt"
}
export -f module sleap-nn
"""


def run_export(workdir, *extra, env_extra=None, fakes=FAKES):
    model = workdir / "model"
    model.mkdir(exist_ok=True)
    out = workdir / "export"
    log = workdir / "sleap_nn_calls.log"
    args = ["--centroid", str(model), "--instance", str(model), "--out", str(out), *extra]
    env = {**os.environ, "SLURM_JOB_ID": "1", "FAKE_LOG": str(log)}
    env.pop("MAX_INSTANCES", None)
    env.update(env_extra or {})
    result = subprocess.run(
        ["bash", "-c", fakes + "bash " + shlex.join([str(SCRIPT), *args])],
        text=True, capture_output=True, timeout=10, env=env)
    calls = log.read_text().splitlines() if log.exists() else []
    return result, out, calls


def metadata(out):
    return json.loads((out / "export_metadata.json").read_text())


def test_default_cap_is_96_at_batch_2_and_reaches_sleap_nn(tmp_path):
    result, out, calls = run_export(tmp_path)
    assert result.returncode == 0, result.stderr
    assert len(calls) == 1
    assert "--max-instances 96" in calls[0] and "--max-batch-size 2" in calls[0]
    assert metadata(out)["max_instances"] == 96


def test_batch_follows_the_cap_unless_set(tmp_path):
    # Measured on A100 (2026-10-05): cap x batch <= 192 crops of 640x640 builds, 240+ does not.
    for cap, batch in (("20", "8"), ("48", "4"), ("64", "3"), ("128", "1")):
        d = tmp_path / ("k" + cap)
        d.mkdir()
        result, _, calls = run_export(d, "--max-instances", cap)
        assert result.returncode == 0, result.stderr
        assert "--max-batch-size %s" % batch in calls[0], (cap, calls)
    d = tmp_path / "explicit"
    d.mkdir()
    result, _, calls = run_export(d, "--max-instances", "96", env_extra={"MAX_BATCH": "1"})
    assert result.returncode == 0, result.stderr
    assert "--max-batch-size 1" in calls[0]


def test_cap_times_batch_over_the_engine_limit_is_refused(tmp_path):
    result, _, calls = run_export(tmp_path, "--max-instances", "96", env_extra={"MAX_BATCH": "3"})
    assert result.returncode != 0
    assert "192" in result.stderr
    assert calls == []
    result, _, calls = run_export(tmp_path, "--max-instances", "256")   # no batch fits at all
    assert result.returncode != 0
    assert calls == []


def test_cap_is_configurable_by_flag_and_env(tmp_path):
    flag_dir, env_dir = tmp_path / "flag", tmp_path / "env"
    flag_dir.mkdir()
    env_dir.mkdir()
    result, _, calls = run_export(flag_dir, "--max-instances", "64")
    assert result.returncode == 0, result.stderr
    assert "--max-instances 64" in calls[0]
    result, _, calls = run_export(env_dir, env_extra={"MAX_INSTANCES": "96"})
    assert result.returncode == 0, result.stderr
    assert "--max-instances 96" in calls[0]


def test_invalid_cap_is_refused_before_exporting(tmp_path):
    for bad in ("0", "-5", "abc", ""):
        result, _, calls = run_export(tmp_path, "--max-instances", bad)
        assert result.returncode != 0, bad
        assert calls == [], bad


def test_existing_export_with_same_cap_is_reused(tmp_path):
    first, _, _ = run_export(tmp_path, "--max-instances", "128")
    assert first.returncode == 0, first.stderr
    (tmp_path / "sleap_nn_calls.log").unlink()
    again, _, calls = run_export(tmp_path, "--max-instances", "128")
    assert again.returncode == 0, again.stderr
    assert calls == []                      # cached engine, no rebuild


def test_existing_export_with_another_batch_is_refused(tmp_path):
    first, _, _ = run_export(tmp_path, "--max-instances", "96", env_extra={"MAX_BATCH": "1"})
    assert first.returncode == 0, first.stderr
    (tmp_path / "sleap_nn_calls.log").unlink()
    again, _, calls = run_export(tmp_path, "--max-instances", "96")      # default batch 2
    assert again.returncode != 0
    assert "max_batch_size" in again.stderr
    assert calls == []


def test_existing_export_with_another_cap_is_refused(tmp_path):
    out = tmp_path / "export"
    out.mkdir()
    (out / "model.trt").write_bytes(b"")
    (out / "model.trt.metadata.json").write_text('{\n  "max_instances": 20\n}\n')
    result, _, calls = run_export(tmp_path, "--max-instances", "128")
    assert result.returncode != 0
    assert "max_instances" in result.stderr and "20" in result.stderr
    assert calls == []                      # never silently reuse the capped engine


def test_export_whose_metadata_disagrees_fails(tmp_path):
    lying = FAKES.replace('--max-instances) k="$2"', '--max-instances) k=20')
    result, _, _ = run_export(tmp_path, "--max-instances", "128", fakes=lying)
    assert result.returncode != 0
    assert "max_instances" in result.stderr


def _bridge_key_block():
    bridge = (ROOT / "templates/bridge.sbatch").read_text()
    start = bridge.index("SLEAP_MAX_INSTANCES=") if "if [[ -z \"${SLEAP_MAX_INSTANCES" not in bridge \
        else bridge.index("if [[ -z \"${SLEAP_MAX_INSTANCES")
    return bridge[start:bridge.index("engine_dir_for()")]


def _bridge_function(name):
    """The text of one shell function defined in bridge.sbatch (top-level, tab-indented body)."""
    bridge = (ROOT / "templates/bridge.sbatch").read_text()
    start = bridge.index(name + "() {")
    return bridge[start:bridge.index("\n}\n", start) + 3]


def test_bridge_derives_the_batch_for_an_env_without_a_cap():
    # A pipeline.env written before the cap existed exports SLEAP_BATCH_SIZE=8 and no
    # SLEAP_MAX_INSTANCES; re-running its bridge must not ask for cap 96 x batch 8.
    for env, want in (("SLEAP_BATCH_SIZE=8", "96 2"),                         # old env
                      ("SLEAP_MAX_INSTANCES=96; SLEAP_BATCH_SIZE=1", "96 1"),  # new env, explicit
                      ("SLEAP_MAX_INSTANCES=20; SLEAP_BATCH_SIZE=8", "20 8"),
                      ("SLEAP_MAX_INSTANCES=128", "128 1")):
        sh = "%s\n%s\necho \"$SLEAP_MAX_INSTANCES $SLEAP_BATCH_SIZE\"" % (env, _bridge_key_block())
        r = subprocess.run(["bash", "-c", sh], text=True, capture_output=True, timeout=10)
        assert r.stdout.strip() == want, (env, r.stdout, r.stderr)


def test_bridge_engine_cache_key_carries_cap_and_batch():
    sh = ("EXPORT_ROOT=/work/x; SAION_PARTITION=largegpu; SLEAP_MODEL_CENTROID=/m/a.centroid;"
          " SLEAP_MODEL_INSTANCE=/m/a.centered_instance\n%s\nengine_dir_for 96 2; engine_dir_for 20 8"
          % _bridge_function("engine_dir_for"))
    r = subprocess.run(["bash", "-c", sh], text=True, capture_output=True, timeout=10)
    assert r.stdout.split() == ["/work/x/a.centroid__a.centered_instance__k96b2__largegpu",
                                "/work/x/a.centroid__a.centered_instance__k20b8__largegpu"], r.stderr
    # The export of a missing engine is asked for that engine's own cap and batch.
    ensure = _bridge_function("ensure_engine")
    assert "--max-instances '$cap'" in ensure
    assert "MAX_BATCH='$batch'" in ensure
