"""lib/engine_crops.sh: the exported engine's crop budget follows the instance model's crop size.

The centered-instance model runs on cap x batch crops at once, and TensorRT cannot build the
engine above a limit that shrinks as the crop grows (192 crops at 640, 140 at 768). A pipeline
that kept the 640 limit for a 768 model would ask for engines that fail after a long build.
"""
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent
LIB = ROOT / "lib/engine_crops.sh"

# The shape sleap-nn writes: crop_size sits under data_config.preprocessing.
CONFIG = """data_config:
  preprocessing:
    max_height: 3036
    crop_size: {crop}
    min_crop_size: 100
model_config:
  head_configs:
    centered_instance:
      confmaps:
        anchor_part: 'aruco '
"""


def lib_call(call):
    return subprocess.run(["bash", "-c", "source %s; %s" % (shlex.quote(LIB.as_posix()), call)],
                          text=True, capture_output=True, timeout=10)


def model_dir(tmp_path, crop):
    d = tmp_path / ("model_%s" % crop)
    d.mkdir()
    (d / "training_config.yaml").write_text(CONFIG.format(crop=crop))
    return d


def test_crop_size_is_read_from_the_training_config(tmp_path):
    for crop in ("640", "768"):
        r = lib_call("instance_crop_size %s" % shlex.quote(model_dir(tmp_path, crop).as_posix()))

        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == crop


def test_a_model_without_a_fixed_crop_is_refused(tmp_path):
    r = lib_call("instance_crop_size %s" % shlex.quote(model_dir(tmp_path, "null").as_posix()))

    assert r.returncode != 0
    assert r.stdout == ""


def test_a_model_without_a_training_config_is_refused(tmp_path):
    r = lib_call("instance_crop_size %s" % shlex.quote(tmp_path.as_posix()))

    assert r.returncode != 0
    assert r.stdout == ""


def test_budget_is_the_measured_limit_for_each_crop():
    # A100, 2026-10-05 (640: 20x8 .. 128x1 build, 240+ fail) and 2026-10-13 (768: 96x1 and
    # 20x7 build, 160 and 192 fail).
    for crop, crops in (("640", "192"), ("768", "140")):
        r = lib_call("max_engine_crops %s" % crop)

        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == crops


def test_an_unmeasured_crop_has_no_budget():
    for crop in ("704", "896", "", "abc"):
        r = lib_call("max_engine_crops %s" % shlex.quote(crop))

        assert r.returncode != 0, crop
        assert r.stdout == "", crop


def test_a_trailing_yaml_comment_is_allowed(tmp_path):
    d = model_dir(tmp_path, "768  # retrained for far antenna tips")

    r = lib_call("instance_crop_size %s" % shlex.quote(d.as_posix()))

    assert r.stdout.strip() == "768"


def run_pipeline_caps(tmp_path, model, env=""):
    """Run pipeline.sh's instance-cap and batch section (no cluster calls in it)."""
    pipeline = (ROOT / "pipeline.sh").read_text()
    section = pipeline[pipeline.index("EXPORTED_PATH=0\n"):pipeline.index("\n# Roots.")]
    table = tmp_path / "nest_cams.tsv"
    table.write_text("2026-07-01 00:00\tcam01\tA\n")
    sh = "\n".join([
        "SKIP_TRT_EXPORT=0 SLEAP_RUNTIME=tensorrt ONLY_CHUNK=0 ONLY_ARUCO=0 ONLY_BACKUP=0",
        "SLEAP_MAX_INSTANCES= SLEAP_BATCH_SIZE= SLEAP_NEST_CAP=96 SLEAP_OTHER_CAP=20 SLEAP_ANCHOR_FILL=1",
        "SLEAP_MODULE=sleap-nn/0.3.3",
        "LIB_DIR=%s NEST_CAMS_TABLE=%s SLEAP_MODEL_INSTANCE=%s" % tuple(
            shlex.quote(p.as_posix()) for p in (ROOT / "lib", table, model)),
        env,
        section,
        'echo "BUDGET=$MAX_ENGINE_CROPS BATCH=$SLEAP_BATCH_SIZE MODE=$SLEAP_CAP_MODE FILL=$SLEAP_ANCHOR_FILL"'])
    return subprocess.run(["bash", "-c", sh], text=True, capture_output=True, timeout=10)


def test_pipeline_takes_the_budget_and_batches_from_a_768_model(tmp_path):
    r = run_pipeline_caps(tmp_path, model_dir(tmp_path, "768"))

    assert r.returncode == 0, r.stderr
    assert "BUDGET=140 BATCH=1 MODE=nest" in r.stdout
    assert "nest cameras 96 (batch 1), others 20 (batch 7)" in r.stdout


def test_pipeline_keeps_the_640_batches_for_a_640_model(tmp_path):
    r = run_pipeline_caps(tmp_path, model_dir(tmp_path, "640"))

    assert r.returncode == 0, r.stderr
    assert "BUDGET=192 BATCH=2 MODE=nest" in r.stdout
    assert "nest cameras 96 (batch 2), others 20 (batch 8)" in r.stdout


def test_pipeline_refuses_an_unmeasured_crop_and_an_oversized_batch(tmp_path):
    r = run_pipeline_caps(tmp_path, model_dir(tmp_path, "704"))
    assert r.returncode == 2
    assert "704" in r.stderr

    r = run_pipeline_caps(tmp_path, model_dir(tmp_path, "768"), "SLEAP_MAX_INSTANCES=20 SLEAP_BATCH_SIZE=8")
    assert r.returncode == 2
    assert "140" in r.stderr


def test_runs_without_a_sleap_leg_need_no_model_config(tmp_path):
    for env in ("ONLY_ARUCO=1", "ONLY_BACKUP=1", "SKIP_TRT_EXPORT=1"):
        r = run_pipeline_caps(tmp_path, tmp_path / "no_such_model", env)

        assert r.returncode == 0, (env, r.stderr)
        assert "BUDGET=192" in r.stdout, env


def test_pipeline_hands_the_budget_to_the_bridge():
    pipeline = (ROOT / "pipeline.sh").read_text()

    assert 'export SLEAP_ENGINE_CROPS="$MAX_ENGINE_CROPS"' in pipeline


def test_the_tag_node_fallback_is_on_by_default_and_only_on_the_exported_path(tmp_path):
    model = model_dir(tmp_path, "768")
    for env, want in (("", "FILL=1"), ("SLEAP_ANCHOR_FILL=0", "FILL=0"),
                      ("SKIP_TRT_EXPORT=1", "FILL=0"), ("SLEAP_RUNTIME=pytorch", "FILL=0")):
        r = run_pipeline_caps(tmp_path, model, env)

        assert r.returncode == 0, (env, r.stderr)
        assert want in r.stdout, (env, r.stdout)
    # The fallback patches sleap-nn 0.3 export code: an older module is refused up front.
    r = run_pipeline_caps(tmp_path, model, "SLEAP_MODULE=sleap-nn/0.2.0")
    assert r.returncode == 2
    assert "--no-sleap-anchor-fill" in r.stderr
    r = run_pipeline_caps(tmp_path, model, "SLEAP_MODULE=sleap-nn/0.2.0 SLEAP_ANCHOR_FILL=0")
    assert r.returncode == 0, r.stderr
    pipeline = (ROOT / "pipeline.sh").read_text()
    assert "\nSLEAP_ANCHOR_FILL=1\n" in pipeline
    assert "--no-sleap-anchor-fill) SLEAP_ANCHOR_FILL=0" in pipeline
    assert 'export SLEAP_ANCHOR_FILL="$SLEAP_ANCHOR_FILL"' in pipeline
