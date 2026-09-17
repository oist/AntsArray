"""SLEAP batch startup must not depend on personal conda/mamba hooks."""
import os
from pathlib import Path
import shlex
import subprocess

import pytest


ROOT = Path(__file__).resolve().parent


@pytest.mark.parametrize("filename", [
    "templates/sleap_predict_array.template.sh",
    "scripts/export_sleap_trt.sh",
])
@pytest.mark.parametrize("modules_ready,load_status", [(True, 0), (False, 0), (True, 23)])
def test_batch_startup_ignores_broken_user_hook(tmp_path, filename, modules_ready, load_status):
    source = (ROOT / filename).read_text()
    if filename.startswith("templates/"):
        source = source.split("# Home is shared", 1)[0]
        source = source.replace("__SLEAP_MODULE__", "sleap-nn/0.3.3")
        args = []
    else:
        model = tmp_path / "model"
        model.mkdir()
        args = ["--centroid", str(model), "--instance", str(model),
                "--out", str(tmp_path / "export")]
    script = tmp_path / "job.sh"
    script.write_text(source + '\nprintf "STARTUP_OK\\n"\n')
    setup = """
fake_module() {
    case "$*" in
        'use /apps/unit/ReiterU/.modulefiles') unit_modules_ready=1 ;;
        'load sleap-nn/0.3.3')
            [[ "${unit_modules_ready:-}" == 1 && "$PYTHONNOUSERSITE" == 1 ]] || return 91
            return "$TEST_MODULE_STATUS" ;;
        *) return 92 ;;
    esac
}
source() {
    if [[ "$1" == /etc/profile ]]; then
        module() { fake_module "$@"; }
    else
        printf 'BROKEN_USER_HOOK\\n' >&2
        return 127
    fi
}
sleap-nn() { printf 'INFERENCE_REACHED\\n'; }
export -f fake_module source sleap-nn
"""
    if modules_ready:
        setup += 'module() { fake_module "$@"; }; export -f module\n'
    else:
        setup += 'unset -f module 2>/dev/null || true\n'
    command = setup + "bash " + shlex.join([str(script), *args])
    result = subprocess.run(
        ["bash", "-c", command], text=True, capture_output=True, timeout=10,
        env={**os.environ, "SLURM_JOB_ID": "1", "SLEAP_MODULE": "sleap-nn/0.3.3",
             "TEST_MODULE_STATUS": str(load_status)},
    )
    assert result.returncode == load_status, result.stderr
    assert "BROKEN_USER_HOOK" not in result.stderr
    assert ("STARTUP_OK" in result.stdout) == (load_status == 0)
