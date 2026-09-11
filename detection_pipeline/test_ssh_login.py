"""Remote Slurm commands use login setup without changing transfer SSH."""
import os
from pathlib import Path
import re
import shlex
import subprocess

import pytest


ROOT = Path(__file__).resolve().parent


def run_bash(command, **env):
    return subprocess.run(["bash", "-c", command], text=True, capture_output=True,
                          env={**os.environ, **env}, timeout=30)


@pytest.mark.parametrize("command,expected", [
    ("shopt -q login_shell && printf login", "login"),
    ("printf '%s' \"spaces and 'quotes'\"", "spaces and 'quotes'"),
    ("printf '%s' \"$REMOTE_SLURM_TEST_VALUE\"", "remote value; not shell syntax"),
    ("printf '%s' '$(printf unintended expansion)'", "$(printf unintended expansion)"),
])
def test_retry_wrapper_preserves_remote_command_and_initializes_login(command, expected):
    script = f"source {shlex.quote(str(ROOT / 'lib/hosts.sh'))}\n" + """
ssh_retry() {
    [[ "$#" == 2 && "$1" == saion ]] || return 90
    bash -c "$2"
}
""" + f"ssh_login_retry saion {shlex.quote(command)}"
    result = run_bash(script, REMOTE_SLURM_TEST_VALUE="remote value; not shell syntax")
    assert result.returncode == 0, result.stderr
    assert result.stdout == expected


def test_retry_wrapper_keeps_stdin_and_exit_status():
    script = f"source {shlex.quote(str(ROOT / 'lib/hosts.sh'))}\n" + """
ssh_retry() { bash -c "$2"; }
printf 'stdin payload\n' | ssh_login_retry saion 'read -r line; printf "%s" "$line"; exit 7'
"""
    result = run_bash(script)
    assert result.stdout == "stdin payload"
    assert result.returncode == 7


def test_retry_wrapper_requires_host_and_command():
    result = run_bash(f"source {shlex.quote(str(ROOT / 'lib/hosts.sh'))}; ssh_login_retry saion")
    assert result.returncode == 2
    assert "usage:" in result.stderr


@pytest.mark.parametrize("filename,variable", [("pipeline.sh", "q_saion"), ("pipeline_multi.sh", "Q_SAION")])
def test_live_wave_queue_checks_initialize_remote_login(filename, variable):
    source = (ROOT / filename).read_text()
    assignment = re.search(rf"{variable}=\$\(ssh.*?\) \|\|", source, re.S).group(0).removesuffix(" ||")
    script = """
squeue() { shopt -q login_shell || return 91; printf 12345; }
export -f squeue
ssh() { bash -c "${@: -1}"; }
""" + assignment + f'\nprintf "%s" "${variable}"'
    result = run_bash(script)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "12345"


def test_all_remote_sbatch_submissions_use_login_wrapper():
    source = (ROOT / "templates/bridge.sbatch").read_text()
    calls = [line for line in source.splitlines() if "ssh_" in line and "sbatch " in line]
    assert len(calls) == 4  # prefetch, SLEAP, upload safety net, and cleanup
    assert all("ssh_login_retry saion" in line for line in calls)
    assert 'ssh_retry saion "mkdir' in source  # File operations stay on ordinary SSH.
    assert 'ssh_login_retry saion "SLEAP_MODULE=' in source  # Export invokes srun.


def test_chunk_initializes_unit_module_path_before_loading_ffmpeg():
    prefix = (ROOT / "templates/chunk.sbatch").read_text().split('JOBS_ROOT=', 1)[0]
    script = """
module() {
    case "$*" in
        'use /apps/unit/ReiterU/.modulefiles') unit_modules_ready=1 ;;
        'load ffmpeg/7.1') [[ "${unit_modules_ready:-}" == 1 ]] ;;
        *) return 92 ;;
    esac
}
""" + prefix
    result = run_bash(script)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("filename", ["aruco_array.sbatch", "bridge.sbatch"])
def test_opencv_load_adds_unit_modules(filename):
    source = (ROOT / "templates" / filename).read_text()
    assert re.search(r"module use /apps/unit/ReiterU/\.modulefiles\s+module load opencv/4\.9\.0", source)


@pytest.mark.parametrize("filename", ["templates/sleap_predict_array.template.sh", "scripts/export_sleap_trt.sh"])
def test_sleap_load_adds_unit_modules(filename):
    source = (ROOT / filename).read_text()
    assert re.search(r"module use /apps/unit/ReiterU/\.modulefiles\s+module load .*SLEAP_MODULE", source)
