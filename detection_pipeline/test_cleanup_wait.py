"""The deigo cleanup must wait only for the chunks saion actually fetches.

bridge.sbatch drops chunks whose SLEAP outputs are already on the bucket and writes the
remaining ones to saion_stage/aruco_worklist.txt; saion fetches only those. cleanup.sbatch
used to wait for every chunk of the wave, so any --only-sleap re-run with bucket-skipped
chunks timed out and kept /flash (2026-10-07: 320 / 611 on 20260928 block01-k96).
"""
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
ALL_ROWS = [("cam01", "002"), ("cam01", "003"), ("cam02", "002"), ("cam02", "003")]


def run_cleanup(tmp_path, sleap_rows, on_saion, bridge_submitted=None, saion_bytes=b"x"):
    """Render templates/cleanup.sbatch for a 4-chunk wave and run it with ssh/rsync stubbed.

    sleap_rows: rows the bridge kept for SLEAP (None = the bridge never wrote its list).
    on_saion: rows whose chunk file is already in saion's input dir.
    bridge_submitted: "before" / "after" the list was written -> jid_bridge.txt mtime
        (None = no jid_bridge.txt). "after" means the list is a previous run's.
    saion_bytes: content of the saion copies (the /flash chunks hold b"x").
    """
    jobs, flash, saion, lib = (tmp_path / d for d in ("jobs", "flash", "saion_input", "lib"))
    for d in (jobs, saion, lib):
        d.mkdir()
    for vname, chunk in ALL_ROWS:
        (flash / vname).mkdir(parents=True, exist_ok=True)
        (flash / vname / f"{vname}_{chunk}.mkv").write_bytes(b"x")
    for vname, chunk in on_saion:
        (saion / f"{vname}_{chunk}.mkv").write_bytes(saion_bytes)
    row = lambda r: f"{r[0]}\t{r[1]}\t96\n"
    (jobs / "aruco_worklist.txt").write_text("".join(row(r) for r in ALL_ROWS))
    if sleap_rows is not None:
        (jobs / "saion_stage").mkdir()
        (jobs / "saion_stage" / "aruco_worklist.txt").write_text("".join(row(r) for r in sleap_rows))
    if bridge_submitted is not None:
        jid, now = jobs / "jid_bridge.txt", time.time()
        jid.write_text("123\n")
        os.utime(jid, (now - 3600, now - 3600) if bridge_submitted == "before" else (now + 3600, now + 3600))
    (lib / "hosts.sh").write_text("ssh_retry() { :; }\nrsync_retry() { :; }\n")
    (jobs / "pipeline.env").write_text(
        f'export FLASH_ROOT="{flash.as_posix()}"\n'
        f'export LIB_DIR="{lib.as_posix()}"\n'
        f'export HPC_LOGS_DIR="{(tmp_path / "hpc_logs").as_posix()}"\n'
        'export OUTPUT_GROUP="grp"\n'
        'export EXP_NAME="exp"\n'
        'export CHUNK_EXT="mkv"\n'
        'export CHUNK_RANGE="2-3"\n'
        f'export CLEANUP_SAION_INPUT_DIR="{saion.as_posix()}"\n'
        'export CHUNK_COPY_TIMEOUT=0\n')
    script = jobs / "cleanup.sbatch"
    script.write_text((ROOT / "templates/cleanup.sbatch").read_text().replace("__JOBS_ROOT__", jobs.as_posix()))
    result = subprocess.run(["bash", script.as_posix()], text=True, capture_output=True, timeout=60,
                            env={**os.environ, "USER": "tester"})
    left = sorted(p.name for p in flash.rglob("*.mkv")) if flash.exists() else []
    return result, left


def test_waits_only_for_the_chunks_the_bridge_kept(tmp_path):
    needed = [("cam01", "002"), ("cam02", "003")]

    result, left = run_cleanup(tmp_path, sleap_rows=needed, on_saion=needed)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "[OK] all 2 / 2 chunks present on saion" in result.stdout
    assert left == []  # the wave's skipped chunks are freed too


def test_a_list_written_by_this_runs_bridge_is_trusted(tmp_path):
    needed = [("cam01", "002"), ("cam02", "003")]

    result, left = run_cleanup(tmp_path, sleap_rows=needed, on_saion=needed, bridge_submitted="before")

    assert result.returncode == 0, result.stdout + result.stderr
    assert left == []


def test_a_previous_runs_list_is_ignored(tmp_path):
    # This run's bridge never rewrote the list (it died early or never ran): an older,
    # smaller list must not let cleanup free chunks saion has not fetched.
    result, left = run_cleanup(tmp_path, sleap_rows=[], on_saion=ALL_ROWS[:2], bridge_submitted="after")

    assert result.returncode == 1
    assert "timeout: 2 / 4" in result.stdout
    assert len(left) == 4


def test_a_same_named_chunk_of_another_size_does_not_count(tmp_path):
    # input/ is shared across runs: a stale copy of a since re-cut chunk is not "fetched".
    needed = [("cam01", "002"), ("cam02", "003")]

    result, left = run_cleanup(tmp_path, sleap_rows=needed, on_saion=needed, saion_bytes=b"stale")

    assert result.returncode == 1
    assert "timeout: 0 / 2" in result.stdout
    assert len(left) == 4


def test_a_missing_needed_chunk_still_keeps_flash(tmp_path):
    needed = [("cam01", "002"), ("cam02", "003")]

    result, left = run_cleanup(tmp_path, sleap_rows=needed, on_saion=needed[:1])

    assert result.returncode == 1
    assert "timeout: 1 / 2" in result.stdout
    assert len(left) == 4


def test_without_the_bridge_list_it_waits_for_the_whole_wave(tmp_path):
    result, left = run_cleanup(tmp_path, sleap_rows=None, on_saion=ALL_ROWS[:2])

    assert result.returncode == 1
    assert "timeout: 2 / 4" in result.stdout
    assert len(left) == 4


def test_whole_wave_on_saion_without_the_bridge_list_frees_flash(tmp_path):
    result, left = run_cleanup(tmp_path, sleap_rows=None, on_saion=ALL_ROWS)

    assert result.returncode == 0, result.stdout + result.stderr
    assert left == []


def test_an_empty_bridge_list_frees_flash_without_waiting(tmp_path):
    result, left = run_cleanup(tmp_path, sleap_rows=[], on_saion=[])

    assert result.returncode == 0, result.stdout + result.stderr
    assert left == []
