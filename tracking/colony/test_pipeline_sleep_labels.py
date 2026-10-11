"""Check the generated sleep-label job and publication dependencies without Slurm."""
from pathlib import Path
import shlex
import subprocess
import sys


def test_recording_timestamp_override_reaches_mapper_and_requires_one_block(tmp_path):
    root = Path(__file__).resolve().parents[2]
    block = tmp_path / "dataset/block01"
    (block / "data").mkdir(parents=True)
    hmats = tmp_path / "H.npz"
    hmats.touch()
    output = tmp_path / "output"
    old_cache = output / "per_track/TrackID_0001_all_left"
    old_cache.mkdir(parents=True)
    (old_cache / "_SUCCESS").touch()  # Old cross-camera motion must be rebuilt.
    command = ["bash", str(root / "tracking/colony/submit_blocks_pipeline.sh"),
               "--blocks_root", str(block.parent), "--hmats", str(hmats),
               "--output_root", str(output), "--experiment_name", "20260726_104256", "--dry_run"]
    subprocess.run(command, check=True, capture_output=True, text=True)
    mapper = output / "jobs/block01/scripts/map_block01.sbatch"
    assert "--experiment_name 20260726_104256" in mapper.read_text()
    subprocess.run(["bash", "-n", str(mapper)], check=True)
    (block.parent / "block02/data").mkdir(parents=True)
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 2
    assert "requires exactly one block" in result.stderr


def test_motion_labels_follow_motion_cache_and_gate_publication(tmp_path):
    root = Path(__file__).resolve().parents[2]
    block = tmp_path / "dataset" / "block01"
    (block / "data").mkdir(parents=True)
    hmats = tmp_path / "homographies.npz"
    hmats.touch()
    output = tmp_path / "output"
    subprocess.run(
        ["bash", str(root / "tracking/colony/submit_blocks_pipeline.sh"),
         "--blocks_root", str(block.parent), "--block_glob", block.name,
         "--hmats", str(hmats), "--output_root", str(output), "--no_eigenposture", "--dry_run"],
        check=True, capture_output=True, text=True,
    )
    job_root = output / "jobs" / block.name
    scripts = job_root / "scripts"
    for script in scripts.iterdir():
        subprocess.run(["bash", "-n", str(script)], check=True)
    labels = (scripts / "sleep_motion_labels_block01.sbatch").read_text()
    assert "--body-threshold 0.5 --antenna-threshold 0.7 --history-seconds 1" in labels
    assert "analysis/compute_sleep_motion_labels.py" in labels
    submitter = (scripts / "submit_per_track_analysis_block01.sbatch").read_text()
    assert "sleep_motion/jobs/sleep_motion_complete_job_id.txt" in submitter
    assert 'motion_dependency_args=(--dependency "afterok:${motion_complete_id}")' in submitter
    assert '"${motion_dependency_args[@]}"' in submitter
    assert "missing sleep-motion completion job ID or marker" in submitter
    assert submitter.count("--no_conda --worker_python_bin") == 6
    manifest = (job_root / "state/per_track_analysis_markers_block01.tsv").read_text()
    assert "sleep_motion_labels/sleep_motion_labels_complete.ok" in manifest


def test_gpu_batch_startup_does_not_source_personal_shell_configuration():
    root = Path(__file__).resolve().parents[2]
    for relative in ("detection_pipeline/scripts/export_sleap_trt.sh",
                     "detection_pipeline/templates/sleap_predict_array.template.sh"):
        script = root / relative
        text = script.read_text()
        assert "source ~/.bashrc" not in text
        assert "module use /apps/unit/ReiterU/.modulefiles" in text
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_per_track_worker_uses_exact_python_executable(tmp_path):
    root = Path(__file__).resolve().parents[2]
    tracks = tmp_path / "dataset/block01/stitched/per_track"
    tracks.mkdir(parents=True)
    (tracks / "TrackID_0001_all_left.parquet").touch()
    output = tmp_path / "output"
    python = tmp_path / "custom_python"
    python.symlink_to(sys.executable)
    result = subprocess.run(
        ["bash", str(root / "scripts/per_track_slurm_fanout.sh"),
         "--per_track_dir", str(tracks), "--flash_output_dir", str(output),
         "--bucket_output_dir", str(tmp_path / "bucket"),
         "--operation_script", "analysis/compute_track_sleep_motion.py",
         "--operation_name", "sleep_motion", "--run_workdir", str(root),
         "--worker_python_bin", str(python), "--python_bin", sys.executable,
         "--no_conda", "--no_transfer_to_bucket", "--skip_existing", "--dry_run"],
        check=True, capture_output=True, text=True,
    )
    worker_path = next(line.partition(": ")[2] for line in result.stdout.splitlines()
                       if line.startswith("[dry-run] worker script:"))
    worker = Path(worker_path).read_text()
    assert f"operation_cmd={python}" in worker
    assert "conda activate" not in worker
    assert "DONE_MARKER=_SUCCESS_same_camera_v1" in worker
    subprocess.run(["bash", "-n", worker_path], check=True)


def test_custom_scratch_analysis_destinations_follow_source_block(tmp_path):
    """Every generated analysis must accept nonstandard scratch paths."""
    root = Path(__file__).resolve().parents[2]
    block = tmp_path / "input datasets/20260729/block01"
    (block / "data").mkdir(parents=True)
    hmats = tmp_path / "H.npz"
    hmats.touch()
    output = tmp_path / "custom scratch/colony_completion/20260729"
    subprocess.run(
        ["bash", str(root / "tracking/colony/submit_blocks_pipeline.sh"),
         "--blocks_root", str(block.parent), "--hmats", str(hmats),
         "--output_root", str(output), "--python_bin", sys.executable,
         "--no_eigenposture", "--dry_run"],
        check=True, capture_output=True, text=True,
    )
    tracks = output / "block01/stitched/per_track"
    tracks.mkdir(parents=True)
    (tracks / "TrackID_0001_all_left.parquet").touch()
    script = output / "jobs/block01/scripts/submit_per_track_analysis_block01.sbatch"
    text = script.read_text().replace("\\\n", "")
    variables = {
        "grid_occupancy_output_name": "grid_occupancy_histograms",
        "grid_occupancy_operation_args_text": "--grid_size_mm 0.25",
        "sleep_motion_operation_args_text": "--fps 24",
        "speed_operation_args_text": "--max_speed_mm_s 20.0",
        "sleep_operation_args_text": "",
    }
    calls = [line.strip() for line in text.splitlines()
             if line.strip().startswith(f'bash "{root}/scripts/per_track_slurm_fanout.sh"')]
    assert len(calls) == 6
    for call in calls:
        call = call.replace('"${sleep_dependency_args[@]}"', "")
        for key, value in variables.items():
            call = call.replace("${" + key + "}", value)
        args = shlex.split(call)
        name = args[args.index("--output_name") + 1]
        destination = args[args.index("--bucket_output_dir") + 1]
        assert destination == str(block / "stitched" / name)
        result = subprocess.run(args + ["--dry_run"], check=True,
                                capture_output=True, text=True)
        assert "worker script:" in result.stdout
