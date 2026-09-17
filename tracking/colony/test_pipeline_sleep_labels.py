"""Check the generated sleep-label job and publication dependencies without Slurm."""
from pathlib import Path
import subprocess
import sys


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
         "--hmats", str(hmats), "--output_root", str(output), "--dry_run"],
        check=True, capture_output=True, text=True,
    )
    job_root = output / "jobs" / block.name
    scripts = job_root / "scripts"
    for script in scripts.iterdir():
        subprocess.run(["bash", "-n", str(script)], check=True)
    labels = (scripts / "sleep_motion_labels_block01.sbatch").read_text()
    assert "--body-threshold 0.5 --antenna-threshold 0.7 --history-seconds 10" in labels
    assert "analysis/compute_sleep_motion_labels.py" in labels
    submitter = (scripts / "submit_per_track_analysis_block01.sbatch").read_text()
    assert "sleep_motion/jobs/sleep_motion_complete_job_id.txt" in submitter
    assert 'motion_dependency_args=(--dependency "afterok:${motion_complete_id}")' in submitter
    assert '"${motion_dependency_args[@]}"' in submitter
    assert "missing sleep-motion completion job ID or marker" in submitter
    assert submitter.count("--no_conda --worker_python_bin") == 5
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
         "--no_conda", "--no_transfer_to_bucket", "--dry_run"],
        check=True, capture_output=True, text=True,
    )
    worker_path = next(line.partition(": ")[2] for line in result.stdout.splitlines()
                       if line.startswith("[dry-run] worker script:"))
    worker = Path(worker_path).read_text()
    assert f"operation_cmd={python}" in worker
    assert "conda activate" not in worker
    subprocess.run(["bash", "-n", worker_path], check=True)
