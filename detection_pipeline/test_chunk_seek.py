"""Wave chunking (-ss seek + stream copy) must start every chunk on its nominal frame."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parent
FPS = 24
GOP = 48                      # keyframe every 2 s
CHUNK_SEC = 4                 # two GOPs per chunk
CHUNK_FRAMES = FPS * CHUNK_SEC
DURATION_SEC = 20             # 480 frames -> 5 chunks


def _has_libx264():
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        return False
    encoders = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    return "libx264" in encoders


pytestmark = pytest.mark.skipif(not _has_libx264(), reason="needs ffmpeg/ffprobe with libx264")


def make_video(path, extra_keyframe_sec=None):
    """Recorder-like MKV: B-frames, closed fixed GOP, no scene cuts, written live (no Cues)."""
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
           "-i", f"testsrc=size=128x96:rate={FPS}:duration={DURATION_SEC}",
           "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-bf", "2",
           "-x264-params", f"keyint={GOP}:min-keyint={GOP}:scenecut=0:open-gop=0:b-adapt=0"]
    if extra_keyframe_sec is not None:
        # Like NVENC's scene-cut IDR: x264 restarts its GOP count at a forced keyframe.
        cmd += ["-force_key_frames", str(extra_keyframe_sec)]
    cmd += ["-f", "matroska", "-live", "1", str(path)]
    subprocess.run(cmd, check=True)


def keyframes(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "packet=pts_time,flags", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    rows = [line.split(",") for line in out.split()]
    return sorted(round(float(pts) * FPS) for pts, flags in rows if "K" in flags)


def frame_md5s(path):
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "framemd5", "-"],
                         capture_output=True, text=True, check=True).stdout
    return [line.split(",")[-1].strip() for line in out.splitlines() if line and not line.startswith("#")]


def run_chunk(tmp_path, video, chunk_range):
    """Render templates/chunk.sbatch against a one-row manifest and run it as array task 0."""
    jobs = tmp_path / "jobs"
    flash = tmp_path / "flash"
    jobs.mkdir()
    n_chunks = -(-DURATION_SEC // CHUNK_SEC)
    (jobs / "manifest.csv").write_text(
        "vname,source_path,ext,fps,frame_count,duration_sec,n_chunks\n"
        f"cam01,{video.as_posix()},.mkv,{FPS},{FPS * DURATION_SEC},{DURATION_SEC},{n_chunks}\n")
    (jobs / "pipeline.env").write_text(
        f'export FLASH_ROOT="{flash.as_posix()}"\n'
        f'export LIB_DIR="{(ROOT / "lib").as_posix()}"\n'
        f'export CHUNK_SEC="{CHUNK_SEC}"\n'
        'export CHUNK_EXT="mkv"\n'
        f'export CHUNK_RANGE="{chunk_range}"\n')
    script = jobs / "chunk.sbatch"
    script.write_text((ROOT / "templates/chunk.sbatch").read_text().replace("__JOBS_ROOT__", jobs.as_posix()))
    result = subprocess.run(["bash", "-c", 'module() { :; }; export -f module; bash "$0"', script.as_posix()],
                            text=True, capture_output=True, timeout=120,
                            env={**os.environ, "SLURM_ARRAY_TASK_ID": "0"})
    return result, flash / "cam01"


def chunk_file(out_dir, idx):
    return out_dir / f"cam01_{idx:03d}.mkv"


def test_wave_chunks_start_on_their_nominal_frame(tmp_path):
    video = tmp_path / "cam01.mkv"
    make_video(video)
    assert 2 * CHUNK_FRAMES in keyframes(video)
    source = frame_md5s(video)

    result, out_dir = run_chunk(tmp_path, video, "2-3")

    assert result.returncode == 0, result.stdout + result.stderr
    for idx in (2, 3):
        start = idx * CHUNK_FRAMES
        assert frame_md5s(chunk_file(out_dir, idx)) == source[start:start + CHUNK_FRAMES], f"chunk {idx}"
    assert not chunk_file(out_dir, 4).exists()


def test_wave_is_refused_when_its_first_boundary_is_not_a_keyframe(tmp_path):
    video = tmp_path / "cam01.mkv"
    make_video(video, extra_keyframe_sec=21 / FPS)   # GOP grid now 21 + 48k
    kf = keyframes(video)
    assert 21 in kf and 2 * CHUNK_FRAMES not in kf

    result, out_dir = run_chunk(tmp_path, video, "2-3")

    assert result.returncode != 0, result.stdout + result.stderr
    assert "keyframe" in result.stderr
    assert not any(chunk_file(out_dir, idx).exists() for idx in (2, 3, 4))


def test_whole_video_run_still_starts_at_frame_zero(tmp_path):
    video = tmp_path / "cam01.mkv"
    make_video(video)
    source = frame_md5s(video)

    result, out_dir = run_chunk(tmp_path, video, "")

    assert result.returncode == 0, result.stdout + result.stderr
    for idx in range(5):
        start = idx * CHUNK_FRAMES
        assert frame_md5s(chunk_file(out_dir, idx)) == source[start:start + CHUNK_FRAMES], f"chunk {idx}"
