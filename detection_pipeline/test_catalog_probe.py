"""Catalog video probe: a split tail block must start where it really starts.

split_block_at_drop.py cuts a recording at a frame drop into block01 (aligned head) and
block02 (the rest) by stream copy, so block02's videos keep the ORIGINAL sidecar, whose
context.startEpochMs is the recording's start; derived.frameOffset says how far in the tail
begins. Ignoring it drew 20260916/block02 on the timeline on top of block01.
"""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "lib"))
from catalog import probe  # noqa: E402

START_MS = 1789566876732          # 2026-09-16 22:54 JST, the original recording start


def probe_with_sidecar(tmp_path, derived=None):
    video = tmp_path / "cam09_cam0_2026-09-16-22-54-36.mkv"
    video.write_bytes(b"")
    side = {"context": {"startEpochMs": START_MS, "fps": 24.0, "status": "closed"},
            "recorder": {"framesEncoded": 4521715}}
    if derived is not None:
        side["derived"] = derived
    Path(str(video) + ".diag.json").write_text(json.dumps(side))
    vn = SimpleNamespace(path=str(video), vname=video.stem, cam_global=9, cam_pc=0,
                         naming_style="new", ext="mkv")
    return probe.probe_video(vn)


def test_a_split_tail_starts_at_its_frame_offset(tmp_path):
    info = probe_with_sidecar(tmp_path, derived={"segment": "tail (-2)", "frameOffset": 11080800})

    assert info.start_epoch_ms == START_MS + 11080800 * 1000 // 24   # 2026-09-22 07:09 JST


def test_an_unsplit_video_keeps_its_recorded_start(tmp_path):
    assert probe_with_sidecar(tmp_path).start_epoch_ms == START_MS
    assert probe_with_sidecar(tmp_path, derived={"tool": "x264 transcode"}).start_epoch_ms == START_MS
