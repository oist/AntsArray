"""Per-video health probe.

Reuses detection_pipeline/lib/manifest.py for sidecar discovery/parsing
(sidecar-first, decode-never by default) and adds recording-health fields.
ffprobe is used only when --allow-ffprobe is set and no sidecar exists.
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import const
from .model import VideoInfo

# Import the sibling manifest.py (detection_pipeline/lib/manifest.py) as-is,
# without turning lib/ into a package.
_LIB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _LIB not in sys.path:
    sys.path.insert(0, _LIB)
import manifest  # noqa: E402


def _num(d, key, cast):
    try:
        if d.get(key) is not None:
            return cast(d[key])
    except (TypeError, ValueError):
        pass
    return None


def probe_video(vn, allow_ffprobe=False, ffprobe_timeout=120, max_probe_sec=180):
    """Probe one VideoName (with .path set) -> VideoInfo."""
    path = Path(vn.path)
    info = VideoInfo(vname=vn.vname, cam_global=vn.cam_global, cam_pc=vn.cam_pc,
                     naming_style=vn.naming_style, ext=vn.ext, source_path=str(path))

    try:
        sc_path = manifest.find_sidecar(path)
    except OSError:
        info.probe_source = "error"
        return info
    if sc_path:
        info.sidecar_path = str(sc_path)
        info.has_sidecar = True
        sc = manifest.read_sidecar(sc_path)
        if sc:
            fps, frames = manifest.from_sidecar(sc)
            ctx = sc.get("context") or {}
            cap = sc.get("capture") or {}
            rec = sc.get("recorder") or {}
            sdk = sc.get("sdk") or {}
            info.probe_source = "sidecar"
            info.fps = fps
            info.frame_count = frames
            info.status = str(ctx.get("status", ""))
            # Prefer the recorder's own cleanClose flag; fall back to status for
            # older sidecars that predate the field.
            cc = ctx.get("cleanClose")
            info.clean_close = (cc if isinstance(cc, bool)
                                else str(ctx.get("status", "")).lower() == "closed")
            info.start_epoch_ms = _num(ctx, "startEpochMs", int)
            info.frames_emitted = _num(cap, "framesEmitted", int)
            info.frames_encoded = _num(rec, "framesEncoded", int)
            info.missed_frames = _num(sdk, "Statistic_Missed_Frame_Count", int)
            info.failed_buffers = _num(sdk, "Statistic_Failed_Buffer_Count", int)
            info.emit_interval_max_ms = _num(cap, "emitIntervalMaxMs", float)
            dur_ms = _num(ctx, "durationMs", float)
            if fps and frames:
                info.duration_sec = frames / fps
            elif dur_ms:
                info.duration_sec = dur_ms / 1000.0
            return info

    # No usable sidecar.
    if allow_ffprobe and path.suffix.lower() in const.VIDEO_EXTS:
        try:
            fps, frames, dur = manifest.ffprobe_fps_and_frames(
                path, max_probe_sec, ffprobe_timeout)
            info.probe_source = "ffprobe"
            info.fps = fps or None
            info.frame_count = frames or None
            info.duration_sec = dur or (frames / fps if fps else None)
        except Exception:
            info.probe_source = "none"
    else:
        info.probe_source = "none"
    return info


def probe_videos(video_names, workers=8, allow_ffprobe=False):
    """Probe a list of VideoName in parallel -> list[VideoInfo] (input order)."""
    if not video_names:
        return []
    results = [None] * len(video_names)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(probe_video, vn, allow_ffprobe): i
                for i, vn in enumerate(video_names)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                results[i] = fut.result()
            except Exception:
                vn = video_names[i]
                results[i] = VideoInfo(
                    vname=vn.vname, cam_global=vn.cam_global, cam_pc=vn.cam_pc,
                    naming_style=vn.naming_style, ext=vn.ext,
                    source_path=vn.path, probe_source="error")
    return results
