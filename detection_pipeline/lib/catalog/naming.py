"""Parse grid-video filenames across the new and legacy camera-naming orders.

Two orders coexist and can mix within one directory (verified on
single_ants/20251210_1):
  new:    cam01_cam0_2026-06-24-20-37-18.mkv   -> global=1,  pc=0
  legacy: cam3_2025-12-10-10-35-51_cam13.avi   -> global=13, pc=3  (global trails)
  global: global_cam7_2026-06-24-20-37-17.mkv  -> overview camera
"""
import os

from . import const
from .model import VideoName


def _split_ext(name: str):
    """Return (stem, ext_without_dot) handling the .mkv.diag.json double suffix."""
    lower = name.lower()
    if lower.endswith(".diag.json"):
        return name[: -len(".diag.json")], "diag.json"
    stem, ext = os.path.splitext(name)
    return stem, ext.lstrip(".").lower()


def is_video(name: str) -> bool:
    return name.lower().endswith(const.VIDEO_EXTS)


def is_sidecar(name: str) -> bool:
    low = name.lower()
    return low.endswith(".diag.json") or low.endswith(".json")


def parse_video_name(name: str) -> VideoName:
    """Classify a filename. Non-video names return naming_style='unknown'.

    A trailing ``_NNN`` chunk index (raw chunked video, e.g. speedTest) is
    detected and stripped before cam parsing.
    """
    stem, ext = _split_ext(name)

    if not is_video(name):
        return VideoName(vname=stem, cam_global=None, cam_pc=None,
                         naming_style="unknown", ext=ext)

    # Detect + strip a trailing chunk index so cam parsing still works.
    chunk_idx = None
    m = const.CHUNK_SUFFIX_RE.search(stem)
    core = stem
    if m and stem.endswith(m.group(0)):
        chunk_idx = int(m.group(1))
        core = stem[: m.start()]

    gm = const.GLOBAL_NAME_RE.match(core)
    if gm:
        return VideoName(vname=stem, cam_global=None, cam_pc=int(gm.group(1)),
                         naming_style="global", timestamp=gm.group(2),
                         is_global=True, chunk_idx=chunk_idx, ext=ext)

    lm = const.LEGACY_NAME_RE.match(core)
    if lm:
        return VideoName(vname=stem, cam_global=int(lm.group(3)),
                         cam_pc=int(lm.group(1)), naming_style="legacy",
                         timestamp=lm.group(2), chunk_idx=chunk_idx, ext=ext)

    nm = const.NEW_NAME_RE.match(core)
    if nm:
        return VideoName(vname=stem, cam_global=int(nm.group(1)),
                         cam_pc=int(nm.group(2)), naming_style="new",
                         timestamp=nm.group(3), chunk_idx=chunk_idx, ext=ext)

    return VideoName(vname=stem, cam_global=None, cam_pc=None,
                     naming_style="unknown", ext=ext)
