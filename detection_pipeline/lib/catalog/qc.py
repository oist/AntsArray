"""Derive QC/hazard flags and health rollups from discovery + footprint + probe.

All signals are cheap (counts, presence, sidecar fields already read). Encodes
the known basler pipeline failure modes as explicit, filterable flags.
"""
import datetime as _dt

from . import const


def _parse_date(s):
    """Parse 'YYYY-MM-DD' -> date (3.6-safe; date.fromisoformat is 3.7+)."""
    return _dt.datetime.strptime(s, "%Y-%m-%d").date()


def _is_unclean(vi) -> bool:
    """Recorder never wrote a clean-close record for this video."""
    if vi.status and vi.status.lower() != "closed":
        return True
    return vi.clean_close is False


def _counts_complete(vi) -> bool:
    """Both independent frame counters are present and agree.

    framesEmitted (capture side) vs framesEncoded (recorder side). Agreement is
    evidence the recording reached its intended end -- but both are last-heartbeat
    values, so it is NOT proof the container was finalized or that every frame
    landed as a decodable cluster. Treated as 'verify', never 'verified'.
    """
    return (vi.frames_emitted is not None and vi.frames_encoded is not None
            and vi.frames_emitted == vi.frames_encoded)


def video_health(vi) -> str:
    """ok | warn | bad | unknown for one VideoInfo.

    An unclean close is only 'bad' when we cannot corroborate that the recording
    finished: if both frame counters are present and agree (and no buffers
    failed), the footage is almost certainly complete and only container
    finalization is missing -> 'warn' (remux/verify before use).
    """
    if not vi.has_sidecar:
        return "unknown"

    bad = warn = False
    if vi.failed_buffers:
        bad = True
    if (vi.frames_emitted is not None and vi.frames_encoded is not None
            and vi.frames_emitted != vi.frames_encoded):
        warn = True

    if _is_unclean(vi):
        if _counts_complete(vi) and not vi.failed_buffers:
            warn = True   # complete capture, unfinalized container
        else:
            bad = True    # cannot verify how much footage landed

    if vi.missed_frames:
        warn = True

    if bad:
        return "bad"
    if warn:
        return "warn"
    return "ok"


def rollup_health(video_infos) -> str:
    if not video_infos:
        return "n/a"
    healths = [video_health(v) for v in video_infos]
    if all(h == "unknown" for h in healths):
        return "unknown"
    if any(h == "bad" for h in healths):
        return "bad"
    if any(h == "warn" for h in healths):
        return "warn"
    return "ok"


def _name_date_mismatch(unit) -> bool:
    if unit.date_kind != "single" or not unit.date_start:
        return False
    try:
        base = _parse_date(unit.date_start)
    except ValueError:
        return False
    for vn in unit.video_names:
        ts = vn.timestamp
        if not ts:
            continue
        try:
            vd = _parse_date(ts[:10])
        except ValueError:
            continue
        if abs((vd - base).days) > const.NAME_DATE_TOL_DAYS:
            return True
    return False


def pipeline_status(unit, fp) -> str:
    if unit.session_kind == "pure_analysis":
        return "analysis_only"
    if not fp.has_data_dir and not fp.has_hpc_logs:
        return "not_started"
    if fp.pipeline_format == "h5_4tuple":
        counts = [fp.n_slp, fp.n_aruco_det, fp.n_aruco_tracks, fp.n_sleap_data]
        if all(c > 0 for c in counts) and len(set(counts)) == 1:
            return "complete"
    return "partial"


def derive_hazards(unit, fp, video_infos) -> list:
    """Return the ordered, de-duplicated hazard flag list for one unit."""
    flags = list(unit.extra_hazards)  # from discovery (dead symlink, legacy naming, ...)
    n_slp, n_det, n_sdat = fp.n_slp, fp.n_aruco_det, fp.n_sleap_data

    if fp.pipeline_format == "legacy":
        flags.append(const.HZ_PIPELINE_FORMAT_LEGACY)

    if fp.pipeline_format == "h5_4tuple":
        if n_slp > 0 and n_sdat == 0:
            flags.append(const.HZ_SLEAP_H5_MISSING)
        if n_slp > 0 and n_det > 0 and n_det != n_slp:
            flags.append(const.HZ_STAGE_SKEW)
        if n_slp > 0 and n_det < n_slp:
            flags.append(const.HZ_ARUCO_MISSING)
        if fp.completeness_state == "internal":
            flags.append(const.HZ_CHUNK_INTERNAL_ONLY)
        elif fp.completeness_state == "unverifiable":
            flags.append(const.HZ_CHUNK_UNVERIFIABLE)

    if fp.truncated_files:
        flags.append(const.HZ_TRUNCATED_ARTIFACT)

    # Recorder died before finalizing, but both frame counters agree: footage
    # almost certainly complete, container not finalized -> remux/verify.
    if any(v.has_sidecar and _is_unclean(v) and _counts_complete(v)
           for v in video_infos):
        flags.append(const.HZ_UNCLEAN_CLOSE)

    # gpu25-style: compute outputs exist but no HPC logs were uploaded.
    if fp.has_data_dir and (n_slp or n_det) and not fp.has_hpc_logs:
        flags.append(const.HZ_SILENT_PARTIAL)

    if (unit.session_kind == "session" and video_infos
            and not any(v.has_sidecar for v in video_infos)):
        flags.append(const.HZ_NO_SIDECAR)

    if unit.session_kind == "session" and unit.video_names:
        if len(unit.video_names) != const.DEFAULT_EXPECTED_CAMS:
            flags.append(const.HZ_CAM_COUNT_OFF)

    if _name_date_mismatch(unit):
        flags.append(const.HZ_NAME_DATE_MISMATCH)

    seen, out = set(), []
    for f in flags:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out
