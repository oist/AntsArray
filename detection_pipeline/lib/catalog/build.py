"""Assemble catalog/video/trial rows for each unit and write the outputs.

Pipeline per unit: footprint scan -> (cache hit? reuse) -> probe videos +
parse sess -> derive QC -> build rows. Emits catalog.csv / videos.csv /
trials.csv / catalog_run.json (+ optional parquet mirror).
"""
import csv
import json
import os
import statistics

from . import (cache as cache_mod, const, discover, footprint as fp_mod,
               labels as labels_mod, probe as probe_mod, provenance, qc,
               recover, sess_parse, viewer)
from .classify import name_hints_stim


# --------------------------------------------------------------------------
# formatting helpers
# --------------------------------------------------------------------------
def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _median(vals):
    vals = [v for v in vals if v is not None]
    return round(statistics.median(vals), 3) if vals else ""


def _mode_fps(vals):
    vals = [round(v, 3) for v in vals if v]
    if not vals:
        return ""
    try:
        return statistics.mode(vals)
    except statistics.StatisticsError:
        return vals[0]


def _naming_style(unit):
    styles = {vn.naming_style for vn in unit.video_names}
    if not styles:
        return ""
    if styles == {"new"}:
        return "new"
    if styles == {"legacy"}:
        return "legacy"
    return "mixed"


def _missing_cam_ids(unit):
    if unit.session_kind != "session" or not unit.video_names:
        return ""
    present = {vn.cam_global for vn in unit.video_names if vn.cam_global is not None}
    missing = [i for i in range(1, const.DEFAULT_EXPECTED_CAMS + 1) if i not in present]
    return const.TOKEN_JOIN.join(str(i) for i in missing)


# --------------------------------------------------------------------------
# row assembly
# --------------------------------------------------------------------------
def _assemble(unit, fp, scanned_at, workers, allow_ffprobe, root, outdir, config_default):
    video_infos = probe_mod.probe_videos(unit.video_names, workers=workers,
                                         allow_ffprobe=allow_ffprobe)
    sess = None
    if unit.sess_paths:
        primary = sorted(unit.sess_paths)[-1]
        sess = sess_parse.parse_sess_file(
            primary, name_stim_hint=name_hints_stim(unit.labels, unit.session_id))

    is_stim = sess.is_stim if sess else (
        True if name_hints_stim(unit.labels, unit.session_id) else None)
    stim_source = sess.stim_source if sess else (
        "foldername" if is_stim else "")
    stim_format = sess.stim_format if sess else "none"
    cam_map = sess.cam_pc_map if sess else {}

    hazards = qc.derive_hazards(unit, fp, video_infos)
    status = qc.pipeline_status(unit, fp)
    health = qc.rollup_health(video_infos)

    vinfo = {vi.vname: (vi.fps or 0, vi.frame_count or 0) for vi in video_infos}
    prov = provenance.read_provenance(unit.path) if fp.has_hpc_logs else {}
    rec = recover.recovery_summary(root, unit.session_id, unit.block, fp, vinfo,
                                   const.HZ_SILENT_PARTIAL in hazards, outdir,
                                   prov, config_default)

    start_epochs = [vi.start_epoch_ms for vi in video_infos if vi.start_epoch_ms]
    earliest = min(start_epochs) if start_epochs else None

    video_rows = []
    for vi in video_infos:
        pc, drive = cam_map.get(vi.cam_global, ("", ""))
        offset = (round((vi.start_epoch_ms - earliest) / 1000.0, 3)
                  if (vi.start_epoch_ms and earliest is not None) else "")
        frame_drop = (vi.frames_emitted - vi.frames_encoded
                      if (vi.frames_emitted is not None
                          and vi.frames_encoded is not None) else "")
        video_rows.append({
            "session_id": unit.session_id, "block": unit.block, "vname": vi.vname,
            "cam_global": _fmt(vi.cam_global), "cam_pc": _fmt(vi.cam_pc),
            "naming_style": vi.naming_style, "ext": vi.ext,
            "source_path": vi.source_path, "sidecar_path": vi.sidecar_path,
            "has_sidecar": _fmt(vi.has_sidecar), "probe_source": vi.probe_source,
            "fps": _fmt(vi.fps), "frame_count": _fmt(vi.frame_count),
            "duration_sec": _fmt(round(vi.duration_sec, 1) if vi.duration_sec else None),
            "n_chunks": _fmt(fp.expected_per_video.get(vi.vname, "") or ""),
            "start_epoch_ms": _fmt(vi.start_epoch_ms), "start_offset_sec": _fmt(offset),
            "status": vi.status, "clean_close": _fmt(vi.clean_close),
            "frames_emitted": _fmt(vi.frames_emitted),
            "frames_encoded": _fmt(vi.frames_encoded), "frame_drop": _fmt(frame_drop),
            "missed_frames": _fmt(vi.missed_frames),
            "failed_buffers": _fmt(vi.failed_buffers),
            "emit_interval_max_ms": _fmt(vi.emit_interval_max_ms),
            "assigned_pc": pc, "assigned_drive": drive,
            "video_health": qc.video_health(vi),
        })

    trial_rows = []
    if sess:
        for t in sess.trials:
            trial_rows.append({
                "session_id": unit.session_id, "block": unit.block,
                "trial": _fmt(t.trial), "iso_time": t.iso_time, "duty": _fmt(t.duty),
                "dur_s": _fmt(t.dur_s), "interval_s": _fmt(t.interval_s),
                "cam_frame_start": _fmt(t.cam_frame_start),
                "cam_frame_end": _fmt(t.cam_frame_end), "fs_hz": _fmt(t.fs_hz),
                "samples": _fmt(t.samples), "gyro_rms_dps": _fmt(t.gyro_rms_dps),
                "gyro_peak_dps": _fmt(t.gyro_peak_dps), "acc_rms_g": _fmt(t.acc_rms_g),
                "acc_peak_g": _fmt(t.acc_peak_g), "temp_mean_C": _fmt(t.temp_mean_C),
                "imu_ok": _fmt(t.imu_ok),
            })

    fps_vals = [vi.fps for vi in video_infos]
    frame_vals = [vi.frame_count for vi in video_infos]
    dur_vals = [vi.duration_sec for vi in video_infos]
    exp_cams = const.DEFAULT_EXPECTED_CAMS if unit.session_kind == "session" else ""

    catalog_row = {
        "session_id": unit.session_id, "block": unit.block,
        "block_id": f"{unit.session_id}/{unit.block or '-'}",
        "session_kind": unit.session_kind, "layout": unit.layout,
        "date_start": unit.date_start, "date_end": unit.date_end,
        "date_kind": unit.date_kind,
        "labels": const.TOKEN_JOIN.join(unit.labels), "is_test": _fmt(unit.is_test),
        "naming_style": _naming_style(unit), "pipeline_format": fp.pipeline_format,
        "is_stim": _fmt(is_stim), "stim_source": stim_source, "stim_format": stim_format,
        "stim_strength": sess.stim_strength if sess else "",
        "stim_duration_s": sess.stim_duration_s if sess else "",
        "stim_interval_s": sess.stim_interval_s if sess else "",
        "stim_trials_cfg": sess.stim_trials_cfg if sess else "",
        "stim_window_min": sess.stim_window_min if sess else "",
        "stim_seed": sess.stim_seed if sess else "",
        "n_trials_observed": _fmt(sess.n_trials if sess else ""),
        "n_colony_videos": len(unit.video_names), "n_global": unit.n_global,
        "expected_cams": _fmt(exp_cams), "missing_cam_ids": _missing_cam_ids(unit),
        "has_sidecars": _fmt(any(vi.has_sidecar for vi in video_infos)),
        "fps_mode": _fmt(_mode_fps(fps_vals)),
        "frames_median": _fmt(_median(frame_vals)),
        "duration_median_sec": _fmt(_median(dur_vals)),
        "health_flag": health, "pipeline_status": status,
        "stage_reached": fp.stage_reached, "chunk_sec": _fmt(fp.chunk_sec),
        "chunk_sec_source": fp.chunk_sec_source,
        "n_slp": fp.n_slp, "n_aruco_det": fp.n_aruco_det,
        "n_aruco_tracks": fp.n_aruco_tracks, "n_sleap_data": fp.n_sleap_data,
        "completeness_pct": _fmt(fp.completeness_pct),
        "completeness_state": fp.completeness_state,
        "downstream": const.TOKEN_JOIN.join(fp.downstream),
        "hazard_flags": const.TOKEN_JOIN.join(hazards),
        "sleap_models": provenance.model_label(prov),
        "sleap_model_centroid": prov.get("sleap_model_centroid", ""),
        "sleap_model_instance": prov.get("sleap_model_instance", ""),
        "saion_partition": prov.get("saion_partition", ""),
        "recover_type": rec["recover_type"], "recover_missing": rec["recover_missing"],
        "recover_cmd": rec["recover_cmd"], "recover_steps": rec["recover_steps"],
        "scan_error": unit.scan_error, "scanned_at": scanned_at,
    }
    return catalog_row, video_rows, trial_rows


# --------------------------------------------------------------------------
# output writers
# --------------------------------------------------------------------------
def _write_csv(path, columns, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})


def _write_run_json(path, scanned_at, root, catalog_rows, ignored):
    kinds, statuses = {}, {}
    for r in catalog_rows:
        kinds[r["session_kind"]] = kinds.get(r["session_kind"], 0) + 1
        statuses[r["pipeline_status"]] = statuses.get(r["pipeline_status"], 0) + 1
    payload = {
        "scanned_at": scanned_at, "root": root, "n_rows": len(catalog_rows),
        "by_session_kind": kinds, "by_pipeline_status": statuses, "ignored": ignored,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_parquet(outdir, catalog_rows, video_rows, trial_rows, log):
    try:
        import pandas as pd  # noqa: WPS433
    except Exception:
        log("[WARN] --parquet: pandas/pyarrow not available; skipping parquet mirror")
        return
    for name, cols, rows in (
        ("catalog", const.CATALOG_COLUMNS, catalog_rows),
        ("videos", const.VIDEO_COLUMNS, video_rows),
        ("trials", const.TRIAL_COLUMNS, trial_rows),
    ):
        pd.DataFrame(rows, columns=cols).to_parquet(
            os.path.join(outdir, name + ".parquet"), index=False)


# --------------------------------------------------------------------------
# top-level run
# --------------------------------------------------------------------------
def run(root, outdir, scanned_at, workers=8, only=None, force=False,
        allow_ffprobe=False, parquet=False, check_sizes=False, mode="all",
        labels_file=None, log=print):
    os.makedirs(outdir, exist_ok=True)
    cache_path = os.path.join(outdir, ".scan_cache.jsonl")
    old_cache = cache_mod.Cache(cache_path)
    old_cache.load()
    config_default = provenance.load_config(outdir)

    catalog_rows, video_rows, trial_rows, ignored = [], [], [], []

    if mode == "build":
        source = old_cache
    else:
        units, ignored = discover.enumerate_units(root, only=only)
        log(f"[scan] discovered {len(units)} units under {root}")
        new_cache = cache_mod.Cache(cache_path)
        new_cache.entries = dict(old_cache.entries)  # carry over untouched blocks
        discovered = set()
        n, hits = len(units), 0
        for i, u in enumerate(units, 1):
            fp = fp_mod.scan_footprint(u, check_sizes=check_sizes)
            key = f"{u.session_id}::{u.block}"
            discovered.add(key)
            fpr = cache_mod.fingerprint(u, fp)
            hit = None if force else old_cache.get(key, fpr)
            if hit:
                crow, vrows, trows = (hit["catalog_row"], hit["video_rows"],
                                      hit["trial_rows"])
                hits += 1
                log(f"[{i}/{n}] cache  {key}")
            else:
                crow, vrows, trows = _assemble(u, fp, scanned_at, workers,
                                               allow_ffprobe, root, outdir, config_default)
                log(f"[{i}/{n}] scan   {key}  kind={crow['session_kind']} "
                    f"status={crow['pipeline_status']} vids={crow['n_colony_videos']} "
                    f"flags={crow['hazard_flags']}")
            new_cache.put(key, fpr, crow, vrows, trows)
        # A full scan prunes blocks no longer on disk; --only leaves them intact.
        if not only:
            for k in list(new_cache.entries):
                if k not in discovered:
                    del new_cache.entries[k]
        new_cache.save()
        log(f"[scan] {hits}/{n} cache hits, {n - hits} freshly scanned")
        source = new_cache

    for rec in source.entries.values():
        catalog_rows.append(rec["catalog_row"])
        video_rows += rec["video_rows"]
        trial_rows += rec["trial_rows"]

    # Per-block label overlay: applied post-cache so edits to the file take
    # effect every run (incl. `build` mode / cache hits) without a rescan.
    labels_path = labels_file or os.path.join(outdir, labels_mod.LABELS_FILENAME)
    block_labels = labels_mod.load_block_labels(labels_path, log=log)
    if block_labels:
        touched = labels_mod.apply_to_rows(catalog_rows, block_labels)
        log("[labels] merged per-block labels into %d/%d rows from %s"
            % (touched, len(catalog_rows), labels_path))

    catalog_rows.sort(key=lambda r: (r["session_id"], r["block"]))
    video_rows.sort(key=lambda r: (r["session_id"], r["block"], r["vname"]))

    if mode in ("build", "all"):
        _write_csv(os.path.join(outdir, "catalog.csv"), const.CATALOG_COLUMNS, catalog_rows)
        _write_csv(os.path.join(outdir, "videos.csv"), const.VIDEO_COLUMNS, video_rows)
        _write_csv(os.path.join(outdir, "trials.csv"), const.TRIAL_COLUMNS, trial_rows)
        _write_run_json(os.path.join(outdir, "catalog_run.json"),
                        scanned_at, root, catalog_rows, ignored)
        viewer.write_html(os.path.join(outdir, "catalog.html"),
                          catalog_rows, video_rows, trial_rows, scanned_at, root)
        if parquet:
            _write_parquet(outdir, catalog_rows, video_rows, trial_rows, log)
        log(f"[write] catalog.csv={len(catalog_rows)} videos.csv={len(video_rows)} "
            f"trials.csv={len(trial_rows)} catalog.html -> {outdir}")

    return catalog_rows, video_rows, trial_rows
