"""Shared constants: regexes, folder taxonomy, thresholds, column orders, flags.

Single source of truth so classify/discover/qc/build agree on names.
"""
import re

# ---------------------------------------------------------------------------
# Video / file recognition
# ---------------------------------------------------------------------------
VIDEO_EXTS = (".mkv", ".mp4", ".avi")

# New-order grid video:  cam01_cam0_2026-06-24-20-37-18.mkv
#   group1 = global cam index, group2 = per-PC index, group3 = timestamp
NEW_NAME_RE = re.compile(
    r"^cam(\d+)_cam(\d+)_(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})", re.IGNORECASE)

# Legacy-order grid video:  cam3_2025-12-10-10-35-51_cam13.avi
#   group1 = per-PC index, group2 = timestamp, group3 = GLOBAL cam index (trailing)
LEGACY_NAME_RE = re.compile(
    r"^cam(\d+)_(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})_cam(\d+)", re.IGNORECASE)

# Overview camera: global_cam8_2026-06-24-20-37-17.mkv
GLOBAL_NAME_RE = re.compile(
    r"^global_cam(\d+)_(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})", re.IGNORECASE)

# Trailing chunk index on a raw or processed file:  ..._000.avi / ..._012.slp
CHUNK_SUFFIX_RE = re.compile(r"_(\d{3})(?=\.|$)")

RE_ENCODED_STEMS = ("_renc", "_nvenc")

# ---------------------------------------------------------------------------
# Top-level folder taxonomy
# ---------------------------------------------------------------------------
# Auxiliary folders: emit one thin row, do NOT recurse for sessions.
KNOWN_AUX = frozenset({
    "Anouk", "Anouk_ant_archive", "Ant_archive_2025", "ColorMarker_test",
    "QRcodes_test", "cameraArray_calib", "lens_FOV_calculation",
    "nest_lowheight_walls", "single_wide_field_lens_test",
})

# Containers that hold several real sub-sessions: recurse exactly one level.
NESTED_SESSION_CONTAINERS = frozenset({"single_ants", "2025_Sep_no_pertubation"})

# Test/dev folders: session_kind=aux + is_test, still scanned for footprint.
TEST_AUX = frozenset({"pipelineTest", "speedTest", "test_tracking"})

# Never walk into these (repo/env trees living on the share).
WALK_BLACKLIST = frozenset({
    ".git", "node_modules", "mambaforge", "miniconda3", "detectron2",
    "__pycache__", "data_old", ".venv", "venv", "site-packages",
    "AntsArray", "conda-meta",
})
MAX_WALK_DEPTH = 2  # below the session/block root

# ---------------------------------------------------------------------------
# Session-name date patterns
# ---------------------------------------------------------------------------
DATE_RANGE_RE = re.compile(r"^(\d{8})_(\d{8})(?:_(.*))?$")     # 20260414_20260417_CustomAruco
BARE_TS_RE = re.compile(r"^(\d{8})-(\d{6})$")                   # 20251118-121513
DATE_ORD_RE = re.compile(r"^(\d{8})(?:_(\d+))?(?:_(.*))?$")     # 20250321_2_test / 20260420
FUZZY_DATE_RE = re.compile(r"^(\d{4})_([A-Za-z]{3,})(?:_(.*))?$")  # 2025_Sep_no_pertubation

BLOCK_DIR_RE = re.compile(r"^block\d+$", re.IGNORECASE)

# Tokens (lowercased) in a session name that hint at stim/vibration.
STIM_NAME_TOKENS = ("stim", "vibration", "vib", "pertubation", "perturbation")

# ---------------------------------------------------------------------------
# Pipeline data/ artifact suffixes
# ---------------------------------------------------------------------------
SLP_RE = re.compile(r"_(\d{3})\.slp$")
ARUCO_DET_RE = re.compile(r"_(\d{3})_aruco_detections\.h5$")
ARUCO_TRK_RE = re.compile(r"_(\d{3})_aruco_tracks_?\.h5$")   # tolerate trailing underscore
SLEAP_DATA_RE = re.compile(r"_(\d{3})_sleap_data\.h5$")
FRAME_COUNTS_RE = re.compile(r"_frame_counts\.csv$")

# Legacy footprint markers (suppress h5-completeness hazards when present).
LEGACY_FOOTPRINT_RE = re.compile(
    r"(aruco_tracks_\.npy$|_aruco_tracks_\.h5$|_sleap_data\.csv$)")

# Downstream analysis directory names (presence -> stage_reached advances).
DOWNSTREAM_DIRS = ("tracks", "stitched", "interactions", "panorama_pkls",
                   "per_track", "per_track_left", "per_track_right",
                   "predictions", "curation", "xy_speed_sleep_pngs",
                   "event_triggered_tensors", "non-colony-cams")

# Pure-analysis footprint markers (a session with these but no raw video).
PURE_ANALYSIS_DIRS = ("per_track", "per_track_left", "per_track_right",
                      "predictions", "xy_speed_sleep_pngs")

HPC_LOG_DIRNAMES = ("hpc_logs", "hpc_log")

# ---------------------------------------------------------------------------
# Thresholds / defaults
# ---------------------------------------------------------------------------
DEFAULT_EXPECTED_CAMS = 25
DEFAULT_WORKERS = 8
TRUNCATED_H5_BYTES = 2048        # below this a data .h5 is almost certainly truncated
NAME_DATE_TOL_DAYS = 2           # folder-name date vs earliest video date tolerance
SCAN_VERSION = 5                 # bump to invalidate cache on logic change

# ---------------------------------------------------------------------------
# Hazard flag names (kept as constants to avoid typos across modules)
# ---------------------------------------------------------------------------
HZ_SLEAP_H5_MISSING = "SLEAP_H5_MISSING"
HZ_STAGE_SKEW = "STAGE_SKEW"
HZ_ARUCO_MISSING = "ARUCO_MISSING"
HZ_TRUNCATED_ARTIFACT = "TRUNCATED_ARTIFACT"
HZ_SILENT_PARTIAL = "SILENT_PARTIAL"
HZ_DEAD_SYMLINK = "DEAD_SYMLINK"
HZ_NONBLOCK_VIDEO_DIR = "NONBLOCK_VIDEO_DIR"
HZ_CAM_NAMING_LEGACY = "CAM_NAMING_LEGACY"
HZ_PIPELINE_FORMAT_LEGACY = "PIPELINE_FORMAT_LEGACY"
HZ_NAME_DATE_MISMATCH = "NAME_DATE_MISMATCH"
HZ_NO_SESS_FILE = "NO_SESS_FILE"
HZ_NO_SIDECAR = "NO_SIDECAR"
HZ_CHUNK_UNVERIFIABLE = "CHUNK_UNVERIFIABLE"
HZ_CHUNK_INTERNAL_ONLY = "CHUNK_INTERNAL_ONLY"
HZ_RAW_CHUNKED = "RAW_CHUNKED"
HZ_CAM_COUNT_OFF = "CAM_COUNT_OFF"

TOKEN_JOIN = "|"   # separator for multi-valued cells (Excel-scannable)

# ---------------------------------------------------------------------------
# Output column orders
# ---------------------------------------------------------------------------
CATALOG_COLUMNS = [
    "session_id", "block", "block_id", "session_kind", "layout",
    "date_start", "date_end", "date_kind", "labels", "is_test",
    "naming_style", "pipeline_format",
    "is_stim", "stim_source", "stim_format",
    "stim_strength", "stim_duration_s", "stim_interval_s", "stim_trials_cfg",
    "stim_window_min", "stim_seed", "n_trials_observed",
    "n_colony_videos", "n_global", "expected_cams", "missing_cam_ids", "has_sidecars",
    "fps_mode", "frames_median", "duration_median_sec", "health_flag",
    "pipeline_status", "stage_reached", "chunk_sec", "chunk_sec_source",
    "n_slp", "n_aruco_det", "n_aruco_tracks", "n_sleap_data",
    "completeness_pct", "completeness_state", "downstream",
    "sleap_models", "saion_partition", "hazard_flags", "recover_type",
    "recover_missing", "scan_error", "scanned_at",
]

VIDEO_COLUMNS = [
    "session_id", "block", "vname", "cam_global", "cam_pc", "naming_style", "ext",
    "source_path", "sidecar_path", "has_sidecar", "probe_source",
    "fps", "frame_count", "duration_sec", "n_chunks", "start_epoch_ms", "start_offset_sec",
    "status", "clean_close", "frames_emitted", "frames_encoded", "frame_drop",
    "missed_frames", "failed_buffers", "emit_interval_max_ms",
    "assigned_pc", "assigned_drive", "video_health",
]

TRIAL_COLUMNS = [
    "session_id", "block", "trial", "iso_time", "duty", "dur_s", "interval_s",
    "cam_frame_start", "cam_frame_end", "fs_hz", "samples",
    "gyro_rms_dps", "gyro_peak_dps", "acc_rms_g", "acc_peak_g", "temp_mean_C", "imu_ok",
]
