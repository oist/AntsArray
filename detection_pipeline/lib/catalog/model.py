"""Plain data-holder classes passed between scan stages.

Written without dataclasses / f-string-only features so the catalog runs under
the login-node's system python3 (3.6), matching manifest.py's portability.
Output ROWS are plain dicts keyed by the column lists in const.py.
"""


class VideoName(object):
    """Result of parsing a grid-video filename."""

    def __init__(self, vname="", cam_global=None, cam_pc=None,
                 naming_style="unknown", timestamp="", is_global=False,
                 chunk_idx=None, ext="", path=""):
        self.vname = vname
        self.cam_global = cam_global      # colony camera index (1..25); None if unparseable
        self.cam_pc = cam_pc              # per-PC/secondary index
        self.naming_style = naming_style  # "new" | "legacy" | "global" | "unknown"
        self.timestamp = timestamp        # raw YYYY-MM-DD-HH-MM-SS token
        self.is_global = is_global
        self.chunk_idx = chunk_idx        # set only for chunked raw video (speedTest)
        self.ext = ext
        self.path = path                  # absolute path, filled by discovery layer


class VideoInfo(object):
    """Per-video health, emitted as one videos.csv row."""

    def __init__(self, vname="", cam_global=None, cam_pc=None, naming_style="",
                 ext="", source_path=""):
        self.vname = vname
        self.cam_global = cam_global
        self.cam_pc = cam_pc
        self.naming_style = naming_style
        self.ext = ext
        self.source_path = source_path
        self.sidecar_path = ""
        self.has_sidecar = False
        self.probe_source = ""            # "sidecar" | "ffprobe" | "none" | "error"
        self.fps = None
        self.frame_count = None
        self.duration_sec = None
        self.start_epoch_ms = None
        self.status = ""                  # sidecar context.status
        self.clean_close = None
        self.frames_emitted = None
        self.frames_encoded = None
        self.missed_frames = None
        self.failed_buffers = None
        self.emit_interval_max_ms = None


class StimTrial(object):
    """One CSV_PULSE row -> one trials.csv row."""

    def __init__(self):
        self.trial = None
        self.iso_time = ""
        self.duty = None
        self.dur_s = None
        self.interval_s = None
        self.cam_frame_start = None
        self.cam_frame_end = None
        self.fs_hz = None
        self.samples = None
        self.gyro_rms_dps = None
        self.gyro_peak_dps = None
        self.acc_rms_g = None
        self.acc_peak_g = None
        self.temp_mean_C = None
        self.imu_ok = None


class SessDoc(object):
    """Parsed session/stim file (sess_*.txt or legacy stim_timing.txt)."""

    def __init__(self, path=""):
        self.path = path
        self.stim_format = "none"   # "sess_v1" | "arduino_serial_v0" | "none" | "unknown"
        self.is_stim = None
        self.stim_source = ""       # "sessfile" | "foldername" | ""
        self.opened = ""
        self.closed = ""
        self.clean_stop = None
        self.stim_strength = ""     # raw, e.g. uni:0.250:1.000
        self.stim_duration_s = ""   # raw, e.g. fix:5.000
        self.stim_interval_s = ""
        self.stim_trials_cfg = ""
        self.stim_window_min = ""
        self.stim_seed = ""
        self.cam_pc_map = {}        # cam_global -> (pc_label, drive_letter)
        self.trials = []            # list[StimTrial]
        self.warnings = []
        self.parse_error = ""

    @property
    def n_trials(self):
        return len(self.trials)


class Footprint(object):
    """detection_pipeline output footprint of one block's data/ dir."""

    def __init__(self):
        self.has_data_dir = False
        self.pipeline_format = "none"   # "h5_4tuple" | "legacy" | "none"
        self.n_slp = 0
        self.n_aruco_det = 0
        self.n_aruco_tracks = 0
        self.n_sleap_data = 0
        self.expected_per_video = {}    # vname -> deepest-stage chunk count
        self.expected_total = 0
        self.chunk_sec = None
        self.chunk_sec_source = "none"  # "frame_counts" | "manifest" | "none"
        self.completeness_pct = None
        self.completeness_state = "n/a"  # "verified" | "internal" | "unverifiable" | "n/a"
        self.has_hpc_logs = False
        self.hpc_log_stages = []
        self.downstream = []
        self.stage_reached = "none"
        self.has_legacy_markers = False
        self.truncated_files = 0
        self.file_count = 0             # files in data/ (for cache fingerprint)
        self.chunk_sets = {}            # {"slp"/"det"/"trk"/"sdat": {vname: set(int idx)}}


class Unit(object):
    """A discovered catalog row skeleton (one block, flat session, or aux entry)."""

    def __init__(self, session_id="", block="", path="", session_kind="session",
                 layout="flat", is_test=False):
        self.session_id = session_id
        self.block = block               # "" for flat/implicit block
        self.path = path                 # absolute path of the block/session dir
        self.session_kind = session_kind  # session | aux | pure_analysis | unknown
        self.layout = layout             # flat | block
        self.is_test = is_test
        self.date_start = ""
        self.date_end = ""
        self.date_kind = ""              # single | range | timestamp | fuzzy | none
        self.labels = []
        self.video_names = []            # list[VideoName] (colony)
        self.n_global = 0
        self.dead_symlinks = 0
        self.sess_paths = []
        self.subdir_names = []           # block-dir immediate subdirs
        self.has_data_dir = False
        self.data_dir = ""               # abspath of data/ if present
        self.extra_hazards = []          # hazards found during discovery
        self.scan_error = ""
