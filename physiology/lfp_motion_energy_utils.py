"""Utilities for aligning SpikeGLX LFP with camera motion energy."""

from __future__ import annotations

import ast
import json
import mmap
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt

try:
    from ephys_functions import GainCorrectIM, SampRate, makeMemMapRaw, readMeta
except Exception as exc:
    print(f"Could not import SpikeGLX helpers from ephys_functions ({type(exc).__name__}: {exc}); using DemoReadSGLXData")
    from DemoReadSGLXData.readSGLX import GainCorrectIM, SampRate, makeMemMapRaw, readMeta


@dataclass(frozen=True)
class CameraSyncSettings:
    channel: int = 1
    threshold_mv: float = 3000.0
    min_separation_samples: int = 85
    force_redetect: bool = False


@dataclass(frozen=True)
class NidqData:
    bin_path: Path
    meta: dict[str, str]
    raw: np.memmap
    sample_rate: float
    n_channels: int
    n_samples: int

    @property
    def first_abs_seconds(self) -> float:
        return float(self.meta.get("firstSample", 0)) / self.sample_rate


@dataclass(frozen=True)
class LfpData:
    bin_path: Path
    meta: dict[str, str]
    raw: np.memmap
    sample_rate: float
    n_saved_channels: int
    n_lfp_channels: int
    n_samples: int

    @property
    def first_abs_seconds(self) -> float:
        return float(self.meta.get("firstSample", 0)) / self.sample_rate

    @property
    def duration_seconds(self) -> float:
        return self.n_samples / self.sample_rate


@dataclass(frozen=True)
class LfpLoadResult:
    t_video_s: np.ndarray
    lfp_uv: np.ndarray
    sample_rate_hz: float


@dataclass(frozen=True)
class TraceResult:
    t_video_s: np.ndarray
    lfp_trace: np.ndarray
    motion_energy: np.ndarray
    sample_rate_hz: float


@dataclass(frozen=True)
class LfpMotionEnergyCacheData:
    path: Path
    channels: np.ndarray
    t_video_s: np.ndarray
    lfp_uv: np.ndarray
    motion_energy: np.ndarray
    sample_rate_hz: float
    attrs: dict[str, object]


@dataclass(frozen=True)
class KilosortSpikes:
    path: Path
    sample_rate_hz: float
    dat_path: Path | None
    sorted_binary_first_abs_seconds: float | None
    sorted_binary_duration_seconds: float | None
    sorted_binary_start_video_seconds: float | None
    sorted_binary_stop_video_seconds: float | None
    video_abs_start_seconds: float | None
    spike_times_samples: np.ndarray
    spike_clusters: np.ndarray
    spike_video_seconds: np.ndarray
    unit_ids: np.ndarray
    unit_labels: dict[int, str]
    params: dict[str, object]


@dataclass(frozen=True)
class SpikeRateResult:
    unit_ids: np.ndarray
    unit_labels: list[str]
    t_video_s: np.ndarray
    spike_rate_hz: np.ndarray
    spike_counts: np.ndarray
    bin_edges_s: np.ndarray
    smoothing_sigma_s: float


@dataclass(frozen=True)
class UnitLeadLagResult:
    unit_ids: np.ndarray
    unit_labels: list[str]
    lags_s: np.ndarray
    correlations: np.ndarray
    zero_lag_r: np.ndarray
    best_lead_s: np.ndarray
    best_lead_r: np.ndarray


@dataclass
class PhysiologyRecording:
    experiment_path: Path
    nidq: NidqData
    lfp: LfpData
    me_path: Path
    motion_energy: np.ndarray
    me_attrs: dict[str, object]
    frame_abs_seconds: np.ndarray
    frame_video_seconds: np.ndarray

    def sample_motion_energy(self, t_video_s: np.ndarray) -> np.ndarray:
        return np.interp(
            t_video_s,
            self.frame_video_seconds,
            self.motion_energy.astype(np.float64, copy=False),
            left=np.nan,
            right=np.nan,
        ).astype(np.float32)

    def apply_motion_energy_artifact_filter(
        self,
        *,
        method: str = "peak_regular",
        diff_height: float | None = None,
        period_frames: int = 250,
        before_frames: int = 5,
        after_frames: int = 15,
    ) -> np.ndarray:
        cleaned, artifact_frames = clean_motion_energy_regular_artifacts(
            self.motion_energy,
            method=method,
            diff_height=diff_height,
            period_frames=period_frames,
            before_frames=before_frames,
            after_frames=after_frames,
        )
        self.motion_energy = cleaned.astype(np.float32, copy=False)
        self.me_attrs = dict(self.me_attrs)
        self.me_attrs["artifact_filter_method"] = method
        self.me_attrs["artifact_filter_n_frames"] = int(len(artifact_frames))
        return artifact_frames

    def lfp_downsample_step(self, target_rate_hz: float) -> int:
        return max(1, int(round(float(self.lfp.sample_rate) / float(target_rate_hz))))

    def lfp_downsampled_rate(self, target_rate_hz: float) -> float:
        return float(self.lfp.sample_rate) / float(self.lfp_downsample_step(target_rate_hz))

    def lfp_video_overlap(self) -> tuple[float, float]:
        video_abs_start = float(self.frame_abs_seconds[0])
        video_stop_s = float(self.frame_video_seconds[-1])
        lfp_start_s = max(0.0, float(self.lfp.first_abs_seconds - video_abs_start))
        lfp_stop_s = min(
            video_stop_s,
            float(self.lfp.first_abs_seconds + self.lfp.duration_seconds - video_abs_start),
        )
        if lfp_stop_s <= lfp_start_s:
            raise ValueError(
                "No LFP/video overlap: "
                f"video starts at {video_abs_start:.3f}s abs, "
                f"LFP is {self.lfp.first_abs_seconds:.3f}-"
                f"{self.lfp.first_abs_seconds + self.lfp.duration_seconds:.3f}s abs"
            )
        return lfp_start_s, lfp_stop_s

    def load_lfp_downsampled(
        self,
        start_s: float,
        duration_s: float,
        channels: np.ndarray,
        *,
        target_rate_hz: float = 1000.0,
        read_chunk_seconds: float = 10.0,
        read_mode: str = "auto",
    ) -> LfpLoadResult:
        """Load LFP near target_rate_hz.

        ``read_mode='auto'`` uses direct memmap slicing for contiguous channel
        ranges, and time-contiguous chunks for one channel or non-contiguous
        channel lists, which are slow as strided memmap reads.
        """
        channels = validate_lfp_channels(channels, self.lfp.n_lfp_channels)
        step = self.lfp_downsample_step(target_rate_hz)
        sample_rate_hz = self.lfp_downsampled_rate(target_rate_hz)

        start_abs_s = float(self.frame_abs_seconds[0]) + float(start_s)
        stop_abs_s = start_abs_s + float(duration_s)
        first_sample = int(np.ceil((start_abs_s - self.lfp.first_abs_seconds) * self.lfp.sample_rate))
        last_sample = int(np.ceil((stop_abs_s - self.lfp.first_abs_seconds) * self.lfp.sample_rate))
        first_sample = max(0, first_sample)
        last_sample = min(self.lfp.n_samples, last_sample)

        if last_sample <= first_sample:
            return LfpLoadResult(
                np.empty(0, dtype=np.float64),
                np.empty((len(channels), 0), dtype=np.float32),
                sample_rate_hz,
            )

        first_downsampled_sample = first_sample + ((step - (first_sample % step)) % step)
        if read_mode not in {"auto", "direct", "sequential"}:
            raise ValueError("read_mode must be 'auto', 'direct', or 'sequential'")
        if read_mode == "auto":
            read_mode = "direct"

        if read_mode == "direct":
            native_samples = np.arange(first_downsampled_sample, last_sample, step, dtype=np.int64)
            if native_samples.size == 0:
                return LfpLoadResult(
                    np.empty(0, dtype=np.float64),
                    np.empty((len(channels), 0), dtype=np.float32),
                    sample_rate_hz,
                )
            sample_slice = slice(first_downsampled_sample, last_sample, step)
            raw_counts = direct_channel_read(self.lfp.raw, channels, sample_slice, self.lfp.n_lfp_channels)
            t_video_s = (
                self.lfp.first_abs_seconds
                + native_samples.astype(np.float64) / float(self.lfp.sample_rate)
                - float(self.frame_abs_seconds[0])
            )
            return LfpLoadResult(t_video_s, gain_correct_lfp_uv(raw_counts, channels, self.lfp.meta), sample_rate_hz)

        chunk_samples = max(step, int(round(float(read_chunk_seconds) * self.lfp.sample_rate)))
        t_parts: list[np.ndarray] = []
        data_parts: list[np.ndarray] = []
        for chunk_start in range(first_sample, last_sample, chunk_samples):
            chunk_stop = min(last_sample, chunk_start + chunk_samples)
            first_keep = chunk_start + ((step - (chunk_start % step)) % step)
            if first_keep >= chunk_stop:
                continue

            native_samples = np.arange(first_keep, chunk_stop, step, dtype=np.int64)
            local_cols = native_samples - chunk_start
            raw_chunk = np.array(self.lfp.raw[:, chunk_start:chunk_stop], copy=True)
            selected_counts = raw_chunk[channels][:, local_cols]
            data_parts.append(gain_correct_lfp_uv(selected_counts, channels, self.lfp.meta))
            t_parts.append(
                self.lfp.first_abs_seconds
                + native_samples.astype(np.float64) / float(self.lfp.sample_rate)
                - float(self.frame_abs_seconds[0])
            )

        if not t_parts:
            return LfpLoadResult(
                np.empty(0, dtype=np.float64),
                np.empty((len(channels), 0), dtype=np.float32),
                sample_rate_hz,
            )

        return LfpLoadResult(
            np.concatenate(t_parts),
            np.concatenate(data_parts, axis=1),
            sample_rate_hz,
        )


def find_one(path: Path, pattern: str) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    matches = sorted(path.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No {pattern!r} file found under {path}")
    if len(matches) > 1:
        examples = "\n  ".join(str(match) for match in matches[:10])
        raise ValueError(f"Multiple {pattern!r} files found; pass one explicitly:\n  {examples}")
    return matches[0]


def open_nidq(path: Path) -> NidqData:
    bin_path = find_one(path, "*.nidq.bin")
    meta = readMeta(bin_path)
    n_channels = int(meta["nSavedChans"])
    n_samples = int(int(meta["fileSizeBytes"]) / (2 * n_channels))
    sample_rate = SampRate(meta)
    raw = np.memmap(bin_path, dtype="int16", mode="r", shape=(n_channels, n_samples), order="F")
    return NidqData(bin_path, meta, raw, sample_rate, n_channels, n_samples)


def open_lfp(path: Path) -> LfpData:
    bin_path = find_one(path, "*.lf.bin")
    meta = readMeta(bin_path)
    raw = makeMemMapRaw(bin_path, meta)
    try:
        raw._mmap.madvise(mmap.MADV_RANDOM)
    except (AttributeError, ValueError):
        pass
    sample_rate = SampRate(meta)
    n_saved_channels = int(meta["nSavedChans"])
    ap_count, lf_count, sync_count = (int(value) for value in meta["snsApLfSy"].split(","))
    if ap_count != 0:
        raise ValueError(f"Expected an LF-only imec file, got snsApLfSy={meta['snsApLfSy']}")
    n_lfp_channels = min(lf_count, n_saved_channels - sync_count)
    return LfpData(bin_path, meta, raw, sample_rate, n_saved_channels, n_lfp_channels, raw.shape[1])


def load_motion_energy(path: Path) -> tuple[Path, np.ndarray, dict[str, object]]:
    me_path = find_one(path, "*.me")
    with h5py.File(me_path, "r") as h5:
        motion_energy = h5["motion_energy"][:].astype(np.float32, copy=False)
        attrs = {key: value for key, value in h5.attrs.items()}
    return me_path, motion_energy, attrs


def load_recording(
    experiment_path: Path,
    *,
    camera: CameraSyncSettings,
) -> PhysiologyRecording:
    experiment_path = Path(experiment_path)
    nidq = open_nidq(experiment_path)
    lfp = open_lfp(experiment_path)
    me_path, motion_energy, me_attrs = load_motion_energy(experiment_path)
    frame_abs_seconds, frame_video_seconds = get_camera_frame_times(
        experiment_path,
        nidq,
        camera=camera,
    )

    if len(frame_video_seconds) != len(motion_energy):
        n = min(len(frame_video_seconds), len(motion_energy))
        print(
            f"WARNING: camera TTL count ({len(frame_video_seconds):,}) != motion energy length "
            f"({len(motion_energy):,}); trimming both to {n:,}"
        )
        frame_abs_seconds = frame_abs_seconds[:n]
        frame_video_seconds = frame_video_seconds[:n]
        motion_energy = motion_energy[:n]

    return PhysiologyRecording(
        experiment_path,
        nidq,
        lfp,
        me_path,
        motion_energy,
        me_attrs,
        frame_abs_seconds,
        frame_video_seconds,
    )


def print_recording_summary(recording: PhysiologyRecording) -> None:
    print(f"nidq: {recording.nidq.bin_path}")
    print(f"lfp:  {recording.lfp.bin_path}")
    print(f"me:   {recording.me_path}")
    print(f"nidq rate={recording.nidq.sample_rate:.3f} Hz; lfp rate={recording.lfp.sample_rate:.3f} Hz")
    print(
        f"frames={len(recording.frame_video_seconds):,}; "
        f"video duration={recording.frame_video_seconds[-1] / 60:.2f} min"
    )
    print(
        f"lfp channels={recording.lfp.n_lfp_channels}; "
        f"lfp duration={recording.lfp.duration_seconds / 60:.2f} min"
    )


def resolve_kilosort_folder(path: Path) -> Path:
    path = Path(path)
    if path.is_dir() and (path / "spike_times.npy").exists():
        return path
    if path.is_file() and path.name == "spike_times.npy":
        return path.parent
    if path.is_dir():
        matches = sorted(folder for folder in path.rglob("*") if folder.is_dir() and (folder / "spike_times.npy").exists())
        if not matches:
            raise FileNotFoundError(f"No Kilosort folder with spike_times.npy found under {path}")
        if len(matches) > 1:
            examples = "\n  ".join(str(match) for match in matches[:10])
            raise ValueError(f"Multiple Kilosort folders found; pass one explicitly:\n  {examples}")
        return matches[0]
    raise FileNotFoundError(f"Kilosort path does not exist: {path}")


def parse_phy_params(params_path: Path) -> dict[str, object]:
    params: dict[str, object] = {}
    if not Path(params_path).exists():
        return params

    tree = ast.parse(Path(params_path).read_text(encoding="utf-8"), filename=str(params_path))
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            params[target.id] = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            continue
    return params


def first_path(value: object, base_path: Path) -> Path | None:
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        candidate = Path(value)
    elif isinstance(value, (list, tuple)) and value:
        candidate = Path(value[0])
    else:
        return None
    if not candidate.is_absolute():
        candidate = base_path / candidate
    return candidate


def kilosort_dat_path(kilosort_path: Path, params: dict[str, object]) -> Path | None:
    path = first_path(params.get("dat_path"), Path(kilosort_path))
    if path is not None:
        return path

    ops_path = Path(kilosort_path) / "ops.npy"
    if not ops_path.exists():
        return None
    try:
        ops = np.load(ops_path, allow_pickle=True).item()
    except Exception:
        return None
    for key in ("filename", "data_file_path"):
        path = first_path(ops.get(key), Path(kilosort_path))
        if path is not None:
            return path
    settings = ops.get("settings")
    if isinstance(settings, dict):
        for key in ("filename", "data_file_path"):
            path = first_path(settings.get(key), Path(kilosort_path))
            if path is not None:
                return path
    return None


def load_cluster_labels(kilosort_path: Path) -> dict[int, str]:
    kilosort_path = Path(kilosort_path)
    for name in ("cluster_group.tsv", "cluster_KSLabel.tsv"):
        path = kilosort_path / name
        if not path.exists():
            continue

        labels: dict[int, str] = {}
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines:
            continue
        header = lines[0].split("\t")
        try:
            cluster_col = header.index("cluster_id")
        except ValueError:
            cluster_col = 0
        label_col = 1 if len(header) > 1 else 0
        for line in lines[1:]:
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) <= max(cluster_col, label_col):
                continue
            labels[int(parts[cluster_col])] = parts[label_col]
        return labels
    return {}


def subset_source_from_name(dat_path: Path) -> tuple[Path | None, int | None]:
    marker = ".ks_"
    if marker not in dat_path.name or not dat_path.name.endswith(".bin"):
        return None, None
    prefix, rest = dat_path.name.split(marker, 1)
    parts = rest.split("_")
    if len(parts) < 2:
        return None, None
    try:
        start_sample = int(parts[0])
    except ValueError:
        return None, None
    return dat_path.with_name(prefix + ".bin"), start_sample


def infer_sorted_binary_first_abs_seconds(dat_path: Path | None, sample_rate_hz: float) -> float | None:
    if dat_path is None:
        return None
    dat_path = Path(dat_path)
    sidecar_path = dat_path.with_suffix(dat_path.suffix + ".json")
    if sidecar_path.exists():
        with sidecar_path.open("r", encoding="utf-8") as stream:
            sidecar = json.load(stream)
        source_bin = Path(str(sidecar.get("source_bin", dat_path)))
        start_sample = int(sidecar.get("start_sample", 0))
        source_meta = readMeta(source_bin)
        source_rate = float(SampRate(source_meta))
        return float(source_meta.get("firstSample", 0)) / source_rate + start_sample / source_rate

    meta_path = dat_path.with_suffix(".meta")
    if meta_path.exists():
        meta = readMeta(dat_path)
        source_rate = float(SampRate(meta))
        return float(meta.get("firstSample", 0)) / source_rate

    source_bin, start_sample = subset_source_from_name(dat_path)
    if source_bin is not None and source_bin.with_suffix(".meta").exists():
        meta = readMeta(source_bin)
        source_rate = float(SampRate(meta))
        return float(meta.get("firstSample", 0)) / source_rate + int(start_sample or 0) / source_rate

    return None


def infer_sorted_binary_duration_seconds(
    dat_path: Path | None,
    params: dict[str, object],
    sample_rate_hz: float,
) -> float | None:
    if dat_path is None or not Path(dat_path).exists():
        return None
    n_channels = params.get("n_channels_dat", params.get("n_chan_bin"))
    if n_channels is None:
        return None
    try:
        dtype = np.dtype(str(params.get("dtype", "int16")))
        offset = int(params.get("offset", 0))
        frame_bytes = int(n_channels) * dtype.itemsize
    except (TypeError, ValueError):
        return None
    if frame_bytes <= 0:
        return None
    data_bytes = max(0, Path(dat_path).stat().st_size - offset)
    return float(data_bytes // frame_bytes) / float(sample_rate_hz)


def label_filter_set(labels: str | list[str] | tuple[str, ...] | set[str] | None) -> set[str] | None:
    if labels is None:
        return None
    if isinstance(labels, str):
        if labels.lower() == "all":
            return None
        return {labels}
    return {str(label) for label in labels}


def load_kilosort_spikes(
    kilosort_path: Path,
    *,
    recording: PhysiologyRecording | None = None,
    include_labels: str | list[str] | tuple[str, ...] | set[str] | None = None,
    exclude_labels: str | list[str] | tuple[str, ...] | set[str] | None = None,
) -> KilosortSpikes:
    kilosort_path = resolve_kilosort_folder(kilosort_path)
    spike_times = np.load(kilosort_path / "spike_times.npy").reshape(-1).astype(np.int64, copy=False)
    spike_clusters = np.load(kilosort_path / "spike_clusters.npy").reshape(-1).astype(np.int64, copy=False)
    if spike_times.shape[0] != spike_clusters.shape[0]:
        raise ValueError("spike_times.npy and spike_clusters.npy have different lengths")

    params = parse_phy_params(kilosort_path / "params.py")
    sample_rate_hz = float(params.get("sample_rate", np.nan))
    if not np.isfinite(sample_rate_hz):
        ops_path = kilosort_path / "ops.npy"
        if ops_path.exists():
            ops = np.load(ops_path, allow_pickle=True).item()
            sample_rate_hz = float(ops.get("fs", np.nan))
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError(f"Could not infer Kilosort sample rate from {kilosort_path}")

    dat_path = kilosort_dat_path(kilosort_path, params)
    sorted_first_abs_s = infer_sorted_binary_first_abs_seconds(dat_path, sample_rate_hz)
    if sorted_first_abs_s is None and recording is not None:
        sorted_first_abs_s = recording.lfp.first_abs_seconds
    sorted_duration_s = infer_sorted_binary_duration_seconds(dat_path, params, sample_rate_hz)

    labels = load_cluster_labels(kilosort_path)
    unit_ids = np.unique(spike_clusters).astype(np.int64, copy=False)
    include = label_filter_set(include_labels)
    exclude = label_filter_set(exclude_labels)
    if include is not None:
        unit_ids = np.asarray([unit for unit in unit_ids if labels.get(int(unit), "") in include], dtype=np.int64)
    if exclude is not None:
        unit_ids = np.asarray([unit for unit in unit_ids if labels.get(int(unit), "") not in exclude], dtype=np.int64)

    if include is not None or exclude is not None:
        keep_spikes = np.isin(spike_clusters, unit_ids)
        spike_times = spike_times[keep_spikes]
        spike_clusters = spike_clusters[keep_spikes]

    video_abs_start = None if recording is None else float(recording.frame_abs_seconds[0])
    spike_seconds = spike_times.astype(np.float64) / sample_rate_hz
    if sorted_first_abs_s is not None and video_abs_start is not None:
        sorted_start_video_s = sorted_first_abs_s - video_abs_start
        sorted_stop_video_s = None if sorted_duration_s is None else sorted_start_video_s + sorted_duration_s
        spike_video_seconds = (sorted_first_abs_s - video_abs_start) + spike_seconds
    else:
        sorted_start_video_s = 0.0 if sorted_duration_s is not None else None
        sorted_stop_video_s = sorted_duration_s
        spike_video_seconds = spike_seconds

    return KilosortSpikes(
        path=kilosort_path,
        sample_rate_hz=sample_rate_hz,
        dat_path=dat_path,
        sorted_binary_first_abs_seconds=sorted_first_abs_s,
        sorted_binary_duration_seconds=sorted_duration_s,
        sorted_binary_start_video_seconds=sorted_start_video_s,
        sorted_binary_stop_video_seconds=sorted_stop_video_s,
        video_abs_start_seconds=video_abs_start,
        spike_times_samples=spike_times,
        spike_clusters=spike_clusters,
        spike_video_seconds=spike_video_seconds.astype(np.float64, copy=False),
        unit_ids=unit_ids,
        unit_labels={int(unit): labels.get(int(unit), "") for unit in unit_ids},
        params=params,
    )


def time_bin_edges_from_centers(t_seconds: np.ndarray) -> np.ndarray:
    t_seconds = np.asarray(t_seconds, dtype=np.float64).reshape(-1)
    if t_seconds.size == 0:
        raise ValueError("t_seconds must be non-empty")
    if t_seconds.size > 1 and np.any(np.diff(t_seconds) <= 0):
        raise ValueError("t_seconds must be strictly increasing")
    if t_seconds.size == 1:
        return np.asarray([t_seconds[0] - 0.5, t_seconds[0] + 0.5], dtype=np.float64)
    midpoints = 0.5 * (t_seconds[:-1] + t_seconds[1:])
    edges = np.empty(t_seconds.size + 1, dtype=np.float64)
    edges[1:-1] = midpoints
    edges[0] = t_seconds[0] - (midpoints[0] - t_seconds[0])
    edges[-1] = t_seconds[-1] + (t_seconds[-1] - midpoints[-1])
    return edges


def gaussian_smooth_rows(values: np.ndarray, sigma_samples: float) -> np.ndarray:
    if sigma_samples <= 0:
        return values.astype(np.float32, copy=False)
    radius = max(1, int(np.ceil(4.0 * float(sigma_samples))))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / float(sigma_samples)) ** 2)
    kernel /= kernel.sum()
    smoothed = np.empty_like(values, dtype=np.float32)
    for row in range(values.shape[0]):
        padded = np.pad(values[row].astype(np.float64, copy=False), radius, mode="edge")
        smoothed[row] = np.convolve(padded, kernel, mode="valid").astype(np.float32, copy=False)
    return smoothed


def gaussian_smooth_1d(values: np.ndarray, sigma_samples: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if sigma_samples <= 0:
        return values.astype(np.float32, copy=False)
    finite = np.isfinite(values)
    if finite.sum() == 0:
        return np.full(values.shape, np.nan, dtype=np.float32)
    filled = values.copy()
    if not finite.all():
        idx = np.arange(values.size, dtype=np.float64)
        filled[~finite] = np.interp(idx[~finite], idx[finite], values[finite])
    smoothed = gaussian_smooth_rows(filled[np.newaxis, :], sigma_samples)[0]
    smoothed[~finite] = np.nan
    return smoothed


def map_spike_rates_to_timebase(
    spikes: KilosortSpikes,
    t_video_s: np.ndarray,
    *,
    unit_ids: np.ndarray | list[int] | None = None,
    smoothing_sigma_s: float = 0.0,
) -> SpikeRateResult:
    t_video_s = np.asarray(t_video_s, dtype=np.float64).reshape(-1)
    if unit_ids is None:
        unit_ids = spikes.unit_ids
    unit_ids = np.asarray(unit_ids, dtype=np.int64).reshape(-1)
    if t_video_s.size == 0:
        empty_rates = np.empty((unit_ids.size, 0), dtype=np.float32)
        return SpikeRateResult(
            unit_ids,
            [spikes.unit_labels.get(int(unit), "") for unit in unit_ids],
            t_video_s,
            empty_rates,
            np.empty((unit_ids.size, 0), dtype=np.uint16),
            np.empty(0, dtype=np.float64),
            float(smoothing_sigma_s),
        )

    bin_edges = time_bin_edges_from_centers(t_video_s)
    bin_widths = np.diff(bin_edges).astype(np.float32, copy=False)
    counts = np.empty((unit_ids.size, t_video_s.size), dtype=np.uint16)
    for row, unit_id in enumerate(unit_ids):
        unit_spikes = spikes.spike_video_seconds[spikes.spike_clusters == int(unit_id)]
        counts[row] = np.histogram(unit_spikes, bins=bin_edges)[0].astype(np.uint16, copy=False)

    rates = counts.astype(np.float32, copy=False) / bin_widths[np.newaxis, :]
    if smoothing_sigma_s > 0 and t_video_s.size > 1:
        median_dt = float(np.nanmedian(np.diff(t_video_s)))
        sigma_samples = float(smoothing_sigma_s) / median_dt
        if sigma_samples > 0:
            rates = gaussian_smooth_rows(rates, sigma_samples)

    return SpikeRateResult(
        unit_ids=unit_ids,
        unit_labels=[spikes.unit_labels.get(int(unit), "") for unit in unit_ids],
        t_video_s=t_video_s,
        spike_rate_hz=rates,
        spike_counts=counts,
        bin_edges_s=bin_edges,
        smoothing_sigma_s=float(smoothing_sigma_s),
    )


def decimate_spike_rate_result(rate_result: SpikeRateResult, step: int) -> SpikeRateResult:
    step = max(1, int(step))
    t_video_s = rate_result.t_video_s[::step]
    return SpikeRateResult(
        unit_ids=rate_result.unit_ids,
        unit_labels=rate_result.unit_labels,
        t_video_s=t_video_s,
        spike_rate_hz=rate_result.spike_rate_hz[:, ::step].astype(np.float32, copy=False),
        spike_counts=rate_result.spike_counts[:, ::step],
        bin_edges_s=time_bin_edges_from_centers(t_video_s) if t_video_s.size else np.empty(0, dtype=np.float64),
        smoothing_sigma_s=rate_result.smoothing_sigma_s,
    )


def _valid_mask_or_all(valid_mask: np.ndarray | None, n_samples: int) -> np.ndarray:
    if valid_mask is None:
        return np.ones(n_samples, dtype=bool)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    if valid.shape[0] != n_samples:
        raise ValueError("valid_mask length must match trace length")
    return valid


def correlate_unit_rates_to_trace(
    rate_result: SpikeRateResult,
    trace: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    trace = np.asarray(trace, dtype=np.float64).reshape(-1)
    if trace.shape[0] != rate_result.spike_rate_hz.shape[1]:
        raise ValueError("trace length must match spike-rate timebase")
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if valid_mask.shape[0] != trace.shape[0]:
            raise ValueError("valid_mask length must match trace length")
        trace = trace.copy()
        trace[~valid_mask] = np.nan
    correlations = np.full(rate_result.spike_rate_hz.shape[0], np.nan, dtype=np.float32)
    for row in range(rate_result.spike_rate_hz.shape[0]):
        rate = rate_result.spike_rate_hz[row]
        if valid_mask is not None:
            rate = rate.copy()
            rate[~valid_mask] = np.nan
        correlations[row], _valid = pearsonr_valid(rate, trace)
    return correlations


def unit_trace_lag_correlations(
    rate_result: SpikeRateResult,
    trace: np.ndarray,
    *,
    sample_rate_hz: float,
    max_lag_seconds: float,
    lag_step_seconds: float,
    valid_mask: np.ndarray | None = None,
    min_lead_seconds: float = 0.0,
) -> UnitLeadLagResult:
    """Correlate each unit with a trace over lags.

    Positive ``lags_s`` mean unit firing leads the trace by that many seconds:
    ``rate(t)`` is compared to ``trace(t + lag)``.
    """
    trace = np.asarray(trace, dtype=np.float64).reshape(-1)
    if trace.shape[0] != rate_result.spike_rate_hz.shape[1]:
        raise ValueError("trace length must match spike-rate timebase")
    valid = _valid_mask_or_all(valid_mask, trace.size)

    max_lag_samples = max(0, int(round(float(max_lag_seconds) * float(sample_rate_hz))))
    step_samples = max(1, int(round(float(lag_step_seconds) * float(sample_rate_hz))))
    positive_lags = np.arange(0, max_lag_samples + 1, step_samples, dtype=np.int64)
    lag_samples = np.r_[-positive_lags[:0:-1], positive_lags].astype(np.int64, copy=False)
    lags_s = lag_samples.astype(np.float64) / float(sample_rate_hz)

    correlations = np.full((rate_result.spike_rate_hz.shape[0], lag_samples.size), np.nan, dtype=np.float32)
    for lag_idx, lag in enumerate(lag_samples):
        if lag > 0:
            trace_slice = slice(int(lag), None)
            rate_slice = slice(0, -int(lag))
            lag_valid = valid[rate_slice] & valid[trace_slice]
        elif lag < 0:
            trace_slice = slice(0, int(lag))
            rate_slice = slice(-int(lag), None)
            lag_valid = valid[rate_slice] & valid[trace_slice]
        else:
            trace_slice = slice(None)
            rate_slice = slice(None)
            lag_valid = valid

        y = trace[trace_slice].copy()
        y[~lag_valid] = np.nan
        for row in range(rate_result.spike_rate_hz.shape[0]):
            x = rate_result.spike_rate_hz[row, rate_slice].astype(np.float64, copy=True)
            x[~lag_valid] = np.nan
            correlations[row, lag_idx], _valid = pearsonr_valid(x, y)

    zero_idx = int(np.flatnonzero(lag_samples == 0)[0])
    zero_lag_r = correlations[:, zero_idx].astype(np.float32, copy=True)
    lead_mask = lags_s >= float(min_lead_seconds)
    lead_mask &= lag_samples > 0
    best_lead_s = np.full(rate_result.spike_rate_hz.shape[0], np.nan, dtype=np.float32)
    best_lead_r = np.full(rate_result.spike_rate_hz.shape[0], np.nan, dtype=np.float32)
    if lead_mask.any():
        lead_corr = correlations[:, lead_mask]
        lead_lags = lags_s[lead_mask]
        finite = np.isfinite(lead_corr)
        for row in range(lead_corr.shape[0]):
            if finite[row].any():
                idx = int(np.nanargmax(lead_corr[row]))
                best_lead_s[row] = float(lead_lags[idx])
                best_lead_r[row] = float(lead_corr[row, idx])

    return UnitLeadLagResult(
        unit_ids=rate_result.unit_ids,
        unit_labels=rate_result.unit_labels,
        lags_s=lags_s,
        correlations=correlations,
        zero_lag_r=zero_lag_r,
        best_lead_s=best_lead_s,
        best_lead_r=best_lead_r,
    )


def future_trace_change(
    trace: np.ndarray,
    *,
    sample_rate_hz: float,
    lead_seconds: float,
    change_window_seconds: float,
    smooth_seconds: float = 0.0,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Trace increase after a future delay.

    For each sample ``t``, returns ``trace(t + lead + window) - trace(t + lead)``.
    This is useful for asking whether current unit rate predicts a later rise
    in LFP magnitude or motion energy.
    """
    trace = np.asarray(trace, dtype=np.float64).reshape(-1)
    valid = _valid_mask_or_all(valid_mask, trace.size) & np.isfinite(trace)
    if smooth_seconds > 0:
        trace_eval = gaussian_smooth_1d(trace, float(smooth_seconds) * float(sample_rate_hz)).astype(np.float64)
    else:
        trace_eval = trace.astype(np.float64, copy=True)

    lead_samples = max(0, int(round(float(lead_seconds) * float(sample_rate_hz))))
    window_samples = max(1, int(round(float(change_window_seconds) * float(sample_rate_hz))))
    change = np.full(trace.shape, np.nan, dtype=np.float32)
    change_valid = np.zeros(trace.shape, dtype=bool)
    stop = trace.size - lead_samples - window_samples
    if stop > 0:
        start_idx = np.arange(stop, dtype=np.int64) + lead_samples
        stop_idx = start_idx + window_samples
        change[:stop] = (trace_eval[stop_idx] - trace_eval[start_idx]).astype(np.float32, copy=False)
        change_valid[:stop] = valid[:stop] & valid[start_idx] & valid[stop_idx]
        change[~change_valid] = np.nan
    return change, change_valid


def channel_label(channels: np.ndarray) -> str:
    channels = np.asarray(channels, dtype=np.int64)
    if channels.size == 1:
        return f"ch{int(channels[0]):03d}"
    if channels.size > 1:
        diffs = np.diff(channels)
        if np.all(diffs == diffs[0]):
            step = int(diffs[0])
            if step == 1:
                return f"ch{int(channels[0]):03d}-{int(channels[-1]):03d}"
            return f"ch{int(channels[0]):03d}-{int(channels[-1]):03d}x{step}"
    return f"ch{int(channels[0]):03d}-n{channels.size}"


def default_lfp_motion_energy_cache_path(
    recording: PhysiologyRecording,
    channels: np.ndarray,
    *,
    target_rate_hz: float = 250.0,
) -> Path:
    channels = validate_lfp_channels(channels, recording.lfp.n_lfp_channels)
    return (
        recording.experiment_path
        / f"lfp_me_cache_{float(target_rate_hz):g}hz_{channel_label(channels)}.h5"
    )


def lfp_motion_energy_cache_matches(
    cache_path: Path,
    channels: np.ndarray,
    *,
    sample_rate_hz: float,
    start_s: float | None = None,
    stop_s: float | None = None,
) -> bool:
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return False

    try:
        with h5py.File(cache_path, "r") as h5:
            if not {"channels", "t_video_s", "lfp_uv", "motion_energy"}.issubset(h5.keys()):
                return False

            cache_rate_hz = float(h5.attrs.get("sample_rate_hz", np.nan))
            if not np.isfinite(cache_rate_hz) or not np.isclose(cache_rate_hz, sample_rate_hz, rtol=0, atol=1e-6):
                return False

            cached_channels = h5["channels"][:].astype(np.int64, copy=False)
            requested_channels = np.asarray(channels, dtype=np.int64)
            if not np.isin(requested_channels, cached_channels).all():
                return False

            if start_s is not None or stop_s is not None:
                if h5["t_video_s"].shape[0] == 0:
                    return False
                cache_start_s = float(h5.attrs.get("start_video_s", h5["t_video_s"][0]))
                cache_stop_s = float(h5.attrs.get("stop_video_s", h5["t_video_s"][-1]))
                tolerance_s = max(1.0 / sample_rate_hz, 1e-6)
                if start_s is not None and float(start_s) < cache_start_s - tolerance_s:
                    return False
                if stop_s is not None and float(stop_s) > cache_stop_s + tolerance_s:
                    return False
    except OSError:
        return False

    return True


def find_lfp_motion_energy_cache(
    recording: PhysiologyRecording,
    channels: np.ndarray,
    *,
    target_rate_hz: float = 250.0,
    start_s: float | None = None,
    stop_s: float | None = None,
    cache_path: Path | None = None,
) -> Path | None:
    channels = validate_lfp_channels(channels, recording.lfp.n_lfp_channels)
    sample_rate_hz = recording.lfp_downsampled_rate(target_rate_hz)
    candidates: list[Path] = []

    if cache_path is not None:
        candidates.append(Path(cache_path))
    candidates.append(default_lfp_motion_energy_cache_path(recording, channels, target_rate_hz=target_rate_hz))
    all_channels = np.arange(recording.lfp.n_lfp_channels, dtype=np.int64)
    candidates.append(default_lfp_motion_energy_cache_path(recording, all_channels, target_rate_hz=target_rate_hz))
    candidates.extend(sorted(recording.experiment_path.glob("lfp_me_cache_*hz_*.h5")))

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if lfp_motion_energy_cache_matches(
            candidate,
            channels,
            sample_rate_hz=sample_rate_hz,
            start_s=start_s,
            stop_s=stop_s,
        ):
            return candidate

    return None


def build_lfp_motion_energy_cache(
    recording: PhysiologyRecording,
    output_path: Path,
    channels: np.ndarray,
    *,
    target_rate_hz: float = 250.0,
    start_s: float | None = None,
    duration_s: float | None = None,
    chunk_seconds: float = 60.0,
    read_chunk_seconds: float = 1.0,
    read_mode: str = "auto",
    overwrite: bool = False,
) -> Path:
    channels = validate_lfp_channels(channels, recording.lfp.n_lfp_channels)
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        print(f"using existing LFP/ME cache: {output_path}")
        return output_path

    overlap_start_s, overlap_stop_s = recording.lfp_video_overlap()
    cache_start_s = overlap_start_s if start_s is None else max(float(start_s), overlap_start_s)
    cache_stop_s = overlap_stop_s if duration_s is None else min(cache_start_s + float(duration_s), overlap_stop_s)
    if cache_stop_s <= cache_start_s:
        raise ValueError(
            f"Requested cache window has no LFP/video overlap: {cache_start_s:.3f}-{cache_stop_s:.3f}s"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    sample_rate_hz = recording.lfp_downsampled_rate(target_rate_hz)
    chunk_samples = max(1, int(round(float(chunk_seconds) * sample_rate_hz)))
    channel_chunk = min(len(channels), 32)

    t0 = perf_counter()
    written = 0
    with h5py.File(tmp_path, "w") as h5:
        h5.attrs["cache_version"] = 1
        h5.attrs["experiment_path"] = str(recording.experiment_path)
        h5.attrs["lfp_bin_path"] = str(recording.lfp.bin_path)
        h5.attrs["me_path"] = str(recording.me_path)
        h5.attrs["target_rate_hz"] = float(target_rate_hz)
        h5.attrs["sample_rate_hz"] = float(sample_rate_hz)
        h5.attrs["native_lfp_sample_rate_hz"] = float(recording.lfp.sample_rate)
        h5.attrs["start_video_s"] = float(cache_start_s)
        h5.attrs["stop_video_s"] = float(cache_stop_s)
        h5.attrs["lfp_units"] = "uV"
        h5.attrs["lfp_processing"] = "gain-corrected and downsampled; not demeaned or filtered"
        h5.attrs["motion_energy_processing"] = "sampled at cached LFP times from recording.motion_energy"
        h5.attrs["read_mode"] = read_mode

        h5.create_dataset("channels", data=channels.astype(np.int64, copy=False))
        t_ds = h5.create_dataset(
            "t_video_s",
            shape=(0,),
            maxshape=(None,),
            chunks=(chunk_samples,),
            dtype="float64",
        )
        me_ds = h5.create_dataset(
            "motion_energy",
            shape=(0,),
            maxshape=(None,),
            chunks=(chunk_samples,),
            dtype="float32",
        )
        lfp_ds = h5.create_dataset(
            "lfp_uv",
            shape=(len(channels), 0),
            maxshape=(len(channels), None),
            chunks=(channel_chunk, chunk_samples),
            dtype="float32",
        )

        current = cache_start_s
        while current < cache_stop_s:
            chunk_duration = min(float(chunk_seconds), cache_stop_s - current)
            loaded = recording.load_lfp_downsampled(
                current,
                chunk_duration,
                channels,
                target_rate_hz=target_rate_hz,
                read_chunk_seconds=read_chunk_seconds,
                read_mode=read_mode,
            )
            n_new = int(loaded.t_video_s.size)
            if n_new:
                new_stop = written + n_new
                t_ds.resize((new_stop,))
                me_ds.resize((new_stop,))
                lfp_ds.resize((len(channels), new_stop))
                t_ds[written:new_stop] = loaded.t_video_s
                me_ds[written:new_stop] = recording.sample_motion_energy(loaded.t_video_s)
                lfp_ds[:, written:new_stop] = loaded.lfp_uv
                written = new_stop

            current += chunk_duration
            elapsed = perf_counter() - t0
            print(
                f"cached {current / 60.0:.2f} / {cache_stop_s / 60.0:.2f} min; "
                f"{written:,} samples; elapsed {elapsed / 60.0:.1f} min"
            )

        h5.attrs["n_samples"] = int(written)
        h5.attrs["n_channels"] = int(len(channels))

    tmp_path.replace(output_path)
    size_gb = output_path.stat().st_size / 1e9
    print(f"saved LFP/ME cache: {output_path} ({size_gb:.2f} GB)")
    return output_path


def load_lfp_motion_energy_cache(
    cache_path: Path,
    *,
    start_s: float | None = None,
    duration_s: float | None = None,
    channels: np.ndarray | None = None,
    load_lfp: bool = True,
) -> LfpMotionEnergyCacheData:
    cache_path = Path(cache_path)
    with h5py.File(cache_path, "r") as h5:
        cached_channels = h5["channels"][:].astype(np.int64, copy=False)
        t_all = h5["t_video_s"][:]
        if t_all.size == 0:
            sample_slice = slice(0, 0)
        else:
            start = float(t_all[0]) if start_s is None else float(start_s)
            stop = float(t_all[-1]) if duration_s is None else start + float(duration_s)
            sample_start = int(np.searchsorted(t_all, start, side="left"))
            sample_stop = int(np.searchsorted(t_all, stop, side="right"))
            sample_slice = slice(sample_start, sample_stop)

        if channels is None:
            channel_rows = np.arange(cached_channels.size, dtype=np.int64)
        else:
            requested = np.asarray(channels, dtype=np.int64)
            channel_lookup = {int(channel): idx for idx, channel in enumerate(cached_channels)}
            missing = [int(channel) for channel in requested if int(channel) not in channel_lookup]
            if missing:
                raise ValueError(f"Cache does not contain requested channel(s): {missing[:10]}")
            channel_rows = np.asarray([channel_lookup[int(channel)] for channel in requested], dtype=np.int64)

        t_video_s = h5["t_video_s"][sample_slice]
        motion_energy = h5["motion_energy"][sample_slice].astype(np.float32, copy=False)
        if load_lfp:
            if channel_rows.size and np.all(np.diff(channel_rows) == 1):
                row_slice = slice(int(channel_rows[0]), int(channel_rows[-1]) + 1)
                lfp_uv = h5["lfp_uv"][row_slice, sample_slice].astype(np.float32, copy=False)
            else:
                row_order = np.argsort(channel_rows)
                sorted_rows = channel_rows[row_order]
                lfp_sorted = h5["lfp_uv"][sorted_rows, sample_slice].astype(np.float32, copy=False)
                lfp_uv = lfp_sorted[np.argsort(row_order)]
        else:
            lfp_uv = np.empty((channel_rows.size, 0), dtype=np.float32)

        attrs = {key: value for key, value in h5.attrs.items()}
        sample_rate_hz = float(attrs.get("sample_rate_hz", np.nan))

    return LfpMotionEnergyCacheData(
        cache_path,
        cached_channels[channel_rows],
        t_video_s,
        lfp_uv,
        motion_energy,
        sample_rate_hz,
        attrs,
    )


def ni_channel_counts(meta: dict[str, str]) -> tuple[int, int, int, int]:
    return tuple(int(value) for value in meta["snsMnMaXaDw"].split(","))  # type: ignore[return-value]


def ni_channel_gain(saved_channel: int, meta: dict[str, str]) -> float:
    saved_mn, saved_ma, _saved_xa, _saved_dw = ni_channel_counts(meta)
    if saved_channel < saved_mn:
        return float(meta["niMNGain"])
    if saved_channel < saved_mn + saved_ma:
        return float(meta["niMAGain"])
    return 1.0


def ni_int16_to_mv(data: np.ndarray, saved_channel: int, meta: dict[str, str]) -> np.ndarray:
    int_to_volts = float(meta["niAiRangeMax"]) / 32768.0
    gain = ni_channel_gain(saved_channel, meta)
    return data.astype(np.float32, copy=False) * np.float32(int_to_volts * 1000.0 / gain)


def rising_threshold_crossings_chunked(
    nidq: NidqData,
    *,
    saved_channel: int,
    threshold_mv: float,
    min_separation_samples: int,
    chunk_samples: int = 5_000_000,
) -> np.ndarray:
    if saved_channel < 0 or saved_channel >= nidq.n_channels:
        raise ValueError(f"saved_channel must be in 0..{nidq.n_channels - 1}, got {saved_channel}")

    crossings: list[int] = []
    last_crossing = -int(min_separation_samples)
    prev_above = False

    for start in range(0, nidq.n_samples, int(chunk_samples)):
        stop = min(nidq.n_samples, start + int(chunk_samples))
        signal_mv = ni_int16_to_mv(np.asarray(nidq.raw[saved_channel, start:stop]), saved_channel, nidq.meta)
        above = signal_mv >= float(threshold_mv)
        if above.size == 0:
            continue

        local_crossings = np.flatnonzero(~above[:-1] & above[1:]) + 1
        if start > 0 and (not prev_above) and bool(above[0]):
            local_crossings = np.r_[0, local_crossings]

        for local_crossing in local_crossings:
            crossing = int(start + local_crossing)
            if crossing - last_crossing >= int(min_separation_samples):
                crossings.append(crossing)
                last_crossing = crossing

        prev_above = bool(above[-1])
        if start == 0 or (start // int(chunk_samples)) % 5 == 0:
            print(f"scanned nidq samples {stop:,}/{nidq.n_samples:,}; crossings={len(crossings):,}")

    return np.asarray(crossings, dtype=np.int64)


def camera_ttl_cache_path(experiment_path: Path) -> Path:
    return Path(experiment_path) / "CAM_TTLs.npy"


def get_camera_frame_times(
    experiment_path: Path,
    nidq: NidqData,
    *,
    camera: CameraSyncSettings,
) -> tuple[np.ndarray, np.ndarray]:
    cache_path = camera_ttl_cache_path(experiment_path)
    if cache_path.exists() and not camera.force_redetect:
        camera_seconds = np.asarray(np.load(cache_path)).reshape(-1)
        print(f"loaded {len(camera_seconds):,} cached camera TTLs from {cache_path}")
    else:
        samples = rising_threshold_crossings_chunked(
            nidq,
            saved_channel=camera.channel,
            threshold_mv=camera.threshold_mv,
            min_separation_samples=camera.min_separation_samples,
        )
        camera_seconds = samples.astype(np.float64) / float(nidq.sample_rate)
        np.save(cache_path, camera_seconds)
        print(f"saved {len(camera_seconds):,} camera TTLs to {cache_path}")

    frame_abs_seconds = nidq.first_abs_seconds + camera_seconds
    return frame_abs_seconds, frame_abs_seconds - frame_abs_seconds[0]


def validate_lfp_channels(channels: np.ndarray, n_lfp_channels: int) -> np.ndarray:
    channels = np.asarray(channels, dtype=np.int64)
    if channels.ndim != 1 or channels.size == 0:
        raise ValueError("channels must be a non-empty 1D array")
    bad = channels[(channels < 0) | (channels >= n_lfp_channels)]
    if bad.size:
        raise ValueError(f"LFP channels outside 0..{n_lfp_channels - 1}: {bad[:10]}")
    return channels


def channel_range(start: int, stop: int, step: int = 1) -> np.ndarray:
    return np.arange(int(start), int(stop), int(step), dtype=np.int64)


def direct_channel_read(
    raw: np.memmap,
    channels: np.ndarray,
    sample_slice: slice,
    n_lfp_channels: int,
) -> np.ndarray:
    """Read selected channels without triggering slow one-row memmap slicing."""
    channels = np.asarray(channels, dtype=np.int64)
    if channels.size == 1:
        channel = int(channels[0])
        span_start = channel if channel + 1 < n_lfp_channels else channel - 1
        span_stop = span_start + 2
        return np.asarray(raw[span_start:span_stop, sample_slice])[channel - span_start : channel - span_start + 1]

    contiguous = bool(np.all(np.diff(channels) == 1))
    span_start = int(channels[0])
    span_stop = int(channels[-1]) + 1
    span_size = span_stop - span_start
    if contiguous or span_size <= max(channels.size + 8, 2 * channels.size):
        raw_span = np.asarray(raw[span_start:span_stop, sample_slice])
        return raw_span[channels - span_start]

    return np.asarray(raw[channels, sample_slice])


def gain_correct_lfp_uv(data_counts: np.ndarray, channels: np.ndarray, meta: dict[str, str]) -> np.ndarray:
    if "imChan0lfGain" in meta:
        im_max_int = float(meta.get("imMaxInt", 512))
        int_to_volts = float(meta["imAiRangeMax"]) / im_max_int
        lf_gain = float(meta["imChan0lfGain"])
        scale = np.float32(1e6 * int_to_volts / lf_gain)
        return (data_counts.astype(np.float32, copy=False) * scale).astype(np.float32, copy=False)

    return (1e6 * GainCorrectIM(data_counts, channels.tolist(), meta)).astype(np.float32)


def interpolate_nans_1d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).copy()
    nan_mask = np.isnan(values)
    if not nan_mask.any():
        return values
    valid = np.flatnonzero(~nan_mask)
    if valid.size == 0:
        return values
    values[nan_mask] = np.interp(np.flatnonzero(nan_mask), valid, values[valid])
    return values


def clean_motion_energy_regular_artifacts(
    motion_energy: np.ndarray,
    *,
    method: str = "peak_regular",
    diff_height: float | None = None,
    period_frames: int = 250,
    before_frames: int = 5,
    after_frames: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove regular ME artifact frames by masking and linear interpolation.

    This mirrors the older OL_toplogy workflow:
    ``find_peaks(np.diff(motion))`` -> infer regular period -> set
    ``p-5:p+15`` to NaN -> interpolate across the gaps.
    """
    values = np.asarray(motion_energy, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return values.astype(np.float32), np.empty(0, dtype=np.int64)
    if method not in {"peak_regular", "fixed"}:
        raise ValueError("method must be 'peak_regular' or 'fixed'")

    if method == "fixed":
        artifact_frames = np.arange(0, values.size, int(period_frames), dtype=np.int64)
    else:
        motion_diff = np.diff(values)
        if diff_height is None:
            finite_diff = motion_diff[np.isfinite(motion_diff)]
            diff_height = float(np.nanmedian(finite_diff) + 8.0 * np.nanstd(finite_diff))
        peaks, _properties = find_peaks(motion_diff, height=float(diff_height))
        peaks = peaks.astype(np.int64) + 1
        if peaks.size >= 4:
            period = max(1, int(round(np.median(np.diff(peaks)))))
            regular_frames = np.arange(peaks[3], peaks[-1] + 1, period, dtype=np.int64)
            artifact_frames = np.unique(np.r_[peaks[:3], regular_frames])
        else:
            artifact_frames = peaks

    masked = values.copy()
    for frame in artifact_frames:
        start = max(0, int(frame) - int(before_frames))
        stop = min(values.size, int(frame) + int(after_frames) + 1)
        masked[start:stop] = np.nan
    return interpolate_nans_1d(masked).astype(np.float32), artifact_frames.astype(np.int64, copy=False)


def subtract_static_channel_mean(lfp_uv: np.ndarray) -> np.ndarray:
    return lfp_uv - np.nanmean(lfp_uv, axis=1, keepdims=True)


def moving_average(values: np.ndarray, window_samples: int) -> np.ndarray:
    window_samples = int(window_samples)
    if window_samples <= 1:
        return values
    kernel = np.ones(window_samples, dtype=np.float64) / float(window_samples)
    return np.convolve(values, kernel, mode="same").astype(np.float32)


def butter_filter_1d(
    values: np.ndarray,
    *,
    sample_rate_hz: float,
    lowpass_hz: float | None = None,
    highpass_hz: float | None = None,
    order: int = 3,
) -> np.ndarray:
    values = values.astype(np.float64, copy=True)
    finite = np.isfinite(values)
    if finite.sum() < max(8, order * 3):
        return np.full(values.shape, np.nan, dtype=np.float32)
    if not finite.all():
        idx = np.arange(len(values), dtype=np.float64)
        values[~finite] = np.interp(idx[~finite], idx[finite], values[finite])

    nyquist = float(sample_rate_hz) / 2.0
    if highpass_hz is not None and lowpass_hz is not None:
        wn = [float(highpass_hz) / nyquist, float(lowpass_hz) / nyquist]
        btype = "bandpass"
    elif highpass_hz is not None:
        wn = float(highpass_hz) / nyquist
        btype = "highpass"
    elif lowpass_hz is not None:
        wn = float(lowpass_hz) / nyquist
        btype = "lowpass"
    else:
        return values.astype(np.float32)

    sos = butter(order, wn, btype=btype, output="sos")
    filtered = sosfiltfilt(sos, values)
    filtered[~finite] = np.nan
    return filtered.astype(np.float32)


def amplitude_envelope_1d(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64, copy=True)
    finite = np.isfinite(values)
    if finite.sum() < 8:
        return np.full(values.shape, np.nan, dtype=np.float32)
    if not finite.all():
        idx = np.arange(len(values), dtype=np.float64)
        values[~finite] = np.interp(idx[~finite], idx[finite], values[finite])

    envelope = np.abs(hilbert(values))
    envelope[~finite] = np.nan
    return envelope.astype(np.float32)


def lagged_pearson(
    x: np.ndarray,
    y: np.ndarray,
    *,
    sample_rate_hz: float,
    max_lag_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    max_lag = int(round(float(max_lag_seconds) * float(sample_rate_hz)))
    lags = np.arange(-max_lag, max_lag + 1, dtype=np.int64)
    r_values = np.full(len(lags), np.nan, dtype=np.float32)
    for idx, lag in enumerate(lags):
        if lag < 0:
            xs = x[:lag]
            ys = y[-lag:]
        elif lag > 0:
            xs = x[lag:]
            ys = y[:-lag]
        else:
            xs = x
            ys = y
        valid = np.isfinite(xs) & np.isfinite(ys)
        if valid.sum() > 3:
            r_values[idx] = np.corrcoef(xs[valid], ys[valid])[0, 1]
    return lags.astype(np.float64) / float(sample_rate_hz), r_values


def decimate_for_lag(values: np.ndarray, source_rate_hz: float, target_rate_hz: float) -> tuple[np.ndarray, float]:
    step = max(1, int(round(float(source_rate_hz) / float(target_rate_hz))))
    return values[::step], float(source_rate_hz) / float(step)


def zscore(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64, copy=False)
    mean = np.nanmean(values)
    std = np.nanstd(values)
    if not np.isfinite(std) or std <= 0:
        return np.zeros(values.shape, dtype=np.float32)
    return ((values - mean) / std).astype(np.float32)


def percentile_minmax(
    values: np.ndarray,
    *,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    clip: bool = False,
) -> np.ndarray:
    values = values.astype(np.float64, copy=False)
    finite = np.isfinite(values)
    if not finite.any():
        return np.full(values.shape, np.nan, dtype=np.float32)

    low = np.nanpercentile(values[finite], lower_percentile)
    high = np.nanpercentile(values[finite], upper_percentile)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        out = np.zeros(values.shape, dtype=np.float32)
        out[~finite] = np.nan
        return out

    out = (values - low) / (high - low)
    if clip:
        out = np.clip(out, 0.0, 1.0)
    return out.astype(np.float32)


def pearsonr_valid(x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() > 2 and np.nanstd(x[valid]) > 0 and np.nanstd(y[valid]) > 0:
        r = np.corrcoef(x[valid], y[valid])[0, 1]
    else:
        r = np.nan
    return float(r), valid


def compute_lfp_magnitude_and_me(
    recording: PhysiologyRecording,
    channels: np.ndarray,
    *,
    start_s: float,
    duration_s: float | None,
    chunk_seconds: float,
    smooth_seconds: float,
    target_rate_hz: float,
    read_chunk_seconds: float,
    read_mode: str = "auto",
    lfp_bandpass_hz: tuple[float, float] | None = None,
    use_cache: bool = True,
    cache_path: Path | None = None,
) -> TraceResult:
    channels = validate_lfp_channels(channels, recording.lfp.n_lfp_channels)
    output_sample_rate_hz = recording.lfp_downsampled_rate(target_rate_hz)
    overlap_start_s, overlap_stop_s = recording.lfp_video_overlap()
    start_s = max(float(start_s), overlap_start_s)
    stop_s = overlap_stop_s if duration_s is None else min(float(start_s) + float(duration_s), overlap_stop_s)

    if use_cache:
        found_cache = find_lfp_motion_energy_cache(
            recording,
            channels,
            target_rate_hz=target_rate_hz,
            start_s=start_s,
            stop_s=stop_s,
            cache_path=cache_path,
        )
        if found_cache is not None:
            print(f"using cached LFP/ME data: {found_cache}")
            cache = load_lfp_motion_energy_cache(
                found_cache,
                start_s=start_s,
                duration_s=stop_s - start_s,
                channels=channels,
            )
            return compute_lfp_magnitude_and_me_from_cache(
                cache,
                smooth_seconds=smooth_seconds,
                lfp_bandpass_hz=lfp_bandpass_hz,
            )

    all_t: list[np.ndarray] = []
    all_mag: list[np.ndarray] = []
    all_me: list[np.ndarray] = []
    if lfp_bandpass_hz is not None:
        low_hz, high_hz = lfp_bandpass_hz
        print(f"bandpass filtering LFP and motion energy: {low_hz:g}-{high_hz:g} Hz; computing envelopes")
    else:
        print("computing LFP and motion-energy envelopes without bandpass filtering")
    current = start_s
    while current < stop_s:
        chunk_duration = min(float(chunk_seconds), stop_s - current)
        loaded = recording.load_lfp_downsampled(
            current,
            chunk_duration,
            channels,
            target_rate_hz=target_rate_hz,
            read_chunk_seconds=read_chunk_seconds,
            read_mode=read_mode,
        )
        output_sample_rate_hz = loaded.sample_rate_hz
        if loaded.t_video_s.size == 0:
            current += chunk_duration
            continue

        lfp_chunk = subtract_static_channel_mean(loaded.lfp_uv)
        me_chunk = recording.sample_motion_energy(loaded.t_video_s)
        if lfp_bandpass_hz is not None:
            for chan_idx in range(lfp_chunk.shape[0]):
                lfp_chunk[chan_idx] = butter_filter_1d(
                    lfp_chunk[chan_idx],
                    sample_rate_hz=output_sample_rate_hz,
                    highpass_hz=low_hz,
                    lowpass_hz=high_hz,
                    order=2,
                )
            me_chunk = butter_filter_1d(
                me_chunk,
                sample_rate_hz=output_sample_rate_hz,
                highpass_hz=low_hz,
                lowpass_hz=high_hz,
                order=2,
            )

        lfp_envelopes = np.empty_like(lfp_chunk, dtype=np.float32)
        for chan_idx in range(lfp_chunk.shape[0]):
            lfp_envelopes[chan_idx] = amplitude_envelope_1d(lfp_chunk[chan_idx])
        lfp_mag = np.nanmean(lfp_envelopes, axis=0).astype(np.float32)
        me_chunk = amplitude_envelope_1d(me_chunk)
        if smooth_seconds > 0:
            win = max(1, int(round(float(smooth_seconds) * output_sample_rate_hz)))
            lfp_mag = moving_average(lfp_mag, win)
            me_chunk = moving_average(me_chunk, win)

        all_t.append(loaded.t_video_s)
        all_mag.append(lfp_mag)
        all_me.append(me_chunk)
        current += chunk_duration
        print(f"processed through {current / 60.0:.2f} / {stop_s / 60.0:.2f} min")

    if not all_t:
        empty = np.empty(0, dtype=np.float32)
        return TraceResult(np.empty(0, dtype=np.float64), empty, empty, output_sample_rate_hz)

    return TraceResult(
        np.concatenate(all_t),
        np.concatenate(all_mag),
        np.concatenate(all_me),
        output_sample_rate_hz,
    )


def compute_lfp_magnitude_and_me_from_cache(
    cache: LfpMotionEnergyCacheData,
    *,
    smooth_seconds: float = 0.0,
    lfp_bandpass_hz: tuple[float, float] | None = None,
) -> TraceResult:
    if cache.lfp_uv.size == 0:
        empty = np.empty(0, dtype=np.float32)
        return TraceResult(cache.t_video_s, empty, cache.motion_energy, cache.sample_rate_hz)

    lfp_chunk = subtract_static_channel_mean(cache.lfp_uv)
    me_chunk = cache.motion_energy.astype(np.float32, copy=True)
    if lfp_bandpass_hz is not None:
        low_hz, high_hz = lfp_bandpass_hz
        print(f"bandpass filtering cached LFP and motion energy: {low_hz:g}-{high_hz:g} Hz; computing envelopes")
        for chan_idx in range(lfp_chunk.shape[0]):
            lfp_chunk[chan_idx] = butter_filter_1d(
                lfp_chunk[chan_idx],
                sample_rate_hz=cache.sample_rate_hz,
                highpass_hz=low_hz,
                lowpass_hz=high_hz,
                order=2,
            )
        me_chunk = butter_filter_1d(
            me_chunk,
            sample_rate_hz=cache.sample_rate_hz,
            highpass_hz=low_hz,
            lowpass_hz=high_hz,
            order=2,
        )
    else:
        print("computing cached LFP and motion-energy envelopes without bandpass filtering")

    lfp_envelopes = np.empty_like(lfp_chunk, dtype=np.float32)
    for chan_idx in range(lfp_chunk.shape[0]):
        lfp_envelopes[chan_idx] = amplitude_envelope_1d(lfp_chunk[chan_idx])
    lfp_mag = np.nanmean(lfp_envelopes, axis=0).astype(np.float32)
    me_chunk = amplitude_envelope_1d(me_chunk)
    if smooth_seconds > 0:
        win = max(1, int(round(float(smooth_seconds) * cache.sample_rate_hz)))
        lfp_mag = moving_average(lfp_mag, win)
        me_chunk = moving_average(me_chunk, win)

    return TraceResult(cache.t_video_s, lfp_mag, me_chunk, cache.sample_rate_hz)


def load_channel_average_lfp(
    recording: PhysiologyRecording,
    channels: np.ndarray,
    *,
    chunk_seconds: float,
    target_rate_hz: float,
    read_chunk_seconds: float,
    reduction: str,
    read_mode: str = "auto",
    use_cache: bool = True,
    cache_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    channels = validate_lfp_channels(channels, recording.lfp.n_lfp_channels)
    if reduction not in {"mean", "mean_abs"}:
        raise ValueError("reduction must be 'mean' or 'mean_abs'")

    start_s, stop_s = recording.lfp_video_overlap()
    output_sample_rate_hz = recording.lfp_downsampled_rate(target_rate_hz)
    if use_cache:
        found_cache = find_lfp_motion_energy_cache(
            recording,
            channels,
            target_rate_hz=target_rate_hz,
            start_s=start_s,
            stop_s=stop_s,
            cache_path=cache_path,
        )
        if found_cache is not None:
            print(f"using cached LFP data: {found_cache}")
            cache = load_lfp_motion_energy_cache(
                found_cache,
                start_s=start_s,
                duration_s=stop_s - start_s,
                channels=channels,
            )
            lfp_chunk = subtract_static_channel_mean(cache.lfp_uv)
            if reduction == "mean_abs":
                lfp_mean = np.nanmean(np.abs(lfp_chunk), axis=0)
            else:
                lfp_mean = np.nanmean(lfp_chunk, axis=0)
            return cache.t_video_s, lfp_mean.astype(np.float32, copy=False), cache.sample_rate_hz

    all_t: list[np.ndarray] = []
    all_lfp_mean: list[np.ndarray] = []
    current = start_s
    while current < stop_s:
        chunk_duration = min(float(chunk_seconds), stop_s - current)
        loaded = recording.load_lfp_downsampled(
            current,
            chunk_duration,
            channels,
            target_rate_hz=target_rate_hz,
            read_chunk_seconds=read_chunk_seconds,
            read_mode=read_mode,
        )
        output_sample_rate_hz = loaded.sample_rate_hz
        if loaded.t_video_s.size == 0:
            current += chunk_duration
            continue

        lfp_chunk = subtract_static_channel_mean(loaded.lfp_uv)
        if reduction == "mean_abs":
            lfp_mean = np.nanmean(np.abs(lfp_chunk), axis=0)
        else:
            lfp_mean = np.nanmean(lfp_chunk, axis=0)

        all_t.append(loaded.t_video_s)
        all_lfp_mean.append(lfp_mean.astype(np.float32, copy=False))
        current += chunk_duration
        print(f"loaded channel-average LFP through {current / 60.0:.2f} / {stop_s / 60.0:.2f} min")

    if not all_t:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float32), output_sample_rate_hz
    return np.concatenate(all_t), np.concatenate(all_lfp_mean), output_sample_rate_hz


def benchmark_lfp_loading(
    recording: PhysiologyRecording,
    channels: np.ndarray,
    *,
    duration_s: float,
    target_rate_hz: float,
    read_chunk_seconds: float,
    read_mode: str = "auto",
) -> dict[str, float]:
    t0 = perf_counter()
    loaded = recording.load_lfp_downsampled(
        0.0,
        duration_s,
        channels,
        target_rate_hz=target_rate_hz,
        read_chunk_seconds=read_chunk_seconds,
        read_mode=read_mode,
    )
    elapsed_s = perf_counter() - t0
    samples = int(loaded.lfp_uv.size)
    return {
        "elapsed_s": float(elapsed_s),
        "channels": float(len(channels)),
        "duration_s": float(duration_s),
        "sample_rate_hz": float(loaded.sample_rate_hz),
        "values_loaded": float(samples),
        "million_values_per_s": float(samples / max(elapsed_s, 1e-9) / 1e6),
    }
