#!/usr/bin/env python3
"""Copy a time subset of SpikeGLX binary data and optionally run Kilosort.

SpikeGLX imec ``*.ap.bin`` files are already interleaved as
``sample0_ch0, sample0_ch1, ...`` int16 data, which is the layout Kilosort
expects when it opens a C-order array shaped ``(samples, channels)``. By
default the full input recording is sorted in-place without copying. A new
binary is written only when a time or channel subset is requested. Pass
``--no-run-kilosort`` to only write the subset.

Examples
--------
Copy minutes 10-20, preserving all saved channels:

    python spikeglx_subset_to_kilosort_bin.py rec.imec0.ap.bin \
        -o rec_10to20min.ap.bin --start-sec 600 --duration-sec 600

Run Kilosort on the full input recording without copying:

    python spikeglx_subset_to_kilosort_bin.py rec.imec0.ap.bin

Write only the binary subset, without sorting:

    python spikeglx_subset_to_kilosort_bin.py rec.imec0.ap.bin \
        -o rec_10to20min.ap.bin --start-sec 600 --duration-sec 600 \
        --no-run-kilosort

Copy the first 5 minutes and drop the trailing sync channel:

    python spikeglx_subset_to_kilosort_bin.py rec.imec0.ap.bin \
        -o rec_first5min_neural.ap.bin --duration-sec 300 --drop-sync

Copy an exact sample range:

    python spikeglx_subset_to_kilosort_bin.py rec.imec0.ap.bin \
        -o rec_samples.ap.bin --start-sample 300000 --stop-sample 900000
"""

from __future__ import annotations

import argparse
import ast
import csv
import fnmatch
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


INT16_DTYPE = np.dtype("<i2")
DEFAULT_CHUNK_SAMPLES = 60_000
DEFAULT_PROGRESS_SECONDS = 10.0
DEFAULT_KILOSORT_PROBE_NAME = "NeuroPixUltra_default.mat"
DEFAULT_KILOSORT_NBLOCKS = 1
DEFAULT_KILOSORT_WHITENING_RANGE = 64
DEFAULT_KILOSORT_NEAREST_CHANS = 32
DEFAULT_KILOSORT_DMIN_UM = 6.0
DEFAULT_KILOSORT_DMINX_UM = 6.0
DEFAULT_KILOSORT_MAX_CHANNEL_DISTANCE_UM = 32.0
DEFAULT_KILOSORT_X_CENTERS = 1
DEFAULT_KILOSORT_VERBOSE_CONSOLE = True
DEFAULT_SAION_WORK_BASE = Path("/work/ReiterU/ant_tmp")
SAION_PARTITION_CAPS = {
    "largegpu": (8, 128, 1024, "0-12"),
    "short-a100": (32, 256, 2048, "0-2"),
    "gpu-a100": (8, 128, 1024, "0-8"),
}


@dataclass(frozen=True)
class SpikeGlxBinary:
    bin_path: Path
    meta_path: Path
    meta: dict[str, str]
    sample_rate: float
    n_channels: int
    n_samples: int

    @property
    def duration_seconds(self) -> float:
        return self.n_samples / self.sample_rate


@dataclass(frozen=True)
class SubsetSpec:
    label: str
    start_sec: float | None = None
    end_sec: float | None = None
    duration_sec: float | None = None
    start_sample: int | None = None
    stop_sample: int | None = None
    dataset_key: str | None = None
    pattern: str | None = None

    @property
    def is_full(self) -> bool:
        return (
            self.start_sec is None
            and self.end_sec is None
            and self.duration_sec is None
            and self.start_sample is None
            and self.stop_sample is None
        )


@dataclass(frozen=True)
class SpikeGlxGeometrySite:
    shank: float
    x: float
    y: float
    connected: bool


def read_meta(bin_path: Path) -> tuple[Path, dict[str, str]]:
    meta_path = bin_path.with_suffix(".meta")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing SpikeGLX metadata file: {meta_path}")

    meta: dict[str, str] = {}
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        meta[key.lstrip("~")] = value
    return meta_path, meta


def find_ap_bin(path: Path) -> Path:
    if path.is_file():
        if path.name.endswith(".bin"):
            return path
        raise ValueError(f"Expected a SpikeGLX *.bin file, got {path}")

    matches = sorted(path.rglob("*.ap.bin"))
    if not matches:
        raise FileNotFoundError(f"No *.ap.bin files found under {path}")
    if len(matches) > 1:
        names = "\n".join(str(match) for match in matches[:25])
        extra = "" if len(matches) <= 25 else f"\n... and {len(matches) - 25} more"
        raise ValueError(f"Found multiple *.ap.bin files; pass one explicitly:\n{names}{extra}")
    return matches[0]


def sample_rate_from_meta(meta: dict[str, str]) -> float:
    if meta.get("typeThis") == "imec" or "imSampRate" in meta:
        return float(meta["imSampRate"])
    if "niSampRate" in meta:
        return float(meta["niSampRate"])
    raise KeyError("Metadata does not contain imSampRate or niSampRate")


def open_spikeglx(path: Path) -> SpikeGlxBinary:
    bin_path = find_ap_bin(path)
    meta_path, meta = read_meta(bin_path)
    n_channels = int(meta["nSavedChans"])
    if n_channels <= 0:
        raise ValueError(f"nSavedChans must be positive in {meta_path}")

    file_size = bin_path.stat().st_size
    frame_bytes = INT16_DTYPE.itemsize * n_channels
    remainder = file_size % frame_bytes
    if remainder:
        raise ValueError(
            f"{bin_path} size ({file_size} bytes) is not divisible by "
            f"2 * nSavedChans ({frame_bytes} bytes per sample)."
        )

    n_samples = file_size // frame_bytes
    meta_size = int(meta.get("fileSizeBytes", file_size))
    if meta_size != file_size:
        print(
            f"[WARN] meta fileSizeBytes={meta_size:,} but actual size={file_size:,}; "
            "using actual file size.",
            file=sys.stderr,
        )

    return SpikeGlxBinary(
        bin_path=bin_path,
        meta_path=meta_path,
        meta=meta,
        sample_rate=sample_rate_from_meta(meta),
        n_channels=n_channels,
        n_samples=n_samples,
    )


def parse_imec_channel_counts(meta: dict[str, str]) -> tuple[int, int, int] | None:
    value = meta.get("snsApLfSy")
    if value is None:
        return None
    parts = [int(part) for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected snsApLfSy to have three comma-separated values, got {value!r}")
    return parts[0], parts[1], parts[2]


def parse_spikeglx_geometry(meta: dict[str, str]) -> list[SpikeGlxGeometrySite] | None:
    value = meta.get("snsGeomMap")
    if not value:
        return None

    sites: list[SpikeGlxGeometrySite] = []
    for group in re.findall(r"\(([^)]*)\)", value):
        fields = [field.strip() for field in group.split(":")]
        if len(fields) != 4:
            continue
        try:
            shank = float(fields[0])
            x = float(fields[1])
            y = float(fields[2])
            connected = bool(int(float(fields[3])))
        except ValueError as exc:
            raise ValueError(f"Could not parse SpikeGLX snsGeomMap entry {group!r}") from exc
        sites.append(SpikeGlxGeometrySite(shank=shank, x=x, y=y, connected=connected))

    if not sites:
        raise ValueError("SpikeGLX metadata has snsGeomMap but no parseable geometry entries")
    return sites


def build_spikeglx_probe_dict(rec: SpikeGlxBinary, channels: np.ndarray) -> dict[str, Any]:
    sites = parse_spikeglx_geometry(rec.meta)
    if sites is None:
        raise ValueError("SpikeGLX metadata does not contain snsGeomMap; cannot auto-generate probe")

    chan_map: list[int] = []
    xc: list[float] = []
    yc: list[float] = []
    kcoords: list[float] = []
    skipped_channels: list[int] = []

    for output_channel, source_channel in enumerate(channels.astype(np.int64, copy=False)):
        source_index = int(source_channel)
        if source_index >= len(sites):
            skipped_channels.append(source_index)
            continue

        site = sites[source_index]
        if not site.connected:
            skipped_channels.append(source_index)
            continue

        chan_map.append(int(output_channel))
        xc.append(float(site.x))
        yc.append(float(site.y))
        kcoords.append(float(site.shank + 1.0))

    if not chan_map:
        raise ValueError("Auto-generated probe would have no connected channels")

    return {
        "chanMap": chan_map,
        "xc": xc,
        "yc": yc,
        "kcoords": kcoords,
        "n_chan": int(channels.size),
        "source_meta": str(rec.meta_path),
        "source_probe_part_number": rec.meta.get("imDatPrb_pn", ""),
        "source_probe_type": rec.meta.get("imDatPrb_type", ""),
        "disconnected_or_unmapped_source_channels": ",".join(str(ch) for ch in skipped_channels),
    }


def write_spikeglx_probe_json(
    path: Path,
    *,
    rec: SpikeGlxBinary,
    channels: np.ndarray,
    dry_run: bool,
) -> tuple[Path, int, int]:
    probe = build_spikeglx_probe_dict(rec, channels)
    path = path.resolve()
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(probe, indent=2) + "\n", encoding="utf-8")
    return path, len(probe["chanMap"]), int(probe["n_chan"])


def parse_index_token(token: str) -> list[int]:
    token = token.strip()
    if not token:
        return []

    if "-" in token and ":" not in token:
        start_text, stop_text = token.split("-", 1)
        start = int(start_text)
        stop = int(stop_text)
        step = 1 if stop >= start else -1
        return list(range(start, stop + step, step))

    if ":" in token:
        parts = token.split(":")
        if len(parts) > 3:
            raise ValueError(f"Invalid slice token {token!r}")
        start = int(parts[0]) if parts[0] else 0
        stop = int(parts[1]) if len(parts) > 1 and parts[1] else None
        step = int(parts[2]) if len(parts) > 2 and parts[2] else 1
        if stop is None:
            raise ValueError(f"Slice token {token!r} needs an explicit stop")
        return list(range(start, stop, step))

    return [int(token)]


def parse_channels(spec: str, rec: SpikeGlxBinary) -> np.ndarray:
    spec_clean = spec.strip().lower()
    if spec_clean == "all":
        channels = np.arange(rec.n_channels, dtype=np.int64)
    elif spec_clean in {"ap", "neural", "neural-only"}:
        counts = parse_imec_channel_counts(rec.meta)
        if counts is None or counts[0] <= 0:
            raise ValueError("--channels ap requires imec metadata with AP channels in snsApLfSy")
        channels = np.arange(counts[0], dtype=np.int64)
    else:
        values: list[int] = []
        for token in spec.split(","):
            values.extend(parse_index_token(token))
        channels = np.asarray(values, dtype=np.int64)

    validate_channels(channels, rec.n_channels)
    return channels


def channels_after_sync_drop(channels: np.ndarray, rec: SpikeGlxBinary) -> np.ndarray:
    counts = parse_imec_channel_counts(rec.meta)
    if counts is None:
        raise ValueError("--drop-sync requires imec metadata field snsApLfSy")
    _ap_count, _lf_count, sync_count = counts
    if sync_count <= 0:
        return channels
    sync_start = rec.n_channels - sync_count
    return channels[channels < sync_start]


def validate_channels(channels: np.ndarray, n_channels: int) -> None:
    if channels.ndim != 1 or channels.size == 0:
        raise ValueError("Channel selection is empty")
    bad = channels[(channels < 0) | (channels >= n_channels)]
    if bad.size:
        raise ValueError(f"Channel indices outside 0..{n_channels - 1}: {bad[:20].tolist()}")
    if np.unique(channels).size != channels.size:
        raise ValueError("Channel selection contains duplicate indices")


def seconds_to_sample(seconds: float, sample_rate: float) -> int:
    return int(round(float(seconds) * float(sample_rate)))


def resolve_sample_bounds(args: argparse.Namespace, rec: SpikeGlxBinary) -> tuple[int, int]:
    if args.start_sample is not None and args.start_sec is not None:
        raise ValueError("Use only one of --start-sample or --start-sec")
    if args.stop_sample is not None and (args.end_sec is not None or args.duration_sec is not None):
        raise ValueError("Use --stop-sample, --end-sec, or --duration-sec, not more than one")
    if args.end_sec is not None and args.duration_sec is not None:
        raise ValueError("Use only one of --end-sec or --duration-sec")

    start = int(args.start_sample) if args.start_sample is not None else seconds_to_sample(args.start_sec or 0.0, rec.sample_rate)

    if args.stop_sample is not None:
        stop = int(args.stop_sample)
    elif args.end_sec is not None:
        stop = seconds_to_sample(args.end_sec, rec.sample_rate)
    elif args.duration_sec is not None:
        stop = start + seconds_to_sample(args.duration_sec, rec.sample_rate)
    else:
        stop = rec.n_samples

    if start < 0:
        raise ValueError("Start sample must be >= 0")
    if stop <= start:
        raise ValueError(f"Stop sample must be greater than start sample ({start}), got {stop}")
    if start >= rec.n_samples:
        raise ValueError(f"Start sample {start} is beyond recording length {rec.n_samples}")
    if stop > rec.n_samples:
        print(
            f"[WARN] requested stop sample {stop:,} exceeds recording length {rec.n_samples:,}; clipping.",
            file=sys.stderr,
        )
        stop = rec.n_samples
    return start, stop


def resolve_chunk_samples(args: argparse.Namespace, sample_rate: float) -> int:
    if args.chunk_samples is not None and args.chunk_sec is not None:
        raise ValueError("Use only one of --chunk-samples or --chunk-sec")
    if args.chunk_sec is not None:
        chunk_samples = seconds_to_sample(args.chunk_sec, sample_rate)
    elif args.chunk_samples is not None:
        chunk_samples = int(args.chunk_samples)
    else:
        chunk_samples = DEFAULT_CHUNK_SAMPLES
    if chunk_samples <= 0:
        raise ValueError("Chunk size must be positive")
    return chunk_samples


def default_output_path(input_path: Path, start_sample: int, stop_sample: int, channels: np.ndarray, rec: SpikeGlxBinary) -> Path:
    channel_tag = "allch" if channels.size == rec.n_channels and np.array_equal(channels, np.arange(rec.n_channels)) else f"{channels.size}ch"
    return input_path.with_name(f"{input_path.stem}.ks_{start_sample}_{stop_sample}_{channel_tag}.bin")


def selects_full_recording(
    rec: SpikeGlxBinary,
    *,
    start_sample: int,
    stop_sample: int,
    channels: np.ndarray,
) -> bool:
    return (
        start_sample == 0
        and stop_sample == rec.n_samples
        and channels.size == rec.n_channels
        and np.array_equal(channels, np.arange(rec.n_channels))
    )


def contiguous_channel_slice(channels: np.ndarray) -> slice | None:
    if channels.size == 0:
        return None
    if channels.size == 1:
        return slice(int(channels[0]), int(channels[0]) + 1)
    if np.all(np.diff(channels) == 1):
        return slice(int(channels[0]), int(channels[-1]) + 1)
    return None


def write_sidecar(
    sidecar_path: Path,
    *,
    rec: SpikeGlxBinary,
    output_path: Path,
    start_sample: int,
    stop_sample: int,
    channels: np.ndarray,
    chunk_samples: int,
    elapsed_s: float,
) -> None:
    payload = {
        "source_bin": str(rec.bin_path),
        "source_meta": str(rec.meta_path),
        "output_bin": str(output_path),
        "dtype": "int16",
        "layout": "C-order sample-major array with shape (n_samples, n_chan_bin)",
        "sample_rate_hz": rec.sample_rate,
        "source_n_chan_bin": rec.n_channels,
        "output_n_chan_bin": int(channels.size),
        "source_n_samples": rec.n_samples,
        "start_sample": int(start_sample),
        "stop_sample_exclusive": int(stop_sample),
        "output_n_samples": int(stop_sample - start_sample),
        "start_sec": float(start_sample / rec.sample_rate),
        "stop_sec": float(stop_sample / rec.sample_rate),
        "duration_sec": float((stop_sample - start_sample) / rec.sample_rate),
        "channels": [int(ch) for ch in channels],
        "chunk_samples": int(chunk_samples),
        "elapsed_s": float(elapsed_s),
        "kilosort_settings_hint": {
            "filename": str(output_path),
            "n_chan_bin": int(channels.size),
            "fs": rec.sample_rate,
            "data_dtype": "int16",
        },
    }
    sidecar_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def format_bytes(n_bytes: int) -> str:
    value = float(n_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{n_bytes} B"


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def copy_subset(
    rec: SpikeGlxBinary,
    output_path: Path,
    *,
    start_sample: int,
    stop_sample: int,
    channels: np.ndarray,
    chunk_samples: int,
    overwrite: bool,
    write_metadata: bool,
    dry_run: bool,
    progress_seconds: float,
) -> Path:
    output_path = output_path.resolve()
    output_samples = stop_sample - start_sample
    output_bytes = output_samples * channels.size * INT16_DTYPE.itemsize
    sidecar_path = output_path.with_suffix(output_path.suffix + ".json")

    print(f"source: {rec.bin_path}")
    print(f"source samples: {rec.n_samples:,} ({rec.duration_seconds:.3f} s)")
    print(f"source channels: {rec.n_channels:,}")
    print(f"sample rate: {rec.sample_rate:.6f} Hz")
    print(f"subset samples: {start_sample:,}:{stop_sample:,} ({output_samples:,} samples)")
    print(f"subset seconds: {start_sample / rec.sample_rate:.6f}:{stop_sample / rec.sample_rate:.6f}")
    print(f"output channels: {channels.size:,}")
    print(f"output bytes: {format_bytes(output_bytes)}")
    print(f"output: {output_path}")
    if dry_run:
        print("dry run: no data written")
        return output_path

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists, pass --overwrite to replace: {output_path}")
    if sidecar_path.exists() and not overwrite and write_metadata:
        raise FileExistsError(f"Sidecar exists, pass --overwrite to replace: {sidecar_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    if tmp_path.exists():
        if overwrite:
            tmp_path.unlink()
        else:
            raise FileExistsError(f"Temporary output exists, pass --overwrite to replace: {tmp_path}")

    raw = np.memmap(
        rec.bin_path,
        dtype=INT16_DTYPE,
        mode="r",
        shape=(rec.n_samples, rec.n_channels),
        order="C",
    )
    channel_slice = contiguous_channel_slice(channels)
    all_channels = channels.size == rec.n_channels and np.array_equal(channels, np.arange(rec.n_channels))
    total_chunks = (output_samples + chunk_samples - 1) // chunk_samples

    started = time.monotonic()
    last_progress = started
    written = 0
    print(f"copy chunks: {total_chunks:,} x up to {chunk_samples:,} samples ({chunk_samples / rec.sample_rate:.3f} s)")
    print(f"progress interval: {progress_seconds:g} s" if progress_seconds > 0 else "progress interval: disabled")
    print(f"temporary output: {tmp_path}")
    print("copying subset...", flush=True)
    try:
        with tmp_path.open("wb") as out:
            for chunk_idx, chunk_start in enumerate(range(start_sample, stop_sample, chunk_samples), start=1):
                chunk_stop = min(stop_sample, chunk_start + chunk_samples)
                if all_channels:
                    block = raw[chunk_start:chunk_stop, :]
                elif channel_slice is not None:
                    block = raw[chunk_start:chunk_stop, channel_slice]
                else:
                    block = raw[chunk_start:chunk_stop, channels]

                np.ascontiguousarray(block, dtype=INT16_DTYPE).tofile(out)
                written += chunk_stop - chunk_start

                now = time.monotonic()
                if progress_seconds > 0 and (now - last_progress >= progress_seconds or chunk_stop == stop_sample):
                    pct = 100.0 * written / output_samples
                    elapsed = now - started
                    rate = written / max(now - started, 1e-9)
                    copied_bytes = written * channels.size * INT16_DTYPE.itemsize
                    throughput = copied_bytes / max(elapsed, 1e-9)
                    remaining_samples = output_samples - written
                    eta = remaining_samples / max(rate, 1e-9)
                    print(
                        "[CONVERT] "
                        f"chunk {chunk_idx:,}/{total_chunks:,}, "
                        f"{written:,}/{output_samples:,} samples ({pct:.1f}%), "
                        f"{format_bytes(copied_bytes)}/{format_bytes(output_bytes)}, "
                        f"{rate:,.0f} samples/s, {format_bytes(int(throughput))}/s, "
                        f"elapsed {format_duration(elapsed)}, ETA {format_duration(eta)}",
                        flush=True,
                    )
                    last_progress = now

        actual_size = tmp_path.stat().st_size
        if actual_size != output_bytes:
            raise RuntimeError(f"Wrote {actual_size} bytes, expected {output_bytes}")
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    elapsed = time.monotonic() - started
    if write_metadata:
        write_sidecar(
            sidecar_path,
            rec=rec,
            output_path=output_path,
            start_sample=start_sample,
            stop_sample=stop_sample,
            channels=channels,
            chunk_samples=chunk_samples,
            elapsed_s=elapsed,
        )
        print(f"sidecar: {sidecar_path}")
    print(f"done in {elapsed / 60.0:.2f} min")
    print(f"average conversion throughput: {format_bytes(int(output_bytes / max(elapsed, 1e-9)))}/s")
    print(f"Kilosort n_chan_bin: {channels.size}")
    print(f"Kilosort fs: {rec.sample_rate:.6f}")
    return output_path


def verify_existing_output(
    output_path: Path,
    *,
    output_samples: int,
    n_channels: int,
) -> None:
    expected_bytes = output_samples * n_channels * INT16_DTYPE.itemsize
    actual_bytes = output_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"Existing output has {actual_bytes:,} bytes, expected {expected_bytes:,} "
            f"for {output_samples:,} samples x {n_channels:,} channels."
    )
    print(f"reusing existing output: {output_path}")
    print(f"existing output shape: {output_samples:,} samples x {n_channels:,} channels")
    print(f"existing output bytes: {format_bytes(actual_bytes)}")


def parse_cli_value(text: str) -> Any:
    value = text.strip()
    lower = value.lower()
    if lower in {"none", "null"}:
        return None
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in {"inf", "+inf", "infinity", "+infinity"}:
        return np.inf
    if lower in {"-inf", "-infinity"}:
        return -np.inf

    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        pass

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value


def parse_key_value(text: str) -> tuple[str, Any]:
    if "=" not in text:
        raise ValueError(f"Expected KEY=VALUE, got {text!r}")
    key, value = text.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Missing key in {text!r}")
    return key, parse_cli_value(value)


def parse_index_spec(spec: str) -> list[int]:
    values: list[int] = []
    for token in spec.split(","):
        values.extend(parse_index_token(token))
    return values


def default_kilosort_results_dir(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_kilosort4")


def safe_slug(value: str, max_len: int = 120) -> str:
    slug = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    slug = "_".join(part for part in slug.split("_") if part)
    slug = slug.strip("._-") or "item"
    return slug[:max_len]


def find_ap_bins_recursive(input_path: Path) -> list[Path]:
    path = input_path.expanduser()
    if path.is_file():
        return [find_ap_bin(path).resolve()]
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    bins = sorted(path.rglob("*.ap.bin"))
    if not bins:
        raise FileNotFoundError(f"No *.ap.bin files found under {path}")
    return [item.resolve() for item in bins]


def format_float_tag(value: float | int) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def subset_label_from_values(
    *,
    start_sec: float | None = None,
    end_sec: float | None = None,
    duration_sec: float | None = None,
    start_sample: int | None = None,
    stop_sample: int | None = None,
) -> str:
    if start_sample is not None or stop_sample is not None:
        start = "start" if start_sample is None else str(int(start_sample))
        stop = "end" if stop_sample is None else str(int(stop_sample))
        return f"samp_{start}_{stop}"
    start = 0.0 if start_sec is None else float(start_sec)
    if duration_sec is not None:
        return f"sec_{format_float_tag(start)}_dur_{format_float_tag(duration_sec)}"
    if end_sec is not None:
        return f"sec_{format_float_tag(start)}_{format_float_tag(end_sec)}"
    return f"sec_{format_float_tag(start)}_end"


def parse_subset_time_spec(text: str) -> SubsetSpec:
    raw = text.strip()
    if not raw:
        raise ValueError("Empty --subset-time value")

    label = None
    if "=" in raw:
        label, raw = raw.split("=", 1)
        label = safe_slug(label.strip(), max_len=60)
        raw = raw.strip()

    if "," in raw and ":" not in raw:
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Expected START,DURATION for subset time, got {text!r}")
        start_sec = float(parts[0])
        duration_sec = float(parts[1])
        return SubsetSpec(
            label=label or subset_label_from_values(start_sec=start_sec, duration_sec=duration_sec),
            start_sec=start_sec,
            duration_sec=duration_sec,
        )

    if ":" not in raw:
        raise ValueError(
            f"Expected START:END, START:+DURATION, or START,DURATION for subset time, got {text!r}"
        )
    start_text, stop_text = [part.strip() for part in raw.split(":", 1)]
    start_sec = float(start_text) if start_text else 0.0
    if stop_text.startswith("+"):
        duration_sec = float(stop_text[1:])
        return SubsetSpec(
            label=label or subset_label_from_values(start_sec=start_sec, duration_sec=duration_sec),
            start_sec=start_sec,
            duration_sec=duration_sec,
        )
    end_sec = float(stop_text)
    return SubsetSpec(
        label=label or subset_label_from_values(start_sec=start_sec, end_sec=end_sec),
        start_sec=start_sec,
        end_sec=end_sec,
    )


def normal_time_args_as_subset(args: argparse.Namespace) -> SubsetSpec | None:
    if not any(
        value is not None
        for value in (
            args.start_sec,
            args.end_sec,
            args.duration_sec,
            args.start_sample,
            args.stop_sample,
        )
    ):
        return None
    return SubsetSpec(
        label=subset_label_from_values(
            start_sec=args.start_sec,
            end_sec=args.end_sec,
            duration_sec=args.duration_sec,
            start_sample=args.start_sample,
            stop_sample=args.stop_sample,
        ),
        start_sec=args.start_sec,
        end_sec=args.end_sec,
        duration_sec=args.duration_sec,
        start_sample=args.start_sample,
        stop_sample=args.stop_sample,
    )


def subset_spec_from_row(row: dict[str, Any]) -> SubsetSpec:
    def get_any(*names: str) -> Any:
        for name in names:
            value = row.get(name)
            if value not in (None, ""):
                return value
        return None

    dataset_key = get_any("dataset", "path", "file", "name", "stem")
    pattern = get_any("pattern", "glob")
    spec_text = get_any("subset", "time", "window")
    label = get_any("label")
    if spec_text is not None:
        spec = parse_subset_time_spec(str(spec_text))
        return SubsetSpec(
            label=safe_slug(str(label), max_len=60) if label is not None else spec.label,
            start_sec=spec.start_sec,
            end_sec=spec.end_sec,
            duration_sec=spec.duration_sec,
            dataset_key=str(dataset_key) if dataset_key is not None else None,
            pattern=str(pattern) if pattern is not None else None,
        )

    start_sec = get_any("start_sec", "start_s", "start")
    end_sec = get_any("end_sec", "stop_sec", "end", "stop")
    duration_sec = get_any("duration_sec", "duration_s", "duration")
    start_sample = get_any("start_sample", "start_samp")
    stop_sample = get_any("stop_sample", "stop_samp")
    values = {
        "start_sec": float(start_sec) if start_sec is not None else None,
        "end_sec": float(end_sec) if end_sec is not None else None,
        "duration_sec": float(duration_sec) if duration_sec is not None else None,
        "start_sample": int(start_sample) if start_sample is not None else None,
        "stop_sample": int(stop_sample) if stop_sample is not None else None,
    }
    return SubsetSpec(
        label=safe_slug(str(label), max_len=60) if label is not None else subset_label_from_values(**values),
        dataset_key=str(dataset_key) if dataset_key is not None else None,
        pattern=str(pattern) if pattern is not None else None,
        **values,
    )


def read_subset_times_file(path: Path) -> list[SubsetSpec]:
    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Subset times file does not exist: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            rows = []
            for key, value in payload.items():
                if isinstance(value, list):
                    for item in value:
                        item = dict(item)
                        item.setdefault("dataset", key)
                        rows.append(item)
                elif isinstance(value, dict):
                    item = dict(value)
                    item.setdefault("dataset", key)
                    rows.append(item)
                else:
                    rows.append({"dataset": key, "subset": value})
        else:
            rows = list(payload)
        return [subset_spec_from_row(dict(row)) for row in rows]
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [subset_spec_from_row(dict(row)) for row in rows]

    text = path.read_text(encoding="utf-8")
    sample = text[:4096]
    dialect = csv.Sniffer().sniff(sample, delimiters=",\t") if sample.strip() else csv.excel
    reader = csv.DictReader(text.splitlines(), dialect=dialect)
    return [subset_spec_from_row(dict(row)) for row in reader]


def dataset_match_values(bin_path: Path, root: Path) -> set[str]:
    values = {
        str(bin_path),
        bin_path.name,
        bin_path.stem,
        bin_path.parent.name,
    }
    try:
        values.add(str(bin_path.relative_to(root)))
    except ValueError:
        pass
    return values


def subset_matches_dataset(spec: SubsetSpec, bin_path: Path, root: Path) -> bool:
    values = dataset_match_values(bin_path, root)
    if spec.dataset_key is not None and spec.dataset_key in values:
        return True
    if spec.pattern is not None:
        return any(fnmatch.fnmatch(value, spec.pattern) for value in values)
    return spec.dataset_key is None and spec.pattern is None


def build_subset_specs_for_dataset(
    *,
    args: argparse.Namespace,
    bin_path: Path,
    root: Path,
    global_specs: list[SubsetSpec],
    file_specs: list[SubsetSpec],
) -> list[SubsetSpec]:
    matched = list(global_specs)
    matched.extend(spec for spec in file_specs if subset_matches_dataset(spec, bin_path, root))
    if matched:
        return matched
    if args.saion_unlisted_datasets == "skip":
        return []
    return [SubsetSpec(label="full")]


def build_worker_args_for_submit(args: argparse.Namespace) -> list[str]:
    worker_args: list[str] = []
    if args.channels != "all":
        worker_args += ["--channels", str(args.channels)]
    if args.drop_sync:
        worker_args.append("--drop-sync")
    if args.chunk_samples is not None:
        worker_args += ["--chunk-samples", str(args.chunk_samples)]
    if args.chunk_sec is not None:
        worker_args += ["--chunk-sec", str(args.chunk_sec)]
    if args.progress_seconds != DEFAULT_PROGRESS_SECONDS:
        worker_args += ["--progress-seconds", str(args.progress_seconds)]
    if args.no_sidecar:
        worker_args.append("--no-sidecar")
    if args.reuse_output:
        worker_args.append("--reuse-output")
    if args.overwrite:
        worker_args.append("--overwrite")
    if not args.run_kilosort:
        worker_args.append("--no-run-kilosort")
    if args.probe_path is not None:
        worker_args += ["--probe-path", str(args.probe_path)]
    elif args.probe_name != DEFAULT_KILOSORT_PROBE_NAME:
        worker_args += ["--probe-name", str(args.probe_name)]
    if not args.auto_spikeglx_probe:
        worker_args.append("--no-auto-spikeglx-probe")
    if args.ks_data_dtype != "int16":
        worker_args += ["--ks-data-dtype", str(args.ks_data_dtype)]
    if args.ks_device != "auto":
        worker_args += ["--ks-device", str(args.ks_device)]
    for flag_name, cli_name in (
        ("ks_no_car", "--ks-no-car"),
        ("ks_invert_sign", "--ks-invert-sign"),
        ("ks_clear_cache", "--ks-clear-cache"),
        ("ks_save_preprocessed_copy", "--ks-save-preprocessed-copy"),
        ("ks_save_extra_vars", "--ks-save-extra-vars"),
        ("ks_verbose_log", "--ks-verbose-log"),
    ):
        if getattr(args, flag_name):
            worker_args.append(cli_name)
    if not args.ks_verbose_console:
        worker_args.append("--ks-quiet-console")
    scalar_options = {
        "--ks-torch-thread-lim": args.ks_torch_thread_lim,
        "--ks-bad-channels": args.ks_bad_channels,
        "--ks-fs": args.ks_fs,
        "--ks-batch-size": args.ks_batch_size,
        "--ks-nblocks": args.ks_nblocks,
        "--ks-th-universal": args.ks_th_universal,
        "--ks-th-learned": args.ks_th_learned,
        "--ks-tmin": args.ks_tmin,
        "--ks-tmax": args.ks_tmax,
        "--ks-highpass-cutoff": args.ks_highpass_cutoff,
        "--ks-whitening-range": args.ks_whitening_range,
        "--ks-nearest-chans": args.ks_nearest_chans,
        "--ks-nearest-templates": args.ks_nearest_templates,
        "--ks-artifact-threshold": args.ks_artifact_threshold,
        "--ks-dmin": args.ks_dmin,
        "--ks-dminx": args.ks_dminx,
        "--ks-max-channel-distance": args.ks_max_channel_distance,
        "--ks-x-centers": args.ks_x_centers,
    }
    defaults = {
        "--ks-nblocks": DEFAULT_KILOSORT_NBLOCKS,
        "--ks-whitening-range": DEFAULT_KILOSORT_WHITENING_RANGE,
        "--ks-nearest-chans": DEFAULT_KILOSORT_NEAREST_CHANS,
        "--ks-dmin": DEFAULT_KILOSORT_DMIN_UM,
        "--ks-dminx": DEFAULT_KILOSORT_DMINX_UM,
        "--ks-max-channel-distance": DEFAULT_KILOSORT_MAX_CHANNEL_DISTANCE_UM,
        "--ks-x-centers": DEFAULT_KILOSORT_X_CENTERS,
    }
    for option, value in scalar_options.items():
        if value is None:
            continue
        if option in defaults and value == defaults[option]:
            continue
        worker_args += [option, str(value)]
    for shank in args.ks_shank_idx or []:
        worker_args += ["--ks-shank-idx", str(shank)]
    for setting in args.ks_setting or []:
        worker_args += ["--ks-setting", str(setting)]
    return worker_args


def time_args_from_subset_spec(spec: SubsetSpec) -> list[str]:
    args: list[str] = []
    if spec.start_sec is not None:
        args += ["--start-sec", str(spec.start_sec)]
    if spec.end_sec is not None:
        args += ["--end-sec", str(spec.end_sec)]
    if spec.duration_sec is not None:
        args += ["--duration-sec", str(spec.duration_sec)]
    if spec.start_sample is not None:
        args += ["--start-sample", str(spec.start_sample)]
    if spec.stop_sample is not None:
        args += ["--stop-sample", str(spec.stop_sample)]
    return args


def dataset_slug_for_path(bin_path: Path, root: Path) -> str:
    try:
        rel = bin_path.relative_to(root)
    except ValueError:
        rel = bin_path.name
    rel_text = str(rel)
    if rel_text.endswith(".bin"):
        rel_text = rel_text[:-4]
    return safe_slug(rel_text, max_len=160)


def is_bucket_path(path: Path) -> bool:
    try:
        resolved = path.expanduser().resolve()
    except OSError:
        resolved = path.expanduser().absolute()
    return str(resolved).startswith("/bucket/")


def default_saion_work_root(output_root: Path) -> Path:
    return DEFAULT_SAION_WORK_BASE / "kilosort_runs" / safe_slug(str(output_root), max_len=120)


def resolve_saion_resources(args: argparse.Namespace) -> tuple[int, int, str, str]:
    caps = SAION_PARTITION_CAPS.get(args.saion_partition)
    if caps is not None:
        gpu_cap, cpu_cap, mem_cap_gb, default_wall = caps
        concurrency = args.saion_concurrency or gpu_cap
        cpus = args.saion_cpus or max(1, cpu_cap // concurrency)
        mem = args.saion_mem or f"{max(1, mem_cap_gb // concurrency)}G"
        time_limit = args.saion_time or default_wall
        if concurrency > gpu_cap:
            print(
                f"[WARN] --saion-concurrency {concurrency} exceeds {args.saion_partition} "
                f"per-user GPU cap {gpu_cap}; extra tasks may pend.",
                file=sys.stderr,
            )
    else:
        concurrency = args.saion_concurrency or 8
        cpus = args.saion_cpus or 16
        mem = args.saion_mem or "128G"
        time_limit = args.saion_time or "0-12"

    if concurrency <= 0:
        raise ValueError("--saion-concurrency must be positive")
    if cpus <= 0:
        raise ValueError("--saion-cpus must be positive")
    return int(concurrency), int(cpus), str(mem), str(time_limit)


def render_saion_runner() -> str:
    return '''#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


def load_task(worklist_path: Path, task_id: int) -> dict:
    with worklist_path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx == task_id:
                return json.loads(line)
    raise IndexError(f"SLURM_ARRAY_TASK_ID={task_id} is outside worklist {worklist_path}")


def run_checked(cmd: list[str], *, label: str) -> None:
    print(f"[{label}] {shlex.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def publish_file(src: str | None, dst: str | None, publish_host: str | None) -> None:
    if not src or not dst:
        return
    src_path = Path(src)
    if not src_path.exists():
        print(f"[PUBLISH] skip missing file: {src_path}", flush=True)
        return
    dst_path = Path(dst)
    direct_cmd = ["rsync", "-a", str(src_path), str(dst_path)]
    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        run_checked(direct_cmd, label="PUBLISH")
        return
    except (OSError, subprocess.CalledProcessError) as exc:
        if not publish_host:
            raise
        code = getattr(exc, "returncode", exc.__class__.__name__)
        print(f"[PUBLISH] direct file sync failed with {code}; retrying through {publish_host}", flush=True)
    run_checked(["ssh", publish_host, "mkdir", "-p", str(dst_path.parent)], label="PUBLISH")
    run_checked(["rsync", "-a", str(src_path), f"{publish_host}:{dst_path}"], label="PUBLISH")


def publish_dir(src: str | None, dst: str | None, publish_host: str | None) -> None:
    if not src or not dst:
        return
    src_path = Path(src)
    if not src_path.exists():
        print(f"[PUBLISH] skip missing dir: {src_path}", flush=True)
        return
    dst_path = Path(dst)
    direct_cmd = ["rsync", "-a", f"{src_path}/", f"{dst_path}/"]
    try:
        dst_path.mkdir(parents=True, exist_ok=True)
        run_checked(direct_cmd, label="PUBLISH")
        return
    except (OSError, subprocess.CalledProcessError) as exc:
        if not publish_host:
            raise
        code = getattr(exc, "returncode", exc.__class__.__name__)
        print(f"[PUBLISH] direct dir sync failed with {code}; retrying through {publish_host}", flush=True)
    run_checked(["ssh", publish_host, "mkdir", "-p", str(dst_path)], label="PUBLISH")
    run_checked(["rsync", "-a", f"{src_path}/", f"{publish_host}:{dst_path}/"], label="PUBLISH")


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} CONFIG_JSON", file=sys.stderr)
        return 2

    config_path = Path(sys.argv[1])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    task = load_task(Path(config["worklist_path"]), task_id)

    output_bin = task.get("output_bin")
    results_dir = task.get("results_dir")
    if output_bin:
        Path(output_bin).parent.mkdir(parents=True, exist_ok=True)
    if results_dir:
        Path(results_dir).mkdir(parents=True, exist_ok=True)

    python_exe = config.get("python_exe") or sys.executable
    cmd = [python_exe, "-u", config["script_path"], task["bin_path"]]
    if output_bin:
        cmd += ["--output", output_bin]
    if results_dir:
        cmd += ["--ks-results-dir", results_dir]
    cmd += task.get("time_args", [])
    cmd += config.get("worker_args", [])

    print(f"[TASK] index={task_id} dataset={task.get('dataset_label')} subset={task.get('subset_label')}", flush=True)
    print(f"[TASK] source={task['bin_path']}", flush=True)
    print(f"[TASK] output_bin={output_bin or '(input binary direct)'}", flush=True)
    print(f"[TASK] results_dir={results_dir}", flush=True)
    print(f"[CMD] {shlex.join(cmd)}", flush=True)
    completed = subprocess.run(cmd)
    if completed.returncode != 0:
        return int(completed.returncode)

    publish_host = config.get("publish_host") or None
    if task.get("publish_output_bin"):
        publish_file(task.get("output_bin"), task.get("publish_output_bin"), publish_host)
        publish_file(
            f"{task.get('output_bin')}.json" if task.get("output_bin") else None,
            f"{task.get('publish_output_bin')}.json" if task.get("publish_output_bin") else None,
            publish_host,
        )
    if task.get("publish_results_dir"):
        publish_dir(task.get("results_dir"), task.get("publish_results_dir"), publish_host)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def render_saion_sbatch(
    *,
    args: argparse.Namespace,
    jobs_root: Path,
    runner_path: Path,
    config_path: Path,
    task_count: int,
    concurrency: int,
    cpus: int,
    mem: str,
    time_limit: str,
) -> str:
    logs_dir = jobs_root / "logs"
    lines = [
        "#!/bin/bash -l",
        f"#SBATCH -t {time_limit}",
        f"#SBATCH -c {cpus}",
        f"#SBATCH --partition={args.saion_partition}",
        f"#SBATCH --mem={mem}",
    ]
    if args.saion_gres:
        lines.append(f"#SBATCH --gres={args.saion_gres}")
    lines.extend(
        [
            f"#SBATCH -J {args.saion_job_name}",
            f"#SBATCH -o {logs_dir}/kilosort_%A_%a.out",
            f"#SBATCH -e {logs_dir}/kilosort_%A_%a.err",
            f"#SBATCH --array=0-{task_count - 1}%{concurrency}",
            "set -euo pipefail",
            "export PYTHONNOUSERSITE=1",
            "export PYTHONDONTWRITEBYTECODE=1",
            f"export OMP_NUM_THREADS=${{OMP_NUM_THREADS:-{cpus}}}",
            f"export MKL_NUM_THREADS=${{MKL_NUM_THREADS:-{cpus}}}",
            f"export NUMEXPR_NUM_THREADS=${{NUMEXPR_NUM_THREADS:-{cpus}}}",
            "export TORCH_NUM_INTEROP_THREADS=${TORCH_NUM_INTEROP_THREADS:-1}",
            "",
            "set +e",
            "source ~/.bashrc",
            "_bashrc_status=$?",
            "set -e",
            'if [[ "${_bashrc_status}" -ne 0 ]]; then',
            '  echo "[WARN] ~/.bashrc returned status ${_bashrc_status}; falling back to direct conda initialization if needed."',
            "fi",
            "if ! command -v conda >/dev/null 2>&1; then",
            "  for _conda_sh in \\",
            "    /bucket/ReiterU/sam/miniforge3/etc/profile.d/conda.sh \\",
            "    /bucket/.deigo/ReiterU/sam/miniforge3/etc/profile.d/conda.sh \\",
            "    $HOME/miniforge3/etc/profile.d/conda.sh \\",
            "    $HOME/miniconda3/etc/profile.d/conda.sh; do",
            '    if [[ -f "${_conda_sh}" ]]; then',
            '      source "${_conda_sh}"',
            "      break",
            "    fi",
            "  done",
            "fi",
            "unset _bashrc_status _conda_sh || true",
        ]
    )
    for module_name in args.saion_module or []:
        lines.append(f"module load {shlex.quote(module_name)}")
    if args.saion_conda_env:
        lines.append(f"conda activate {shlex.quote(args.saion_conda_env)}")
    lines.extend(
        [
            "",
            'echo "[INFO] start=$(date)"',
            'echo "[INFO] host=$(hostname)"',
            'echo "[INFO] job=${SLURM_JOB_ID:-manual} array=${SLURM_ARRAY_JOB_ID:-manual} task=${SLURM_ARRAY_TASK_ID:-0}"',
            'echo "[INFO] partition=${SLURM_JOB_PARTITION:-unknown} cpus=${SLURM_CPUS_PER_TASK:-unset}"',
            'python --version',
            "nvidia-smi || true",
            f"python -u {shlex.quote(str(runner_path))} {shlex.quote(str(config_path))}",
            'echo "[INFO] done=$(date)"',
            "",
        ]
    )
    return "\n".join(lines)


def submit_saion_jobs(args: argparse.Namespace) -> None:
    if args.output is not None:
        raise ValueError("--output is for single-recording mode; use --saion-output-root with --submit-saion")
    if args.ks_results_dir is not None:
        raise ValueError("--ks-results-dir is for single-recording mode; Saion mode creates one results dir per task")

    input_path = args.input.expanduser().resolve()
    scan_root = input_path.parent if input_path.is_file() else input_path
    ap_bins = find_ap_bins_recursive(input_path)
    publish_root = (args.saion_publish_root or args.saion_output_root or (scan_root / "kilosort_saion")).expanduser().resolve()
    if args.saion_work_root is not None:
        work_root = args.saion_work_root.expanduser().resolve()
    elif is_bucket_path(publish_root):
        work_root = default_saion_work_root(publish_root).resolve()
    else:
        work_root = publish_root
    jobs_root = (args.saion_jobs_root or (work_root / "jobs")).expanduser().resolve()
    logs_dir = jobs_root / "logs"
    jobs_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    publish_enabled = publish_root != work_root

    global_specs: list[SubsetSpec] = []
    normal_spec = normal_time_args_as_subset(args)
    if normal_spec is not None:
        global_specs.append(normal_spec)
    for spec_text in args.subset_time or []:
        global_specs.append(parse_subset_time_spec(spec_text))
    file_specs = read_subset_times_file(args.subset_times_file) if args.subset_times_file is not None else []

    worker_args = build_worker_args_for_submit(args)
    tasks: list[dict[str, Any]] = []
    label_counts: dict[tuple[str, str], int] = {}
    needs_conversion_for_channels = args.channels != "all" or bool(args.drop_sync)
    for bin_path in ap_bins:
        dataset_label = dataset_slug_for_path(bin_path, scan_root)
        specs = build_subset_specs_for_dataset(
            args=args,
            bin_path=bin_path,
            root=scan_root,
            global_specs=global_specs,
            file_specs=file_specs,
        )
        for spec in specs:
            subset_label = safe_slug(spec.label or "full", max_len=80)
            key = (dataset_label, subset_label)
            label_counts[key] = label_counts.get(key, 0) + 1
            if label_counts[key] > 1:
                subset_label = f"{subset_label}_{label_counts[key]:02d}"

            task_dir = work_root / dataset_label / subset_label
            publish_task_dir = publish_root / dataset_label / subset_label if publish_enabled else None
            results_dir = task_dir / "kilosort4"
            conversion_needed = needs_conversion_for_channels or not spec.is_full
            output_bin = None
            publish_output_bin = None
            if conversion_needed:
                output_bin = task_dir / f"{safe_slug(bin_path.stem, max_len=80)}_{subset_label}.bin"
                if publish_task_dir is not None:
                    publish_output_bin = publish_task_dir / output_bin.name
            tasks.append(
                {
                    "task_index": len(tasks),
                    "bin_path": str(bin_path),
                    "dataset_label": dataset_label,
                    "subset_label": subset_label,
                    "output_bin": str(output_bin) if output_bin is not None else None,
                    "results_dir": str(results_dir),
                    "publish_output_bin": str(publish_output_bin) if publish_output_bin is not None else None,
                    "publish_results_dir": str(publish_task_dir / "kilosort4") if publish_task_dir is not None else None,
                    "time_args": time_args_from_subset_spec(spec),
                    "start_sec": spec.start_sec,
                    "end_sec": spec.end_sec,
                    "duration_sec": spec.duration_sec,
                    "start_sample": spec.start_sample,
                    "stop_sample": spec.stop_sample,
                }
            )

    if not tasks:
        raise ValueError("No Saion tasks were generated. Check --saion-unlisted-datasets and subset filters.")

    worklist_path = jobs_root / "kilosort_worklist.jsonl"
    config_path = jobs_root / "kilosort_submit_config.json"
    runner_path = jobs_root / "run_kilosort_task.py"
    sbatch_path = jobs_root / "kilosort_array.sbatch"
    manifest_path = jobs_root / "kilosort_submit_manifest.json"

    with worklist_path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task, sort_keys=True) + "\n")

    runner_path.write_text(render_saion_runner(), encoding="utf-8")
    runner_path.chmod(0o755)

    concurrency, cpus, mem, time_limit = resolve_saion_resources(args)
    config = {
        "script_path": str(Path(__file__).resolve()),
        "worklist_path": str(worklist_path),
        "python_exe": args.saion_python or "python",
        "worker_args": worker_args,
        "publish_host": args.saion_publish_host,
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    sbatch_path.write_text(
        render_saion_sbatch(
            args=args,
            jobs_root=jobs_root,
            runner_path=runner_path,
            config_path=config_path,
            task_count=len(tasks),
            concurrency=concurrency,
            cpus=cpus,
            mem=mem,
            time_limit=time_limit,
        ),
        encoding="utf-8",
    )
    sbatch_path.chmod(0o755)

    manifest = {
        "input": str(input_path),
        "scan_root": str(scan_root),
        "publish_root": str(publish_root),
        "work_root": str(work_root),
        "jobs_root": str(jobs_root),
        "publish_host": args.saion_publish_host,
        "ap_dataset_count": len(ap_bins),
        "task_count": len(tasks),
        "partition": args.saion_partition,
        "gres": args.saion_gres,
        "concurrency": concurrency,
        "cpus": cpus,
        "mem": mem,
        "time": time_limit,
        "conda_env": args.saion_conda_env,
        "worklist": str(worklist_path),
        "runner": str(runner_path),
        "config": str(config_path),
        "sbatch": str(sbatch_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("Saion Kilosort submission:")
    print(f"  input: {input_path}")
    print(f"  AP datasets: {len(ap_bins):,}")
    print(f"  tasks: {len(tasks):,}")
    print(f"  work_root: {work_root}")
    print(f"  publish_root: {publish_root}")
    if publish_enabled:
        print(f"  publish_host: {args.saion_publish_host or '(direct only)'}")
    print(f"  jobs_root: {jobs_root}")
    print(f"  worklist: {worklist_path}")
    print(f"  sbatch: {sbatch_path}")
    print(
        "  resources: "
        f"partition={args.saion_partition} gres={args.saion_gres or '(none)'} "
        f"array=0-{len(tasks) - 1}%{concurrency} cpus={cpus} mem={mem} time={time_limit}"
    )
    if global_specs:
        print(f"  global subsets: {', '.join(spec.label for spec in global_specs)}")
    if file_specs:
        print(f"  subset file specs: {len(file_specs):,}")

    submit_cmd = [args.saion_sbatch_bin, "--parsable", str(sbatch_path)]
    if args.dry_run:
        print("dry run: sbatch not submitted")
        print(f"submit command: {shlex.join(submit_cmd)}")
        return

    print(f"submitting: {shlex.join(submit_cmd)}")
    try:
        completed = subprocess.run(submit_cmd, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        if exc.stdout:
            print(exc.stdout, file=sys.stderr, end="")
        if exc.stderr:
            print(exc.stderr, file=sys.stderr, end="")
        raise
    job_id = completed.stdout.strip()
    (jobs_root / "kilosort_array_job_id.txt").write_text(job_id + "\n", encoding="utf-8")
    print(f"submitted Slurm job: {job_id}")
    print(f"logs: {logs_dir}/kilosort_{job_id}_<task>.out")


def format_setting_value(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def build_kilosort_settings(
    *,
    args: argparse.Namespace,
    rec: SpikeGlxBinary,
    channels: np.ndarray,
    output_path: Path,
    results_dir: Path,
    probe_path: Path | None,
) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "n_chan_bin": int(channels.size),
        "fs": float(args.ks_fs if args.ks_fs is not None else rec.sample_rate),
    }
    optional_settings = {
        "batch_size": args.ks_batch_size,
        "nblocks": args.ks_nblocks,
        "Th_universal": args.ks_th_universal,
        "Th_learned": args.ks_th_learned,
        "tmin": args.ks_tmin,
        "tmax": args.ks_tmax,
        "highpass_cutoff": args.ks_highpass_cutoff,
        "whitening_range": args.ks_whitening_range,
        "nearest_chans": args.ks_nearest_chans,
        "nearest_templates": args.ks_nearest_templates,
        "artifact_threshold": args.ks_artifact_threshold,
        "dmin": args.ks_dmin,
        "dminx": args.ks_dminx,
        "max_channel_distance": args.ks_max_channel_distance,
        "x_centers": args.ks_x_centers,
    }
    for key, value in optional_settings.items():
        if value is not None:
            settings[key] = value

    for item in args.ks_setting or []:
        key, value = parse_key_value(item)
        settings[key] = value

    # Keep the settings tied to the converted binary even if a generic
    # --ks-setting tries to override these by mistake.
    settings["filename"] = str(output_path)
    settings["results_dir"] = str(results_dir)
    settings["n_chan_bin"] = int(channels.size)
    settings["fs"] = float(args.ks_fs if args.ks_fs is not None else rec.sample_rate)
    if probe_path is not None:
        settings["probe_path"] = str(probe_path)
    return settings


def resolve_kilosort_probe(
    *,
    args: argparse.Namespace,
    rec: SpikeGlxBinary,
    channels: np.ndarray,
    output_path: Path,
    results_dir: Path,
) -> tuple[str | None, Path | None]:
    if args.probe_path is not None:
        return None, args.probe_path.expanduser().resolve()

    if (
        args.auto_spikeglx_probe
        and args.probe_name == DEFAULT_KILOSORT_PROBE_NAME
        and rec.meta.get("snsGeomMap")
    ):
        probe_path = results_dir / f"{safe_slug(output_path.stem, max_len=80)}.spikeglx_probe.json"
        generated_path, connected, total = write_spikeglx_probe_json(
            probe_path,
            rec=rec,
            channels=channels,
            dry_run=bool(args.dry_run),
        )
        action = "would write" if args.dry_run else "wrote"
        print(
            f"auto SpikeGLX probe: {action} {generated_path} "
            f"({connected:,}/{total:,} connected channels)"
        )
        return None, generated_path

    if args.auto_spikeglx_probe and args.probe_name == DEFAULT_KILOSORT_PROBE_NAME:
        print(
            "[WARN] SpikeGLX snsGeomMap not found; using Kilosort probe cache "
            f"entry {args.probe_name}.",
            file=sys.stderr,
        )
    return args.probe_name, None


def kilosort_device(device_text: str):
    if device_text == "auto":
        return None
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required to set --ks-device. Run from the kilosort "
            "environment or omit --ks-device."
        ) from exc
    return torch.device(device_text)


def run_kilosort_on_output(
    *,
    args: argparse.Namespace,
    rec: SpikeGlxBinary,
    channels: np.ndarray,
    output_path: Path,
) -> None:
    output_path = output_path.resolve()
    results_dir = (args.ks_results_dir or default_kilosort_results_dir(output_path)).resolve()
    probe_name, probe_path = resolve_kilosort_probe(
        args=args,
        rec=rec,
        channels=channels,
        output_path=output_path,
        results_dir=results_dir,
    )
    settings = build_kilosort_settings(
        args=args,
        rec=rec,
        channels=channels,
        output_path=output_path,
        results_dir=results_dir,
        probe_path=probe_path,
    )
    bad_channels = parse_index_spec(args.ks_bad_channels) if args.ks_bad_channels else None
    shank_idx = args.ks_shank_idx
    if shank_idx is not None and len(shank_idx) == 1:
        shank_idx = shank_idx[0]

    print("Kilosort run:")
    print(f"  binary: {output_path}")
    print(f"  results: {results_dir}")
    print(f"  probe_name: {probe_name if probe_name is not None else '(using --probe-path)'}")
    print(f"  probe_path: {probe_path if probe_path is not None else ''}")
    print(f"  n_chan_bin: {settings['n_chan_bin']}")
    print(f"  fs: {settings['fs']}")
    print(f"  console log level: {'DEBUG' if args.ks_verbose_console else 'INFO'}")
    print(f"  debug log statements: {'enabled' if args.ks_verbose_log else 'disabled'}")
    print(f"  log file: {results_dir / 'kilosort4.log'}")
    print("  settings:")
    for key in sorted(settings):
        print(f"    {key}: {format_setting_value(settings[key])}")
    if args.dry_run:
        print("dry run: Kilosort not launched")
        return

    print("importing Kilosort...", flush=True)
    try:
        from kilosort import run_kilosort
    except ImportError as exc:
        raise RuntimeError(
            "Kilosort is not importable in this Python environment. Run this "
            "script with `conda activate kilosort`, or pass --no-run-kilosort "
            "to only write the subset binary."
        ) from exc

    print("launching Kilosort; built-in stage logs and progress bars follow.", flush=True)
    started = time.monotonic()
    ops, st, clu, _tF, _Wall, _similar_templates, _is_ref, _est_contam_rate, kept_spikes = run_kilosort(
        settings=settings,
        probe_name=probe_name,
        filename=output_path,
        results_dir=results_dir,
        data_dtype=args.ks_data_dtype,
        do_CAR=not args.ks_no_car,
        invert_sign=bool(args.ks_invert_sign),
        device=kilosort_device(args.ks_device),
        save_extra_vars=bool(args.ks_save_extra_vars),
        clear_cache=bool(args.ks_clear_cache),
        save_preprocessed_copy=bool(args.ks_save_preprocessed_copy),
        bad_channels=bad_channels,
        shank_idx=shank_idx,
        verbose_console=bool(args.ks_verbose_console),
        verbose_log=bool(args.ks_verbose_log),
        torch_thread_lim=args.ks_torch_thread_lim,
    )
    elapsed = time.monotonic() - started
    print(f"Kilosort done in {elapsed / 60.0:.2f} min")
    print(f"results_dir: {results_dir}")
    print(f"log file: {results_dir / 'kilosort4.log'}")
    print(f"spikes: {len(st):,}")
    print(f"clusters: {len(np.unique(clu)):,}")
    print(f"kept_spikes: {int(np.sum(kept_spikes)):,}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Trim a SpikeGLX binary into a Kilosort-compatible int16 binary, "
            "then run Kilosort by default. Pass a *.ap.bin file, a directory "
            "containing exactly one *.ap.bin, or use --submit-saion to fan out "
            "all recursive *.ap.bin datasets as a Saion Slurm array."
        )
    )
    convert = parser.add_argument_group("conversion")
    convert.add_argument("input", type=Path, help="SpikeGLX *.ap.bin file, or a directory containing one.")
    convert.add_argument("-o", "--output", type=Path, default=None, help="Output .bin path. Defaults beside input.")
    convert.add_argument("--start-sec", type=float, default=None, help="Subset start time in seconds.")
    convert.add_argument("--end-sec", type=float, default=None, help="Subset stop time in seconds, exclusive.")
    convert.add_argument("--duration-sec", type=float, default=None, help="Subset duration in seconds.")
    convert.add_argument("--start-sample", type=int, default=None, help="Subset start sample.")
    convert.add_argument("--stop-sample", type=int, default=None, help="Subset stop sample, exclusive.")
    convert.add_argument(
        "--channels",
        default="all",
        help=(
            "Saved channels to write. Use 'all' (default), 'ap'/'neural', comma-separated "
            "indices, inclusive dash ranges like 0-383, or Python slices like 0:384."
        ),
    )
    convert.add_argument(
        "--drop-sync",
        action="store_true",
        help="Drop trailing imec sync channel(s) from the selected channels using snsApLfSy.",
    )
    convert.add_argument("--chunk-samples", type=int, default=None, help=f"Samples per copy chunk. Default {DEFAULT_CHUNK_SAMPLES}.")
    convert.add_argument("--chunk-sec", type=float, default=None, help="Seconds per copy chunk.")
    convert.add_argument("--progress-seconds", type=float, default=DEFAULT_PROGRESS_SECONDS, help=f"Progress print interval. Use 0 to disable. Default: {DEFAULT_PROGRESS_SECONDS}.")
    convert.add_argument("--no-sidecar", action="store_true", help="Do not write output .json settings sidecar.")
    convert.add_argument("--reuse-output", action="store_true", help="If the converted output already exists and has the expected size, skip copying and run Kilosort on it.")
    convert.add_argument("--overwrite", action="store_true", help="Replace existing output and sidecar.")
    convert.add_argument("--dry-run", action="store_true", help="Print planned conversion and Kilosort run without writing or sorting.")

    kilosort = parser.add_argument_group("kilosort")
    kilosort.set_defaults(
        run_kilosort=True,
        ks_verbose_console=DEFAULT_KILOSORT_VERBOSE_CONSOLE,
        auto_spikeglx_probe=True,
    )
    kilosort.add_argument("--run-kilosort", dest="run_kilosort", action="store_true", help="Run Kilosort after writing the subset binary. This is the default.")
    kilosort.add_argument("--no-run-kilosort", dest="run_kilosort", action="store_false", help="Only write the subset binary; do not run Kilosort.")
    kilosort.add_argument("--probe-name", default=DEFAULT_KILOSORT_PROBE_NAME, help=f"Kilosort probe filename in the probe cache. Default: {DEFAULT_KILOSORT_PROBE_NAME}.")
    kilosort.add_argument("--probe-path", type=Path, default=None, help="Full path to a Kilosort probe .mat/.prb/.json file. Overrides --probe-name.")
    kilosort.add_argument(
        "--auto-spikeglx-probe",
        dest="auto_spikeglx_probe",
        action="store_true",
        help=(
            "When using the default NeuroPixUltra probe, write a per-run Kilosort probe "
            "from SpikeGLX snsGeomMap and use it as --probe-path. This is the default."
        ),
    )
    kilosort.add_argument(
        "--no-auto-spikeglx-probe",
        dest="auto_spikeglx_probe",
        action="store_false",
        help="Use --probe-name exactly and do not auto-generate a probe from SpikeGLX metadata.",
    )
    kilosort.add_argument("--ks-results-dir", type=Path, default=None, help="Kilosort results directory. Defaults to OUTPUT_STEM_kilosort4 beside the output binary.")
    kilosort.add_argument("--ks-data-dtype", default="int16", help="Kilosort data_dtype. Default: int16.")
    kilosort.add_argument("--ks-device", default="auto", help="Kilosort device: auto, cpu, cuda, cuda:0, etc. Default: auto.")
    kilosort.add_argument("--ks-no-car", action="store_true", help="Disable Kilosort common average reference preprocessing.")
    kilosort.add_argument("--ks-invert-sign", action="store_true", help="Run Kilosort with invert_sign=True.")
    kilosort.add_argument("--ks-clear-cache", action="store_true", help="Run Kilosort with clear_cache=True.")
    kilosort.add_argument("--ks-save-preprocessed-copy", action="store_true", help="Save Kilosort preprocessed temp_wh.dat and point Phy at it.")
    kilosort.add_argument("--ks-save-extra-vars", action="store_true", help="Save extra Kilosort variables.")
    kilosort.add_argument("--ks-verbose-console", dest="ks_verbose_console", action="store_true", help="Use DEBUG logging on the console. This is the default.")
    kilosort.add_argument("--ks-quiet-console", dest="ks_verbose_console", action="store_false", help="Use INFO logging on the console instead of DEBUG.")
    kilosort.add_argument("--ks-verbose-log", action="store_true", help="Use extra DEBUG logging in the Kilosort log.")
    kilosort.add_argument("--ks-torch-thread-lim", type=int, default=None, help="Limit PyTorch CPU thread count.")
    kilosort.add_argument("--ks-bad-channels", default=None, help="Bad output-binary channel rows, e.g. 0,2,10-15.")
    kilosort.add_argument("--ks-shank-idx", type=float, action="append", default=None, help="Restrict sorting to a shank index. Repeat to sort multiple shanks.")
    kilosort.add_argument("--ks-fs", type=float, default=None, help="Override Kilosort fs. Defaults to SpikeGLX imSampRate.")
    kilosort.add_argument("--ks-batch-size", type=int, default=None, help="Kilosort setting: batch_size.")
    kilosort.add_argument("--ks-nblocks", type=int, default=DEFAULT_KILOSORT_NBLOCKS, help=f"Kilosort setting: nblocks. Default: {DEFAULT_KILOSORT_NBLOCKS}.")
    kilosort.add_argument("--ks-th-universal", type=float, default=None, help="Kilosort setting: Th_universal.")
    kilosort.add_argument("--ks-th-learned", type=float, default=None, help="Kilosort setting: Th_learned.")
    kilosort.add_argument("--ks-tmin", type=float, default=None, help="Kilosort setting: tmin, in seconds within the converted subset.")
    kilosort.add_argument("--ks-tmax", type=float, default=None, help="Kilosort setting: tmax, in seconds within the converted subset.")
    kilosort.add_argument("--ks-highpass-cutoff", type=float, default=None, help="Kilosort setting: highpass_cutoff.")
    kilosort.add_argument("--ks-whitening-range", type=int, default=DEFAULT_KILOSORT_WHITENING_RANGE, help=f"Kilosort setting: whitening_range. Default: {DEFAULT_KILOSORT_WHITENING_RANGE}.")
    kilosort.add_argument("--ks-nearest-chans", type=int, default=DEFAULT_KILOSORT_NEAREST_CHANS, help=f"Kilosort setting: nearest_chans. Default: {DEFAULT_KILOSORT_NEAREST_CHANS}.")
    kilosort.add_argument("--ks-nearest-templates", type=int, default=None, help="Kilosort setting: nearest_templates.")
    kilosort.add_argument("--ks-artifact-threshold", type=float, default=None, help="Kilosort setting: artifact_threshold.")
    kilosort.add_argument("--ks-dmin", type=float, default=DEFAULT_KILOSORT_DMIN_UM, help=f"Kilosort setting: dmin. Default: {DEFAULT_KILOSORT_DMIN_UM}.")
    kilosort.add_argument("--ks-dminx", type=float, default=DEFAULT_KILOSORT_DMINX_UM, help=f"Kilosort setting: dminx. Default: {DEFAULT_KILOSORT_DMINX_UM}.")
    kilosort.add_argument("--ks-max-channel-distance", type=float, default=DEFAULT_KILOSORT_MAX_CHANNEL_DISTANCE_UM, help=f"Kilosort setting: max_channel_distance. Default: {DEFAULT_KILOSORT_MAX_CHANNEL_DISTANCE_UM}.")
    kilosort.add_argument("--ks-x-centers", type=parse_cli_value, default=DEFAULT_KILOSORT_X_CENTERS, help=f"Kilosort setting: x_centers. Default: {DEFAULT_KILOSORT_X_CENTERS}.")
    kilosort.add_argument(
        "--ks-setting",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Additional Kilosort setting. Values are parsed as Python literals, e.g. --ks-setting nskip=25 --ks-setting drift_smoothing='[0.5,0.5,0.5]'.",
    )

    cluster = parser.add_argument_group("saion slurm submission")
    cluster.add_argument(
        "--submit-saion",
        action="store_true",
        help="Submit one Saion Slurm array task for each recursive *.ap.bin dataset/subset under INPUT.",
    )
    cluster.add_argument(
        "--subset-time",
        action="append",
        default=None,
        metavar="LABEL=START:END",
        help=(
            "Time subset to apply to every dataset in --submit-saion mode. "
            "Repeat for multiple windows. Forms: LABEL=START:END, LABEL=START:+DURATION, or LABEL=START,DURATION. Times are seconds."
        ),
    )
    cluster.add_argument(
        "--subset-times-file",
        type=Path,
        default=None,
        help=(
            "CSV/TSV/JSON/JSONL file of per-dataset subset times. Columns/keys may include "
            "dataset/path/file/name/stem, pattern/glob, label, subset/time/window, start_sec, end_sec, duration_sec."
        ),
    )
    cluster.add_argument(
        "--saion-unlisted-datasets",
        choices=("full", "skip"),
        default="full",
        help="When --subset-times-file is used, sort unlisted AP datasets as full recordings or skip them. Default: full.",
    )
    cluster.add_argument(
        "--saion-output-root",
        type=Path,
        default=None,
        help="Final root for per-dataset Kilosort outputs. Default: INPUT/kilosort_saion for directory input.",
    )
    cluster.add_argument(
        "--saion-publish-root",
        type=Path,
        default=None,
        help="Alias/override for the final publish root. Defaults to --saion-output-root.",
    )
    cluster.add_argument(
        "--saion-work-root",
        type=Path,
        default=None,
        help=(
            "Compute-writable root for intermediate binaries and Kilosort runs. "
            "Default: same as publish root, except /bucket publish roots use "
            f"{DEFAULT_SAION_WORK_BASE}/kilosort_runs/..."
        ),
    )
    cluster.add_argument(
        "--saion-publish-host",
        default="saion",
        help=(
            "SSH host used by compute jobs to publish results when direct rsync fails "
            "(for example, /bucket is read-only on compute nodes). Default: saion. "
            "Pass an empty string to disable SSH fallback."
        ),
    )
    cluster.add_argument(
        "--saion-jobs-root",
        type=Path,
        default=None,
        help="Directory for generated worklist, runner, sbatch script, and logs. Default: SAION_WORK_ROOT/jobs.",
    )
    cluster.add_argument(
        "--saion-partition",
        default="largegpu",
        help="Saion Slurm partition. Default: largegpu.",
    )
    cluster.add_argument(
        "--saion-gres",
        default="gpu:1",
        help="Slurm --gres value per task. Default: gpu:1. Pass an empty string to omit.",
    )
    cluster.add_argument(
        "--saion-concurrency",
        type=int,
        default=None,
        help="Array concurrency cap. Default auto-derived from partition; largegpu -> 8.",
    )
    cluster.add_argument(
        "--saion-cpus",
        type=int,
        default=None,
        help="CPUs per task. Default auto-derived from partition/concurrency; largegpu -> 16.",
    )
    cluster.add_argument(
        "--saion-mem",
        default=None,
        help="Memory per task. Default auto-derived from partition/concurrency; largegpu -> 128G.",
    )
    cluster.add_argument(
        "--saion-time",
        default=None,
        help="Wall time per task. Default auto-derived from partition; largegpu -> 0-12.",
    )
    cluster.add_argument(
        "--saion-conda-env",
        default="kilosort",
        help="Conda env to activate inside Saion tasks. Default: kilosort. Pass an empty string to skip activation.",
    )
    cluster.add_argument(
        "--saion-python",
        default=None,
        help="Python executable used by the generated runner after activation. Default: python.",
    )
    cluster.add_argument(
        "--saion-module",
        action="append",
        default=None,
        help="Module to load inside Saion tasks before conda activation. Repeat as needed.",
    )
    cluster.add_argument(
        "--saion-sbatch-bin",
        default="sbatch",
        help="sbatch command/path. Default: sbatch.",
    )
    cluster.add_argument(
        "--saion-job-name",
        default="ks_sort",
        help="Slurm job name for the Kilosort array. Default: ks_sort.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        if args.submit_saion:
            submit_saion_jobs(args)
            return 0

        rec = open_spikeglx(args.input)
        channels = parse_channels(args.channels, rec)
        if args.drop_sync:
            channels = channels_after_sync_drop(channels, rec)
            validate_channels(channels, rec.n_channels)
        start_sample, stop_sample = resolve_sample_bounds(args, rec)
        chunk_samples = resolve_chunk_samples(args, rec.sample_rate)
        conversion_needed = not selects_full_recording(
            rec,
            start_sample=start_sample,
            stop_sample=stop_sample,
            channels=channels,
        )
        if conversion_needed:
            output_path = args.output or default_output_path(rec.bin_path, start_sample, stop_sample, channels, rec)
            output_path = output_path.resolve()
        else:
            output_path = rec.bin_path.resolve()
            print("No time/channel subset requested; using input SpikeGLX binary directly.")
            print(f"source: {rec.bin_path}")
            print(f"source samples: {rec.n_samples:,} ({rec.duration_seconds:.3f} s)")
            print(f"source channels: {rec.n_channels:,}")
            print(f"sample rate: {rec.sample_rate:.6f} Hz")
            print(f"Kilosort binary: {output_path}")
            print(f"Kilosort n_chan_bin: {channels.size}")
            if args.output is not None:
                print(f"[WARN] ignoring --output because the full input binary is being used directly: {args.output}")

        if not conversion_needed:
            if args.reuse_output:
                print("[INFO] --reuse-output has no effect when using the input binary directly.")
            if args.overwrite:
                print("[INFO] --overwrite has no effect when using the input binary directly.")
            if args.no_sidecar:
                print("[INFO] --no-sidecar has no effect because no conversion sidecar is written.")
        elif args.reuse_output and output_path.exists() and not args.dry_run:
            verify_existing_output(
                output_path,
                output_samples=stop_sample - start_sample,
                n_channels=int(channels.size),
            )
        elif conversion_needed:
            output_path = copy_subset(
                rec,
                output_path,
                start_sample=start_sample,
                stop_sample=stop_sample,
                channels=channels,
                chunk_samples=chunk_samples,
                overwrite=bool(args.overwrite),
                write_metadata=not bool(args.no_sidecar),
                dry_run=bool(args.dry_run),
                progress_seconds=float(args.progress_seconds),
            )

        if args.run_kilosort:
            run_kilosort_on_output(
                args=args,
                rec=rec,
                channels=channels,
                output_path=output_path,
            )
        else:
            print("Kilosort skipped (--no-run-kilosort).")
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
