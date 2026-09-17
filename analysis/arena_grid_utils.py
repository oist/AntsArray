"""Rebuild occupancy grids from finished tracks using annotated arena borders."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

import numpy as np

from analysis import compute_track_grid_occupancy as compute
from analysis import grid_occupancy_utils as go


GRID_SETTINGS_FILENAME = "grid_occupancy_settings.json"


def configured_grid_size_mm(block_dir: Path, override: float | None = None) -> float:
    """Resolve an explicit bin width or the block's saved analysis preference.

    Without either, use the preprocessing default (0.25 mm), including when the
    source metadata describe an older, coarser grid. The preference lives
    beside panorama annotations, including when the caller passes ``stitched``.
    """
    root = Path(block_dir)
    if root.name == "stitched":
        root = root.parent
    value = override
    settings_path = root / GRID_SETTINGS_FILENAME
    if value is None and settings_path.is_file():
        settings = json.loads(settings_path.read_text())
        if not isinstance(settings, dict) or "grid_size_mm" not in settings:
            raise ValueError(f"Expected a grid_size_mm setting in {settings_path}")
        value = settings["grid_size_mm"]
    if value is None:
        return compute.DEFAULT_GRID_SIZE_MM
    if isinstance(value, bool):
        raise ValueError("grid_size_mm must be a positive finite number")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("grid_size_mm must be a positive finite number") from exc
    if not np.isfinite(value) or value <= 0:
        raise ValueError("grid_size_mm must be a positive finite number")
    return value


def arena_grid_output_name(grid_size_mm: float | None) -> str:
    """Separate requested resolutions from legacy caches and their analyses."""
    if grid_size_mm is None:
        return "grid_occupancy_histograms_arena"
    size = format(grid_size_mm, ".12g").replace(".", "p")
    return f"grid_occupancy_histograms_arena_{size}mm"


def arena_bounds_from_regions(regions):
    arenas = regions[regions.region_type.eq("arena")]
    if len(arenas) != 2 or not arenas["shape"].eq("rectangle").all() or set(arenas.side) != {"left", "right"}:
        raise ValueError("Draw exactly two rectangular regions labelled arena, one per side, and Save in the panorama GUI")
    result = {}
    for row in arenas.itertuples():
        bounds = {f"{key}_px": float(getattr(row, f"tracking_{key}_px"))
                  for key in ("x_min", "x_max", "y_min", "y_max")}
        if (not np.isfinite(list(bounds.values())).all()
                or bounds["x_min_px"] >= bounds["x_max_px"]
                or bounds["y_min_px"] >= bounds["y_max_px"]):
            raise ValueError(f"Invalid arena rectangle: {row.name}")
        result[row.side] = bounds
    if result["left"]["x_max_px"] - result["right"]["x_min_px"] > go.ARENA_BORDER_TOLERANCE_PX:
        raise ValueError("Arena rectangles overlap in x; draw separate left and right arenas")
    return result


def prepare_arena_grid_cache(block_dir: Path, source_grid_root: Path, regions_path: Path,
                             *, max_workers: int = 2, grid_size_mm: float | None = None) -> Path:
    """Reuse per-ant grids only when bounds, source tracking, and bin size match.

    Keep the pipeline grids untouched: arrays clipped by the old boundary
    cannot be repaired by cropping or renormalizing them. Re-read raw positions.
    """
    block_dir, source_grid_root = Path(block_dir), Path(source_grid_root)
    requested_size = configured_grid_size_mm(block_dir, grid_size_mm)
    stitched_root = go.resolve_stitched_root(block_dir)
    regions = go.load_panorama_regions(regions_path)
    bounds = arena_bounds_from_regions(regions)
    scales = regions.loc[regions.region_type.eq("arena"), "mm_per_pixel"].unique()
    if len(scales) != 1 or not np.isfinite(scales[0]) or scales[0] <= 0:
        raise ValueError("Arena annotations must use one positive calibrated mm/pixel scale")
    scale = float(scales[0])
    source_paths = go.metadata_paths(source_grid_root)
    if not source_paths:
        raise FileNotFoundError(f"No source occupancy metadata: {source_grid_root}")
    output_root = stitched_root / arena_grid_output_name(requested_size)
    print(f"Annotated arena bounds (tracking pixels): {bounds}", flush=True)
    if requested_size is not None:
        print(f"Arena grid spacing: {requested_size:g} mm; cache: {output_root}", flush=True)
    overlap = bounds["left"]["x_max_px"] - bounds["right"]["x_min_px"]
    if overlap > 0:
        print(f"Adjacent arena borders overlap by {overlap:.2f} px; preserving the drawn bounds "
              "and using their midpoint for side labels.", flush=True)

    def build(path):
        source = json.loads(path.read_text())
        name, side = source["track_name"], source["side"]
        track_path = stitched_root / "per_track" / name
        if not track_path.is_file():
            raise FileNotFoundError(f"Missing finished track for arena occupancy: {track_path}")
        if source.get("input_x_is_side_local", False):
            raise ValueError(f"Arena occupancy requires raw/global track coordinates: {name}")
        if not np.isclose(float(source["mm_per_px"]), scale, rtol=0, atol=1e-9):
            raise ValueError(f"Arena and tracking scales disagree: {name}")
        stat = track_path.stat()
        inputs = dict(version=1, track_path=str(track_path), source_size=stat.st_size,
                      source_mtime_ns=stat.st_mtime_ns, arena_bounds_px=bounds[side], side=side,
                      grid_size_mm=requested_size if requested_size is not None else float(source["grid_size_mm"]), mm_per_px=scale,
                      x_col=source.get("x_col", "TrackX"), y_col=source.get("y_col", "TrackY"),
                      bodypoint=int(source.get("bodypoint_filter", 0)))
        output = output_root / "per_track" / Path(name).stem
        metadata_path = output / "grid_occupancy_metadata.json"
        filenames = ("grid_occupancy_f4.npy", "grid_x_edges_mm.npy", "grid_y_edges_mm.npy")
        if metadata_path.is_file() and all((output / filename).is_file() for filename in filenames):
            saved = json.loads(metadata_path.read_text())
            if saved.get("arena_cache_inputs") == inputs:
                return name, False, int(saved["n_out_of_grid_detected_frames"])
        xy = compute.load_track_xy(track_path, frame_col="Frame", x_col=inputs["x_col"],
                                   y_col=inputs["y_col"], bodypoint=inputs["bodypoint"])
        split = (bounds["left"]["x_max_px"] + bounds["right"]["x_min_px"]) / 2
        parameters = dict(side=side, mm_per_px=scale, grid_size_mm=inputs["grid_size_mm"],
                          x_min_px=bounds["left"]["x_min_px"], x_split_px=split,
                          x_max_px=bounds["right"]["x_max_px"],
                          y_min_px=bounds[side]["y_min_px"], y_max_px=bounds[side]["y_max_px"],
                          input_x_is_side_local=False, same_shape_sides=False, grid_pad_mm=0.0,
                          arena_bounds_px=bounds[side])
        histogram, x_edges, y_edges, stats = compute.compute_grid_occupancy(xy, **parameters)
        output.mkdir(parents=True, exist_ok=True)
        for filename, values in zip(filenames, (histogram, x_edges, y_edges)):
            np.save(output / filename, values)
        metadata = {**source, **parameters, **stats, "track_path": str(track_path),
                    "frame_min": int(xy.Frame.min()), "frame_max": int(xy.Frame.max()),
                    "n_observed_frames": len(xy), "histogram_shape_yx": list(histogram.shape),
                    "occupancy_path": str(output / filenames[0]), "x_edges_mm_path": str(output / filenames[1]),
                    "y_edges_mm_path": str(output / filenames[2]), "bounds_json": None,
                    "bounds_source": "panorama_arena_regions", "arena_regions_path": str(regions_path),
                    "arena_cache_inputs": inputs, "normalization": "bin_count / n_detected_frames",
                    "occupancy_units": "fraction_of_detected_frames", "dtype": str(histogram.dtype)}
        temporary = metadata_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(metadata, indent=2) + "\n")
        temporary.replace(metadata_path)
        return name, True, int(stats["n_out_of_grid_detected_frames"])

    rebuilt, outside = 0, 0
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
        futures = [pool.submit(build, path) for path in source_paths]
        for number, future in enumerate(as_completed(futures), 1):
            _, changed, excluded = future.result()
            rebuilt += int(changed)
            outside += excluded
            if number == 1 or number % 10 == 0 or number == len(source_paths):
                print(f"Arena occupancy: {number}/{len(source_paths)} ants ({rebuilt} rebuilt)", flush=True)
    print(f"Arena occupancy excludes {outside:,} detected ant-frames outside drawn borders; "
          "per-ant excluded counts remain in metadata.", flush=True)
    return output_root
