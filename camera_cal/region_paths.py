"""Block-local panorama annotations and optional earlier-date lookup for tracking."""

from pathlib import Path
import csv
from datetime import date, datetime
import math
import re


TRACKING_ARENA_LABELS = {"left": "arena_left", "right": "arena_right"}
_ARENA_LABEL = re.compile(r"arena(?:[_\s-]*(left|right|l|r))?", re.IGNORECASE)


def arena_label_side(label: str) -> str | None:
    """Recognize the side of an explicitly labelled arena, including old spellings."""
    match = _ARENA_LABEL.fullmatch(label.strip())
    if match is None or match[1] is None:
        return None
    return "left" if match[1].lower() in {"l", "left"} else "right"


def _recording_date(name: str) -> date | None:
    """Recognize recording folders such as 20260515 or 20251117_2_stim."""
    match = re.match(r"^(\d{8})(?:$|[_-])", name)
    if match is None:
        return None
    try:
        return datetime.strptime(match[1], "%Y%m%d").date()
    except ValueError:
        return None


def panorama_dataset_dir(block_dir: Path) -> Path:
    block_dir = Path(block_dir)
    if (block_dir.name.startswith("block")
            and _recording_date(block_dir.parent.name) is not None):
        return block_dir.parent
    return block_dir


def panorama_regions_path(
    block_dir: Path, suffix: str = ".csv", *, for_write: bool = False,
    search_earlier_dates: bool = False,
) -> Path:
    """Save in the selected block; prefer local annotations when reading.

    Share the immediate dated parent of a block. Tracking can opt into looking
    under earlier recording folders in the same dataset root, including their
    block*/ annotations. Rank by recording date first, then file modification
    time within that date, with a deterministic path tie-break. Never borrow
    same-day sibling-block or future-date annotations. Writes stay block-local.
    """
    if suffix not in {".csv", ".json"}:
        raise ValueError("Panorama annotations must be .csv or .json")
    block_dir = Path(block_dir)
    dataset_dir = panorama_dataset_dir(block_dir)
    shared = dataset_dir / f"panorama_regions{suffix}"
    local = block_dir / shared.name
    if for_write or local.is_file():
        return local
    if shared.is_file():
        return shared
    recording_date = _recording_date(dataset_dir.name)
    if search_earlier_dates and recording_date is not None and dataset_dir.parent.is_dir():
        earlier_dirs: dict[date, list[Path]] = {}
        for path in dataset_dir.parent.iterdir():
            day = _recording_date(path.name)
            if day is not None and day < recording_date and path.is_dir():
                earlier_dirs.setdefault(day, []).append(path)
        for day in sorted(earlier_dirs, reverse=True):
            candidates = []
            for directory in earlier_dirs[day]:
                paths = [directory / local.name, *directory.glob(f"block*/{local.name}")]
                candidates.extend(path for path in paths if path.is_file())
            if candidates:
                return max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)))
    return local


def panorama_tracking_split(
    block_dir: Path, *, regions_path: Path | None = None,
) -> tuple[float, Path]:
    """Read the split between full arenas in raw tracking coordinates."""
    path = (Path(regions_path) if regions_path is not None else
            panorama_regions_path(block_dir, search_earlier_dates=True))
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return tracking_split_from_region_rows(rows, source=str(path)), path


def tracking_split_from_region_rows(rows: list[dict], *, source: str = "panorama regions") -> float:
    """Share arena validation between tracking and the GUI's live split preview.

    Colony rectangles describe nests, which need not be centered in their arenas.
    They must never be used to infer the boundary between the full arenas.
    """
    anchors = []
    for row in rows:
        if _ARENA_LABEL.fullmatch(row["semantic_label"].strip()):
            anchors.append(row)
    if len(anchors) != 2 or any(row["shape"].lower() != "rectangle" for row in anchors):
        raise ValueError(
            f"{source}: expected two arena rectangles for tracking split. "
            "Use the Left arena and Right arena buttons to outline the full arenas; "
            "colony/nest regions do not define this split."
        )
    bounds = []
    for row in anchors:
        xmin, xmax = (float(row[f"tracking_x_{edge}_px"]) for edge in ("min", "max"))
        if not all(map(math.isfinite, (xmin, xmax))) or xmin >= xmax:
            raise ValueError(f"{source}: invalid tracking x bounds for {row['name']}")
        bounds.append((xmin, xmax, row))
    bounds.sort(key=lambda item: (item[0] + item[1]) / 2)
    left, right = bounds
    tolerance = 3.0
    if left[1] - right[0] > tolerance:
        raise ValueError(f"{source}: arena rectangles overlap in x")
    for (_, _, row), side in zip(bounds, ("left", "right")):
        label_side = arena_label_side(row["semantic_label"])
        if label_side is not None and label_side != side:
            raise ValueError(f"{source}: label/geometry side mismatch for {row['name']}")
    return (left[1] + right[0]) / 2
