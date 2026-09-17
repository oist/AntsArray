#!/usr/bin/env python3
"""Make GUI-re-saved SLEAP packages loadable again by cutting the source_video chain.

Every time a ``.pkg.slp`` is opened and re-saved by the SLEAP GUI, the previous
project is nested whole into the new file's ``videos_json`` under ``source_video``.
After a handful of review rounds each video entry carries a chain 7-14 levels deep,
and ``sleap_io.load_slp`` blows up with ``RecursionError`` while parsing it. Since
``sleap_nn``'s ``LabelsReader`` calls ``load_slp``, such a file cannot be trained on
at all.

The chain is pure provenance. The ``backend`` at the top of each entry already
describes where the images live::

    {"type": "HDF5Video", "filename": ".", "dataset": "video0/video",
     "has_embedded_images": true, "shape": [100, 3036, 4024, 1]}

``filename: "."`` means "this same file", so dropping ``source_video`` loses nothing
that training needs. This script copies each input and rewrites ``videos_json``
without the chain, leaving ``frames``, ``instances``, ``points`` and the embedded
``videoN/video`` image datasets byte-for-byte untouched.

Labels exported without their images (a bare ``.slp``) instead point ``backend
.filename`` at the package they were reviewed against. When that path is stale --
e.g. a collaborator's ``C:/Users/.../OneDrive/Desktop/...pkg.slp`` -- pass
``--images-from`` to re-bind it to a package that holds the same videos. The
``--images-from`` package is validated to have matching per-video frame counts
before anything is written.

Examples
--------
Flatten the six recorrected packages into an output directory::

    python flatten_slp_videos.py --out-dir flat/ raw/*.pkg.slp

Re-bind a reviewed export to the package holding its images::

    python flatten_slp_videos.py --out-dir flat/ \
        --images-from raw/chunk06_..._recorrected.pkg.slp \
        raw/chunk06_..._recorrected.slp
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np

from topdown_common import stem_of, write_report

VIDEOS_JSON = "videos_json"
SOURCE_VIDEO = "source_video"


def chain_depth(entry: dict) -> int:
    """Number of nested source_video levels hanging off one video entry."""
    depth = 0
    node = entry.get(SOURCE_VIDEO)
    while isinstance(node, dict) and depth < 1000:
        depth += 1
        node = node.get(SOURCE_VIDEO)
    return depth


def read_video_entries(h5: h5py.File) -> list[dict]:
    entries = []
    for item in h5[VIDEOS_JSON][:]:
        text = item.decode() if isinstance(item, (bytes, np.bytes_)) else str(item)
        entries.append(json.loads(text))
    return entries


def video_shapes(entries: list[dict]) -> list[list[int] | None]:
    return [entry.get("backend", {}).get("shape") for entry in entries]


def embedded_video_groups(h5: h5py.File) -> list[str]:
    return [
        key
        for key in h5
        if key.startswith("video") and key != VIDEOS_JSON and isinstance(h5[key], h5py.Group)
    ]


def validate_images_from(images_from: Path, entries: list[dict]) -> None:
    """Refuse to re-bind unless the donor package really holds the same videos."""
    with h5py.File(images_from, "r") as donor:
        donor_entries = read_video_entries(donor)
        embedded = embedded_video_groups(donor)

    if not embedded:
        raise SystemExit(f"[ERR] --images-from has no embedded video groups: {images_from}")

    want = video_shapes(entries)
    have = video_shapes(donor_entries)
    if len(want) != len(have):
        raise SystemExit(
            f"[ERR] --images-from holds {len(have)} videos, labels reference {len(want)}: {images_from}"
        )
    for idx, (labelled, donated) in enumerate(zip(want, have)):
        # Frame count and geometry must agree. A mismatch means these are different
        # recordings, and re-binding would silently pair wrong images with labelled
        # coordinates -- corrupt training data that still loads cleanly.
        if labelled != donated:
            raise SystemExit(
                f"[ERR] video{idx} shape mismatch: labels say {labelled}, --images-from says {donated}"
            )


def flatten_entry(entry: dict, images_from: Path | None) -> dict:
    out = {key: value for key, value in entry.items() if key != SOURCE_VIDEO}
    backend = dict(out.get("backend", {}))
    backend.pop(SOURCE_VIDEO, None)

    if images_from is not None:
        backend["filename"] = str(images_from)
        out["filename"] = str(images_from)

    out["backend"] = backend
    return out


def rewrite_videos_json(h5: h5py.File, entries: list[dict]) -> None:
    payload = [json.dumps(entry).encode() for entry in entries]
    width = max(len(item) for item in payload)
    del h5[VIDEOS_JSON]
    h5.create_dataset(VIDEOS_JSON, data=np.array(payload, dtype=f"|S{width}"), maxshape=(None,))


def drop_source_video_datasets(h5: h5py.File) -> int:
    dropped = 0
    for key in embedded_video_groups(h5):
        if SOURCE_VIDEO in h5[key]:
            del h5[key][SOURCE_VIDEO]
            dropped += 1
    return dropped


def process(src: Path, dst: Path, images_from: Path | None, quiet: bool) -> dict:
    with h5py.File(src, "r") as h5:
        entries = read_video_entries(h5)
        self_contained = bool(embedded_video_groups(h5))

    depths = [chain_depth(entry) for entry in entries]

    if images_from is not None:
        validate_images_from(images_from, entries)
    elif not self_contained:
        raise SystemExit(
            f"[ERR] {src.name} has no embedded images and no --images-from was given; "
            "training would have no pixels to read."
        )

    if not quiet:
        print(
            f"[..] {src.name}: {len(entries)} video(s), "
            f"source_video depth {min(depths)}-{max(depths)}, "
            f"{'embedded' if self_contained else 'external'} images"
        )

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

    flat = [flatten_entry(entry, images_from) for entry in entries]

    with h5py.File(dst, "r+") as h5:
        rewrite_videos_json(h5, flat)
        dropped = drop_source_video_datasets(h5)

    # Re-read to prove the written file parses and that no chain survived.
    with h5py.File(dst, "r") as h5:
        remaining = max((chain_depth(entry) for entry in read_video_entries(h5)), default=0)
    if remaining:
        raise SystemExit(f"[ERR] {dst.name} still has source_video depth {remaining}")

    if not quiet:
        print(
            f"[OK] {dst.name}: chain removed (was {max(depths)}), "
            f"{dropped} embedded source_video dataset(s) dropped"
        )

    return {
        "src": str(src),
        "dst": str(dst),
        "n_videos": len(entries),
        "max_source_depth_before": max(depths),
        "embedded_images": self_contained,
        "images_from": str(images_from) if images_from else None,
        "source_video_datasets_dropped": dropped,
    }


def output_path(src: Path, out_dir: Path, suffix: str) -> Path:
    """Keep the double extension intact: foo.pkg.slp -> foo<suffix>.pkg.slp."""
    extension = ".pkg.slp" if src.name.endswith(".pkg.slp") else src.suffix
    return out_dir / f"{stem_of(src)}{suffix}{extension}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("inputs", nargs="+", type=Path, help=".slp / .pkg.slp files to flatten")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--images-from",
        type=Path,
        default=None,
        help=(
            "Package holding the images for a bare .slp export. Its per-video shapes "
            "must match the labels exactly. Only valid with a single input."
        ),
    )
    parser.add_argument("--suffix", default="", help="Appended to each output stem.")
    parser.add_argument("--report", type=Path, default=None, help="Write a JSON summary here.")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.images_from is not None and len(args.inputs) != 1:
        raise SystemExit("[ERR] --images-from applies to exactly one input file")

    results = []
    for src in args.inputs:
        if not src.is_file():
            raise SystemExit(f"[ERR] not a file: {src}")
        results.append(
            process(src, output_path(src, args.out_dir, args.suffix), args.images_from, args.quiet)
        )

    write_report(args.report, results, quiet=args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
