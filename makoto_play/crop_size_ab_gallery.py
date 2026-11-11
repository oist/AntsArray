#!/usr/bin/env python3
"""Picture examples of what a crop-size change fixes and what it breaks.

Reads the ``arrays.npz`` written by crop_size_ab.py and draws, for one candidate crop
against the trained 640, randomly sampled instances from each outcome:

  recovered     an antenna tip > 320 px from the tag: gross at 640, good at the candidate
  tag_jump      the tag node: good at 640, gross at the candidate
  tip_broken    an antenna tip <= 256 px from the tag: good at 640, gross at the candidate
  body_broken   occiput / petiole / gaster_tip: good at 640, gross at the candidate

Samples are random (fixed seed), not the worst cases, so a panel shows what a typical
member of its category looks like. Each panel is the labelled frame around the tag:
labels in green, 640 in blue, the candidate in red, both crop squares drawn, and the
node that decided the category ringed.

    & "C:\\Program Files\\uv-tools\\tools\\sleap-nn\\Scripts\\python.exe" `
        makoto_play\\crop_size_ab_gallery.py --run-dir D:\\crop_ab\\run --crop 768
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import sleap_io as sio  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

BASELINE = 640
GOOD_PX, GROSS_PX = 10.0, 20.0
VIEW_PX = 880                      # side of the image window drawn around the tag
TIPS = ["antenna_L3", "antenna_R_3"]
BODY = ["occiput", "petiole", "gaster_tip"]
COLOURS = {"gt": "#2ca02c", "base": "#1f77b4", "cand": "#d62728"}


def fold(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def categories(z, crop: int) -> dict[str, list[tuple[int, int]]]:
    """category -> [(row, node)] of every qualifying instance, one entry per row."""
    names = [fold(n) for n in z["names"]]
    gt = z["gt"]
    e0 = np.linalg.norm(gt - z[f"pred_{BASELINE}"], axis=-1)
    e1 = np.linalg.norm(gt - z[f"pred_{crop}"], axis=-1)
    reach = z["reach"]
    tag = names.index("aruco")
    tips = [names.index(fold(n)) for n in TIPS]
    body = [names.index(fold(n)) for n in BODY]

    def first(nodes, cond):
        hits = {}
        for node in nodes:
            for row in np.flatnonzero(cond(node)):
                hits.setdefault(int(row), node)
        return sorted(hits.items())

    def broke(n):
        return (e0[:, n] <= GOOD_PX) & (e1[:, n] > GROSS_PX)

    return {
        "recovered": first(tips, lambda n: (reach[:, n] > BASELINE / 2)
                           & (e0[:, n] > GROSS_PX) & (e1[:, n] <= GOOD_PX)),
        "tag_jump": first([tag], broke),
        "tip_broken": first(tips, lambda n: broke(n) & (reach[:, n] <= 256)),
        "body_broken": first(body, broke),
    }


class FrameSource:
    """Labelled images, loaded once per package and looked up by (video, frame)."""

    def __init__(self, packages: list[str]):
        self.packages = packages
        self.cache: dict[int, dict] = {}

    def image(self, pkg: int, video: int, frame: int) -> np.ndarray:
        if pkg not in self.cache:
            labels = sio.load_slp(self.packages[pkg])
            self.cache[pkg] = {(labels.videos.index(lf.video), int(lf.frame_idx)): lf for lf in labels}
        img = self.cache[pkg][(video, frame)].image
        return img[..., 0] if img.ndim == 3 else img


def draw_skeleton(ax, points, edges, colour, size, origin):
    xy = points - origin
    for a, b in edges:
        if not (np.isnan(xy[a]).any() or np.isnan(xy[b]).any()):
            ax.plot(*zip(xy[a], xy[b]), color=colour, lw=1.0, alpha=0.9)
    ax.scatter(xy[:, 0], xy[:, 1], s=size, c=colour, edgecolors="black", linewidths=0.3, zorder=3)


def panel(ax, z, src: FrameSource, row: int, node: int, crop: int, edges, names):
    gt = z["gt"][row]
    centre = gt[[fold(n) for n in names].index("aruco")]
    if np.isnan(centre).any():
        centre = np.nanmean(gt, axis=0)
    img = src.image(int(z["package"][row]), int(z["video"][row]), int(z["frame"][row]))
    h, w = img.shape[:2]
    x0 = int(np.clip(centre[0] - VIEW_PX / 2, 0, max(w - VIEW_PX, 0)))
    y0 = int(np.clip(centre[1] - VIEW_PX / 2, 0, max(h - VIEW_PX, 0)))
    origin = np.array([x0, y0], float)
    ax.imshow(img[y0:y0 + VIEW_PX, x0:x0 + VIEW_PX], cmap="gray", vmin=0, vmax=255)

    for size, colour in ((BASELINE, COLOURS["base"]), (crop, COLOURS["cand"])):
        corner = centre - size / 2 - origin
        ax.add_patch(Rectangle(corner, size, size, fill=False, ec=colour, lw=1.0, ls="--"))
    draw_skeleton(ax, gt, edges, COLOURS["gt"], 14, origin)
    draw_skeleton(ax, z[f"pred_{BASELINE}"][row], edges, COLOURS["base"], 10, origin)
    draw_skeleton(ax, z[f"pred_{crop}"][row], edges, COLOURS["cand"], 10, origin)
    if not np.isnan(gt[node]).any():
        ax.add_patch(Circle(gt[node] - origin, 22, fill=False, ec="yellow", lw=1.4))

    e0 = float(np.linalg.norm(gt[node] - z[f"pred_{BASELINE}"][row][node]))
    e1 = float(np.linalg.norm(gt[node] - z[f"pred_{crop}"][row][node]))
    pkg = Path(str(z["packages"][int(z["package"][row])])).name.split("_round1_")[-1][:7]
    ax.set_title(f"{pkg} f{int(z['frame'][row])} #{int(z['position'][row])} "
                 f"{names[node].strip()}\n{BASELINE}: {e0:.0f}px  {crop}: {e1:.0f}px  "
                 f"ants {int(z['density'][row])}", fontsize=7)
    ax.set_xlim(0, VIEW_PX)
    ax.set_ylim(VIEW_PX, 0)
    ax.set_axis_off()
    return {"package": pkg, "video": int(z["video"][row]), "frame": int(z["frame"][row]),
            "position": int(z["position"][row]), "node": names[node].strip(),
            f"err_{BASELINE}": round(e0, 1), f"err_{crop}": round(e1, 1)}


def skeleton_edges(package: str) -> list[tuple[int, int]]:
    skeleton = sio.load_slp(package).skeletons[0]
    return [(skeleton.node_names.index(e.source.name), skeleton.node_names.index(e.destination.name))
            for e in skeleton.edges]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--crop", type=int, default=768, help="Candidate crop to set against 640.")
    parser.add_argument("--n", type=int, default=8, help="Panels per category.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    z = dict(np.load(args.run_dir / "arrays.npz", allow_pickle=False))
    names = [str(n) for n in z["names"]]
    packages = [str(p) for p in z["packages"]]
    out_dir = args.run_dir / f"gallery_{args.crop}"
    out_dir.mkdir(parents=True, exist_ok=True)
    edges = skeleton_edges(packages[0])
    src = FrameSource(packages)
    rng = np.random.default_rng(args.seed)

    index = {}
    for category, hits in categories(z, args.crop).items():
        pick = [hits[i] for i in sorted(rng.choice(len(hits), min(args.n, len(hits)), replace=False))]
        print(f"{category:<12} {len(hits):>6} instances, drawing {len(pick)}", flush=True)
        if not pick:
            continue
        cols = 4
        rows = int(np.ceil(len(pick) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 4.0), squeeze=False)
        index[category] = {"n_total": len(hits), "panels": []}
        for ax, (row, node) in zip(axes.ravel(), pick):
            index[category]["panels"].append(panel(ax, z, src, row, node, args.crop, edges, names))
        for ax in axes.ravel()[len(pick):]:
            ax.set_axis_off()
        fig.suptitle(f"{category}: {len(hits)} instances (random {len(pick)}). "
                     f"green = label, blue = {BASELINE}, red = {args.crop}, "
                     f"dashed = crop squares, yellow ring = node in question", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.0)
        fig.savefig(out_dir / f"{category}.png", dpi=130)
        plt.close(fig)

    (out_dir / "index.json").write_text(json.dumps(index, indent=2))
    print(f"[OK] {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
