#!/usr/bin/env python3
"""sleap-nn `export` of a top-down model whose anchor (tag) node falls back to the centroid.

The centroid model is trained on the anchor part, and the instance crop is centred on its peak,
so the centroid is the model's own estimate of where the anchor is. The centered-instance model
predicts the anchor again inside the crop, and where that peak is below the instance threshold
the exported engine returns NaN for it. Tracking cannot use such an instance (map_combine drops
NaN rows; the tracker needs the anchor to assign an instance), and the crop-768 instance model
leaves its tag node below the threshold about four times as often as the crop-640 one (20260928
chunk 010, cam09: 0.37 % -> 1.57 % of instances).

This wrapper patches TopDownONNXWrapper.forward so that, where the anchor's peak value is below
the instance threshold, the anchor takes the centroid's coordinates. peak_vals is not touched:
upstream already sets it to 0 below the threshold, so a filled anchor stays recognisable
downstream as finite X/Y with Score_node 0 (a real peak scores at least the threshold).
Every flag and every other output is upstream sleap-nn's.

Usage on a saion GPU node with a sleap-nn module loaded (same arguments as `sleap-nn export`):

    "$(dirname "$(readlink -f "$(command -v sleap-nn)")")/python" \\
        $SCRIPTS_DIR/sleap_nn_export_anchor_fill.py \\
        export <centroid-dir> <instance-dir> -o <out-dir> -f tensorrt --precision fp16 \\
        --max-batch-size 1 --max-instances 96 --device cuda

Writes <out-dir>/anchor_fill.json recording the anchor node and the rule.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RECORD = "anchor_fill.json"


def fill_anchor(outputs, anchor_index, threshold):
    """Return the top-down outputs with a below-threshold anchor replaced by the centroid.

    outputs: dict with "centroids" (B, I, 2), "peaks" (B, I, N, 2) and "peak_vals" (B, I, N),
    optionally "instance_valid" (B, I); empty instance slots are never filled. Only the
    anchor's coordinates change; the dict keeps its key order (the ONNX output order).
    """
    import torch

    peaks = outputs["peaks"]
    missing = outputs["peak_vals"][:, :, anchor_index] < threshold
    if "instance_valid" in outputs:
        missing = missing & outputs["instance_valid"]
    missing = missing.unsqueeze(-1)
    anchor = torch.where(missing, outputs["centroids"], peaks[:, :, anchor_index, :])
    filled = torch.cat(
        [peaks[:, :, :anchor_index, :], anchor.unsqueeze(2), peaks[:, :, anchor_index + 1:, :]],
        dim=2,
    )
    return {**outputs, "peaks": filled}


def find_anchor(argv):
    """(anchor name, node index) from the centered-instance model dir among the CLI arguments."""
    from sleap_nn.export.utils import load_training_config, resolve_anchor_part, resolve_node_names

    for i, arg in enumerate(argv):
        if i > 0 and argv[i - 1] in ("-o", "--output"):
            continue                    # the export dir, not a model
        path = Path(arg)
        if not path.is_dir():
            continue
        try:
            cfg = load_training_config(path)
        except FileNotFoundError:
            continue
        if cfg.model_config.head_configs.get("centered_instance") is None:
            continue
        anchor = resolve_anchor_part(cfg, "centered_instance")
        nodes = resolve_node_names(cfg, "centered_instance")
        if anchor not in nodes:
            raise SystemExit("[anchor-fill] anchor part %r is not a node of %s: %s" % (anchor, path, nodes))
        return anchor, nodes.index(anchor)
    raise SystemExit("[anchor-fill] no centered-instance model dir among the arguments")


def output_dir(argv):
    for flag in ("-o", "--output"):
        if flag in argv and argv.index(flag) + 1 < len(argv):
            return Path(argv[argv.index(flag) + 1])
    raise SystemExit("[anchor-fill] pass the export dir with -o: the fallback must be recorded next to it")


def require_sleap_nn_03():
    """The patch targets sleap-nn 0.3's export wrapper; refuse anything older up front."""
    import sleap_nn

    version = getattr(sleap_nn, "__version__", "0")
    major_minor = tuple(int(p) for p in version.split(".")[:2] if p.isdigit())
    if major_minor < (0, 3):
        raise SystemExit("[anchor-fill] needs sleap-nn 0.3 or newer, found %s" % version)


def install(anchor_index):
    from sleap_nn.export.wrappers import TopDownONNXWrapper

    upstream = TopDownONNXWrapper.forward

    def forward(self, image):
        return fill_anchor(upstream(self, image), anchor_index, self.instance_peak_threshold)

    TopDownONNXWrapper.forward = forward


def main(argv):
    """Export with the fallback installed; the record is written only after a clean export."""
    out = output_dir(argv)
    require_sleap_nn_03()
    anchor, index = find_anchor(argv)
    install(index)
    print("[anchor-fill] node %d %r falls back to the centroid below the instance threshold" % (index, anchor),
          file=sys.stderr, flush=True)

    from sleap_nn.cli import cli

    # standalone_mode=False returns a ctx.exit() code instead of exiting.
    rc = cli(args=argv, standalone_mode=False)
    if rc not in (None, 0):
        print("[anchor-fill] sleap-nn export exited %s; no %s written" % (rc, RECORD), file=sys.stderr)
        return int(rc)
    (out / RECORD).write_text(json.dumps({
        "anchor_part": anchor,
        "anchor_index": index,
        "rule": "anchor peak_val < instance peak threshold -> anchor xy = centroid xy; peak_val stays 0",
        "script": Path(__file__).name,
    }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
