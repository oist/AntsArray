#!/usr/bin/env python3
"""sleap-nn `export` with an enlarged TensorRT build workspace.

sleap-nn 0.2.0 hardcodes the TRT build workspace at 2 GB in its CLI; for
full-frame topdown models this caps the achievable --max-batch-size around 8
(the autotuner can't fit tactics for the instance crop at higher batches).
A100/SXM4-80GB have ~70 GB of free VRAM at runtime even with max-batch 8, so
the constraint is purely the build-time workspace, not the engine's runtime
footprint.

This wrapper monkey-patches `export_to_tensorrt`'s default `workspace_size`
to WORKSPACE_GB (env, default 16 GB), then hands off to the regular sleap-nn
CLI. Every other flag (--max-batch-size, --precision, --device, ...) behaves
exactly as upstream documents.

Usage on a saion largegpu node, with sleap-nn/0.2.0 module loaded:

    WORKSPACE_GB=16 \\
    "$UV_TOOL_DIR/sleap-nn/bin/python" \\
        $SCRIPTS_DIR/sleap_nn_export_bigws.py \\
        export <centroid-dir> <instance-dir> \\
        -o <out-dir> -f tensorrt --precision fp16 \\
        --max-batch-size 16 --device cuda

The wrapper only changes the *default*; if the upstream CLI ever starts
passing `workspace_size=` explicitly, the explicit value wins.
"""

from __future__ import annotations

import os
import sys


def _bump_workspace_default() -> None:
    """Patch sleap_nn.export.exporters.export_to_tensorrt to use a larger default workspace."""
    workspace_bytes = int(float(os.environ.get("WORKSPACE_GB", "16")) * (1 << 30))

    import sleap_nn.export.exporters as exporters_pkg

    orig = exporters_pkg.export_to_tensorrt

    def patched(*args, **kwargs):
        if "workspace_size" not in kwargs:
            kwargs["workspace_size"] = workspace_bytes
            print(
                f"[bigws] export_to_tensorrt workspace_size={workspace_bytes / (1 << 30):.1f} GB",
                file=sys.stderr,
                flush=True,
            )
        return orig(*args, **kwargs)

    exporters_pkg.export_to_tensorrt = patched

    # Also patch the source submodule in case any caller reaches in directly.
    try:
        import sleap_nn.export.exporters.tensorrt_exporter as trt_mod

        if hasattr(trt_mod, "export_to_tensorrt"):
            trt_mod.export_to_tensorrt = patched
    except Exception:
        pass


if __name__ == "__main__":
    _bump_workspace_default()
    from sleap_nn.cli import cli

    cli()
