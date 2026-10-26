#!/usr/bin/env python3
"""Adjust a SLEAP 0.3 top-down export's centroid cutoff, preserving learned weights.

Run with the colony sleap-nn interpreter (onnx/torch/TensorRT are optional for
normal tracking). A100 compilation is a separate command so CPU validation of
an ONNX variant can precede the expensive GPU build. The source export is never
modified. The node-name checks intentionally fail on unfamiliar export graphs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def adjust(source: Path, target: Path, threshold: float) -> dict:
    import onnx
    from onnx import numpy_helper

    if not 0 < threshold < 1:
        raise ValueError("Centroid threshold must be between 0 and 1")
    if source.resolve() == target.resolve() or (target / "model.onnx").exists():
        raise ValueError("Use a fresh export directory; existing exports are immutable")
    metadata = json.loads((source / "export_metadata.json").read_text())
    if metadata["model_type"] != "topdown":
        raise ValueError("Only paired top-down exports are supported")
    model = onnx.load(source / "model.onnx")
    weights = [tensor.SerializeToString() for tensor in model.graph.initializer]
    constants = {n.output[0]: n for n in model.graph.node if n.op_type == "Constant"}
    changes = []
    for node in model.graph.node:
        if node.op_type != "Greater" or len(node.input) != 2:
            continue
        is_centroid_map = "CentroidConfmapsHead" in node.input[0]
        is_centroid_valid = "instance_valid" in node.output
        if not (is_centroid_map or is_centroid_valid):
            continue
        if node.input[1] not in constants:
            continue  # local-maximum comparisons also consume centroid maps
        constant = constants[node.input[1]]
        attr = next(a for a in constant.attribute if a.name == "value")
        value = numpy_helper.to_array(attr.t).copy()
        if value.size != 1 or abs(float(value.item()) - metadata["peak_threshold"]) > 1e-6:
            raise ValueError(f"Unexpected centroid cutoff at {constant.name}")
        changes.append(dict(node=constant.name, old=float(value.item()), new=threshold))
        value[...] = threshold
        attr.t.CopyFrom(numpy_helper.from_array(value))
    if len(changes) != 2:
        raise ValueError(f"Expected two centroid comparisons, found {len(changes)}")
    if weights != [tensor.SerializeToString() for tensor in model.graph.initializer]:
        raise AssertionError("Learned weights changed")
    metadata.update(centroid_peak_threshold=threshold, instance_peak_threshold=metadata["peak_threshold"],
                    export_timestamp=datetime.now(timezone.utc).isoformat(),
                    single_ant_variant=dict(source_export=str(source), source_onnx_sha256=sha256(source / "model.onnx"),
                                            changes=changes, learned_weights_unchanged=True))
    for prop in model.metadata_props:
        if prop.key == "sleap_nn_metadata":
            prop.value = json.dumps(metadata)
    onnx.checker.check_model(model)
    target.mkdir(parents=True, exist_ok=True)
    onnx.save(model, target / "model.onnx")
    metadata["onnx_sha256"] = sha256(target / "model.onnx")
    (target / "export_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for config in source.glob("training_config_*.yaml"):
        shutil.copyfile(config, target / config.name)
    return metadata


def build_trt(target: Path, height: int, width: int, batch: int) -> dict:
    import torch
    import tensorrt as trt
    from sleap_nn.export.exporters.tensorrt_exporter import _export_tensorrt_onnx

    metadata = json.loads((target / "export_metadata.json").read_text())
    if sha256(target / "model.onnx") != metadata["onnx_sha256"]:
        raise ValueError("ONNX graph changed after validation")
    build = target / "build"
    build.mkdir(exist_ok=True)
    link = build / "model.onnx"
    if not link.exists():
        link.symlink_to((target / "model.onnx").resolve())
    engine_path = _export_tensorrt_onnx(
        None, build / "model.trt", (batch, 1, height, width), torch.uint8,
        (1, 1, height, width), (batch, 1, height, width), (batch, 1, height, width),
        "fp16", 8 << 30, True,
    )
    logger = trt.Logger(trt.Logger.WARNING)
    with trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if engine is None:
            raise ValueError("Compiled TensorRT engine cannot be loaded")
        del engine
    metadata.update(export_format="tensorrt", precision="fp16", max_batch_size=batch,
                    fixed_resolution=[height, width], engine_sha256=sha256(engine_path),
                    export_timestamp=datetime.now(timezone.utc).isoformat())
    engine_path.replace(target / "model.trt")
    (target / "model.trt.metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (target / "export_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("adjust")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--target", type=Path, required=True)
    p.add_argument("--centroid-threshold", type=float, required=True)
    p = sub.add_parser("build-trt")
    p.add_argument("--target", type=Path, required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    result = (adjust(args.source, args.target, args.centroid_threshold) if args.command == "adjust"
              else build_trt(args.target, args.height, args.width, args.batch_size))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
