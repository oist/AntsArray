"""Optional ONNX export tests; inference environments provide onnx."""
import json

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper

from tracking.single_ant.tune_export import adjust


def test_only_centroid_thresholds_change_and_source_is_immutable(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    nodes = [helper.make_node("Constant", [], [name], name=name,
                              value=numpy_helper.from_array(np.array(.2, np.float32)))
             for name in ("centroid_cutoff", "valid_cutoff", "pose_cutoff")]
    nodes += [helper.make_node("Greater", ["CentroidConfmapsHead_output", "centroid_cutoff"], ["cm_valid"]),
              helper.make_node("Greater", ["CentroidConfmapsHead_output", "candidate_values"], ["local_max"]),
              helper.make_node("Greater", ["candidate_values", "valid_cutoff"], ["instance_valid"]),
              helper.make_node("Less", ["keypoint_values", "pose_cutoff"], ["pose_invalid"])]
    inputs = [helper.make_tensor_value_info(n, TensorProto.FLOAT, [1])
              for n in ("CentroidConfmapsHead_output", "candidate_values", "keypoint_values")]
    outputs = [helper.make_tensor_value_info(n, TensorProto.BOOL, [1])
               for n in ("cm_valid", "instance_valid", "pose_invalid")]
    weight = numpy_helper.from_array(np.array([1.25, -3.5], dtype=np.float32), name="learned_weight")
    model = helper.make_model(helper.make_graph(nodes, "thresholds", inputs, outputs, [weight]),
                              opset_imports=[helper.make_opsetid("", 18)])
    onnx.save(model, source / "model.onnx")
    (source / "export_metadata.json").write_text(json.dumps(dict(model_type="topdown", peak_threshold=.2)))
    original = (source / "model.onnx").read_bytes()
    metadata = adjust(source, target, .01)
    assert (source / "model.onnx").read_bytes() == original
    tuned = onnx.load(target / "model.onnx")
    values = {node.name: float(numpy_helper.to_array(node.attribute[0].t))
              for node in tuned.graph.node if node.op_type == "Constant"}
    assert values == pytest.approx(dict(centroid_cutoff=.01, valid_cutoff=.01, pose_cutoff=.2))
    assert tuned.graph.initializer[0].SerializeToString() == weight.SerializeToString()
    assert metadata["single_ant_variant"]["learned_weights_unchanged"]
    with pytest.raises(ValueError, match="immutable"):
        adjust(source, target, .05)


def test_refuses_unrecognized_export_graph(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    model = helper.make_model(helper.make_graph([], "unknown", [], []))
    onnx.save(model, source / "model.onnx")
    (source / "export_metadata.json").write_text(json.dumps(dict(model_type="topdown", peak_threshold=.2)))
    with pytest.raises(ValueError, match="Expected two centroid"):
        adjust(source, tmp_path / "target", .01)
