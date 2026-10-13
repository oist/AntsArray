"""scripts/sleap_nn_export_anchor_fill.py: a below-threshold tag node falls back to the centroid.

The centroid model is anchored on the tag, so the centroid is where the tag is estimated to be.
The engine must use it only where the centered-instance model found no tag peak, and must leave
every other node, and every instance whose tag it did find, exactly as upstream exports them.
Needs torch (and sleap-nn for the wrapper test); skipped where they are not installed.
"""
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
import sleap_nn_export_anchor_fill as anchor_fill  # noqa: E402

NAN = float("nan")


def outputs(peak_vals):
    """One frame, two instances, three nodes; node 0 is the anchor."""
    return {
        "centroids": torch.tensor([[[20.0, 30.0], [50.0, 60.0]]]),
        "centroid_vals": torch.tensor([[0.9, 0.8]]),
        "peaks": torch.tensor([[[[NAN, NAN], [16.0, 27.0], [22.0, 34.0]],
                                [[51.0, 59.0], [45.0, 55.0], [NAN, NAN]]]]),
        "peak_vals": torch.tensor(peak_vals),
        "instance_valid": torch.tensor([[True, True]]),
    }


def test_a_missing_anchor_takes_the_centroid_and_nothing_else_changes():
    before = outputs([[[0.0, 0.9, 0.8], [0.7, 0.9, 0.0]]])

    after = anchor_fill.fill_anchor(before, 0, 0.2)

    assert list(after) == list(before)                       # ONNX output order
    assert after["peaks"][0, 0, 0].tolist() == [20.0, 30.0]  # filled from the centroid
    assert after["peaks"][0, 1, 0].tolist() == [51.0, 59.0]  # found tag kept, not the centroid
    assert torch.equal(after["peaks"][0, 0, 1:], before["peaks"][0, 0, 1:])
    assert after["peaks"][0, 1, 1].tolist() == [45.0, 55.0]
    assert torch.isnan(after["peaks"][0, 1, 2]).all()        # other missing nodes stay NaN
    for key in ("centroids", "centroid_vals", "peak_vals", "instance_valid"):
        assert after[key] is before[key]
    assert after["peak_vals"][0, 0, 0] == 0.0                # filled anchor stays recognisable


def test_an_empty_instance_slot_is_never_filled():
    before = outputs([[[0.0, 0.9, 0.8], [0.0, 0.0, 0.0]]])
    before["instance_valid"] = torch.tensor([[True, False]])
    before["peaks"][0, 1] = NAN

    after = anchor_fill.fill_anchor(before, 0, 0.2)

    assert after["peaks"][0, 0, 0].tolist() == [20.0, 30.0]
    assert torch.isnan(after["peaks"][0, 1]).all()


def test_another_anchor_index_is_filled_in_place():
    before = outputs([[[0.9, 0.0, 0.8], [0.7, 0.9, 0.0]]])
    before["peaks"][0, 0, 1] = NAN

    after = anchor_fill.fill_anchor(before, 1, 0.2)

    assert after["peaks"][0, 0, 1].tolist() == [20.0, 30.0]
    assert after["peaks"][0, 1, 1].tolist() == [45.0, 55.0]
    assert after["peaks"].shape == before["peaks"].shape


class _Fixed(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


def test_the_patched_sleap_nn_wrapper_fills_only_the_missing_anchor():
    wrappers = pytest.importorskip("sleap_nn.export.wrappers")

    def centroid_cms(image):
        cms = torch.zeros(image.shape[0], 1, 64, 64)
        cms[:, 0, 30, 20] = 1.0                    # one ant at x=20, y=30
        return cms

    def instance_cms(crops):
        cms = torch.zeros(crops.shape[0], 3, 16, 16)
        cms[:, 0] = 0.1                            # tag never above the 0.2 threshold
        cms[:, 1, 5, 4] = 0.9
        cms[:, 2, 12, 10] = 0.9
        return cms

    wrapper = wrappers.TopDownONNXWrapper(
        centroid_model=_Fixed(centroid_cms), instance_model=_Fixed(instance_cms),
        max_instances=2, crop_size=(16, 16), centroid_output_stride=1, instance_output_stride=1,
        n_nodes=3)
    image = torch.zeros(1, 1, 64, 64, dtype=torch.uint8)
    upstream = wrappers.TopDownONNXWrapper.forward
    plain = wrapper(image)
    try:
        anchor_fill.install(0)
        filled = wrapper(image)
    finally:
        wrappers.TopDownONNXWrapper.forward = upstream

    assert torch.isnan(plain["peaks"][0, 0, 0]).all()
    assert filled["peaks"][0, 0, 0].tolist() == plain["centroids"][0, 0].tolist() == [20.0, 30.0]
    assert torch.equal(filled["peaks"][0, 0, 1:], plain["peaks"][0, 0, 1:])
    assert filled["peaks"][0, 0, 1].tolist() == [16.0, 27.0]          # crop peak + crop offset
    assert torch.equal(filled["peaks"][0, 1], plain["peaks"][0, 1])  # invalid slot untouched
    assert list(filled) == list(plain)


def model_dirs(tmp_path, nodes=("head", "aruco ", "tail")):
    centroid, instance = tmp_path / "c", tmp_path / "i"
    skeleton = "  skeletons:\n  - nodes:\n" + "".join("    - name: '%s'\n" % n for n in nodes)
    for d, head, other in ((centroid, "centroid", "centered_instance"),
                           (instance, "centered_instance", "centroid")):
        d.mkdir(parents=True)
        (d / "training_config.yaml").write_text(
            "data_config:\n" + skeleton + "model_config:\n  head_configs:\n    %s: null\n"
            "    %s:\n      confmaps:\n        anchor_part: 'aruco '\n" % (other, head))
    return centroid, instance


def test_the_anchor_comes_from_the_instance_model_config(tmp_path):
    pytest.importorskip("sleap_nn")
    centroid, instance = model_dirs(tmp_path)

    assert anchor_fill.find_anchor(["export", str(centroid), str(instance), "-o", "x"]) == ("aruco ", 1)
    with pytest.raises(SystemExit):
        anchor_fill.find_anchor(["export", str(centroid)])
    with pytest.raises(SystemExit):
        anchor_fill.output_dir(["export", str(centroid), str(instance)])

    # The export dir is never read as a model, even if it holds a model config.
    _, decoy = model_dirs(tmp_path / "decoy", nodes=("aruco ", "head", "tail"))
    argv = ["export", "-o", str(decoy), str(centroid), str(instance)]
    assert anchor_fill.find_anchor(argv) == ("aruco ", 1)


def test_the_record_is_written_only_after_a_clean_export(tmp_path):
    cli_mod = pytest.importorskip("sleap_nn.cli")
    wrappers = pytest.importorskip("sleap_nn.export.wrappers")
    centroid, instance = model_dirs(tmp_path)
    out = tmp_path / "export"
    out.mkdir()
    argv = ["export", str(centroid), str(instance), "-o", str(out)]
    upstream_cli, upstream_forward = cli_mod.cli, wrappers.TopDownONNXWrapper.forward
    try:
        cli_mod.cli = lambda args, standalone_mode: 1          # sleap-nn ctx.exit(1)
        assert anchor_fill.main(argv) == 1
        assert not (out / anchor_fill.RECORD).exists()
        cli_mod.cli = lambda args, standalone_mode: None
        assert anchor_fill.main(argv) == 0
        assert '"anchor_index": 1' in (out / anchor_fill.RECORD).read_text()
    finally:
        cli_mod.cli, wrappers.TopDownONNXWrapper.forward = upstream_cli, upstream_forward
