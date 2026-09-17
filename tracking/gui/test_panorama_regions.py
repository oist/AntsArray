import json

import pytest
from PIL import Image

from camera_cal.region_paths import panorama_regions_path
from tracking.gui import panorama_region_annotator as annotator
from analysis import grid_occupancy_utils as go
from analysis import return_sleep_utils as rs


def test_local_save_and_chunk_copy_round_trip_with_other_block_origin(tmp_path):
    block = tmp_path / "20260810" / "block02-w000-031"
    block.mkdir(parents=True)
    panorama = block / "panorama.png"
    Image.new("RGB", (100, 80), "white").save(panorama)
    metadata = dict(homographies=str(tmp_path / "hmats.npz"), mm_per_pixel=0.016,
                    image_to_raw_tracking_offset_px=[-20, -30],
                    image_to_raw_tracking_matrix=[[1, 0, -20], [0, 1, -30], [0, 0, 1]])
    regions = [dict(region_id="r1", semantic_label="colony_L", name="colony_left", shape="rectangle",
                    geometry=dict(x_min=10, y_min=20, x_max=40, y_max=50)),
               dict(region_id="r2", semantic_label="water_R", name="water_right", shape="circle",
                    geometry=dict(center_x=80, center_y=60, radius=5))]
    annotation_path = panorama_regions_path(block, ".json", for_write=True)
    csv_path = panorama_regions_path(block, for_write=True)
    later = block.with_name("block02-w149-197")
    annotated = annotator.save_annotations(regions, annotation_path, csv_path, panorama,
                                           block / "metadata.json", metadata, copy_regions_to=[later])
    assert annotation_path.parent == block
    assert annotation_path.is_file() and csv_path.is_file() and annotated.is_file()
    assert not (block.parent / "panorama_regions.csv").exists()
    assert annotator.load_regions(annotation_path, metadata) == regions
    later_csv = later / "panorama_regions.csv"
    assert go.panorama_regions_path(later) == rs.panorama_regions_path(later) == later_csv
    assert csv_path.read_bytes() == later_csv.read_bytes()
    assert annotation_path.read_bytes() == (later / "panorama_regions.json").read_bytes()
    loaded = go.load_panorama_regions(go.panorama_regions_path(later), x_split_px=30)
    assert loaded.side.tolist() == ["left", "right"]
    assert loaded.tracking_x_min_px.iloc[0] == -10
    old_fingerprint = rs.fingerprint(csv_path)
    annotator.save_annotations(regions[:1], annotation_path, csv_path, panorama, block / "metadata.json", metadata,
                               copy_regions_to=[later])
    assert rs.fingerprint(csv_path) != old_fingerprint
    assert csv_path.read_bytes() == later_csv.read_bytes()
    other_metadata = {**metadata, "image_to_raw_tracking_offset_px": [-25, -40],
                      "image_to_raw_tracking_matrix": [[1, 0, -25], [0, 1, -40], [0, 0, 1]]}
    shifted = annotator.load_regions(annotation_path, other_metadata)
    assert shifted[0]["geometry"] == dict(x_min=15, y_min=30, x_max=45, y_max=60)
    payload, _ = annotator.build_annotation_payload(shifted, panorama, block / "metadata.json", other_metadata)
    original = json.loads(annotation_path.read_text())["regions"][0]
    assert payload["regions"][0]["tracking_geometry_px"] == original["tracking_geometry_px"]


def test_mismatched_calibration_rejected(tmp_path):
    path = tmp_path / "regions.json"
    path.write_text(json.dumps(dict(homographies=str(tmp_path / "old.npz"), regions=[])))
    with pytest.raises(ValueError, match="different homographies"):
        annotator.load_regions(path, dict(homographies=str(tmp_path / "new.npz")))


def test_matching_date_calibration_replaces_pilot_default(tmp_path):
    block = tmp_path / "20260810" / "block02"
    block.mkdir(parents=True)
    correct = tmp_path / "cameraArray_calib" / "20260810_calib_elevated_by_2mm" / "frame0" / "aruco_stitch" / "aruco_H_mats.npz"
    correct.parent.mkdir(parents=True)
    correct.touch()
    metadata_path = block / "metadata.json"
    metadata_path.write_text(json.dumps(dict(homographies=str(tmp_path / "pilot.npz"))))
    assert annotator.resolve_homographies(block, None, metadata_path) == correct
    explicit = tmp_path / "explicit.npz"
    assert annotator.resolve_homographies(block, explicit, metadata_path) == explicit


def test_ambiguous_date_calibration_requires_explicit_choice(tmp_path):
    block = tmp_path / "20260810" / "block02"
    for name in ("20260810_calib_a", "20260810_calib_b"):
        path = tmp_path / "cameraArray_calib" / name / "frame0" / "aruco_stitch" / "aruco_H_mats.npz"
        path.parent.mkdir(parents=True)
        path.touch()
    with pytest.raises(ValueError, match="Multiple dataset calibrations"):
        annotator.resolve_homographies(block, None, block / "metadata.json")


def test_fixed_arena_buttons_round_trip_and_preview_match_tracking(tmp_path, monkeypatch):
    """Draw using the actual buttons, save CSV, and resolve the tracking boundary."""
    from camera_cal.region_paths import panorama_tracking_split
    try:
        root = annotator.tk.Tk()
    except annotator.tk.TclError:
        pytest.skip("GUI test requires a display (or xvfb-run)")
    root.withdraw()
    panorama = tmp_path / "panorama.png"
    Image.new("RGB", (100, 80), "white").save(panorama)
    metadata = dict(homographies=str(tmp_path / "H.npz"), mm_per_pixel=0.016,
                    image_to_raw_tracking_offset_px=[-20, -30],
                    image_to_raw_tracking_matrix=[[1, 0, -20], [0, 1, -30], [0, 0, 1]])
    app = annotator.PanoramaRegionAnnotator(
        root, panorama, tmp_path / "metadata.json", tmp_path / "panorama_regions.json",
        tmp_path / "panorama_regions.csv", metadata, [])

    def widgets(widget):
        for child in widget.winfo_children():
            yield child
            yield from widgets(child)

    buttons = {w.cget("text"): w for w in widgets(root) if isinstance(w, annotator.ttk.Button)}

    def draw(side, x0, x1):
        buttons[f"{side.title()} arena"].invoke()
        assert str(app.label_box.cget("state")) == "disabled"
        app.drag_start = (x0, 10)
        monkeypatch.setattr(app, "_image_xy", lambda event: (x1, 70))
        app._on_left_release(None)

    try:
        assert "unavailable" in app.tracking_split_var.get()
        draw("left", 5, 40)
        draw("right", 60, 95)
        assert [r["semantic_label"] for r in app.regions] == ["arena_left", "arena_right"]
        assert app.canvas.find_withtag("tracking_split")
        assert "X = 30.00" in app.tracking_split_var.get()
        # Redraw replaces the selected side and Undo restores its old boundary.
        draw("left", 5, 50)
        assert len(app.regions) == 2
        assert annotator.arena_split_in_image(app.regions) == 55
        app._undo()
        assert annotator.arena_split_in_image(app.regions) == 50
        app._save()
        assert panorama_tracking_split(tmp_path)[0] == 30
        assert annotator.load_regions(app.annotation_path, metadata) == app.regions
        saved = Image.open(tmp_path / "panorama_annotated.png").convert("RGB")
        assert saved.getpixel((50, 75)) == (255, 51, 68)
    finally:
        app.original_image.close()
        root.destroy()
