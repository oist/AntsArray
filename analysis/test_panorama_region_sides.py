import numpy as np
import pandas as pd
import pytest

from analysis.grid_occupancy_utils import load_panorama_regions


def _write_regions(tmp_path, rows=None):
    if rows is None:
        rows = [dict(region_id="a", name="colony_1", semantic_label="colony", shape="rectangle",
                     tracking_x_min_px=0, tracking_x_max_px=40, tracking_center_x_px=20),
                dict(region_id="b", name="colony_2", semantic_label="colony", shape="rectangle",
                     tracking_x_min_px=260, tracking_x_max_px=300, tracking_center_x_px=280)]
        for i, (x, label) in enumerate([(120, "food"), (320, "food"), (50, "water"), (250, "water")]):
            rows.append(dict(region_id=f"r{i}", name=f"{label}_{i}", semantic_label=label,
                             shape="circle", tracking_center_x_px=x, radius_px=5))
    table = pd.DataFrame(rows)
    table["tracking_y_min_px"] = 0
    table["tracking_y_max_px"] = 40
    table["tracking_center_y_px"] = 20
    table["mm_per_pixel"] = 0.016
    table["area_mm2"] = 1.0
    path = tmp_path / "panorama_regions.csv"
    table.to_csv(path, index=False)
    return path, table


def test_unsuffixed_names_infer_sides_from_position_not_row_order(tmp_path):
    path, table = _write_regions(tmp_path)
    table.iloc[::-1].to_csv(path, index=False)
    regions = load_panorama_regions(path).set_index("name")
    assert regions.side.to_dict() == dict(colony_1="left", food_0="left", water_2="left",
                                         colony_2="right", food_1="right", water_3="right")
    assert regions.side_split_x_px.eq(150).all()
    assert regions.side_source.eq("tracking_geometry").all()
    assert regions.label_side.isna().all()
    assert set(regions.index) == set(table.name)


def test_explicit_split_supports_single_colony_and_rejects_nonfinite_values(tmp_path):
    path, table = _write_regions(tmp_path)
    table[table.name != "colony_2"].to_csv(path, index=False)
    with pytest.raises(ValueError, match="two separated colony rectangles"):
        load_panorama_regions(path)
    assert set(load_panorama_regions(path, x_split_px=150).side) == {"left", "right"}
    with pytest.raises(ValueError, match="must be finite"):
        load_panorama_regions(path, x_split_px=np.nan)


def test_rectangles_with_missing_centers_use_bounds_for_side_order(tmp_path):
    path, table = _write_regions(tmp_path)
    table.loc[table["shape"] == "rectangle", "tracking_center_x_px"] = np.nan
    table.iloc[::-1].to_csv(path, index=False)
    regions = load_panorama_regions(path).set_index("name")
    assert regions.loc["colony_1", "side"] == "left"
    assert regions.loc["colony_2", "side"] == "right"
    assert regions.side_split_x_px.eq(150).all()


def test_overlapping_colonies_fail_loudly(tmp_path):
    path, table = _write_regions(tmp_path)
    table.loc[table.name == "colony_1", "tracking_x_max_px"] = 270
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="overlap in x"):
        load_panorama_regions(path)


def test_explicit_side_label_still_checked_against_inferred_geometry(tmp_path):
    path, table = _write_regions(tmp_path)
    table.loc[table.name == "water_2", "semantic_label"] = "water_R"
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="label/geometry side mismatch"):
        load_panorama_regions(path)
