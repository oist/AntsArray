from tracking.single_ant.cup_order import row_major


def test_staggered_rows_use_image_order_with_geometry_and_identity_mapping():
    cups = [dict(cup_id=old, name=f"old_{old}", center_mm=[x, y], radius_mm=10)
            for old, x, y in [(1, 20, .4), (2, 10, 15), (3, 0, 0),
                              (4, 30, 14.8), (5, 20, 30), (6, 0, 29.5)]]
    ordered, mapping = row_major(cups)
    assert [m["PreviousCupID"] for m in mapping] == [3, 1, 2, 4, 6, 5]
    assert [(c["row"], c["column"]) for c in ordered] == [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)]
    assert [c["cup_id"] for c in ordered] == list(range(1, 7))
    assert [c["name"] for c in ordered] == [f"cup_{i}" for i in range(1, 7)]
    for cup, entry in zip(ordered, mapping):
        old = next(c for c in cups if c["cup_id"] == entry["PreviousCupID"])
        assert cup["center_mm"] == old["center_mm"] and cup["radius_mm"] == old["radius_mm"]
