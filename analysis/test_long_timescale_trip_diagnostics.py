import numpy as np
import pandas as pd

from analysis import long_timescale_trip_diagnostics as td


def test_equal_clock_weight_is_not_pooled_event_weight():
    rows, bins = [], []
    for rec in ("a", "b", "c"):
        for slot, values in ((0, [1.] * 100), (1, [10.])):
            for x in values:
                rows.append(dict(side="left", ant="left:001", recording=rec, phase="light",
                                 clock_stratum=slot, duration_minutes=x, observed_coverage=1.))
            bins.append(dict(ant="left:001", recording=rec, clock_stratum=slot, trip_observed_hours=1.))
    matched, support = td.clock_matched(pd.DataFrame(rows), pd.DataFrame(bins), ["a", "b", "c"])
    np.testing.assert_allclose(matched["mean"], 5.5)
    np.testing.assert_allclose(matched.long10, 50.)
    assert (matched.shared_strata == 2).all()
    assert support.n_trips.sum() == 606  # all + light, explicitly separate phases


def test_night_is_assigned_to_date_darkness_begins():
    events = pd.DataFrame(dict(exit_timestamp=pd.to_datetime(["2026-07-24 23:00", "2026-07-25 03:00", "2026-07-25 06:00"]), source_block="block"))
    meta = dict(light_on_hour=5.5, light_off_hour=19.5, windows=[dict(block="block", start="2026-07-24 00:00", stop="2026-07-26 00:00")])
    out = td.add_calendar(events, meta)
    assert out.cycle_day.tolist() == ["2026-07-24", "2026-07-24", "2026-07-25"]
    assert out.phase.tolist() == ["dark", "dark", "light"]


def test_trip_tail_does_not_imply_every_trip_lengthens():
    data = pd.DataFrame(dict(duration_minutes=[1., 1., 1., 17.], observed_coverage=1.))
    result = td.duration_summary(data)
    assert result["mean"] == 5 and result["median"] == 1
    assert result["long10"] == 25


def test_change_summary_pairs_by_identity_not_input_order():
    table = pd.DataFrame([dict(side="left", phase="all", ant=ant, recording=rec,
                               mean=value, median=value, p90=value, long10=value)
                          for ant, values in [("left:001", [1, 2, 4]), ("left:002", [9, 8, 3])]
                          for rec, value in zip(["a", "b", "c"], values)])
    result = td.changes(table.sample(frac=1, random_state=2), ["a", "b", "c"], bootstrap=100)
    assert (result.n_ants == 2).all() and (result.increased == 1).all()
    np.testing.assert_allclose(result.median_fold, (4 + 1/3)/2)


def test_recording_handoff_preserves_real_intra_recording_gap():
    from analysis.long_timescale_plots import recording_handoffs
    manifest = dict(bin_minutes=30, windows=[
        dict(block="/20260723/block01", start="2026-07-23 11:43", stop="2026-07-23 14:57"),
        dict(block="/20260723/block02", start="2026-07-23 19:31", stop="2026-07-24 09:29:50"),
        dict(block="/20260724/block01", start="2026-07-24 09:31:10", stop="2026-07-26 10:40")])
    handoffs = recording_handoffs(manifest)
    assert len(handoffs) == 1
    assert handoffs[0]["gap_seconds"] == 80
    assert handoffs[0]["previous"] == "/20260723/block02"
