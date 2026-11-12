"""Number manual cups in panorama reading order without changing geometry."""
from __future__ import annotations

import numpy as np


def row_major(cups):
    """Group nearby y centers, then number rows top-down and columns left-right.

    Raw calibration and displayed panorama axes differ only by translation,
    so millimetre centers have the same reading order as the image. A half
    median cup radius tolerates slight row misalignment in the manual labels.
    """
    tolerance = .5 * float(np.median([c["radius_mm"] for c in cups]))
    rows = []
    for cup in sorted(cups, key=lambda c: c["center_mm"][1]):
        if not rows or cup["center_mm"][1] - rows[-1][0]["center_mm"][1] > tolerance:
            rows.append([])
        rows[-1].append(cup)
    ordered, mapping = [], []
    for row, group in enumerate(rows, 1):
        for column, cup in enumerate(sorted(group, key=lambda c: c["center_mm"][0]), 1):
            cid = len(ordered) + 1
            ordered.append(dict(cup, cup_id=cid, name=f"cup_{cid}", row=row, column=column))
            mapping.append(dict(CupID=cid, Cup=f"cup_{cid}", Row=row, Column=column,
                                PreviousCupID=cup["cup_id"], PreviousCup=cup["name"],
                                CenterXMM=cup["center_mm"][0], CenterYMM=cup["center_mm"][1]))
    return ordered, mapping
