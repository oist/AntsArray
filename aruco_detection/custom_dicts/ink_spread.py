"""Pre-compensate UV-printer black ink spread on rendered ArUco markers.

Black ink printed over the white base spreads outward by roughly the same distance in
every direction, so the white modules come out smaller than designed. To cancel that, the
black area of the rendered marker is eroded by a disc of radius d (black_shrink_mm): a
pixel stays black only if its centre lies at least d from every white pixel. Distances are
measured to the white pixel's *square*, not its centre, so a straight edge moves in by d
rounded to the nearest pixel (an exact half rounds down) and corners follow the true disc. The whole black area is eroded at
once, so edge-adjacent black modules stay joined with no seam between them.

Spread then grows the eroded black back by d: edges return to the design, white module
corners re-square, and only the outer corners of black come out rounded (radius d), which
no pre-compensation can avoid.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MM_PER_INCH = 25.4
PREVIEW_PANEL_PX = 630   # zoomed tag width in the before/after preview
RED = (0, 0, 255)        # BGR: black the shrink turned white
GRID = (235, 200, 160)   # BGR: design module boundaries


def check_shrink(black_shrink_mm: float, module_mm: float) -> None:
    """Reject a shrink that is negative or would erase a lone black module (half a module)."""
    if not 0 <= black_shrink_mm < module_mm / 2:   # also False for NaN
        raise ValueError(
            f"black shrink {black_shrink_mm} mm must be >= 0 and below half a module "
            f"({module_mm / 2:.3f} mm), or a lone black module would vanish")


def shrink_kernel(shrink_px: float) -> np.ndarray:
    """Offsets of the white pixels that lie closer than shrink_px to a pixel centre."""
    r = int(np.ceil(shrink_px + 0.5))
    gap = np.maximum(np.abs(np.arange(-r, r + 1)) - 0.5, 0.0)   # centre-to-square gap per axis
    return (np.hypot(gap[:, None], gap[None, :]) < shrink_px).astype(np.uint8)


def shrink_black(marker: np.ndarray, shrink_px: float) -> np.ndarray:
    """Erode the black of a 0/255 marker image by shrink_px; removed black becomes white.

    Everything outside the image counts as white, which matches the white quiet zone the
    marker sits in on the sheet. The result holds only 0 and 255.
    """
    if shrink_px <= 0:
        return marker.copy()
    black = (marker < 128).astype(np.uint8)
    kept = cv2.erode(black, shrink_kernel(shrink_px),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return np.where(kept > 0, 0, 255).astype(np.uint8)


@dataclass(frozen=True)
class ShrinkPlan:
    """How a requested black shrink lands on the pixel grid of a `dpi` raster."""

    mm: float
    dpi: float

    @property
    def px(self) -> float:
        # rounded so float noise (0.03 mm @ 2540 dpi = 2.9999999999999996) cannot flip a tie
        return round(self.mm * self.dpi / MM_PER_INCH, 6)

    @property
    def edge_px(self) -> int:
        """Whole pixels a straight black edge moves in: px to the nearest pixel, a half rounds down."""
        if self.px <= 0:
            return 0
        k = shrink_kernel(self.px)
        r = k.shape[0] // 2
        return int(k[r, r + 1:].sum())

    @property
    def edge_mm(self) -> float:
        return self.edge_px * MM_PER_INCH / self.dpi

    @property
    def error_mm(self) -> float:
        return self.edge_mm - self.mm

    def lone_module_mm(self, module_mm: float) -> tuple[float, float]:
        """(black, white) width of a module whose neighbours on both sides are the other colour."""
        return module_mm - 2 * self.edge_mm, module_mm + 2 * self.edge_mm


def preview_strip(marker: np.ndarray, margin_px: int, modules: int, module_mm: float,
                  plans: list[ShrinkPlan], title: str) -> np.ndarray:
    """Side-by-side zoomed tag per shrink (BGR): kept black, removed black in red, design grid."""
    marker_px = marker.shape[0]
    tag = cv2.copyMakeBorder(marker, margin_px, margin_px, margin_px, margin_px,
                             cv2.BORDER_CONSTANT, value=255)
    zoom = max(1, round(PREVIEW_PANEL_PX / tag.shape[0]))
    font = cv2.FONT_HERSHEY_SIMPLEX
    panels = []
    for plan in plans:
        shrunk = cv2.copyMakeBorder(shrink_black(marker, plan.px), margin_px, margin_px,
                                    margin_px, margin_px, cv2.BORDER_CONSTANT, value=255)
        panel = np.full(tag.shape + (3,), 255, np.uint8)
        panel[shrunk == 0] = 0
        panel[(tag == 0) & (shrunk == 255)] = RED
        panel = cv2.resize(panel, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
        for i in range(modules + 1):   # where generateImageMarker put each module edge
            p = (margin_px + round(i * marker_px / modules)) * zoom
            cv2.line(panel, (p, 0), (p, panel.shape[0] - 1), GRID, 1)
            cv2.line(panel, (0, p), (panel.shape[1] - 1, p), GRID, 1)
        black_mm, white_mm = plan.lone_module_mm(module_mm)
        lines = [f"black shrink {plan.mm:g} mm",
                 f"= {plan.px:.2f} px @ {plan.dpi:g} dpi -> {plan.edge_px} px = {plan.edge_mm:.4f} mm",
                 f"lone module: black {black_mm:.3f} / white {white_mm:.3f} mm"]
        caption = np.full((30 * len(lines) + 12, panel.shape[1], 3), 255, np.uint8)
        for j, text in enumerate(lines):
            cv2.putText(caption, text, (8, 28 + 30 * j), font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        panels.append(np.vstack([panel, caption]))
    gap = np.full((panels[0].shape[0], 16, 3), 255, np.uint8)
    strip = np.hstack([x for p in panels for x in (p, gap)][:-1])
    header = np.full((44, strip.shape[1], 3), 255, np.uint8)
    cv2.putText(header, title, (8, 30), font, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([header, strip])
