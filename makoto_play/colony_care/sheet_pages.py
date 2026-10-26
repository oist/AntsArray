"""Drawing of the care-sheet pages: the shared feeding sheet and the per-colony sheet.

Coordinates are mm from the top left of an A4 landscape page. Every page carries four ArUco
markers (DICT_4X4_50, ids 0-3 clockwise from top left) and a QR code with its SheetId, so a
photo of a filled sheet can be rectified and identified automatically.
"""
import textwrap
from datetime import timedelta

import cv2
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

import sheet_model as M

A4_LONG, A4_SHORT = 297.0, 210.0
MARGIN, MARKER, QR, QR_GAP = 8.0, 11.0, 25.0, 4.0
GREY, SHADE, ARENA_SHADE = "0.45", "0.92", "0.80"
MM_PER_CHAR_PT = 0.22       # mean width of a bold DejaVu Sans character, mm per point of size
WIDE_MM_PER_CHAR_PT = 0.40  # widest bold character (W measures 0.387): where a name must never overflow
MIN_STATUS_PT = 6.0
INFO_X = 126.0              # left edge of the information lines in the colony sheet header
FEEDING_SLOTS = 16          # colony column pairs on a feeding sheet (blank ones for new colonies)
MAX_FEEDING_COLONIES = 18   # more than this no longer fits the width
EVENT_TYPES = ("tray", "nest", "tagging #", "count", "arena in", "arena out", "removal", "other")
EVENT_TYPE_STEP = 14.0      # mm between the event-type tick boxes
EVENT_ROWS, EVENT_H = 8, 6.5
LEFTOVER_SCALE = "0 none, 1 a little, 2 about half, 3 most or all"

# (title, hint, width mm); None = the rest of the line
# No adult count: the number of adults is computed (last full count + new adults - dead - removed).
COLONY_COLS = (("Days", "", 20.0), ("Paint day", "date, initials", 34.0),
               ("Colour", "Y G R B Or W Gy, G+R", 28.0), ("New adults", "number painted", 22.0),
               ("Pupae", "count", 18.0), ("Dead: tag IDs", "tagged ants, photo", 40.0),
               ("Comments", "", None))
FEEDING_FIXED = (("Date", "", 12.0), ("Day", "", 9.0), ("Plan", "", 14.0), ("Time", "start", 12.0),
                 ("By", "initials", 11.0))
EVENT_COLS = (("Date", "", 18.0), ("Time", "", 14.0), ("Type", "", 112.0), ("Details", "", None),
              ("By", "", 18.0))

COLONY_FOOTER = (
    "Paint day (once a week): paint the new adults and write how many, count the pupae, photograph "
    "this sheet. Paint: Y first after collection, then G R B Or W Gy (1 dot), then 2 dots, e.g. G+R (= R+G).",
    "Adults are not counted every week: the number is computed from the last full count (Events: count), "
    "new adults, dead and removed ants.",
    "Dead ants are counted on the feeding sheet; here only the tag IDs of tagged ants{photo}. "
    "Events: the time to the minute while the colony is in the arena.",
)
FEEDING_FOOTER = (
    "D = dead females found (number; dead males go in the colony sheet Comments).  L = food left over: "
    "%s.  Slash both cells when a colony was not fed." % LEFTOVER_SCALE,
    "By: your initials.  Time: start of the round; to the minute for colonies in the arena (shaded "
    "names).  A blank column is for a new colony: write its name.",
    "MW mealworm, RR red roach, J jelly.  J* = jelly on Saturday OR Sunday (one of the two).  "
    "Anything else: Comments / Events on the colony sheet.",
)


# --- primitives ----------------------------------------------------------------------------

def _new_page(width, height):
    fig = Figure(figsize=(width / 25.4, height / 25.4))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_axis_off()
    return fig, ax


def _text(ax, x, y, s, size=7.0, color="black", weight="normal", ha="left", va="top"):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va)


def _box(ax, x, y, w, h, lw=0.6, fill=None):
    ax.add_patch(Rectangle((x, y), w, h, fill=fill is not None, facecolor=fill or "none",
                           lw=lw, edgecolor="black"))


def _image(ax, img, x, y, size):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest",
              extent=(x, x + size, y + size, y))


def _columns(x0, x1, spec):
    """[(x, width, title, hint)] laid out from x0; a None width takes what is left up to x1."""
    fixed = sum(w for _, _, w in spec if w is not None)
    out, x = [], x0
    for title, hint, w in spec:
        w = w if w is not None else x1 - x0 - fixed
        out.append((x, w, title, hint))
        x += w
    return out


def _draw_head(ax, cols, y, h):
    for x, w, title, hint in cols:
        _box(ax, x, y, w, h, lw=0.8)
        _text(ax, x + 1.2, y + 1.0, title, size=7.5, weight="bold")
        if hint:
            _text(ax, x + 1.2, y + 4.8, hint, size=5.5, color=GREY)


def _draw_frame(ax, code, footer):
    """Corner markers, QR code with its text, and the footer between the bottom markers."""
    width, height = A4_LONG, A4_SHORT
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    far_x, far_y = width - MARGIN - MARKER, height - MARGIN - MARKER
    for marker_id, (x, y) in enumerate(((MARGIN, MARGIN), (far_x, MARGIN), (far_x, far_y), (MARGIN, far_y))):
        _image(ax, cv2.aruco.generateImageMarker(dictionary, marker_id, 6, borderBits=1), x, y, MARKER)
    qr_x = far_x - QR_GAP - QR
    _image(ax, cv2.QRCodeEncoder.create().encode(code), qr_x, MARGIN, QR)
    _text(ax, qr_x + QR / 2, MARGIN + QR + 0.3, code, size=4.5, color=GREY, ha="center")
    for i, line in enumerate(footer):
        _text(ax, MARGIN + MARKER + 3, far_y + 0.6 + i * 3.4, line, size=6)


def _status_line(ax, x, y, status, max_width):
    """Arena line; a long text shrinks (to 6 pt) and then wraps, so it never runs into the QR."""
    if status is None:
        return
    text, highlight = status
    size = max(MIN_STATUS_PT, min(8.0, max_width / (MM_PER_CHAR_PT * len(text))))
    width_chars = int(max_width / (MM_PER_CHAR_PT * size))
    _text(ax, x, y, "\n".join(textwrap.wrap(text, width_chars)), size=size,
          weight="bold" if highlight else "normal", color="black" if highlight else GREY)


def _title(ax, big, small, max_width=None):
    """Big bold title (22 pt, smaller when it would be wider than max_width mm) and a subtitle."""
    left = MARGIN + MARKER + 4
    size = 22.0 if max_width is None else min(22.0, max_width / (WIDE_MM_PER_CHAR_PT * len(big)))
    _text(ax, left, MARGIN, big, size=size, weight="bold")
    _text(ax, left, MARGIN + 11, small, size=9)


# --- per-colony sheet ----------------------------------------------------------------------

def draw_colony_page(colony, month, arena):
    """Paint, count, dead tag IDs, comments and events of one colony for a month.

    arena = M.arena_header(...) or None.
    """
    fig, ax = _new_page(A4_LONG, A4_SHORT)
    photo = " (photograph tags into the tag_photos inbox C%s_dead)" % colony if colony.isdigit() else ""
    _draw_frame(ax, M.colony_code(colony, month), [line.format(photo=photo) for line in COLONY_FOOTER])
    _title(ax, "COLONY %s" % colony, "Colony sheet   %s" % month.strftime("%B %Y"),
           max_width=INFO_X - 3 - (MARGIN + MARKER + 4))
    _text(ax, MARGIN + MARKER + 4, MARGIN + 17, "Feeding, dead counts and left-over food: feeding sheet.",
          size=7.5, color=GREY)
    qr_x = A4_LONG - MARGIN - MARKER - QR_GAP - QR
    _status_line(ax, INFO_X, MARGIN + 1, arena, qr_x - 3 - INFO_X)
    _text(ax, INFO_X, MARGIN + 8, "Last tray change: ______________   (every 2-3 months)", size=8)
    _text(ax, INFO_X, MARGIN + 15, "Collected: ______________", size=8)
    cols = _columns(MARGIN, A4_LONG - MARGIN, COLONY_COLS)
    top, head = 37.0, 9.0
    _draw_head(ax, cols, top, head)
    weeks = M.week_ranges(month)
    row_h = (118.0 - top - head) / len(weeks)
    for k, (first, last) in enumerate(weeks):
        _draw_colony_week(ax, cols, top + head + k * row_h, row_h, first, last)
    _draw_events(ax, 122.0, MARGIN, A4_LONG - MARGIN)
    return fig


def _draw_colony_week(ax, cols, y, h, first, last):
    for x, w, _, _ in cols:
        _box(ax, x, y, w, h)
    wx, px = cols[0][0], cols[1][0]
    _text(ax, wx + 1.2, y + 1.2, "%s -\n%s" % (first.strftime("%m/%d"), last.strftime("%m/%d")),
          size=7.5, weight="bold")
    _text(ax, px + 1.2, y + 1.5, "Date ___/___", size=7, color=GREY)
    _text(ax, px + 1.2, y + h / 2 + 0.5, "By ________", size=7, color=GREY)


def _draw_events(ax, top, x0, x1):
    _text(ax, x0, top, "Events - date, time (to the minute while in the arena), tick the type, "
          "details, initials", size=7.5, weight="bold")
    cols = _columns(x0, x1, EVENT_COLS)
    head_y = top + 4.0
    for x, w, title, _ in cols:
        _box(ax, x, head_y, w, 5.0, lw=0.8)
        _text(ax, x + 1.0, head_y + 0.9, title, size=6.5, weight="bold")
    type_x = cols[2][0]
    for r in range(EVENT_ROWS):
        y = head_y + 5.0 + r * EVENT_H
        for x, w, _, _ in cols:
            _box(ax, x, y, w, EVENT_H, lw=0.5)
        _text(ax, x0 + 1.0, y + 1.8, "___/___", size=6.5, color=GREY)
        for i, name in enumerate(EVENT_TYPES):
            bx = type_x + 1.0 + i * EVENT_TYPE_STEP
            _box(ax, bx, y + 2.1, 2.2, 2.2, lw=0.5)
            _text(ax, bx + 3.0, y + 1.9, name, size=5.8, color=GREY)


# --- shared feeding sheet ------------------------------------------------------------------

def slot_name_size(name, slot_w):
    """Font size (pt) of a colony name in a feeding-sheet column: 8 pt, smaller if it would not fit."""
    return min(8.0, (slot_w - 1.0) / (WIDE_MM_PER_CHAR_PT * max(len(name), 1)))


def feeding_slots(colonies):
    """Column headers of a feeding sheet: the colonies, then blanks up to FEEDING_SLOTS."""
    if len(colonies) > MAX_FEEDING_COLONIES:
        raise M.InputError("a feeding sheet holds at most %d colonies, got %d"
                           % (MAX_FEEDING_COLONIES, len(colonies)))
    return list(colonies) + [""] * (FEEDING_SLOTS - len(colonies))


def draw_feeding_page(month, page, colonies, arena_rows):
    """One line per day (half a month) with a dead / left-over pair per colony.

    arena_rows = rows of arena.csv or None (unknown).
    """
    first, last = M.feeding_pages(month)[page - 1]
    fig, ax = _new_page(A4_LONG, A4_SHORT)
    _draw_frame(ax, M.feeding_code(month, page), FEEDING_FOOTER)
    _title(ax, "FEEDING   %s" % month.strftime("%B %Y"),
           "Days %d - %d   (sheet %d of 2)    Plan:  Mon / Wed / Fri  %s    Tue / Thu  %s    Sat or Sun  %s"
           % (first.day, last.day, page, M.FEED_ALL, M.FEED_JELLY, M.FEED_JELLY))
    qr_x = A4_LONG - MARGIN - MARKER - QR_GAP - QR
    _status_line(ax, MARGIN + MARKER + 4, MARGIN + 17, M.feeding_arena_line(arena_rows, first, last),
                 qr_x - 3 - (MARGIN + MARKER + 4))
    in_arena = {M.colony_key(c) for c in M.arena_colonies(arena_rows, first, last)}
    fixed = _columns(MARGIN, MARGIN + sum(w for _, _, w in FEEDING_FIXED), FEEDING_FIXED)
    slots = feeding_slots(colonies)
    x0 = fixed[-1][0] + fixed[-1][1]
    slot_w = (A4_LONG - MARGIN - x0) / len(slots)
    top, head = 37.0, 11.0
    _draw_feeding_head(ax, fixed, slots, x0, slot_w, top, head, in_arena)
    days = [first + timedelta(days=k) for k in range((last - first).days + 1)]
    row_h = (186.0 - top - head) / len(days)
    for i, day in enumerate(days):
        y = top + head + i * row_h
        _draw_feeding_day(ax, fixed, len(slots), x0, slot_w, y, row_h, day)
        if day.weekday() == 0 and i:
            ax.plot([MARGIN, A4_LONG - MARGIN], [y, y], color="black", lw=1.6)
    return fig


def _draw_feeding_head(ax, fixed, slots, x0, slot_w, top, head, in_arena):
    for x, w, title, hint in fixed:
        _box(ax, x, top, w, head, lw=0.8)
        _text(ax, x + 1.0, top + 1.0, title, size=7, weight="bold")
        if hint:
            _text(ax, x + 1.0, top + 5.0, hint, size=5, color=GREY)
    for k, name in enumerate(slots):
        x = x0 + k * slot_w
        shade = ARENA_SHADE if name and M.colony_key(name) in in_arena else None
        _box(ax, x, top, slot_w, head - 4.0, lw=0.8, fill=shade)
        _text(ax, x + slot_w / 2, top + 0.8, name, size=slot_name_size(name, slot_w), weight="bold", ha="center")
        for j, sub in enumerate(("D", "L")):
            _box(ax, x + j * slot_w / 2, top + head - 4.0, slot_w / 2, 4.0, lw=0.5)
            _text(ax, x + (j + 0.5) * slot_w / 2, top + head - 3.4, sub, size=5.5, color=GREY, ha="center")


def _draw_feeding_day(ax, fixed, n_slots, x0, slot_w, y, h, day):
    shade = SHADE if day.weekday() >= 5 else None
    for x, w, _, _ in fixed:
        _box(ax, x, y, w, h, lw=0.5, fill=shade)
    for k in range(n_slots):
        _box(ax, x0 + k * slot_w, y, slot_w, h, lw=0.8, fill=shade)
        ax.plot([x0 + (k + 0.5) * slot_w] * 2, [y, y + h], color="0.6", lw=0.4)
    mid = y + h / 2
    (dx, _, _, _), (wx, _, _, _), (px, _, _, _), (tx, tw, _, _) = fixed[:4]
    _text(ax, dx + 1.0, mid, day.strftime("%m/%d"), size=7, va="center")
    _text(ax, wx + 1.0, mid, M.DAY_NAMES[day.weekday()], size=7, va="center")
    plan = M.planned_food(day)
    _text(ax, px + 1.0, mid, plan, size=6.5, va="center", weight="bold" if plan == M.FEED_ALL else "normal")
    _text(ax, tx + tw - 1.0, mid, "__:__", size=6, color=GREY, ha="right", va="center")
