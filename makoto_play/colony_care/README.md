# colony_care - husbandry records for reared colonies

Paper sheets that are filled in while caring for the colonies, and the table they become.
The colony name is the key everywhere: on the paper, in `husbandry_log.csv`, and in the tables
of `basler/_colonies/` (`colonies.csv`, `tag_assignments.csv`, `arena.csv`, ...). Names are what
the lab writes: `22`, `10#1`, `Mix` (a number 1-999, a split `<number>#<1-99>`, or a word of at
most 6 letters/digits; case does not matter). A name is given once and never reused.

Paper is where things are written; `husbandry_log.csv` is the record. Summaries (last feeding,
dead counts and food left over, paint cohorts and ages, population curves, tray due dates, which
colony was in the arena) are derived from the log, never typed a second time.

## The monthly set (`care_sheet.py`)

One PDF per calendar month, all pages A4 landscape:

- **Feeding sheets** (2: days 1-15 and 16-end). All colonies are fed in one round, so each day
  is one line, with a **D | L** pair of cells per colony: D = dead ants found, L = food left
  over (0 none, 1 a little, 2 about half, 3 most or all). 16 colony columns; blank ones are for
  a new colony. Colonies in the arena have a shaded name.
- **Colony sheets** (one per colony, kept next to it): one line per 7 days (1-7, 8-14, 15-21,
  22-28, 29-end) for the paint day, colour, new adults (number painted) and pupae; tag IDs of
  dead tagged ants; comments; and events. Adults are not counted every week (see below).

```powershell
python makoto_play/colony_care/care_sheet.py --month 2026-11 `
    --colonies-dir Z:\ReiterU\Ants\basler\_colonies `
    --out Z:\ReiterU\Ants\basler\_colonies\sheets\2026-11\care_2026-11.pdf `
    --png-dir Z:\ReiterU\Ants\basler\_colonies\sheets\2026-11\png
```

The PDFs (and scans of the filled sheets) are kept in `basler/_colonies/sheets/<YYYY-MM>/`, not
in this repository. The colonies come from the roster in `colonies.csv` (below), in file
order; `--colonies 8,Mix,22-29` overrides it (e.g. to reprint one colony sheet with
`--no-feeding`). Without `--colonies-dir` and `--colonies`, only the feeding sheets are
printed, with blank columns. A log of the run is written next to the PDF; it warns when a
colony is in the arena but has no colony sheet.

**Records already in the log are printed into the sheets** (in blue), so a sheet printed
again, or late in the month, shows what was done and the carers write only what is missing:

- feeding sheets: time and initials of each feeding round, each colony's D and L (D = dead
  females; dead rows are logged only when D > 0, so an L code 0-3 read from a sheet with no
  dead row is D 0), a slash for `not_fed`;
- colony sheets: `Collected` (`colonies.csv` `collected_on`) and `Last tray change` (latest
  `tray_changed` up to the end of the month); per 7-day line the paint day (date of painting,
  else of the pupa count, and the initials), colours, new adults (0 when the pupae were
  counted but nobody was painted), pupae (last count), tag IDs of dead ants and the `note`
  rows as comments; the events of the month (`tray_changed`, `nest_changed`, `tagged`,
  `census adult` = count, `arena_in`/`arena_out`, `removed`, `tag_lost` = other) with the
  type ticked; when more than 8, the last line says how many more are in the log.
- A number that was not recorded (a `painted` or `dead` row without n or IDs) is printed as
  `?` (`3+?` with known ones), never as 0.

The run log lists the days and colonies that got records. `--blank` prints the sheets
without them. A missing `husbandry_log.csv` prints blank sheets (with the collection dates);
a bad row in it stops the run with its line number.

What goes where:

| sheet | part | when | write |
| --- | --- | --- | --- |
| feeding | Time, By | every feeding round | start time (to the minute for colonies in the arena), your initials |
| feeding | D, L | every feeding round | dead ants found; food left over 0-3; slash both cells if the colony was not fed |
| colony | 7-day line | paint day, once a week | date, initials, colour painted, new adults (number painted), pupae |
| colony | Dead: tag IDs | when a tagged ant is found dead | its tag ID; photograph the tag into the tag_photos inbox `C<colony>_dead` |
| colony | Comments | any time | free text |
| colony | Events | when it happens | date, time (to the minute while in the arena), type, details, initials; type `count` = a full count of the adults (number in details) |

### Adults: computed, not counted every week

A full count of a colony is done only now and then (Events: `count`; or the count at
collection). In between, the number of adults is computed:

```text
adults(t) = last full count on or before t
          + new adults painted since then      (colony sheet, "New adults")
          - dead ants found since then         (feeding sheet, D)
          - ants removed since then            (Events: removal)
```

"Since then" = a later day, or the same day at a later time when both have a time; an untimed
event on the day of the count is taken to be in the count. Of two counts at the same date and
time, the later row (a recount) wins. Changes whose number was not recorded, and a result below
zero, are listed with the number instead of being guessed. `status.html` shows the number per
colony with the date of the last full count; for tagged colonies in the arena it can be checked
against the tag census from video. Each colony needs one full count to start from.

Feeding plan printed on the sheets: Mon/Wed/Fri `MW RR J` (mealworm, red roach, jelly),
Tue/Thu `J`, Saturday or Sunday `J` (`J*`; weekend lines are shaded).

Paint codes: `Y` is the first paint after collection; then single dots `G R B Or W Gy`
(green, red, blue, orange, white, grey); after that two dots, written `G+R`. `G+R` and `R+G`
are the same code. A cohort is identified by (colony, date painted); the colour is only its
visible label.

Arena lines: without `--colonies-dir` there are none. With it, `arena.csv` must exist and be
valid (an Excel BOM is fine; a half holds one colony name or nothing). A layout change inside
the month is listed, e.g. `ARENA: 10/01 right half, 10/16 out`.

Machine-readable parts: a QR code (`FEED1 <YYYY-MM> P<1|2>` or `CARE1 C<colony key> <YYYY-MM>`,
where the key is the name in upper case with `#` as `.`, e.g. `10.1`; kept short so it stays a
version-1 code with ~1 mm modules) and four ArUco markers (`DICT_4X4_50`, ids 0-3 clockwise
from top left) for reading photos of filled sheets automatically later. Photograph a whole
sheet so that it fills the frame.

Code: `sheet_model.py` (months, feeding plan, names, codes, roster, arena.csv), `sheet_fill.py`
(what the log says for each cell), `sheet_pages.py` (drawing), `care_sheet.py` (PDF and command
line).

## Roster: `colonies.csv` columns `care_from`, `care_until`

A colony is on the care sheets of a month when `care_from` <= the last day of the month and
`care_until` is blank or >= the first day. A row without `care_from` (a colony kept only in the
analysis tables) is not printed. Add a row when a colony is collected or split; fill
`care_until` when it ends or is merged. `collected_on` can be filled in later.

## husbandry_log.csv (in `basler/_colonies/`)

One row per care action or event, appended from photos of the filled sheets after review.
Rows of the same day are ordered by time; a row without a time counts as earlier than a timed
one. A `not_fed` row cancels that day's feeding of its colony.

| column | meaning |
| --- | --- |
| `date` | YYYY-MM-DD |
| `time` | HH:MM, blank when not written |
| `colony` | colony name; `all` for a feeding round |
| `event` | see below |
| `what` | food codes, paint code, census item, left-over code, tray/nest type, arena side |
| `n` | count |
| `ids` | tag IDs, space separated |
| `block` | recording block (`<date>/block**`) when the event happened during a recording |
| `by` | initials |
| `source` | photo of the sheet + row/cell it was read from |
| `note` | free text |
| `provisional` | `y` = entered from photos, tables or notes before the paper records were merged |

| event | `what` / `n` | feeds |
| --- | --- | --- |
| `collected` | site / adults | `colonies.csv` (collection date) |
| `fed` | `MW+RR+J`, `J`, ...; colony `all` for the round | time since feeding (analysis covariate) |
| `not_fed` | - | a slashed D/L pair |
| `leftover` | `0`-`3`; `y` = some left, amount not recorded (Forms records) | food intake per colony |
| `water` | - | water exchanged (Forms records) |
| `dead` | - / number of dead females found, `ids` of tagged ones; `male` / number of dead males | mortality per colony; computed adults (females only) |
| `painted` | paint code / number painted | paint cohorts -> age |
| `census` | `pupa` / count (weekly); `adult` / count (full count, occasional) | computed adults; population curve; checked against tag census from video |
| `tray_changed`, `nest_changed` | type | tray due date; disturbance record |
| `arena_in`, `arena_out` | side | `arena.csv` -> catalog `colony_left` / `colony_right` |
| `tagged` | paint code / number, `ids` | new rows of `tag_assignments.csv` |
| `removed` | - / number, `ids` | computed adults; ends rows of `tag_assignments.csv` |
| `tag_lost` | - / number, `ids` | loose tags found (the ants are still there) |
| `note` | - | comments and anything else |

A colony was fed on a day when there is a `fed` row for `all` or for that colony and no
`not_fed` row for it. Tag IDs of dead or removed ants also end their rows in
`tag_assignments.csv` (tag_photos does that from the photos).

## Status page (`status_page.py`)

```powershell
python makoto_play/colony_care/status_page.py --colonies-dir Z:\ReiterU\Ants\basler\_colonies
```

writes `_colonies/status.html` (open it in a browser; self-contained) and a run log in
`_colonies/logs/`. It has an overview with one line per colony on the care roster (arena,
computed adults, last full count, pupae, new adults and deaths in the last 28 days, last fed,
left over, water change, tray status, last paint, tags in use) and a section per colony with the
same facts and its newest 200 history rows. Colonies whose care ended before the month are in a
"Legacy colonies" table (care period, collection, feedings, dead females and males, notes).
Provisional rows are highlighted; `*` marks a number that uses them and `-?` one that misses
changes of unknown size. Code: `husbandry.py` (reading the log, computing), `status_page.py`
(HTML).

## Legacy records: the Microsoft Forms import (`import_forms.py`)

Until October 2026 each feeding was recorded per box with a Microsoft Form; from then on the
paper feeding sheet replaces it. The Forms export (`Check of Ants Box.xlsx`, 2024-02 to
2026-10) was imported once into `husbandry_log.csv` (rows with source `forms ID <n>`; the
workbook is kept in `_colonies/evidence/legacy/`):

```powershell
python makoto_play/colony_care/import_forms.py --xlsx "Check of Ants Box.xlsx" --colonies-dir Z:\ReiterU\Ants\basler\_colonies         # dry run
python makoto_play/colony_care/import_forms.py --xlsx "Check of Ants Box.xlsx" --colonies-dir Z:\ReiterU\Ants\basler\_colonies --yes   # write
```

- Each response: `fed` (or `not_fed`), `dead` females, `dead male`, `water`, and from the
  remarks `leftover y`, `tray_changed`, `nest_changed`, `note wiped out`. The Forms "Dead number
  (total)" included the males, so dead females = total - males; before the Forms had a male
  column, the males come from remarks that say how many males died (in Japanese).
  The remark itself is kept in the note of the `fed` row; free-text counts ("about 10") are
  summed and noted. Exact repeated responses are left out (listed in the run log).
- Box names are unified: full-width digits, `mix`/`MIX`/`miz` -> `Mix`, the `13tag-R`
  spellings, `4 with queen` -> `4`. Box `10` is `10#1` and `10q` is `10#2` from the first day
  `10q` was recorded (2026-05-13); before that `10` is the undivided colony `10`.
- `colonies.csv` got every box with its first and last record as `care_from` / `care_until`
  (colonies on the current roster keep a blank `care_until`), and the Box List details
  (collection date, place, size, queen, males) for boxes 1-6.
- A second import is refused; the tables from before the import are in `_colonies/backups/`.
- `xlsx_read.py` reads the workbook with the standard library (no openpyxl needed).

## Next steps

- Tagging checklist printed with the tag sheet: one tick per ID, spare copies marked "not for
  the same colony" (avoids the 2026-10-01 duplicate IDs).
- Photo -> inbox -> read -> review -> apply into `husbandry_log.csv` (as in tag_photos).
- Catalog colony columns; charts of computed adults and deaths per colony.
