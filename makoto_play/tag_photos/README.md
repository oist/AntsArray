# tag_photos — tag IDs from photos of dead ants and loose tags

Lab members photograph ants found dead and tags found lying in the arena. This tool reads the
ArUco ID (dictionary `custom_4x4_A100_d4`) from each photo, shows every read for a quick visual
check, and then ends the matching row of `basler/_colonies/tag_assignments.csv`
(`ended_on`, `end_reason`, `evidence`) and files the photo under `evidence/`.

Both colonies reuse tag IDs 0-99, so a photo alone cannot say which colony an ant came from.
**The folder a photo is put in is the only record of its colony.**

## 1. Put the photos in the inbox

```text
Z:\ReiterU\Ants\basler\_colonies\evidence\inbox\      (= /bucket/ReiterU/Ants/basler/_colonies/evidence/inbox)
    C21_dead\        ants of colony 21 found dead        -> end_reason found_dead
    C22_dead\
    C21_tag\         loose tags found on colony 21's side -> end_reason tag_lost
    C22_tag\
    C21_removed\     ants taken out of the colony         -> end_reason removed
    C21_check\       live ants, to check who wears which tag -> read and reviewed, never written
```

`C22_dropped` is the same as `C22_tag`; `Colony22_dead`, `c22_DEAD` etc. also work. The colony must be listed in `colonies.csv`.
JPEG/PNG/TIFF are read directly; iPhone HEIC needs `pip install pillow-heif` (or export as JPEG).
`ended_on` is the photo's EXIF capture date, so photograph on the day the ant is found. Photos
sent through a messenger or copied by some apps lose that date; for them the date found must be
typed into `date_override` (apply never uses file times).

Photo tips (from the validation below):

- Fill the frame: tags should be at least ~60 px wide in the photo. Tags of ~35 px are often
  missed; 60-110 px tags were all read.
- Shoot from straight above, without flash; tilt the phone if the film shows a bright reflection.
- Several ants or tags per photo are fine. Turn loose tags face up.
- For anything important, take two or three shots: misses differ between shots.

## 2. Scan

```powershell
python makoto_play/tag_photos/tag_photos.py scan --colonies Z:\ReiterU\Ants\basler\_colonies
```

Writes `inbox/_review/<YYYYMMDD_HHMMSS>/`:

| file | what |
|---|---|
| `review.html` | the interactive review page (section 3); photos are embedded, so it opens straight from disk |
| `review.csv` | one row per tag; `accept` is pre-set to `y` for `ok` reads |
| `crops/` | per tag: the warped crop, the dictionary drawing of the ID it was read as (turned the same way), the surroundings |
| `annotated/` | each photo with every read outlined and labelled `#n ID` (n = the row's `n`) |
| `scan.json` | inbox and colony folder, SHA-256 of every photo, photos that could not be read |
| `tag_photos_scan_log_<time>.txt` | everything printed, including skipped folders and files |

Status of a read:

| status | meaning | what to do |
|---|---|---|
| `ok` | >= 2 exact reads of one ID, no real competitor | glance at the crop |
| `check` | only 1-bit-off reads, a single read, a competing ID, or `mirrored` | compare crop and drawing, set `accept` |
| `unreadable` | a black-bordered square in a white margin that matches no ID | read it by eye, type it into `id_override` |

`mirrored` means the tag was read from behind through its clear film (a loose tag lying face
down); the dictionary drawing is mirrored the same way. IDs 0 6 21 38 47 67 92 97 look the same
from behind; 8/22 and 10/88 are each other's mirror image 1 bit off.

## 3. Review

Open `review.html` in Chrome or Edge. Each photo shows every read outlined (green ok, orange
check, red unreadable) as `#n ID`, and a table with the crops:

- `accept`: choose `y` or `n`. `ok` rows come pre-set to `y`. A row left blank is reported by
  apply, and its photo stays in the inbox until every tag on it has a decision.
- `id_override`: the ID you read; applies unless `accept` is `n`.
- **A tag the scan missed** (no outline on it): click near it on the photo, then click the four
  outer corners of its black border in the zoom view (any order; `2x/3x/4x` change the zoom,
  `undo corner` removes the last click). The page reads the tag from those corners and lists
  candidate IDs with how many of 27 slightly shifted reads agree, each beside the dictionary
  drawing turned the way the tag lies. `use <ID>` adds it as a manual row (`m1`, `m2`, ...);
  `use typed ID` takes an ID you read yourself. Typical misses: an ant leg or the body lying
  across the black border, torn or crumpled film, a tag seen at a steep angle. Few agreeing reads
  (e.g. 6/27) mean the print is too damaged to trust the candidates.
- **Save review.csv** (top or bottom of the page) and save it over `review.csv` in the review
  folder (or move the downloaded file there). Until then decisions are kept in this browser.

`review.csv` can also be edited in Excel instead (it may save as cp932 or with 2026/10/20 dates):
the same columns, `date_override` for photos whose `date_source` is not `exif`, and a missed tag
as an extra row with `photo` (e.g. `C21_dead/IMG_3618.JPEG`), any `n` and the ID in `id_override`.

## 4. Apply

```powershell
python makoto_play/tag_photos/tag_photos.py apply --review <inbox>\_review\<time>\review.csv          (dry run)
python makoto_play/tag_photos/tag_photos.py apply --review <inbox>\_review\<time>\review.csv --yes
```

For every accepted `(colony, tag_id)` the one active row (blank `ended_on`) is ended:
`ended_on` = earliest date found, `end_reason` from the folder, and
`photo Colony22_founddead_IMG_1234.JPEG (tag_photos)` appended to `evidence`. The evidence name
gets `_<YYYYMMDD>` added when another photo already has it; the dry run shows the final names.

With `--yes`, in this order:

1. each photo that ends a row is copied into `evidence/` and the copy is verified (SHA-256);
2. the previous table is saved as `tag_assignments.before.csv` in the review folder and the new
   table is written - unchanged rows keep their exact bytes;
3. a photo leaves the inbox only when every accepted tag on it is recorded.

If step 1 or 2 fails (for example the table is open in Excel), the copies are removed and the
table is left as it was. Running apply again after any interruption is safe: rows already ended
from the same photo show as `[RECORDED]`, and the leftover inbox photo is removed.

Not applied (exit code 1): a photo changed since the scan (`[SKIP]`, scan again), no EXIF date and
no `date_override`, and `[PROBLEM]` rows - no row for that colony and ID, the row already ended
for another reason, two active rows, found before `tagged_on`, or one ID photographed in two kinds
of folders. `apply` refuses to write if `tag_assignments.csv` changed while it ran, or if a row
of it does not have as many fields as the header.

## How a tag is read

`photo_decode.py`: every quad OpenCV finds, accepted or rejected, is warped to an 8x8 grid (white
margin ring, black border ring, 4x4 bits) and its bits are matched against every code in all four
rotations. The tag is then re-detected in a crop at three scales, with and without CLAHE, and each
read within 1 bit of a code votes (as a mirrored vote when the mirror image matches strictly
better). Cells are read by their median so glare specks and speckled UV prints do not flip them;
a complete white margin ring lets the border pass with 12 of 20 black cells. Reference bits come
from `aruco.generateImageMarker` (the npz `patterns` array is 1 = white).

Validation on earlier photos (IDs established by hand): dead-ant photos 7/7 IDs read
(`IMG_3560` incl. the glare-covered 34; `IMG_3586`; `IMG_3587`), 6/6 large loose tags
(`IMG_3547`, 2 as `check`), 52/57 small (~35-40 px) loose tags over three shots of the same 19 tags,
each of the 19 read in at least one shot. No wrong ID in any photo.

## Tests

```powershell
python -m pytest makoto_play/tag_photos -q
```

Synthetic photos (`synth_scene.py`): rotated, tilted, blurred, noisy tags of 24-140 px, a glare
spot, clutter without tags; status rules; table planning, round trip and conflicts; scan → review →
apply end to end, including an Excel (cp932) edited review.
