# tag_roster

Which ArUco IDs each colony (left/right half of the arena) is using, which tags have dropped off,
and cuttable sheets of the IDs to print for re-tagging.

## Run

On a deigo login node (2-4), once the block's ArUco detections are on the bucket:

```bash
bash /apps/unit/ReiterU/AntsArray/current/aruco_detection/tag_roster/tag_roster.sh \
    --block /bucket/ReiterU/Ants/basler/20260928/block01
```

It picks every chunk whose ArUco files exist for all cameras, the newest calibration recorded on or
before the block's date, and the dictionary from `data/PIPELINE_STATE.json`, then submits one
compute task per chunk plus a summary job (a few minutes in total). Results:

```
/flash/ReiterU/$USER/tag_roster/<date>_<block>_c<first>-<last>/out/
    needed_ids.txt       what to print, per side
    retag_left.png/.svg  cuttable sheet (open in Liene), 2 tags per ID
    retag_right.png/.svg
    roster_status.csv    every ID x side: status, rates per chunk, last seen
```

On the lab Windows PCs the same folder is `V:\<user>\tag_roster\...`.

Options: `--chunks A-B` (a window; required with `--after JOBID` to queue behind the pipeline's
`aruco_datacp` job), `--recent N` (chunks that decide the current status, default 4 = 2 h),
`--copies N`, `--previous <earlier out>/roster_status.csv`, `--hmats`, `--x-threshold`, `--out`.

## Status of an ID, per side

Detection rate = fraction of a chunk's frames in which the ID is seen.

| status | rule | printed |
|---|---|---|
| present | mean rate of the last `--recent` chunks >= 0.10 | no |
| dropped | >= 0.10 in some chunk, recent < 0.02 (last seen time given); or present in `--previous`, absent now | yes |
| absent | recent < 0.02 and never present: unused, or dropped before the window | yes |
| uncertain | anything else | no - look at those ants first |

Misreads of real tags reach ~0.014 on laminate tags (20260810/block02), hence the 0.02 floor.
Both colonies reuse IDs 0-99, so every ID is judged separately on each side.
`absent` cannot tell "never used" from "dropped before this window": pass `--previous` with the
last run's `roster_status.csv` to turn IDs that were present then into `dropped`.

## Print only some IDs by hand

```bash
python aruco_detection/custom_dicts/generate_cuttable_sheet.py \
    --npz aruco_detection/custom_dicts/custom_4x4_A100_d4_20260410_103938.npz \
    --output retag.png --ids "3,13,57-59" --copies 2 --title "LEFT retag"
```
