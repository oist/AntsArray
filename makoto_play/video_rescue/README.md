# video_rescue

Moves PylonRecorder output off the auto-purged acquisition share
(`/hpacquire/users/<user>`, `Y:\` on the recording PCs; files vanish after
about three weeks) into `/bucket/StephensU/<user>/<date>/blockNN/`, re-encoding
each video on deigo so the archived copy is what the recorder would now produce
with `--ffmpeg-crf 33`.

Everything runs from a deigo login shell:

```bash
scp -r makoto_play/video_rescue deigo:~/    # from the repo root; $HOME is shared with saion
R=~/video_rescue/rescue.sh
$R inventory                            # sessions.tsv + manifest.tsv under /flash/ReiterU/$USER/video_rescue
$R copy  --all                          # insurance: verbatim copy, oldest session first (datacp partition)
$R probe --all                          # exact frame count per video (compute array, I/O only)
$R xcode --all                          # window encode -> join -> verify -> upload (compute array)
$R status
$R prune-raw --session 20260819/block01 --yes   # only after xcode verified every video of the session
```

`--session` accepts a session id (`20260819_181951`) or its destination
(`20260819/block01`). `xcode --dry-run` prints the plan without submitting;
`--files 'cam05_*'` limits a run to matching videos.

## What lands where

| path | content |
|---|---|
| `$DEST_ROOT/<date>/blockNN/<name>.mkv` | re-encoded video, same file name as the recording |
| `$DEST_ROOT/<date>/blockNN/<name>.mkv.xcode.json` | provenance: source size/frames, encoder line, verification |
| `$DEST_ROOT/<date>/blockNN/<name>.mkv.diag.json` | recorder sidecar, copied verbatim by `copy` |
| `$DEST_ROOT/<date>/short_HHMMSS/` | stub sessions (a recorder restart loop leaves thousands of sub-MB files), copied verbatim |
| `$RAW_ROOT/<date>/blockNN/<name>.mkv` | verbatim copy made by `copy`; delete with `prune-raw` once the transcode is verified |
| `$WORK/state/<date>/blockNN/` | `copy.done`, `<name>.probe.json`, `<name>.uploaded` |
| `$WORK/parts/<date>/blockNN/<name>/wNNNN.mkv` | encoded windows, removed after upload |

Defaults live in `lib.sh` and are overridable from the environment
as `VR_<NAME>` (`VR_DEST_ROOT`, `VR_RAW_ROOT`, `VR_WORK`, `VR_X264_PRESET`, `VR_X264_CRF`, `VR_WINDOW_SEC`,
`VR_TASK_CPUS`, `VR_ARRAY_THROTTLE`, ...).

## Encoder equivalence

The recorder's new target is `h264_nvenc -profile main -preset p3 -rc vbr
-cq 33 -maxrate 150M -bufsize 240M -rc-lookahead 4 -surfaces 16 -g 60s`.
No cluster GPU offers NVENC, so the archive is encoded with libx264 on deigo.
Calibration on a 60 s sample of a cq31 recording (4024x3036, 24 fps), all
measured against the same cq31 source:

| encode | size | SSIM (Y) | PSNR (Y) |
|---|---|---|---|
| NVENC p3 cq33 (RTX 5070 Ti, exact recorder flags) | 14.7 MB | 0.9878 | 47.20 |
| libx264 medium crf29 | 14.4 MB | 0.9890 | 47.72 |
| libx264 **faster crf29** (default) | 13.5 MB | 0.9891 | 47.64 |
| libx264 medium crf30 | 12.3 MB | 0.9883 | 47.26 |
| libx264 veryfast crf29 | 9.0 MB | 0.9868 | 46.29 |

`faster crf29` matches the NVENC cq33 target on size and beats it on both
quality metrics. The output keeps profile main, yuv420p, a 60 s GOP with no
scene-cut keyframes, and the 150/240 Mbit VBV, so `chunk.sbatch` splits it on
the same keyframe grid as a direct recording.

Two caveats. This is a second lossy generation: the archive is slightly
worse than a direct cq33 recording would have been, and it is about 30% of
the cq31 file size rather than the ~25% saving cq31 -> cq33 gives at record
time. And the cluster libx264 is not the recorder's NVENC. A first ArUco
spot-check (70 s of cam06, 20260828/block02, 19 tagged ants, production
detector) put x264 faster crf29 at -1.2% detections versus the cq31 source
(NVENC cq33: -7.2%, medium crf29: +5.1%), with matched corners within 0.2 px
mean / 0.7 px p99 for every encode; per-frame flips run either way at this
marginal tag size. SLEAP behaviour is still unchecked, so verify a whole
block through the pipeline before the raw copies are pruned.

## How xcode works

1. `probe` counts packets in each source (the recorder writes live-mode
   Matroska with no cues or duration, so this is the only exact count).
2. `xcode` splits every video into `WINDOW_SEC` windows (default 4 h) that
   start on multiples of the 60 s GOP, i.e. on source keyframes. Each array
   task encodes one window with an input-side `-ss` and `-frames:v`, checks
   the frame count it produced, and writes `wNNNN.mkv` + `.ok`.
3. The task that completes a video's last window takes a `mkdir` lock, joins
   the windows with the concat demuxer under `-c copy`, scans the packets of
   the joined file (frame count must equal the probe, keyframe count must
   equal `ceil(frames / GOP)`), writes the `.xcode.json`, rsyncs both to
   `$DEST_ROOT` through the login node (compute nodes mount `/bucket`
   read-only), verifies the remote size, and removes its scratch.
4. Re-running `xcode` is idempotent: uploaded videos and finished windows are
   skipped, a crashed finalize is retried.

Measured throughput on deigo compute: x264 faster on 4024x3036 does 9.6 fps
with 2 cores, 14.9 with 4, 15.6 with 8, 23.6 with 16, so tasks use 4 cores.
At the default `ARRAY_THROTTLE=400` (1600 of the account's 2000 cores) the
full 2026-08/09 backlog (~690 M frames) needs roughly 35 h of wall time when
the queue is free.

## Sources that already live on /bucket

The same flow compresses a block that was archived verbatim (e.g.
`/bucket/ReiterU/Ants/basler/20260810/block02`, 8 TB): point `VR_SRC_ROOT` at
the block directory, give the campaign its own `VR_WORK` (`inventory` rewrites
`manifest.tsv`), name the session with `--dest-rel` (auto-numbering would say
`block01`), skip `copy` except `--sidecars-only`, and finish with
`prune-source --symlink` instead of `prune-raw`:

```bash
export VR_SRC_ROOT=/bucket/ReiterU/Ants/basler/20260810/block02
export VR_WORK=/flash/ReiterU/$USER/video_rescue_0810
$R inventory --dest-rel 20260810/block02
$R copy  --session 20260810/block02 --sidecars-only
$R probe --session 20260810/block02
$R xcode --session 20260810/block02
$R prune-source --session 20260810/block02 --symlink        # dry run
$R prune-source --session 20260810/block02 --symlink --yes  # all-or-nothing: 26/26 verified or nothing goes
```

`--symlink` leaves an absolute link to each transcode where the original was,
so the block directory, its view blocks (relative `../blockNN/` links) and the
detection pipeline keep resolving the same file names. Compute nodes and saion
read `/bucket/StephensU`, so detection can continue through the links; Windows
SMB shares of `/bucket/ReiterU` may not follow them.

## Order of operations for purge-threatened data

`copy` is deliberately separate from `xcode`: the datacp nodes stream at
~370 MB/s into `/bucket`, so a session is safe within hours, whereas the
transcode takes days. Run `copy` first for anything close to the purge age,
then `probe`/`xcode` (which read the original while it exists and fall back
to the raw copy afterwards), and `prune-raw` last.
