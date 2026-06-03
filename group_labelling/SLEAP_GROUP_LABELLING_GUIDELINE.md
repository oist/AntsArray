# Procedure for SLEAP group labelling - problem-frame rescue round

## Overview

Hi everyone! Thank you for helping with this SLEAP labelling round.

This time, each person will receive one separate SLEAP package file (`.pkg.slp`). The video frames are embedded in the file, so you should not need to locate or download the original videos separately.

Each file contains only the assigned suggested/problem frames for that labeler. Please go through the suggested frames in your assigned file, correct the labels, add missing useful ant instances, and remove false positives or duplicated predictions.

When you are finished, please save the corrected file with your name appended to the filename and return it to the shared folder.

Example:

```text
problem_frames_chunk01_Makoto_corrected.pkg.slp
```

## Purpose of This Round

These frames were selected because the current model produced problematic predictions, such as missing keypoints, incomplete ants, false positives, duplicated instances, or difficult partial views.

The corrected labels will be merged into a master SLEAP project and used to train improved top-down models:

- a centroid/anchor model
- a centered-instance model

## SLEAP Installation and GUI Help

Please use **SLEAP 1.6.2** for this labelling round. The package files were prepared for the current SLEAP workflow, and older SLEAP versions may behave differently when opening or saving `.pkg.slp` files.

If you already have an older SLEAP installation, please install this version as a separate `uv` tool instead of modifying your existing environment.

Recommended `uv` installation:

```powershell
uv tool install --python 3.13 "sleap[nn]==1.6.2" --torch-backend auto
```

Check the installed version:

```powershell
sleap doctor
```

Start the GUI:

```powershell
sleap
```

General SLEAP installation instructions:

https://docs.sleap.ai/latest/installation/

GUI tutorial:

https://docs.sleap.ai/latest/tutorial/correcting-predictions

You only need the GUI labelling/editing workflow. The project files are already prepared.

## What to Label

Please work only on the suggested frames in your assigned file.

For each suggested frame, inspect the predicted ant instances and decide whether each one should be corrected, deleted, or left hidden/partial according to the rules below.

## Annotation Rules

### 1. Correct real ant predictions

If a predicted instance is a real ant, correct the visible keypoints. Move wrong keypoints to the correct visible body parts.

Keep occluded, out-of-frame, or uncertain keypoints hidden. Do not guess keypoints that you cannot place with confidence.

### 2. Add missing useful ant instances

If an ant is missing and is useful for training, add a new instance and label the visible keypoints.

An ant is useful when the anchor/body center is visible and enough of the body is visible to make a meaningful training example.

### 3. Delete false positives

If a predicted instance is not an ant, delete that instance.

### 4. Delete duplicated predictions

If two predicted instances are on the same ant, keep the better one, correct it, and delete the duplicate.

### 5. Partial ants

For worker ants, label a partial ant only if the ArUco tag/anchor point is visible.

If the tag is visible but some body parts are outside the image or occluded, label the visible keypoints and keep the missing parts hidden.

If only legs, antennae, or a small body edge are visible and the ArUco tag/anchor is not visible, delete or ignore that instance.

This rule is important because the centroid/anchor model needs reliable anchor examples. Partial ants without a reliable anchor can hurt training more than help.

### 6. Queen ants

The queen does not have an ArUco tag. If the painted thorax dot is visible, use the painted dot as the `aruco`/anchor keypoint.

If the painted dot is not visible, do not invent an anchor and do not substitute a different body part as the `aruco`/anchor point. Label only clearly visible body landmarks and keep the anchor hidden.

If the queen is too partial or ambiguous to label confidently, skip/delete that instance.

### 7. Visible keypoints only

Please label what is visible in the image.

Hidden means the keypoint is occluded, outside the frame, or too uncertain to place confidently.

### 8. Do not change project structure

Please do not rename nodes, add or remove skeleton nodes, change the skeleton, delete videos, or change project settings.

Only edit/add/delete instances and keypoints.

## Practical Tips

It may be easier **to set the GUI colors by node** rather than by instance, because this helps distinguish left and right antennae.

You can adjust node size and hide node names in the GUI if the display is crowded.

Please save regularly while working.

## When You Finish

Save the corrected `.pkg.slp` file with your name appended to the filename.

Upload or return the corrected file to the shared folder and let me know that you are done.

If you are unsure about a difficult case, please leave a note or send me a screenshot. Thank you very much for helping with this annotation round!
