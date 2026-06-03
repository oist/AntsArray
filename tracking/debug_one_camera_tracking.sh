#!/usr/bin/env bash
set -euo pipefail

# One-camera debug run for get_complete_tracks().
# Edit BASE to inspect a different camera/chunk.

PYTHON_BIN="${PYTHON_BIN:-/home/sam-reiter/miniforge3/envs/ants/bin/python}"

BASE="${BASE:-/home/sam-reiter/bucket/ReiterU/Ants/basler/20260414_20260417_CustomAruco/block02/data/cam05_cam4_2026-04-16-09-00-03_002}"
OUT_DIR="${OUT_DIR:-/home/sam-reiter/bucket/ReiterU/Ants/basler/20260414_20260417_CustomAruco/block02/debug_tracking/cam01_chunk000}"

VIDEO_FILE="${VIDEO_FILE:-${BASE}.avi}"
ARUCO_H5="${ARUCO_H5:-${BASE}_aruco_detections.h5}"
SLEAP_H5="${SLEAP_H5:-${BASE}_sleap_data.h5}"

TRACKS_OUT="${TRACKS_OUT:-${OUT_DIR}/debug_tracks.parquet}"
VIDEO_OUT="${VIDEO_OUT:-${OUT_DIR}/debug_tracking_video.mp4}"

mkdir -p "${OUT_DIR}"

echo "Debug tracking inputs"
echo "  video: ${VIDEO_FILE}"
echo "  aruco: ${ARUCO_H5}"
echo "  sleap: ${SLEAP_H5}"
echo
echo "Debug tracking outputs"
echo "  tracks: ${TRACKS_OUT}"
echo "  video: ${VIDEO_OUT}"
echo
echo "Press q in the OpenCV window to stop early."

"${PYTHON_BIN}" -c "
import pandas as pd
from tracking.core.tracking_utils import get_complete_tracks

get_complete_tracks(
    output_path='${TRACKS_OUT}',
    aruco_detection=pd.read_hdf('${ARUCO_H5}', key='detections'),
    sleap_detection=pd.read_hdf('${SLEAP_H5}', key='sleap_data').dropna(),
    video_file='${VIDEO_FILE}',
    video_out_path='${VIDEO_OUT}',
    debug_viz=True,
    debug_layout='stack',
    debug_show_aruco_raw=False,
    debug_show_aruco=False,
    debug_show_sleap=False,
    debug_show_track_output=True,
)
"
