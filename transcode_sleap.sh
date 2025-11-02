#!/bin/bash -l

# ------------------------------------------------------------
#  video_saion_pipeline.sh  — split → parallel encode → Saion
# ------------------------------------------------------------
#  * job1-<video>   : lossless split into _raw_ chunks     (compute/shortish)
#  * job2-<video>   : spawn encoders (job2a per chunk)     (compute)
#      - throttled: max 200 concurrent enc-* jobs (per user)
#      - skip if output segment already exists
#  * job3-<video>   : wait enc, rsync reencoded+CSV → Saion, submit sleap
#      - skip sleap per segment if .slp or .csv already exists
#  * monitor        : move local outputs, collect from Saion
# ------------------------------------------------------------
set -euo pipefail
shopt -s nullglob

echo "Starting transcode_sleap.sh …"
# ─────────────── CLI parsing ───────────────────────────────
DIR=""
SAION_NODE="largegpu"
usage(){ echo "Usage: $0 --dir <folder> [--node <saion-partition>]"; exit 1; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir)  DIR="$2"; shift 2 ;;
    --node) SAION_NODE="$2"; shift 2 ;;
    *) usage ;;
  esac
done
DIR=${DIR%/}; [[ -z "${DIR:-}" || ! -d "$DIR" ]] && usage

# ─────────────── constants & folders ───────────────────────
email="samuel.reiter@oist.jp"

SEG_SEC="${SEG_SEC:-3600}"      # 120 min segments by default
ENC_CAP="${ENC_CAP:-100000}"       # max concurrent enc-* jobs per user

base_folder="$(basename "$DIR")"
data_folder="$DIR/data"
flash_folder="/flash/ReiterU/ant_tmp/$base_folder"           # local fast storage
saion_work_remote="/work/ReiterU/ant_tmp/$base_folder"       # remote target path on Saion

JOBS_DIR="/flash/ReiterU/ant_tmp/output/jobs/$base_folder"

mkdir -p "$JOBS_DIR" "$data_folder" "$flash_folder"
chmod 2775 "$JOBS_DIR" "$data_folder" "$flash_folder"

command -v ffmpeg >/dev/null || { echo "[ERR] ffmpeg missing"; exit 2; }
command -v ffprobe >/dev/null || { echo "[ERR] ffprobe missing"; exit 2; }
command -v squeue  >/dev/null || echo "[WARN] squeue not found locally; throttling may fail here"

# ─────────────── per-video jobs ────────────────────────────
for video in "$DIR"/*.avi; do
  b="$(basename "$video")"
  [[ "$b" =~ ^\. ]] && continue
  [[ "$b" =~ _renc\.avi$ || "$b" =~ _nvenc\.avi$ ]] && continue
  vname="${b%.avi}"

  # ---------- job1 : lossless split -------------------------
job1="$JOBS_DIR/job1-$vname.sh"
cat > "$job1" <<'EOF1'
#!/bin/bash -l
#SBATCH -t 0-6
#SBATCH -c 4
#SBATCH --partition=compute
#SBATCH --mem=8G
#SBATCH -J split-__VNAME__
#SBATCH -o __JOBS_DIR__/split-__VNAME___%j.out
#SBATCH -e __JOBS_DIR__/split-__VNAME___%j.err
set -euo pipefail
shopt -s nullglob

video="__VIDEO__"
flash="__FLASH__"
seg_sec=__SEG_SEC__
mkdir -p "$flash"
base="$(basename "${video%.avi}")"

# ─────────────── skip logic ────────────────────────────────
chunks=( "$flash/${base}_raw_"*.avi )

if (( ${#chunks[@]} > 0 )); then
  echo "[SKIP] found ${#chunks[@]} raw chunks for $video"
  exit 0
fi

# ─────────────── split ─────────────────────────────────────
ffmpeg -hide_banner -y -i "$video" \
  -c copy -map 0:v:0 -f segment -segment_time "$seg_sec" \
  -reset_timestamps 1 "$flash/${base}_raw_%03d.avi"

echo "[INFO] Split done for $video → $flash"
EOF1

# Replace placeholders
sed -i "s#__VNAME__#$vname#g"     "$job1"
sed -i "s#__VIDEO__#$video#g"     "$job1"
sed -i "s#__FLASH__#$flash_folder#g" "$job1"
sed -i "s#__SEG_SEC__#$SEG_SEC#g" "$job1"
sed -i "s#__JOBS_DIR__#$JOBS_DIR#g" "$job1"



# ---------- job2 : spawn encoders (job2a per raw chunk) ---
job2="$JOBS_DIR/job2-$vname.sh"
cat > "$job2" <<'EOF2'
#!/bin/bash -l
#SBATCH -t 0-24
#SBATCH -c 2
#SBATCH --partition=compute
#SBATCH --mem=8G
#SBATCH -J encstage-__VNAME__
#SBATCH -o __JDIR__/encstage-__VNAME___%j.out
#SBATCH -e __JDIR__/encstage-__VNAME___%j.err
set -euo pipefail
shopt -s nullglob

JDIR="__JDIR__"
EM="__EMAIL__"
flash="__FLASH__"
DATA="__DATA__"
base="__VNAME__"
frame_csv="$flash/${base}_frame_counts_tmp.csv"
: > "$frame_csv"

USER_NAME="$(id -un)"
saion_work="__SAION_WORK__"

for raw in "$flash/${base}_raw_"*.avi; do
  [[ -e "$raw" ]] || continue
  seg="${raw/_raw/}"
  segbase="$(basename "${seg%.avi}")"
  segfile="$(basename "$seg")"
  slp="$saion_work/${segbase}.slp"
  csv="$saion_work/${segbase}.csv"
  echo "[DEBUG] segbase=$segbase"
  echo "[DEBUG] saion_work=$saion_work"
  echo "[DEBUG] slp=$slp"
  echo "[DEBUG] csv=$csv"
  # ---------- encoder submission ----------
  if [[ ! -f "$seg" ]]; then
    job2a="$JDIR/job2a-$segbase.sh"
    cat > "$job2a" <<'EOJ'
#!/bin/bash -l
#SBATCH -t 0-12
#SBATCH -c 8
#SBATCH --partition=compute
#SBATCH --mem=16G
#SBATCH -J enc-SEGBASE
#SBATCH -o JOBS_DIR_P/enc-SEGBASE_%j.out
#SBATCH -e JOBS_DIR_P/enc-SEGBASE_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=EMAIL_P
set -euo pipefail

ffmpeg -hide_banner -y -i "RAWFILE" -c:v libx264 -pix_fmt yuv420p \
  -preset fast -crf 23 -threads 8 "OUTFILE"

nb=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames \
     -of default=nk=1:nw=1 "OUTFILE" || echo 0)
echo "SEGFILE,$nb" >> "FRAMECSV"

rsync -ah "OUTFILE" "__DATA__/"
rsync -ah "FRAMECSV" "__DATA__/"
EOJ

    sed -i \
      -e "s#RAWFILE#${raw}#g" \
      -e "s#OUTFILE#${seg}#g" \
      -e "s#FRAMECSV#${frame_csv}#g" \
      -e "s#JOBS_DIR_P#${JDIR}#g" \
      -e "s#SEGBASE#${segbase}#g" \
      -e "s#SEGFILE#${segfile}#g" \
      -e "s#__DATA__#${DATA}#g" \
      "$job2a"

    chmod +x "$job2a"
    sbatch "$job2a"
  else
    echo "[SKIP] already encoded: $seg"
  fi

  # ---------- sleap submission (on saion) ----------

  ready_flag="$DATA/.collect_done-$base"

  # Wait until monitor/collect has finished copying this video to bucket
  until [[ -f "$ready_flag" ]]; do
    echo "[WAIT] monitor not finished for $base. Sleeping 60s…"
    sleep 60
  done

  if ssh -x saion "test -f $(printf %q "$slp")"; then
  echo "[SKIP] sleap output already exists for $segbase in $saion_work"
  continue
  fi
    # ensure remote workdir exists
    ssh -x saion "mkdir -p '$saion_work'"

    # write remote job3a with placeholders
    ssh -x saion "cat > '$saion_work/job3a-$segbase.sh'" <<'EOJ2'
#!/bin/bash -l
#SBATCH -t 0-24
#SBATCH -c 8
#SBATCH --partition=largegpu
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH -J sleap-__BASE__
#SBATCH -o __WORK__/__BASE___%j.out
#SBATCH -e __WORK__/__BASE___%j.err
set -euo pipefail

source ~/.bashrc
conda activate sleap

sleap-track "__DATA__/__BASE__.avi" -m __MODEL1__ -m __MODEL2__ \
  --tracking.tracker none -o "__WORK__/__BASE__.slp" \
  --verbosity json --no-empty-frames --batch_size 2

python3 /home/sam-reiter/saionHome/AntsArray/sleap2h5.py "__WORK__/__BASE__.slp"
EOJ2

    # substitute placeholders on remote
    model1="/bucket/ReiterU/Ants/SLEAP_files/Simple_skeleton/20250408_models_LATESTWORKINGMODEL/250408_141245.centroid/training_config.json"
    model2="/bucket/ReiterU/Ants/SLEAP_files/Simple_skeleton/20250408_models_LATESTWORKINGMODEL/250408_141245.centered_instance/training_config.json"

    ssh -x saion "sed -i \
      -e s#__BASE__#$segbase#g \
      -e s#__WORK__#$saion_work#g \
      -e s#__MODEL1__#$model1#g \
      -e s#__MODEL2__#$model2#g \
      -e s#__DATA__#$DATA#g \
      '$saion_work/job3a-$segbase.sh'"

    ssh -x saion "bash -lc 'chmod +x $saion_work/job3a-$segbase.sh && sbatch $saion_work/job3a-$segbase.sh'"
    
  done
EOF2

sed -i \
  -e "s#__VNAME__#$vname#g" \
  -e "s#__JDIR__#$JOBS_DIR#g" \
  -e "s#__EMAIL__#$email#g" \
  -e "s#__FLASH__#$flash_folder#g" \
  -e "s#__ENC_CAP__#$ENC_CAP#g" \
  -e "s#__SAION_WORK__#$saion_work_remote#g" \
  -e "s#__DATA__#$data_folder#g" \
  "$job2"
chmod +x "$job2"




  # ---------- submit chain: job1 → job2  ------
  id1=$(sbatch "$job1" | awk '{print $4}')
  id2=$(sbatch --dependency=afterok:$id1 "$job2" | awk '{print $4}')
  echo "Submitted chain for $vname: job1=$id1, job2=$id2"
done   # closes per-video loop



# ─────────────── Monitor and collect ───────────────────────
echo "Monitoring and collecting…"
for video in "$DIR"/*.avi; do
  b="$(basename "$video")"
  [[ "$b" =~ ^\. ]] && continue
  [[ "$b" =~ _renc\.avi$ || "$b" =~ _nvenc\.avi$ ]] && continue
  vname="${b%.avi}"

  mkdir -p "$data_folder"

  echo "[DEBUG] flash_folder=$flash_folder"
  echo "[DEBUG] data_folder=$data_folder"
  echo "[DEBUG] vname=$vname"

  # Wait until all job2a-* encoders for this video are finished
  echo "Waiting for all 'enc*' jobs..."
while :; do
  running=$(
    # list only your jobs, only active states, print just the job name
    squeue -u "$(id -un)" -h -t R,PD -o "%j" \
      | awk '/^enc/ {c++} END {print c+0}'   # count names starting with "enc"
  )
  (( running == 0 )) && break
  echo "  $running enc* jobs still running..."
  sleep 60
done

  echo "All job2a encoders finished for $vname. Collecting outputs..."

  rsync -ah \
    --include="${vname}_[0-9]*.avi" \
    --include="${vname}_frame_counts.csv" \
    --exclude="*" \
    "$flash_folder/" "$data_folder/"

  touch "$data_folder/.collect_done-$vname"
  echo "✓ Collected outputs for $vname"

done

# ─────────────── ArUco detection stage ───────────────────────
echo "Submitting ArUco detection jobs..."

DATA_FOLDER="$data_folder"
OUTPUT_FOLDER="$flash_folder/aruco"
SCRIPT_PATH="$HOME/AntsArray/aruco_detection/opencv_aruco.py"


mkdir -p "$OUTPUT_FOLDER"
mkdir -p "$DATA_FOLDER"

shopt -s nullglob  # prevents literal patterns when no matches

for video_file in "$DATA_FOLDER"/*.{mp4,avi,mov}; do
  filename=$(basename -- "$video_file")
  base_name="${filename%.*}"

  marker_file="$OUTPUT_FOLDER/${base_name}_aruco_detections.csv"

  if [[ -f "$marker_file" ]]; then
    echo "[SKIP] $base_name already processed ($marker_file exists)"
    continue
  fi

  sbatch_script="$OUTPUT_FOLDER/run_${base_name}.sh"
  cat <<EOF > "$sbatch_script"
#!/bin/bash -l
#SBATCH -t 0-24
#SBATCH -c 4
#SBATCH --partition=compute
#SBATCH --mem=4G
#SBATCH -J aruco-$base_name
#SBATCH -o $OUTPUT_FOLDER/%x_%j.out
#SBATCH -e $OUTPUT_FOLDER/%x_%j.err

set -euo pipefail
source ~/.bashrc
conda activate aruco_env

python3 "$SCRIPT_PATH" \\
  --video-file "$video_file" \\
  --output-path "$OUTPUT_FOLDER" \\
  --dictionary-size 1000 \\
  --max-gap 100 \\
  --min-fraction 0.125
EOF

  sbatch "$sbatch_script"
  echo "[SUBMIT] ArUco detection for $base_name"
done

# ─────────────── Wait for ArUco jobs and rsync results ───────────────────────
echo "Waiting for all ArUco detection jobs to finish..."
sleep 60
while true; do
  running=$(squeue -u "$(id -un)" -h -n "aruco-" | wc -l)
  if (( running == 0 )); then
    break
  fi
  echo "  $running ArUco jobs still running..."
  sleep 60
done

echo "All ArUco jobs finished. Syncing results to bucket..."

rsync -ah \
  --include="*_aruco_detections.csv" \
  --exclude="*" \
  "$OUTPUT_FOLDER/" "$DATA_FOLDER/"

echo "✓ All ArUco outputs synced to bucket."


# ─────────────── Wait for Saion SLEAP outputs & rsync back ───────────────
for video in "$DIR"/*.avi; do
  b="$(basename "$video")"
  [[ "$b" =~ ^\. ]] && continue
  [[ "$b" =~ _renc\.avi$ || "$b" =~ _nvenc\.avi$ ]] && continue
  vname="${b%.avi}"

  echo "Waiting for SLEAP outputs on Saion for $vname…"

  # Build the list of expected segment bases from the encoded segments we just collected
  shopt -s nullglob
  seg_avs=( "$data_folder/${vname}_"*.avi )
  if (( ${#seg_avs[@]} == 0 )); then
    echo "[WARN] No encoded segments found for $vname in $data_folder; skipping SLEAP wait/sync."
  else
    # If we’ve already fetched all .slp files for these segments, skip the wait
    existing_slp=( "$data_folder/${vname}_"*.slp )
    if (( ${#existing_slp[@]} >= ${#seg_avs[@]} )); then
      echo "[SKIP] All SLEAP .slp files for $vname appear to be present locally."
    else
      # Wait for each expected .slp to materialize on Saion
      for f in "${seg_avs[@]}"; do
        segbase="$(basename "${f%.*}")"                # e.g., name_003
        remote_slp="$saion_work_remote/${segbase}.slp" # /work/.../name_003.slp

        # Poll until remote .slp exists and has nonzero size
        until ssh -x saion "[ -s $(printf %q "$remote_slp") ]"; do
          # Optional: show whether the SLURM job with this base is still queued/running
          # (best-effort; not required for correctness)
          ssh -x saion "squeue -u $(id -un) -h -n sleap-${segbase} -o '%T %j' || true" 2>/dev/null || true
          echo "[WAIT] $segbase.slp not ready on Saion. Sleeping 60s…"
          sleep 60
        done
        echo "[READY] ${segbase}.slp is present on Saion."
      done

      # Fetch .slp (and any .h5 created by sleap2h5.py) for this video
      echo "Syncing SLEAP outputs for $vname → $data_folder …"
      rsync -ah \
        --include="${vname}_[0-9]*.slp" \
        --include="${vname}_[0-9]*.h5" \
        --exclude="*" \
        "saion:$saion_work_remote/" "$data_folder/"

      touch "$data_folder/.sleap_collect_done-$vname"
      echo "✓ Retrieved SLEAP outputs for $vname"
    fi
  fi
done


