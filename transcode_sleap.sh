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
model1="/bucket/ReiterU/Ants/SLEAP_files/Simple_skeleton/20250408_models_LATESTWORKINGMODEL/250408_141245.centroid/training_config.json"
model2="/bucket/ReiterU/Ants/SLEAP_files/Simple_skeleton/20250408_models_LATESTWORKINGMODEL/250408_141245.centered_instance/training_config.json"

SEG_SEC="${SEG_SEC:-1200}"      # 20 min segments by default
ENC_CAP="${ENC_CAP:-1000}"       # max concurrent enc-* jobs per user

base_folder="$(basename "$DIR")"
data_folder="$DIR/data_test"
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
base="__VNAME__"
frame_csv="$flash/${base}_frame_counts_tmp.csv"
: > "$frame_csv"

USER_NAME="$(id -un)"

# throttle() {
#   # Limit concurrent enc-* jobs to __ENC_CAP__
#   while true; do
#     # squeue may not be in PATH on some nodes; fallback to no throttle
#     if ! command -v squeue >/dev/null 2>&1; then
#       break
#     fi
#     cnt="$(squeue -h -u "$USER_NAME" -o %j | grep -c '^enc-' || true)"
#     if [[ "$cnt" -lt "__ENC_CAP__" ]]; then
#       break
#     fi
#     sleep 10
#   done
# }

for raw in "$flash/${base}_raw_"*.avi; do
  [[ -e "$raw" ]] || continue
  seg="${raw/_raw/}"    
  segbase="$(basename "${seg%.avi}")"
  segfile="$(basename "$seg")"

  if [[ -f "$seg" ]]; then
    echo "[SKIP] already encoded: $seg"
    continue
  fi

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

# Re-encode
ffmpeg -hide_banner -y -i "RAWFILE" -c:v libx264 -pix_fmt yuv420p \
  -preset fast -crf 23 -threads 8 "OUTFILE"

# Frame count and append CSV
nb=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames \
     -of default=nk=1:nw=1 "OUTFILE" || echo 0)
echo "SEGFILE,$nb" >> "FRAMECSV"

rsync -ah "OUTFILE" "FRAMECSV" /bucket/ReiterU/ant_tmp/__VNAME__/

### CHANGED: Submit sleap job on Saion (points to bucket directly)
ssh saion bash -lc "
set -euo pipefail
SAION_NODE='__SAION_NODE__'
email='__EMAIL__'
model1='__MODEL1__'
model2='__MODEL2__'
workdir='/bucket/ReiterU/ant_tmp/__VNAME__'
base='SEGBASE'

case \"\$SAION_NODE\" in
  gpu) gputype='gpu:V100:1' ;;
  largegpu) gputype='gpu:1' ;;
  *) gputype='gpu:1' ;;
esac

j3a=\"\$workdir/job3a-\$base.sh\"
cat > \"\$j3a\" <<'EOJ2'
#!/bin/bash -l
#SBATCH -t 0-24
#SBATCH -c 32
#SBATCH --partition=__SAION_NODE__
#SBATCH --mem=128G
#SBATCH --gres=__GRES__
#SBATCH -J sleap-__BASE__
#SBATCH -o __WORK__/__BASE___%j.out
#SBATCH -e __WORK__/__BASE___%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=__EMAIL__
set -euo pipefail

source /bucket/ReiterU/miniforge3/etc/profile.d/conda.sh
conda activate sleap2

sleap-track \"__WORK__/__BASE__.avi\" -m __MODEL1__ -m __MODEL2__ \
  --tracking.tracker none -o \"__WORK__/__BASE__.slp\" \
  --verbosity json --no-empty-frames

python3 /home/sam-reiter/saionHome/AntsArray/sleap2csv.py \"__WORK__/__BASE__.slp\"
EOJ2

sed -i \
  -e \"s#__BASE__#\$base#g\" \
  -e \"s#__WORK__#\$workdir#g\" \
  -e \"s#__EMAIL__#$email#g\" \
  -e \"s#__MODEL1__#$model1#g\" \
  -e \"s#__MODEL2__#$model2#g\" \
  -e \"s#__SAION_NODE__#$SAION_NODE#g\" \
  -e \"s#__GRES__#\$gputype#g\" \
  \"\$j3a\"

chmod +x \"\$j3a\"
sbatch \"\$j3a\"
"
EOJ

  sed -i \
    -e "s#RAWFILE#${raw}#g" \
    -e "s#OUTFILE#${seg}#g" \
    -e "s#FRAMECSV#${frame_csv}#g" \
    -e "s#JOBS_DIR_P#${JDIR}#g" \
    -e "s#EMAIL_P#${EM}#g" \
    -e "s#SEGBASE#${segbase}#g" \
    -e "s#SEGFILE#${segfile}#g" \
    "$job2a"

  chmod +x "$job2a"
  throttle
  sbatch "$job2a"
done
EOF2


  # Bake values in job2
  sed -i \
    -e "s#__VNAME__#$vname#g" \
    -e "s#__JDIR__#$JOBS_DIR#g" \
    -e "s#__EMAIL__#$email#g" \
    -e "s#__FLASH__#$flash_folder#g" \
    -e "s#__ENC_CAP__#$ENC_CAP#g" \
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

  # Copy encoded files + CSVs from local flash to permanent data folder
  rsync -ah \
    --include="${vname}_[0-9]*.avi" \
    --include="${vname}[0-9]*.avi" \
    --include="${vname}_frame_counts.csv" \
    --exclude="*" \
    "$flash_folder/" "$data_folder/"

  # If desired, you can also rsync Saion-produced sleap outputs back here.
  # Example (uncomment to enable):
  # rsync -ah "saion:/bucket/ReiterU/ant_tmp/$vname/"*.{slp,csv,npy} "$data_folder/" || true

  echo "✓ Collected outputs for $vname"
done   # closes monitor loop







#   # ---------- job3 : wait enc, rsync, submit sleap on Saion --
#   job3="$JOBS_DIR/job3-$vname.sh"
#   cat > "$job3" <<'EOF3'
# #!/bin/bash -l
# #SBATCH -t 0-24
# #SBATCH -c 2
# #SBATCH --partition=compute
# #SBATCH --mem=8G
# #SBATCH -J stage-__VNAME__
# #SBATCH -o __JOBS_DIR__/stage-__VNAME___%j.out
# #SBATCH -e __JOBS_DIR__/stage-__VNAME___%j.err
# set -euo pipefail
# shopt -s nullglob

# # Wait until all enc-__VNAME__ jobs finish
# while true; do
#   if command -v squeue >/dev/null 2>&1; then
#     left=$(squeue -h -o "%j" -u "$(id -un)" | grep -c "^enc-__VNAME__" || true)
#     [[ $left -eq 0 ]] && break
#   else
#     break
#   fi
#   sleep 30
# done

# flash="__FLASH__"
# csv_tmp="$flash/__VNAME___frame_counts_tmp.csv"
# csv="$flash/__VNAME___frame_counts.csv"
# [[ -f "$csv" ]] || { [[ -f "$csv_tmp" ]] && mv "$csv_tmp" "$csv" || true; }

# # copy only reencoded segments + CSV to Saion
# #scp_target="saion:__SAION_WORK__"
# #rsync -ah --ignore-existing "$flash/__VNAME___"[0-9][0-9][0-9].avi "$csv" "$scp_target/"

# # Remote submit sleap jobs
# ssh saion bash -lc '
# set -euo pipefail
# SAION_NODE="__SAION_NODE__"
# email="__EMAIL__"
# model1="__MODEL1__"
# model2="__MODEL2__"
# saion_work="__SAION_WORK__"
# vname="__VNAME__"

# case "$SAION_NODE" in
#   gpu)      gputype="gpu:V100:1" ;;
#   largegpu) gputype="gpu:1" ;;
#   *)        gputype="gpu:1" ;;
# esac

# mkdir -p "$saion_work"
# source /bucket/ReiterU/miniforge3/etc/profile.d/conda.sh
# conda activate sleap2

# for seg in "$saion_work/${vname}_"[0-9][0-9][0-9].avi; do
#   [ -e "$seg" ] || continue
#   base=${seg##*/}; base=${base%.avi}

#   if [[ -f "$saion_work/$base.slp" || -f "$saion_work/$base.csv" ]]; then
#     echo "[SKIP] sleap outputs present for $base"
#     continue
#   fi

#   j3a="$saion_work/job3a-$base.sh"
#   cat > "$j3a" <<'EOJ'

# #!/bin/bash -l
# #SBATCH -t 0-24
# #SBATCH -c 32
# #SBATCH --partition=__SAION_NODE__
# #SBATCH --mem=128G
# #SBATCH --gres=__GRES__
# #SBATCH -J sleap-__BASE__
# #SBATCH -o __LOGDIR__/%x_%j.out
# #SBATCH -e __LOGDIR__/%x_%j.err
# #SBATCH --mail-type=FAIL
# #SBATCH --mail-user=__EMAIL__
# set -euo pipefail

# source /bucket/ReiterU/miniforge3/etc/profile.d/conda.sh
# conda activate sleap2

# sleap-track "__WORK__/__BASE__.avi" -m __MODEL1__ -m __MODEL2__ \
#   --tracking.tracker none -o "__WORK__/__BASE__.slp" \
#   --verbosity json --no-empty-frames

# python3 /home/sam-reiter/saionHome/AntsArray/sleap2csv.py "__WORK__/__BASE__.slp"
# EOJ

#   sed -i \
#     -e "s#__BASE__#$base#g" \
#     -e "s#__WORK__#$saion_work#g" \
#     -e "s#__LOGDIR__#$saion_work#g" \
#     -e "s#__EMAIL__#$email#g" \
#     -e "s#__MODEL1__#$model1#g" \
#     -e "s#__MODEL2__#$model2#g" \
#     -e "s#__SAION_NODE__#$SAION_NODE#g" \
#     -e "s#__GRES__#$gputype#g" \
#     "$j3a"

#   chmod +x "$j3a"
#   sbatch "$j3a"
# done
# '
# EOF3

#   # Replace placeholders in job3
#   sed -i \
#     -e "s#__VNAME__#$vname#g" \
#     -e "s#__JOBS_DIR__#$JOBS_DIR#g" \
#     -e "s#__FLASH__#$flash_folder#g" \
#     -e "s#__SAION_WORK__#$saion_work_remote#g" \
#     -e "s#__SAION_NODE__#$SAION_NODE#g" \
#     -e "s#__EMAIL__#$email#g" \
#     -e "s#__MODEL1__#$model1#g" \
#     -e "s#__MODEL2__#$model2#g" \
#     "$job3"
    
#   chmod +x "$job3"

#   # submit chain: job1 → job2 → job3
#   id1=$(sbatch "$job1" | awk '{print $4}')
#   id2=$(sbatch --dependency=afterok:$id1 "$job2" | awk '{print $4}')
#   id3=$(sbatch --dependency=afterok:$id2 "$job3" | awk '{print $4}')
#   echo "Submitted chain for $vname: job1=$id1, job2=$id2, job3=$id3"
# done

# # ─────────────── Monitor and collect ───────────────────────
# echo "Monitoring and collecting…"

# for video in "$DIR"/*.avi; do
#   b="$(basename "$video")"
#   [[ "$b" =~ ^\. ]] && continue
#   [[ "$b" =~ _renc\.avi$ || "$b" =~ _nvenc\.avi$ ]] && continue
#   vname="${b%.avi}"

#   # Wait for stage-$vname to finish
#   job3_out=$(ls -t "$JOBS_DIR/stage-${vname}_"*.out 2>/dev/null | head -1 || true)
#   if [[ -n "${job3_out:-}" ]]; then
#     job3_id=$(basename "$job3_out" .out | sed 's/.*_//')
#     while squeue -j "$job3_id" -h 1>/dev/null 2>&1; do
#       sleep 30
#     done
#   fi

#   # Move local reencoded segments + CSV to data_folder (safe)
#   # Move local reencoded segments + CSV to data_folder (safe)
#   mkdir -p "$data_folder"

#   echo "[DEBUG] flash_folder=$flash_folder"
#   echo "[DEBUG] data_folder=$data_folder"
#   echo "[DEBUG] vname=$vname"
#   rsync -ah  \
#     --include="${vname}_[0-9]*.avi" \
#     --include="${vname}[0-9]*.avi" \
#     --include="${vname}_frame_counts.csv" \
#     --exclude="*" \
#     "$flash_folder/" "$data_folder/"

#   # # Collect Saion outputs until sleap-* for this video are done
#   # while true; do
#   #   mapfile -t remote_files < <(
#   #     ssh saion bash -l -c "ls -1 ${saion_work}/${vname}_*.slp ${saion_work}/${vname}_*.npy ${saion_work}/${vname}_*.csv 2>/dev/null" || true
#   #   )

#   #   if [[ ${#remote_files[@]} -eq 0 ]]; then
#   #     jobs_left=$(ssh saion bash -l -c "squeue -h -o %j | grep -c '^sleap-${vname}_[0-9]\\{3\\}\$' || true")
#   #     if [[ $jobs_left -eq 0 ]]; then
#   #       break
#   #     fi
#   #     echo "  … No files yet, still $jobs_left jobs for $vname on Saion. Sleeping 60s."
#   #     sleep 60
#   #     continue
#   #   fi

#   #   for rf in "${remote_files[@]}"; do
#   #     echo "  ↪ Rsyncing $(basename "$rf") ..."
#   #     rsync -ah --remove-source-files "saion:$rf" "$data_folder/" || true
#   #   done

#   #   sleep 30
#   # done

#   echo "✓ Collected SLEAP outputs for $vname"
# done   # closes: for video in "$DIR"/*.avi

