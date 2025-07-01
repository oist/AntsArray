#!/bin/bash -l
# ------------------------------------------------------------
#  video_saion_pipeline.sh
# ------------------------------------------------------------
#  * job1-<video>   : split *.avi into 20-min chunks      (deigo/compute)
#  * job2-<video>   : copy chunks to saion & spawn jobs   (deigo/short)
#  * job3-<segment> : SLEAP-track + slp2csv              (saion/$SAION_NODE)
#  * monitor-<video>: move .avi/.csv/.npy/.slp back home  (deigo/compute)
# ------------------------------------------------------------
set -euo pipefail

# ──────────────── CLI parsing ───────────────────────────────
DIR=""
SAION_NODE="largegpu"          # saion partition
usage() { echo "Usage: $0 --dir <folder> [--node <saion-partition>]"; exit 1; }

while [[ $# -gt 0 ]]; do
  case $1 in
    --dir)  DIR="$2"; shift 2 ;;
    --node) SAION_NODE="$2"; shift 2 ;;
    *) usage ;;
  esac
done
DIR=${DIR%/}; [[ -z $DIR ]] && usage

echo "Input dir : $DIR"
echo "Saion node : $SAION_NODE"

# ─────────────── constants & folders ────────────────────────
email="samuel.reiter@oist.jp"
[[ $SAION_NODE == gpu ]] && gputype="gpu:V100:1" || gputype="gpu:1"

base_folder=$(basename "$DIR")
data_folder="$DIR/data"
flash_folder="/flash/ReiterU/ant_tmp/$base_folder"
deigo_folder="/deigo_flash/ReiterU/ant_tmp/$base_folder"
saion_work="/work/ReiterU/ant_tmp/$base_folder"


model1="/bucket/ReiterU/Ants/SLEAP_files/Simple_skeleton/20250408_models_LATESTWORKINGMODEL/250408_141245.centroid/training_config.json"
model2="/bucket/ReiterU/Ants/SLEAP_files/Simple_skeleton/20250408_models_LATESTWORKINGMODEL/250408_141245.centered_instance/training_config.json"

mkdir -p "$HOME/output/jobs" "$data_folder" "$flash_folder" 
chmod 2775 "$data_folder" "$flash_folder" 

# ─────────────── iterate over videos ────────────────────────
for video in "$DIR"/*.avi; do
  [[ $video =~ (^\.|_renc\.avi$|_nvenc\.avi$) ]] && continue
  vname=$(basename "$video" .avi)

  # ---------- job1 : segmentation on deigo ------------------
  job1="$flash_folder/job1-$vname.sh"
  cat > "$job1" <<EOF1
#!/bin/bash -l
#SBATCH -t 0-72
#SBATCH -c 32
#SBATCH --partition=compute
#SBATCH --mem=32G
#SBATCH -J seg-$vname
#SBATCH -o /flash/ReiterU/ant_tmp/$base_folder/%x_%j.out
#SBATCH -e /flash/ReiterU/ant_tmp/$base_folder/%x_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=$email
set -euo pipefail

video="$video"
flash="$flash_folder"

# Skip everything if at least the first segment already exists
if [[ -f "$flash_folder/${vname}_000.avi" ]]; then
    echo "[INFO] $flash_folder/${vname}_000.avi already present – skipping transcoding."
    exit 0
fi

fps=\$(ffprobe -v 0 -select_streams v:0 -show_entries stream=r_frame_rate \
       -of csv=p=0 "$video" | awk -F/ '{print \$1/\$2}')
frames=\$(ffprobe -v 0 -count_frames -select_streams v:0 \
          -show_entries stream=nb_read_frames -of csv=p=0 "$video")

seg_sec=1200
seg_frames=\$(printf "%.0f" "\$(echo \$fps*\$seg_sec | bc -l)")

points=()
for ((p=seg_frames; p<frames; p+=seg_frames)); do points+=(\$p); done
IFS=,; SPLIT="\${points[*]}"; unset IFS
[[ -z \$SPLIT ]] && SPLIT=\$seg_frames

csv="\$flash/${vname}_frame_counts.csv"
tmp="\${csv%.csv}_tmp.csv"
rm -f "\$csv" "\$tmp"

ffmpeg -fflags +genpts -y -i "\$video" -threads 32 -c:v libx264 \
  -pix_fmt yuv420p -preset slow -crf 23 -f segment -reset_timestamps 1 \
  -segment_list "\$csv" -segment_frames "\$SPLIT" \
  -break_non_keyframes 1 -force_key_frames "expr:gte(t,n_forced*\$seg_sec)" \
  "\$flash/${vname}_%03d.avi"

while IFS=, read -r seg s e; do
  nb=\$(ffprobe -v 0 -select_streams v:0 -show_entries stream=nb_frames \
       -of csv=p=0 "\$flash/\$seg")
  echo "\$seg,\$s,\$e,\$nb" >> "\$tmp"
done < "\$csv"
mv "\$tmp" "\$csv"
EOF1
  chmod +x "$job1"

  # ---------- job2 : create & submit saion jobs -------------
  job2="$flash_folder/job2-$vname.sh"
  cat > "$job2" <<EOF2
#!/bin/bash -l
#SBATCH -t 0-2
#SBATCH -c 2
#SBATCH --partition=short
#SBATCH --mem=8G
#SBATCH -J stage-$vname
#SBATCH -o /flash/ReiterU/ant_tmp/$base_folder/%x_%j.out
#SBATCH -e /flash/ReiterU/ant_tmp/$base_folder/%x_%j.err
set -euo pipefail

flash="$flash_folder"; deigo="$deigo_folder"
scp_target="saion:$saion_work"
csv="\$flash/${vname}_frame_counts.csv"

# copy segments & csv to saion
rsync -ah "\$flash/${vname}"_*.avi "\$scp_target/"
#rsync -ah "\$flash/${vname}"_*.avi "\$csv" "\$scp_target/"

# generate & submit job3 for each segment (on saion)
ssh saion bash -ls <<'EOSA'
set -euo pipefail
SAION_NODE="__SAION_NODE__"
gputype="__GPUTYPE__"
email="__EMAIL__"
model1="__MODEL1__"; model2="__MODEL2__"
work="__WORK__"; vname="__VNAME__"
#csv="\$work/\${vname}_frame_counts.csv"

mkdir -p "\$work"
source ~/mambaforge/etc/profile.d/conda.sh
conda activate sleap2

# Find all segments for this video and submit jobs
for seg in "\$work/\${vname}"_*.avi; do
  [[ -e "\$seg" ]] || continue
  base=\${seg##*/}; base=\${base%.avi}
  [[ \$base =~ ^\${vname}_[0-9]{3}$ ]] || continue

  # Get frame count for this segment
  frames=\$(ffprobe -v 0 -select_streams v:0 -show_entries stream=nb_frames \
       -of csv=p=0 "\$seg")

  j3="\$work/job3-\${base}.sh"
  cat > "\$j3" <<EOJ
#!/bin/bash -l
#SBATCH -t 0-24
#SBATCH -c 32
#SBATCH --partition=\$SAION_NODE
#SBATCH --mem=128G
#SBATCH --gres=\$gputype
#SBATCH -J sleap-\${base}
#SBATCH -o /work/ReiterU/ant_tmp/$base_folder/%x_%j.out
#SBATCH -e /work/ReiterU/ant_tmp/$base_folder/%x_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=\$email
set -euo pipefail

source ~/mambaforge/etc/profile.d/conda.sh
conda activate sleap2

frames=\$frames
#sleap-track "\$work/\${base}.avi" --batch_size 1 --frames 0-\$((frames-1)) \
  -m \$model1 -m \$model2 --tracking.tracker none \
  -o "\$work/\${base}.slp" --verbosity json --no-empty-frames

# Convert SLEAP file to CSV
python3 /home/sam-reiter/saionHome/AntsArray/sleap2csv.py "\$work/\${base}.slp"
EOJ
  chmod +x "\$j3"; sbatch "\$j3"
done
EOSA
EOF2
  sed -i -e "s#__SAION_NODE__#$SAION_NODE#g" \
         -e "s#__GPUTYPE__#$gputype#g" \
         -e "s#__EMAIL__#$email#g" \
         -e "s#__MODEL1__#$model1#g" \
         -e "s#__MODEL2__#$model2#g" \
         -e "s#__WORK__#$saion_work#g" \
         -e "s#__VNAME__#$vname#g" \
         "$job2"
  chmod +x "$job2"

# ---------- submit chain ----------------------------------
id1=$(sbatch "$job1" | awk '{print $4}')
id2=$(sbatch --dependency=afterok:$id1 "$job2" | awk '{print $4}')
echo "Submitted job chain for $vname: job1=$id1, job2=$id2"

done

##############################################################################
#  Now wait for all jobs to complete and move files to bucket
##############################################################################
echo "All job chains submitted. Now monitoring for completion..."

for video in "$DIR"/*.avi; do
  [[ $video =~ (^\.|_renc\.avi$|_nvenc\.avi$) ]] && continue
  vname=$(basename "$video" .avi)

  # ───── wait for job1 (segmentation) ──────────────────────────────────────
  echo "Waiting for job1-$vname to complete..."
  job1_out=$(ls -t "/flash/ReiterU/ant_tmp/$base_folder/seg-${vname}_"*.out 2>/dev/null | head -1)
  if [[ -z "$job1_out" ]]; then
      echo "Warning: could not find job1 output file for $vname"
      continue
  fi
  job1_id=$(basename "$job1_out" .out | sed 's/.*_//')
  while squeue -j "$job1_id" -h 1>/dev/null 2>&1; do sleep 30; done

  state=$(sacct -j "$job1_id" -X -n -o State 2>/dev/null | head -1 | tr -d '[:space:]')
  if [[ "$state" != "COMPLETED" ]]; then
      echo "Warning: job1-$vname ended with state $state"
      continue
  fi
  echo "job1-$vname completed."

  # ───── move AVIs from flash → bucket ────────────────────────────────────
  echo "Moving AVIs for $vname to $data_folder ..."
  mkdir -p "$data_folder"
  mv "$flash_folder/${vname}"_*.avi "$data_folder/" 2>/dev/null \
      && echo "  ✓ AVIs moved." \
      || echo "  ⚠  No AVIs found (maybe already moved)."

done

for video in "$DIR"/*.avi; do
    [[ $video =~ (^\.|_renc\.avi$|_nvenc\.avi$) ]] && continue
    vname=$(basename "$video" .avi)

    # ───── wait for job2 (rsync to Saion & remote submits) ───────────────────
    echo "Waiting for job2-$vname to complete..."
    job2_out=$(ls -t "/flash/ReiterU/ant_tmp/$base_folder/stage-${vname}_"*.out 2>/dev/null | head -1)
    if [[ -n "$job2_out" ]]; then
        job2_id=$(basename "$job2_out" .out | sed 's/.*_//')
        while squeue -j "$job2_id" -h 1>/dev/null 2>&1; do sleep 30; done
        echo "job2-$vname completed."
    else
        echo "Warning: could not find job2 output file for $vname"
    fi

    # ───── collect processed files from Saion → bucket ──────────────────────
    echo "Collecting processed files for $vname from Saion ..."
    while true; do
        mapfile -t remote_files < <(
            ssh saion "ls -1 ${saion_work}/${vname}_*.{slp,npy} 2>/dev/null"
        )

        if [[ ${#remote_files[@]} -eq 0 ]]; then
            # Check if any Saion SLURM jobs with this name are still running
            jobs_left=$(ssh saion "squeue -h -o '%j' | grep -c '^${vname}' || true")
            if [[ $jobs_left -eq 0 ]]; then
                echo "  ✓ All processed files collected for $vname."
                break
            fi
            echo "  … No files yet, Saion jobs still running ($jobs_left). Sleeping 60 s."
            sleep 60
            continue
        fi

        for rf in "${remote_files[@]}"; do
            base=$(basename "$rf")
            echo "  ↪  Rsyncing $base ..."
            rsync -ah --remove-source-files saion:"$rf" "$data_folder/" \
                && echo "     ✓ transferred."
        done
        echo "  Waiting 60 s for more files ..."
        sleep 60
    done
done

echo "✓ All AVIs and processed files are now in $data_folder."