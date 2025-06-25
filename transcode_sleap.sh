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
SAION_NODE="gpu"          # saion partition
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
saion_prefix="/saion_work/ReiterU/ant_tmp/$base_folder"

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
rsync -ah "\$flash/${vname}"_*.avi "\$csv" "\$scp_target/"

# generate & submit job3 for each segment (on saion)
ssh saion bash -s <<'EOSA'
set -euo pipefail
SAION_NODE="__SAION_NODE__"
gputype="__GPUTYPE__"
email="__EMAIL__"
model1="__MODEL1__"; model2="__MODEL2__"
work="__WORK__"; vname="__VNAME__"
csv="\$work/\${vname}_frame_counts.csv"

mkdir -p "\$work"
while IFS=, read -r seg _ _ frames; do
  [[ -z \$seg ]] && continue
  base=\${seg%.avi}
  [[ \$base =~ ^\${vname}_[0-9]{3}$ ]] || continue

  j3="\$work/job3-\${base}.sh"
  cat > "\$j3" <<EOJ
#!/bin/bash -l
#SBATCH -t 0-48
#SBATCH -c 8
#SBATCH --partition=\$SAION_NODE
#SBATCH --mem=128G
#SBATCH --gres=\$gputype
#SBATCH -J sleap-\${base}
#SBATCH -o /saion_work/ReiterU/ant_tmp/$base_folder/%x_%j.out
#SBATCH -e /saion_work/ReiterU/ant_tmp/$base_folder/%x_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=\$email
set -euo pipefail

ml use /apps/unit/ReiterU/.modulefiles
ml load sleap
frames=\$frames
sleap-track "\$work/\${base}.avi" --batch_size 1 --frames 0-\$((frames-1)) \
  -m \$model1 -m \$model2 --tracking.tracker none \
  -o "\$work/\${base}.slp" --verbosity json --no-empty-frames

module --ignore_cache load matlab
matlab -nosplash -nodisplay -nojvm -nodesktop -r \
  "addpath('/apps/unit/ReiterU/makoto/mfiles'); slp2csv('\$work/\${base}.slp'); exit;"
EOJ
  chmod +x "\$j3"; sbatch "\$j3"
done < "\$csv"
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

  # ---------- monitor job -----------------------------------
  monitor="$data_folder/monitor-$vname.sh"
  cat > "$monitor" <<EOF3
#!/bin/bash -l
#SBATCH -t 0-48
#SBATCH -c 1
#SBATCH --partition=compute
#SBATCH --mem=8G
#SBATCH -J mon-$vname
#SBATCH -o /flash/ReiterU/ant_tmp/$base_folder/%x_%j.out
#SBATCH -e /flash/ReiterU/ant_tmp/$base_folder/%x_%j.err
set -euo pipefail

flash="$flash_folder"; saion="$saion_prefix"; target="$data_folder"
while true; do
  moved=0
  for avi in \$flash/${vname}_*.avi; do
    [[ -e \$avi ]] || continue
    base=\${avi##*/}; base=\${base%.avi}

    for ext in csv npy slp; do
      remote="\$saion/\$base.\$ext"
      if ssh saion test -e "\$remote"; then
        ssh saion mv "\$remote" "\$target/"
        moved=1
      fi
    done
    if [[ \$moved -eq 1 ]]; then mv "\$avi" "\$target/"; fi
  done
  [[ \$moved -eq 0 ]] && { echo "All files collected."; break; }
  sleep 120
done
EOF3
  chmod +x "$monitor"

  # ---------- submit chain ----------------------------------
  id1=$(sbatch "$job1" | awk '{print $4}')
  id2=$(sbatch --dependency=afterok:$id1 "$job2" | awk '{print $4}')
  sbatch --dependency=afterok:$id2 "$monitor"
done
