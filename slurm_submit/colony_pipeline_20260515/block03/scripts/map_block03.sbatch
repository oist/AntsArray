#!/usr/bin/env bash
#SBATCH -J map_block03
#SBATCH -p compute
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-12:00:00
#SBATCH -o /home/s/samuel-reiter/AntsArray/slurm_submit/colony_pipeline_20260515/block03/logs/map_block03_%j.out
#SBATCH -e /home/s/samuel-reiter/AntsArray/slurm_submit/colony_pipeline_20260515/block03/logs/map_block03_%j.err

set -euo pipefail
source ~/.bashrc
conda activate "ants"

cd "/home/s/samuel-reiter/AntsArray"
mkdir -p "/flash/ant_tmp/samuel-reiter/colony_pipeline/20260515/block03/colony_work/panorama_pkls" "/flash/ant_tmp/samuel-reiter/colony_pipeline/20260515/block03/colony_work/tracks" "/flash/ant_tmp/samuel-reiter/colony_pipeline/20260515/block03/colony_work/slurm_colony_pipeline"
python "/home/s/samuel-reiter/AntsArray/tracking/colony/pipeline.py" \
  --hmats "/bucket/ReiterU/Ants/basler/cameraArray_calib/20260414_calibration_dataset/set0_patterns_elevated_by_2mm/frame0/initial_H_mats.npz" \
  --data_dir "/bucket/ReiterU/Ants/basler/20260515/block03/data" \
  --work_dir "/flash/ant_tmp/samuel-reiter/colony_pipeline/20260515/block03/colony_work" \
  --panorama_dir "/flash/ant_tmp/samuel-reiter/colony_pipeline/20260515/block03/colony_work/panorama_pkls" \
  --tracks_dir "/flash/ant_tmp/samuel-reiter/colony_pipeline/20260515/block03/colony_work/tracks" \
  --map_mode "both" \
  --skip_combine \
  --skip_stitch \
  --skip_existing
