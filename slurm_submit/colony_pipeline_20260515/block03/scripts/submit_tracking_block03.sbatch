#!/usr/bin/env bash
#SBATCH -J submit_track_block03
#SBATCH -p compute
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 0-01:00:00
#SBATCH -o /home/s/samuel-reiter/AntsArray/slurm_submit/colony_pipeline_20260515/block03/logs/submit_tracking_block03_%j.out
#SBATCH -e /home/s/samuel-reiter/AntsArray/slurm_submit/colony_pipeline_20260515/block03/logs/submit_tracking_block03_%j.err

set -euo pipefail
source ~/.bashrc
conda activate "ants"

cd "/home/s/samuel-reiter/AntsArray"
mkdir -p "/flash/ant_tmp/samuel-reiter/colony_pipeline/20260515/block03/colony_work/tracks" "/flash/ant_tmp/samuel-reiter/colony_pipeline/20260515/block03/colony_work/slurm_colony_pipeline/chunk_tracking"
python "/home/s/samuel-reiter/AntsArray/tracking/colony/combine_batch.py" \
  --input_folder "/flash/ant_tmp/samuel-reiter/colony_pipeline/20260515/block03/colony_work/panorama_pkls" \
  --output_path "/flash/ant_tmp/samuel-reiter/colony_pipeline/20260515/block03/colony_work/tracks" \
  --side "both" \
  --runner slurm \
  --logs_dir "/flash/ant_tmp/samuel-reiter/colony_pipeline/20260515/block03/colony_work/slurm_colony_pipeline/chunk_tracking" \
  --partition "compute" \
  --cpus "32" \
  --mem "32G" \
  --time "0-24:00:00" \
  --job_name "track_block03" \
  --conda_env "ants" \
  --max_distance "90.0" \
  --lost_track_max_frames "120" \
  --skip_existing
