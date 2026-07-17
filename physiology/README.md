# Ant Physiology Motion Energy

Chunked Deigo pipeline for long ant-on-ball physiology videos.

Example from a Deigo login node:

```bash
cd /home/s/samuel-reiter/AntsArray
physiology/submit_motion_energy_cluster.sh \
  --experiment-path /bucket/ReiterU/Ants/physiology/20260707_ant_on_ball \
  --output-path /bucket/ReiterU/Ants/physiology/20260707_ant_on_ball \
  --mask-path /bucket/ReiterU/Ants/physiology/20260707_ant_on_ball/cam1_mask.png \
  --work-root /flash/ReiterU/ant_tmp/samuel-reiter/ant_physiology_motion_energy \
  --overwrite
```

The wrapper activates conda `aruco_env` before calling Python. The generated Slurm jobs also activate `aruco_env` by default.
Both use `/bucket/ReiterU/sam/miniforge3/envs/aruco_env/bin/python`.

If you want to run the Python driver directly, activate the environment first:

```bash
source /bucket/ReiterU/sam/miniforge3/etc/profile.d/conda.sh
conda activate aruco_env
/bucket/ReiterU/sam/miniforge3/envs/aruco_env/bin/python physiology/calc_motion_energy_cluster.py submit \
  --experiment-path /path/to/experiment \
  --output-path /path/to/experiment \
  --mask-path /path/to/mask.png \
  --overwrite
```

To use a different batch environment, pass an activation snippet:

```bash
physiology/submit_motion_energy_cluster.sh \
  --experiment-path /path/to/experiment \
  --output-path /path/to/experiment \
  --mask-path /path/to/mask.png \
  --env-activate 'source ~/.bashrc; conda activate YOUR_ENV' \
  --overwrite
```

Pipeline stages:

1. A short Slurm job cuts each AVI into 30-minute chunks under the selected work root.
2. A compute-node array processes each chunk independently.
3. A short Slurm job assembles chunk `.me` files into one flash-local final `.me`.
4. A `datacp` job publishes the final `.me` to `--output-path`.

By default the work root is chosen from an existing writable scratch path, preferring `/flash/ReiterU/ant_tmp/$USER`. You can also set `ANT_PHYSIOLOGY_WORK_ROOT` or pass `--work-root` explicitly.

Each chunk output stores its first and last grayscale frame. During assembly, the first sample of each chunk after the first is replaced with the motion-energy difference between the previous chunk's last frame and the current chunk's first frame, so chunk boundaries do not get artificial zeros.

Check active processing progress:

```bash
tail -f /flash/ReiterU/ant_tmp/samuel-reiter/ant_physiology_motion_energy/*/logs/process_*.out
```

Chunk workers print `[PROGRESS]` lines every 5,000 frames or 30 seconds. Existing workers that were launched before this logging change may only show the initial `[INFO]` line; for those, check growing `*.me.tmp` files under `chunk_motion_energy/`.
