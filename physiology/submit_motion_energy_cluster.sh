#!/bin/bash -l
set -euo pipefail

if command -v conda >/dev/null 2>&1; then
	eval "$(conda shell.bash hook)"
elif [[ -f "/bucket/ReiterU/sam/miniforge3/etc/profile.d/conda.sh" ]]; then
	source "/bucket/ReiterU/sam/miniforge3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
	source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
	source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
	echo "[ERR] Could not find conda. Tried PATH, /bucket/ReiterU/sam/miniforge3, and \$HOME conda installs." >&2
	exit 2
fi

conda activate aruco_env

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_exe="/bucket/ReiterU/sam/miniforge3/envs/aruco_env/bin/python"
if [[ ! -x "$python_exe" ]]; then
	echo "[ERR] aruco_env Python not executable: $python_exe" >&2
	exit 2
fi
exec "$python_exe" "$script_dir/calc_motion_energy_cluster.py" submit "$@"
