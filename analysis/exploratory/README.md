# Exploratory analyses

These scripts are retained for development and comparison, not as supported
analysis entry points. They may overlap, use dataset-specific defaults, or
need more validation.

- Sleep and interaction scripts are active tuning work.
- `early_late_activity.py` compares the 20260810 early and late blocks using
  existing annotated-arena grids, motion contexts, and sleep/activity clock
  tables. It matches workers by colony side and tag ID, then matches their
  clock bins before comparing activity. See the comparison section in
  `../README.md` for inputs, coverage controls, and outputs.
- `colony_speed.py`, `cluster_time_of_day_occupancy.py`, and
  `grid_occupancy_subclusters.py` preserve analyses that overlap the focused
  spatial-job workflow in `../grid_occupancy.py`.
- The remaining scripts are narrow debugging or legacy experiments.

Promote a result back to the supported workflow only after its definitions and
parameters are settled.
