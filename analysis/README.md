# Analysis workflows

## Supported interactive workflow

[`grid_occupancy.py`](grid_occupancy.py) is the single routine interactive
analysis. Its `# %%` cells form one end-to-end workflow:

1. load normalized spatial occupancy and speed-vector outputs;
2. filter poorly observed tracks;
3. cluster ants by spatial position, separately for each colony side;
4. save `track_cluster_ids.csv`;
5. inspect cluster maps and example ants;
6. plot speed through time for each spatial job cluster with light/dark context;
7. use `panorama_regions.csv` as a putative next step to compare
   colony-restricted versus in/out clusters; and
8. optionally summarize roaming ants from complete colony-outside-colony trips,
   including trip investment, trip and resource timing, inside/outside time,
   and correlations between colony use and trip investment.

Edit `DATASET_ROOT` and `GRID_OUTPUT_NAME` near the top, then run the cells in
order. The required per-track inputs are produced by
`compute_track_grid_occupancy.py` and `compute_track_speed_vector.py`.

The panorama-region section measures the fraction of detected time inside and
outside each colony annotation from the existing occupancy histograms. Exact
food/water detections are then extracted from raw tracks for the resource-time
heatmap.

The final trip-summary experiment is deliberately removable. Set
`RUN_OPTIONAL_TRIP_PHENOTYPING = False` to skip it, or delete the marked final
block and `trip_phenotyping_utils.py` to remove it completely. Its cached
tables and figures are kept under `optional_trip_phenotyping/`, separate from
the routine outputs. The summaries treat variation among roaming ants as
continuous; there is no second-stage trip or resource clustering.

## Exploratory work

[`exploratory/`](exploratory/) contains older, overlapping, or still-tuning
interactive analyses. In particular, sleep and interaction results there are
not treated as stable conclusions. These files remain available so ongoing
work and parameter choices are not lost, but they are not routine entry
points.

## Other top-level files

Top-level `compute_*`, `export_*`, classifier, GUI, and utility modules are
supporting preprocessing or tooling rather than additional interactive
analysis workflows. `commands.sh` records the per-track preprocessing fanout
commands.
