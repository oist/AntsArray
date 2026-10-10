"""Run numbered cells to inspect the saved Leiden analysis; no refitting or HTML."""
# %% 1. Paths and measurements
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analysis.eigenposture_leiden_features import balanced_matrix

RESULTS = Path('/home/sam-reiter/bucket/ReiterU/Ants/basler/20260724/block01/analysis_outputs/leiden_states_20261010')
summary = json.loads((RESULTS / 'summary.json').read_text())
with np.load(RESULTS / 'states.npz') as saved:
    values = saved['values']               # ant x time bin x 49 measurements
    tasks = saved['tasks']                 # -1 means missing
    names = saved['names']
    columns = saved['selected_columns']
    proportions = saved['proportions']
valid = tasks >= 0
x, center, scale = balanced_matrix(values[valid], columns)
print(summary)

# %% 2. How many states, and which features matter?
resolution = pd.read_csv(RESULTS / 'state_resolution_sweep.csv')
importance = pd.read_csv(RESULTS / 'feature_importance.csv')
reduction = pd.read_csv(RESULTS / 'feature_reduction.csv')
print(resolution.to_string(index=False))
print(importance.sort_values('permutation_importance', ascending=False).to_string(index=False))
print(reduction[['kind', 'name', 'n_features', 'k', 'ari_to_broad', 'preserves']].to_string(index=False))

# %% 3. Inspect retained features and the fixed-label UMAP
centroids = np.stack([x[tasks[valid] == state].mean(axis=0) for state in range(summary['n_states'])])
fig, ax = plt.subplots(figsize=(12, 5), layout='constrained')
im = ax.imshow(centroids, aspect='auto', cmap='RdBu_r')
ax.set_xticks(range(len(columns)), names[columns], rotation=75, ha='right', fontsize=8)
ax.set(ylabel='State'); fig.colorbar(im, ax=ax)
with np.load(RESULTS / 'umap.npz') as saved:
    coordinates = saved['coordinates'][int(saved['chosen'])]
    labels = saved['labels']
fig, ax = plt.subplots()
ax.scatter(*coordinates.T, c=labels, s=3, cmap='tab20')
ax.set(xlabel='UMAP 1', ylabel='UMAP 2')

# %% 4. Every ant through time, then proportions and roles
ants = pd.read_csv(RESULTS / 'ant_roles.csv')
fig, ax = plt.subplots(figsize=(14, 9))
ax.imshow(np.ma.masked_less(tasks, 0), aspect='auto', interpolation='nearest',
          extent=[0, 48, len(ants)-.5, -.5], cmap='tab20')
ax.set(xlabel='Hours from July 24 10:00 JST', ylabel='Ant index')
for side in ('left', 'right'):
    print(side, pd.read_csv(RESULTS / f'{side}_role_k.csv').to_string(index=False))
print(ants[['ant', 'observed_bin_hours', 'role', 'spatial_cluster']].to_string(index=False))

# %% 5. Frozen low-motion candidate compared with the existing sleep rule
sleep = pd.read_csv(RESULTS / 'sleep_by_state.csv')
print(sleep.to_string(index=False))
print('Candidate fixed before sleep labels:', summary['low_motion_candidate'])
fig, ax = plt.subplots()
ax.bar(sleep.state, sleep.sleep_fraction)
ax.set(xlabel='State', ylabel='Sleep-rule fraction (equal ant weight)', ylim=(0, 1))
plt.show()
