"""Static UMAP sensitivity, feature/state summaries, timelines, roles and sleep."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, PowerNorm
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits
from umap import UMAP

from analysis.eigenposture_leiden import SEED
from analysis.eigenposture_leiden_features import balanced_matrix, FAMILIES


def layouts(output, raw, tasks, columns):
    x, _, _ = balanced_matrix(raw, columns)
    rng = np.random.default_rng(SEED)
    sample = rng.choice(len(x), min(10000, len(x)), replace=False)
    audit = rng.choice(len(sample), min(1800, len(sample)), replace=False)
    labels = tasks[sample]
    rows, coordinates = [], []
    for neighbors in (10, 30, 60):
        for distance in (0., .2):
            embedding = UMAP(n_neighbors=neighbors, min_dist=distance, repulsion_strength=2.,
                n_epochs=300, random_state=SEED, n_jobs=1).fit_transform(x[sample])
            score = silhouette_score(embedding[audit], labels[audit])
            trust = trustworthiness(x[sample[audit]], embedding[audit], n_neighbors=15)
            rows.append(dict(n_neighbors=neighbors, min_dist=distance, repulsion_strength=2.,
                             silhouette_2d=score, trustworthiness=trust))
            coordinates.append(embedding)
            print('UMAP', rows[-1], flush=True)
    table = pd.DataFrame(rows)
    candidates = table[table.trustworthiness >= .95]
    chosen = int((table.trustworthiness if candidates.empty else candidates.silhouette_2d).idxmax())
    table['selected_view'] = table.index == chosen
    table.to_csv(output / 'umap_parameters.csv', index=False)
    np.savez_compressed(output / 'umap.npz', sample=sample, coordinates=np.stack(coordinates),
                        labels=labels, chosen=chosen)
    return sample, coordinates, table, chosen


def plot(source, inputs, output, reuse_umap=False, clips=None):
    threadpool_limits(1)
    plt.close('all')
    summary = json.loads((output / 'summary.json').read_text())
    ants = pd.read_csv(output / 'ant_roles.csv')
    with np.load(output / 'states.npz') as z:
        tasks, values, proportions, counts = z['tasks'], z['values'], z['proportions'], z['counts']
        columns, names, minutes = z['selected_columns'], z['names'], int(z['minutes'])
    valid = tasks >= 0
    raw = values[valid]
    k = summary['n_states']
    palette = plt.get_cmap('tab20' if k <= 20 else 'turbo')(np.linspace(0, 1, k))
    cmap = ListedColormap(palette);cmap.set_bad('#ddd')
    norm = BoundaryNorm(np.arange(k + 1) - .5, k)
    profiles = pd.read_csv(output / 'state_profiles.csv').set_index('state')
    sweep = pd.read_csv(output / 'state_resolution_sweep.csv')
    reduction = pd.read_csv(output / 'feature_reduction.csv')
    importance = pd.read_csv(output / 'feature_importance.csv')
    if reuse_umap:
        with np.load(output / 'umap.npz') as z:
            sample, coordinates, chosen = z['sample'], z['coordinates'], int(z['chosen'])
        layout_table = pd.read_csv(output / 'umap_parameters.csv')
    else:
        sample, coordinates, layout_table, chosen = layouts(output, raw, tasks[valid], columns)
    labels = tasks[valid][sample]

    # 1: All bin widths and resolutions, including rejected partitions.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    for width, group in sweep.groupby('minutes'):
        for ax, y in zip(axes, ['k', 'seed_ari', 'ant_subsample_ari']):
            ax.plot(group.resolution, group[y], 'o-', label=f'{width} min')
            rejected = group[~group.qualifies]
            ax.scatter(rejected.resolution, rejected[y], marker='x', color='red', s=55)
            ax.set(xlabel='Leiden resolution', ylabel=y.replace('_', ' '))
    axes[1].axhline(.9, color='.5', ls='--');axes[2].axhline(.8, color='.5', ls='--')
    axes[0].legend();fig.suptitle('49 measured features: state resolution and stability; red × = fails any support rule')

    # 2: What affects the dictionary, and what survives feature reduction?
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout='constrained')
    family = reduction[reduction.kind == 'family_ablation']
    axes[0].barh(family.name.str.replace('without_', ''), family.ari_to_broad, color='#4c78a8')
    axes[0].axvline(.9, color='black', ls='--')
    axes[0].set(xlabel='ARI after removing family', xlim=(0, 1.02), title='Leave-one-family-out Leiden refits')
    top = importance.sort_values('permutation_importance').tail(15)
    axes[1].barh(top.feature, top.permutation_importance, color='#59a14f')
    axes[1].tick_params(axis='y', labelsize=7)
    axes[1].set(xlabel='Drop in surrogate balanced accuracy', title='Predictive importance of measured features')
    prefix = reduction[reduction.kind == 'ranked_prefix']
    axes[2].plot(prefix.n_features, prefix.ari_to_broad, 'o-')
    rejected = prefix[~prefix.preserves]
    axes[2].scatter(rejected.n_features, rejected.ari_to_broad, marker='x', color='red',
                    label='Fails preservation/stability')
    axes[2].legend(fontsize=7)
    axes[2].axhline(.9, color='black', ls='--')
    axes[2].axvline(len(columns), color='#e15759', ls='--')
    axes[2].set(xlabel='Number of retained features', ylabel='ARI to broad states',
                title=f'Final: {len(columns)} features, {k} states', ylim=(-.05, 1.05))

    # 3: Minimal feature profiles and physically interpretable motion.
    x, _, _ = balanced_matrix(raw, columns)
    centroids = np.stack([x[tasks[valid] == state].mean(axis=0) for state in range(k)])
    fig, axes = plt.subplots(1, 2, figsize=(13, max(5, len(columns)*.22)), layout='constrained', gridspec_kw={'width_ratios':[1.2,1]})
    image = axes[0].imshow(centroids.T, aspect='auto', cmap='RdBu_r', vmin=-1.5, vmax=1.5)
    axes[0].set_yticks(range(len(columns)), names[columns], fontsize=7)
    axes[0].set_xticks(range(k), [f'T{i}' for i in range(k)])
    axes[0].set_title('State means in retained balanced features')
    fig.colorbar(image, ax=axes[0], label='Balanced feature mean')
    for state in range(k):
        selected = tasks[valid] == state
        axes[1].scatter(np.median(raw[selected, 0]), np.median(raw[selected, 43]),
                        color=palette[state], s=100)
        axes[1].annotate(f'T{state}', (np.median(raw[selected, 0]), np.median(raw[selected, 43])), xytext=(5,5), textcoords='offset points')
    axes[1].set(xlabel='Median forward peak (mm/s)', ylabel='Median unsigned antennal speed (mm/s)',
                xscale='log', yscale='log', title='Body and antenna motion by state')
    axes[1].margins(.2)

    # 4: Layout parameters vary; state labels are fixed throughout.
    fig, axes = plt.subplots(3, 2, figsize=(10, 13), layout='constrained')
    for i, ax in enumerate(axes.flat):
        row = layout_table.iloc[i]
        ax.scatter(*coordinates[i].T, c=labels, cmap=cmap, norm=norm, s=2, alpha=.65, rasterized=True)
        ax.set(title=f"neighbors={int(row.n_neighbors)}, min_dist={row.min_dist:g}\n2D silhouette={row.silhouette_2d:.2f}, trust={row.trustworthiness:.2f}", xticks=[], yticks=[])
    fig.suptitle('UMAP layout comparison: identical Leiden state labels; visual gaps do not establish new states')

    # 5: Selected UMAP view and physical velocity gradients.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout='constrained')
    for ax, color, label, discrete in [(axes[0], labels, 'Leiden states', True),
            (axes[1], np.log1p(raw[sample, 0]), 'log(1 + forward peak mm/s)', False),
            (axes[2], np.log1p(raw[sample, 43]), 'log(1 + unsigned antennal mm/s)', False)]:
        im = ax.scatter(*coordinates[chosen].T, c=color, cmap=cmap if discrete else 'viridis',
                        norm=norm if discrete else None, s=3, alpha=.7, rasterized=True)
        fig.colorbar(im, ax=ax, label=label, ticks=range(k) if discrete else None)
        ax.set(xlabel='UMAP 1', ylabel='UMAP 2', xticks=[], yticks=[])
    fig.suptitle(f'{minutes}-minute bins · {len(columns)} retained features · {k} Leiden states')

    # 6: Wavelet power and rates remain available for interpretation even if removed.
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), layout='constrained')
    for ax, chosen_columns, label in zip(axes, [np.arange(24,36), np.arange(20,24), np.arange(43,49)],
                                         ['Morlet power (2/3/4 Hz)', 'Posture rates', 'Unsigned antennal motion']):
        block = profiles.iloc[:, chosen_columns].to_numpy()
        image = ax.imshow(np.log1p(block), aspect='auto', cmap='magma')
        ax.set_xticks(range(len(chosen_columns)), names[chosen_columns], rotation=75, ha='right', fontsize=7)
        ax.set_yticks(range(k), [f'T{i}' for i in range(k)])
        ax.set_title(label)
        fig.colorbar(image, ax=ax, label='log(1 + mean physical value)')

    # 7: Existing role criterion on proportions, with K=1 allowed.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for ax, side in zip(axes, ('left','right')):
        table = pd.read_csv(output / f'{side}_role_k.csv')
        ax.plot(table.k, table.bic - table.bic.min(), 'o-')
        rejected = table[(table.k > 1) & ~table.qualifies]
        ax.scatter(rejected.k, rejected.bic - table.bic.min(), marker='x', color='red', label='Fails BIC/size/stability')
        ax.axvline(summary['roles'][side]['k'], color='.5', ls='--')
        ax.set(title=f"{side}: K={summary['roles'][side]['k']}, n={summary['roles'][side]['n']}",
               xlabel='Role K', ylabel='BIC − minimum', xticks=[1,2,3,4])
        ax.legend(fontsize=7)

    ordered = {}
    for side in ('left', 'right'):
        rows = np.flatnonzero(ants.side.eq(side))
        expected_speed = np.nan_to_num(proportions[rows] @ profiles.forward_peak.to_numpy())
        order = np.lexsort((expected_speed, np.where(ants.role.iloc[rows] < 0, 99, ants.role.iloc[rows])))
        ordered[side] = rows[order]
    # 8: All identities; missing remains gray.
    fig, axes = plt.subplots(2, 1, figsize=(15, 15), sharex=True, layout='constrained')
    for ax, side in zip(axes, ('left', 'right')):
        rows = ordered[side]
        im = ax.imshow(np.ma.masked_less(tasks[rows], 0), aspect='auto', interpolation='nearest',
                       extent=[0,48,len(rows)-.5,-.5], cmap=cmap, norm=norm)
        ax.set_yticks(range(len(rows)), [f"{ants.ant.iloc[i]} G{ants.role.iloc[i]}" if ants.role.iloc[i]>=0 else f"{ants.ant.iloc[i]} (low coverage)" for i in rows], fontsize=6)
        for boundary in np.flatnonzero(np.diff(ants.role.iloc[rows])):
            ax.axhline(boundary+.5, color='black', lw=1)
        ax.set(title=side, ylabel='Ant / proportion-derived role; gray = missing')
        ax.axvline(24, color='black', ls='--')
    axes[-1].set(xticks=[0,12,24,36,48], xticklabels=['Jul24 10:00','Jul24 22:00','Jul25 10:00','Jul25 22:00','Jul26 10:00'], xlabel='Time (JST)')
    fig.colorbar(im, ax=axes, ticks=range(k), label='State', shrink=.5)

    # 9: State proportions, grouped by role.
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), layout='constrained')
    for ax, side in zip(axes, ('left','right')):
        rows = ordered[side];bottom = np.zeros(len(rows))
        for state in range(k):
            ax.bar(range(len(rows)), proportions[rows,state], bottom=bottom, color=palette[state], label=f'T{state}')
            bottom += proportions[rows,state]
        ax.set_xticks(range(len(rows)), [f'{ants.ant.iloc[i]}\nG{ants.role.iloc[i]}' for i in rows], rotation=90, fontsize=6)
        ax.set(ylabel='Fraction of observed bins', title=side, ylim=(0,1))
    axes[0].legend(ncols=min(k,10), fontsize=7)

    # 10–11: Spatial outcomes opened only after behavioral states / roles freeze.
    occupancy = source / 'reproduction/occupancy/per_track'
    for side in ('left','right'):
        sub = ants[ants.side.eq(side) & ants.eligible]
        maps, titles = [], []
        for column, prefix in [('spatial_cluster','Spatial'), ('role','Role')]:
            for group, selected in sub.groupby(column):
                histograms=[]
                for ant in selected.itertuples():
                    folder=occupancy/Path(ant.track_name).stem
                    histogram=np.load(folder/'grid_occupancy_f4.npy').astype(float)
                    histograms.append(histogram/histogram.sum())
                    xe=np.load(folder/'grid_x_edges_mm.npy');ye=np.load(folder/'grid_y_edges_mm.npy')
                maps.append(np.mean(histograms,axis=0));titles.append(f'{side} {prefix} {int(group)} · n={len(selected)}')
        fig,axes=plt.subplots(1,len(maps),figsize=(3*len(maps),5),layout='constrained')
        vmax=max(np.quantile(m,.995) for m in maps)
        for ax,m,title in zip(axes,maps,titles):
            im=ax.imshow(m,origin='upper',extent=[xe[0],xe[-1],ye[-1],ye[0]],cmap='magma',norm=PowerNorm(.5,vmin=0,vmax=vmax))
            ax.set(title=title,xlabel='x (mm)',ylabel='y (mm)')
        fig.colorbar(im,ax=axes,label='Mean occupancy fraction / bin',shrink=.7)

    if (output/'sleep_by_state.csv').is_file():
        sleep=pd.read_csv(output/'sleep_by_state.csv')
        candidate=summary['low_motion_candidate']
        # 12: Sleep candidate picked from movement BEFORE reading sleep labels.
        fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
        axes[0].bar(sleep.state,sleep.sleep_fraction,color=palette,
                    yerr=np.stack([sleep.sleep_fraction-sleep.lower,sleep.upper-sleep.sleep_fraction]),capsize=3)
        axes[0].bar([candidate],[sleep.sleep_fraction.iloc[candidate]],facecolor='none',edgecolor='black',lw=2)
        axes[0].set(xlabel='State',ylabel='Existing sleep-rule fraction (equal ant weight)',ylim=(0,1),title=f'Frozen low-motion candidate: T{candidate}',xticks=range(k))
        axes[1].bar(sleep.state,sleep.persistence,color=palette)
        axes[1].set(xlabel='State',ylabel='P(same state in next observed adjacent bin)',ylim=(0,1),xticks=range(k),title=f'Persistence at {minutes}-minute resolution')
        for state,row in sleep.iterrows():
            axes[2].scatter(row.body_forward_peak,row.antenna_unsigned_mean,color=palette[state],s=100)
            axes[2].annotate(f'T{state}',(row.body_forward_peak,row.antenna_unsigned_mean),xytext=(4,4),textcoords='offset points')
        axes[2].set(xlabel='Median body forward peak (mm/s)',ylabel='Median relative antennal speed (mm/s)',xscale='log',yscale='log',title='Independent of sleep labels during fitting')
        axes[2].margins(.2)
        # 13: Full trajectories and sleep-rule fractions on the same ants/time.
        with np.load(output/'sleep_comparison.npz') as z: sf=z['sleep_fraction']
        fig,axes=plt.subplots(2,2,figsize=(15,14),sharex=True,layout='constrained')
        for row,side in enumerate(('left','right')):
            rows=ordered[side]
            for ax,data,title in [(axes[row,0],np.where(tasks[rows]>=0,(tasks[rows]==candidate).astype(float),np.nan),'Candidate-state bins'),(axes[row,1],sf[rows],'Existing sleep fraction')]:
                im=ax.imshow(np.ma.masked_invalid(data),aspect='auto',interpolation='nearest',extent=[0,48,len(rows)-.5,-.5],cmap='Blues',vmin=0,vmax=1)
                ax.set_yticks(range(len(rows)),ants.ant.iloc[rows],fontsize=6)
                ax.set(title=f'{side}: {title}',xlabel='Hours from Jul24 10:00 JST',xticks=[0,12,24,36,48])
        fig.colorbar(im,ax=axes,label='Fraction / indicator',shrink=.5)

    # Example waveforms are measured clips, never concatenations of minute samples.
    if clips is not None:
        with np.load(clips / 'clip_features.npz') as z:
            example_ant, example_minute, scores = z['example_ant'], z['example_minute'], z['example_scores']
        example_bin = example_minute // minutes
        example_state = tasks[example_ant, example_bin]
        row_index = np.full(tasks.shape, -1, dtype=int)
        row_index[valid] = np.arange(valid.sum())
        fig, axes = plt.subplots(k, 1, figsize=(10, max(3, 2*k)), squeeze=False, layout='constrained')
        examples = []
        for state, ax in enumerate(axes[:, 0]):
            candidates = np.flatnonzero(example_state == state)
            if not len(candidates):
                ax.text(.5, .5, 'No saved complete example clip', ha='center', transform=ax.transAxes)
                continue
            rows = row_index[example_ant[candidates], example_bin[candidates]]
            chosen_example = candidates[np.argmin(((x[rows] - centroids[state])**2).sum(axis=1))]
            time = np.arange(scores.shape[1]) / 12
            for pc in range(4):
                ax.plot(time, scores[chosen_example, :, pc], label=f'PC{pc+1}', lw=1.5)
            ant = ants.ant.iloc[example_ant[chosen_example]]
            ax.set(ylabel=f'T{state}: PC score', title=f'{ant}, minute {example_minute[chosen_example]}')
            examples.append(dict(state=state, ant=ant, minute=int(example_minute[chosen_example]),
                                 example_index=int(chosen_example)))
        axes[0, 0].legend(ncols=4)
        axes[-1, 0].set_xlabel('Time within sampled clip (s)')
        fig.suptitle('Measured PC traces: one saved clip from a central bin of each state; bins can mix movements')
        pd.DataFrame(examples).to_csv(output / 'waveform_examples.csv', index=False)

    figures=output/'figures';figures.mkdir(exist_ok=True)
    with PdfPages(output/'leiden_behavior_summary.pdf') as pdf:
        for number in plt.get_fignums():
            fig=plt.figure(number)
            if not summary['broad_supported']:
                title = fig._suptitle.get_text() if fig._suptitle else ''
                fig.suptitle((title + '\n' if title else '') +
                             'Exploratory state partition: fails the predefined stability criteria',
                             fontsize=11, color='#9c3600')
            fig.savefig(figures/f'figure_{number:02d}.png',dpi=150)
            pdf.savefig(fig)
    plt.close('all')
    print('Saved static figures and PDF',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('source','inputs','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--reuse-umap',action='store_true')
    parser.add_argument('--clips',type=Path,help='Optional clip_features.npz directory for measured waveform examples')
    a=parser.parse_args()
    plot(a.source,a.inputs,a.output,a.reuse_umap,a.clips)
