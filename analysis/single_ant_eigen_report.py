"""Reuse the colony report's numerical cells for a single cohort of cup ants."""
import json
import os
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

# Cells 7, 10 and 11 require colony grids, nests, trips or social contacts.
COMMON_CELLS = (0, 1, 2, 3, 4, 5, 7, 8)
EDGES = [(0, 1), (0, 2), (2, 3), (1, 4), (4, 5), (5, 6),
         (1, 7), (7, 8), (8, 9)]


def bootstrap(namespace):
    block, ants, output = (namespace[k] for k in ('BLOCK', 'ants', 'OUTPUT'))
    stitched = block/'stitched' if (block/'stitched/speed_vectors').is_dir() else block/'analysis'
    os.environ['ANTS_MOTION_CACHE_ROOT'] = str(stitched/'sleep_motion')
    grid = namespace['go'].load_speed_tracks(stitched/'speed_vectors')
    table = ants[['side', 'track_id', 'track_name', 'ant_class']].copy()
    table['leiden_cluster'] = table.ant_class
    table['cluster_id'] = 'single_behavior_' + table.ant_class.astype(str)
    ids = table[['side', 'track_id', 'track_name', 'cluster_id', 'leiden_cluster']]
    ids.to_csv(output/'behavior_class_ids.csv', index=False)
    namespace.update(stitched=stitched, grid=grid, cluster_table=table, cluster_ids=ids)


def save(figure, namespace, section, title, caption):
    figure._report_title, figure._report_caption = title, caption
    figure._report_section, figure._report_index = section, 0
    figure._right_saved = True
    figure.savefig(namespace['OUTPUT']/f'{section}_{figure.number:02d}.png', dpi=160, bbox_inches='tight')


def supplementary(namespace):
    source, pca = namespace['SOURCE'], namespace['pca']
    fig, axes = plt.subplots(2, 4, figsize=(14, 8), layout='constrained')
    for i, ax in enumerate(axes.flat):
        for sign, color in [(-1, '#2166ac'), (0, '.45'), (1, '#b2182b')]:
            xy = (pca['center'] + sign*2*np.sqrt(pca['eigenvalues'][i])*pca['modes'][i]).reshape(10, 2)
            for a, b in EDGES:
                ax.plot(xy[[a, b], 0], xy[[a, b], 1], color=color, alpha=.75, lw=1.3)
            ax.scatter(xy[:, 0], xy[:, 1], s=8, color=color, label=f'{sign*2:+} SD')
        ax.set_aspect('equal'); ax.invert_yaxis()
        ax.set_title(f'PC{i+1}: {pca["variance"][i]:.1%}')
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Eigenpostures: mean and ±2 standard deviations along each mode')
    save(fig, namespace, '01_eigenposture_shapes', 'Reconstructed eigenpostures',
         'The first eight PCA modes displace the mean skeleton by minus two (blue) and plus two (red) standard deviations. '
         'Gray is the mean. Coordinates are normalized by each ant’s median tag-to-petiole distance. '
         'These linear reconstructions illustrate covariance modes, rather than observed example postures.')

    manifest = json.loads((source/'tracking_manifest.json').read_text())
    ants, colors = namespace['ants'], namespace['class_colors']
    cups = {int(c['track_id']): c for c in manifest['cups'] if c.get('track_id') is not None}
    classes = dict(zip(ants.track_id, ants.ant_class))
    fig, ax = plt.subplots(figsize=(11, 10), layout='constrained')
    assigned = {c['cup_id']: c.get('track_id') for c in manifest['cups']}
    for cup in manifest.get('physical_cups', manifest['cups']):
        tid = assigned.get(cup['cup_id'])
        center, radius = cup['center_mm'], cup['radius_mm']
        color = colors[classes[tid]] if tid in classes else '.8'
        ax.add_patch(Circle(center, radius, fc=color, ec='.3', alpha=.55))
        label = cup['cup_id']
        ax.text(*center, f'{label}\nC{classes[tid]}' if tid in classes else f'{label}\n'+('no track' if tid is None else 'QC excluded'),
                ha='center', va='center', fontsize=8)
    ax.autoscale_view(); ax.invert_yaxis(); ax.set_aspect('equal')
    ax.set(xlabel='Panorama X (mm)', ylabel='Panorama Y (mm)', title='Manual cups and behavioral ant classes')
    save(fig, namespace, '07_cup_classes', 'Behavioral classes on the cup map',
         'Circles use the manually labeled cup geometry. Numbers retain the row-then-column cup ordering. '
         'Colors show classes learned only from state proportions; cup position does not enter PCA, state learning or class selection. '
         'Gray marks tracks excluded by the colony workflow’s position-coverage rule or lacking pose rows.')

    histograms = []
    for row in ants.itertuples():
        cup = cups[int(row.track_id)]
        xy = np.load(source/'sequences'/row.ant.replace(':', '_')/'position.npy', mmap_mode='r')
        xy = (xy*.016 - np.asarray(cup['center_mm']))/cup['radius_mm']
        good = np.isfinite(xy).all(axis=1)
        h, _, _ = np.histogram2d(xy[good, 1], xy[good, 0], bins=45, range=[[-1, 1], [-1, 1]])
        if h.sum() == 0: raise ValueError(f'No position observations for {row.ant}')
        histograms.append(h/h.sum())
    histograms = np.asarray(histograms)
    np.savez_compressed(namespace['OUTPUT']/'within_cup_occupancy.npz',
                        track_ids=ants.track_id.to_numpy(), occupancy=histograms)

    def panel(ax, h, title):
        image = ax.imshow(h*100, extent=(-1, 1, 1, -1), cmap='magma', vmin=0)
        ax.add_patch(Circle((0, 0), 1, fill=False, ec='cyan', lw=.7))
        ax.set(title=title, xticks=[-1, 0, 1], yticks=[-1, 0, 1], aspect='equal')
        return image

    k = namespace['k_roles']
    fig, axes = plt.subplots(1, k, figsize=(max(5, 4*k), 4.5), squeeze=False, layout='constrained')
    for c, ax in enumerate(axes.flat):
        selected = ants.ant_class.eq(c).to_numpy()
        im = panel(ax, histograms[selected].mean(axis=0), f'C{c}: {selected.sum()} ants')
        fig.colorbar(im, ax=ax, label='Equal-ant mean observed time (%) / bin', shrink=.8)
    save(fig, namespace, '07_cup_occupancy', 'Within-cup occupancy by behavioral class',
         'Each ant’s full-recording tracked positions are centered on its manually labeled cup and divided by its radius. '
         'Histograms first normalize each ant’s observed time, then average ants equally within a class. '
         'The circle is the labeled cup boundary. Missing positions contribute no time. Spatial occupancy is a post hoc description.')
    fig, axes = plt.subplots(int(np.ceil(len(ants)/6)), 6, figsize=(15, 2.5*np.ceil(len(ants)/6)), layout='constrained')
    for ax, row, h in zip(axes.flat, ants.itertuples(), histograms):
        panel(ax, h, f'Cup {row.track_id} · C{row.ant_class}')
    for ax in list(axes.flat)[len(ants):]: ax.set_visible(False)
    save(fig, namespace, '07_individual_cup_occupancy', 'Every cohort ant’s within-cup occupancy',
         'All ants retained for posture and state analysis are shown in cup order. Each histogram sums to 100% of that ant’s observed position time. '
         'Coordinates are relative to the manual cup center and radius; each panel scales its color to its own maximum to show shape. '
         'Unobserved time is omitted rather than filled.')


def single_captions(namespace):
    no_quiet = not namespace['motion_profiles'].quiet_candidate.any()
    readout_notes = {
        '00_workflow': 'Tracking, normalized posture PCA, state learning and ant classes use the colony workflow. S0 is retained as the lowest-motion reference state. No state meets the colony absolute quiet limits, so the downstream S0 and calibrated-motion readouts do not establish sleep.',
        '04_quiet_states': 'Points show each state’s median body and antenna motion. Dotted lines retain the colony descriptive limits: body ≤0.05 mm/s and antenna ≤0.15 mm/s. No state passes both. The outline identifies the lowest-motion reference S0, without assigning a sleep interpretation.',
        '05_motion_calibration': 'Equal-ant body and antenna motion distributions are measured from the same raw coordinates as the colony analysis. S0 95th percentiles define a relative motion envelope. The companion panel checks overlap and held-out time blocks. These thresholds preserve the colony calculation; they do not validate sleep.',
        '05_motion_sleep': 'Curves show the equal-ant percentage of time inside the S0-calibrated motion envelope. The original trailing one-second window, 50% valid-data and 90% below-threshold rules are unchanged. Unknown periods are excluded. This is a relative motion readout; none of the learned states passes the colony absolute quiet limits.',
        '05_quiet_bouts': 'Curves show contiguous runs of direct S0 membership. Missing observations and recording boundaries censor runs rather than count as state transitions. S0 is a relative low-motion state; its run lengths are not validated sleep bouts.',
        '05_class_sleep_speed': 'Points and class means summarize direct S0 occupancy and trajectory speed. Curves use ten-minute bins. Classes were fitted to state proportions, so differences in S0 occupancy are descriptive of the class construction. S0 is not assigned a validated sleep interpretation here.',
        '05_speed_sleep': 'The panels show direct S0 occupancy, its observation support, or individual S0 time courses. The original colony coverage rules and equal-ant weighting are unchanged. Missing intervals remain unknown. S0 membership denotes the relative low-motion state, not established sleep.',
        '06_clock_profiles': 'Matrices align each ant’s speed and direct S0 occupancy to clock time. Available complete light-cycle days and original coverage rules are used. S0 is not validated sleep, and this recording does not establish an endogenous circadian rhythm.',
        '06_motion_clock_profiles': 'The same clock and coverage rules are applied to the broader S0-calibrated motion envelope. Unknown intervals remain missing. The readout describes relative motion and cannot establish physiological sleep or an endogenous circadian rhythm.'}
    from matplotlib.text import Text
    for number in plt.get_fignums():
        fig = plt.figure(number)
        text = fig._report_caption
        text = text.replace('Both colonies contribute posture, motion and exact contact features',
                            'The single-ant cohort contributes posture and motion features')
        text = text.replace('both colonies', 'the single-ant cohort').replace('Both colonies', 'The single-ant cohort')
        text = text.replace('selected colony', 'single-ant cohort').replace('Selected-colony', 'Single-ant')
        text = text.replace('Posture, movement and concurrent contacts jointly define a state',
                            'Posture and movement define a state')
        text = text.replace('motion and contact measurements', 'motion measurements')
        text = text.replace('Posture, motion, and exact contacts', 'Posture and motion')
        text = text.replace('S1 is also quiet but remains a separate behavioral state. ', '')
        text = text.replace('Overlap with S1 is expected:', 'Other low-motion states may overlap:')
        text = re.sub(r'S1 is also quiet but remains a separate\s+behavioral state\. ', '', text)
        text = text.replace('for return-contact analyses', 'for individual activity analysis')
        text = text.replace('later used for return analysis', 'used for the motion-only readout')
        text = text.replace('whereas the return figures use the separately calibrated motion labels',
                            'alongside the separately calibrated motion labels')
        text = text.replace('used in the return analysis', 'used in the motion-only readout')
        fig._report_caption = text
        if fig._report_section == '03_state_umaps':
            fig._report_title = 'Views of the same learned states'
        if fig._report_section == '06_state_coverage':
            fig._report_caption = ('Bars show detected position, usable full-body posture, valid motion, feature support and assigned state coverage. '
                'Each stage uses full-recording time as its denominator. Geometry bounds are calibrated for this cohort and recorded in geometry.json. '
                'Camera-local derivatives, missing observations and recording boundaries constrain valid segments. No social-contact requirement is applied.')
        if no_quiet:
            changed = False
            for artist in fig.findobj(match=Text):
                old = artist.get_text()
                new = old.replace('Single colony', 'Single-ant cohort')
                new = new.replace('S0 = putative sleep state', 'S0 = relative low-motion state')
                new = re.sub(r'(?i)\b(?:putative\s+)?sleep\b', 'S0-based readout', new)
                if new != old: artist.set_text(new); changed = True
            fig._report_title = re.sub(r'(?i)\b(?:putative\s+)?sleep\b', 'S0-based readout', fig._report_title)
            if fig._report_section in readout_notes:
                fig._report_caption = readout_notes[fig._report_section]
            if changed:
                fig.savefig(namespace['OUTPUT']/f'{fig._report_section}_{number:02d}.png',dpi=160,bbox_inches='tight')


def run_cells(source):
    os.environ['ANTS_COLONY_SIDE'] = 'single'
    cells = [c for c in re.split(r'(?m)(?=^# %%)', source.read_text()) if c.strip()]
    if len(cells) != 11: raise ValueError('Review changed colony report cells before single-ant reuse')
    namespace = dict(__file__=str(source), __name__='single_ant_eigenposture_report')
    for index in COMMON_CELLS:
        if index == 7: bootstrap(namespace)
        print(f'SINGLE_REPORT colony cell {index+1}', flush=True)
        exec(compile(cells[index], str(source)+f':cell{index+1}', 'exec'), namespace)
    supplementary(namespace)
    single_captions(namespace)
    summary = {key: namespace[key] for key in ['n_states', 'k_roles', 'selection', 'role_choice', 'complete_cycles']}
    interpretation = dict(reference_state=0,quiet_candidates=namespace['motion_profiles'].loc[
        namespace['motion_profiles'].quiet_candidate,'state'].tolist(),
        body_limit_mm_s=.05,antenna_limit_mm_s=.15,
        status='relative_S0_readouts_only_no_quiet_candidate' if not namespace['motion_profiles'].quiet_candidate.any()
               else 'putative_sleep_not_physiologically_validated',
        native_thresholds_changed=False)
    (namespace['SOURCE']/'state_interpretation.json').write_text(json.dumps(interpretation,indent=2)+'\n')
    summary.update(side='single', n_ants=len(namespace['ants']),
        class_sizes=namespace['ants'].ant_class.value_counts().sort_index().to_dict(),
        quiet_states=namespace['quiet_states'].tolist(), source=str(namespace['SOURCE']),
        geometry=json.loads((namespace['SOURCE']/'geometry.json').read_text()),
        protocol=json.loads((namespace['SOURCE']/'analysis_protocol.json').read_text()),
        colony_cells_reused=[i+1 for i in COMMON_CELLS],
        sleep_interpretation=interpretation,
        motion_sleep_definition=namespace['motion_definition'])
    (namespace['OUTPUT']/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return namespace
