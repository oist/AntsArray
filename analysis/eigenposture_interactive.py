# %% 1. Inputs: both colonies for states, right colony for ant classes
"""Full-body behavior -> pooled state proportions -> ant classes -> sleep/space.

Run these ten cells in VS Code/Jupyter. Expensive full-track preprocessing and
method comparisons are produced by full_behavior_extract/states/roles.py.
Every eligible ant's usable segments are assigned; no hourly clip sampling.
"""
from pathlib import Path
import os
import sys
import json
repo=Path(__file__).resolve().parents[1] if '__file__' in globals() else Path.cwd()
if str(repo) not in sys.path:sys.path.insert(0,str(repo))
if os.environ.get('MPLBACKEND','').lower()!='agg':
    try:get_ipython().run_line_magic('matplotlib','qt')
    except (NameError,ImportError):pass
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, homogeneity_score
from threadpoolctl import threadpool_limits
from umap import UMAP
from analysis import grid_occupancy_utils as go
from analysis import sleep_motion_analysis_utils as sma
from analysis import return_sleep_utils as rs
from analysis.full_behavior_extract import SEED,HZ,FPS
from analysis.full_behavior_states import NAMES,encode
from analysis.full_behavior_roles import fit_labels
from analysis.full_behavior_sleep import build as build_state_sleep,select_quiet_states

BLOCK=Path(os.environ.get('ANTS_DATASET_ROOT','/home/sam-reiter/bucket/ReiterU/Ants/basler/20260724/block01'))
SOURCE=Path(os.environ.get('ANTS_BEHAVIOR_SOURCE',str(BLOCK/'analysis_outputs/full_body_behavior_20261011')))
OUTPUT=Path(os.environ.get('ANTS_ROLE_OUTPUT',str(SOURCE/'report')))
LIGHT_ON,LIGHT_OFF=5.5,19.5
# None chooses quiet clusters from the motion-profile table in cell 3. This is
# POST HOC interpretation, never training labels. Edit to explicit IDs to assess.
SLEEP_STATES=None
MIN_SLEEP_SECONDS=0  # Primary result: cluster membership; inspect bout durations separately.
threadpool_limits(limits=4);OUTPUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})


def show(name):
    for number in plt.get_fignums():
        figure=plt.figure(number)
        if not getattr(figure,'_right_saved',False):
            figure.savefig(OUTPUT/f'{name}_{number:02d}.png',dpi=160,bbox_inches='tight')
            figure._right_saved=True
    plt.show(block=False)


inventory=pd.read_csv(SOURCE/'inventory.csv')
cohort=inventory[inventory.selected].reset_index(drop=True)
assert (cohort.detection_fraction>.6).all()
ants=cohort[cohort.side.eq('right')].reset_index(drop=True)
print('State cohort:',cohort.groupby('side').size().to_dict(),'right ant classes:',len(ants),flush=True)

# %% 2. PCA of all ten landmarks after body alignment, throughout the recording
# Tag at origin, petiole->tag is anterior. Divide by each ant's median body-axis
# length, preserving articulation and antenna identity. All four body points AND
# six antennal points enter PCA. This skeleton has no leg landmarks.
# All valid full-recording observations at 12 Hz fit the covariance, not day 1.
with np.load(SOURCE/'full_body_pca.npz') as archive:pca=dict(archive)
fig,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained')
axes[0].plot(np.arange(1,21),np.cumsum(pca['variance']),'o-')
for count in (4,8):axes[0].axvline(count,color='.5',ls=':',label=f'Tested: {count} PCs')
axes[0].legend();axes[0].set(xlabel='Full-body posture PCs',ylabel='Cumulative variance',
    title=f"1. PCA from {int(pca['n_samples']):,} observations")
image=axes[1].imshow(pca['modes'][:8],aspect='auto',cmap='coolwarm',vmin=-.8,vmax=.8)
landmarks=['tag','head','petiole','gaster','A1','A2','A3','B1','B2','B3']
axes[1].set(xticks=np.arange(20),xticklabels=[f'{node} {axis}' for node in landmarks for axis in ('x','y')],
    yticks=np.arange(8),yticklabels=[f'PC{i+1}' for i in range(8)],title='Full-body coordinate loadings')
axes[1].tick_params(axis='x',rotation=90);fig.colorbar(image,ax=axes[1],label='PCA loading')
show('01_full_body_pca')

# %% 3. State discovery: feature expansion, reproducibility, and UMAP views
# Windows partition every valid segment into <=1-second pieces, retaining tails
# >=250 ms. Gaps/camera changes break segments. No rates cross gaps. States are
# assigned throughout the recording; an ant-balanced graph dictionary makes
# the large dataset tractable. Exact frame-wise contacts include ongoing contact,
# summarized as mean and maximum simultaneous partners; no new-bout feature.
# Compare 4/8 mean posture PCs, then within-window SD and RMS PC rates. Families
# receive equal total variance; Leiden works in that space, never in 2-D UMAP.
selection=json.loads((SOURCE/'models/selection.json').read_text())
state_search=pd.read_csv(SOURCE/'models/search.csv')
with np.load(SOURCE/'models/dictionary.npz') as archive:dictionary=dict(archive)
n_states=int(dictionary['state'].max()+1)
normalizer=json.loads((SOURCE/'models'/(selection['representation']+'_normalizer.json')).read_text())
np.testing.assert_allclose(encode(dictionary['raw'],normalizer),dictionary['x'],atol=2e-6)
state_stability=pd.read_csv(SOURCE/'models'/f"{selection['representation']}_{selection['resolution']:g}_state_stability.csv")
jaccard=state_stability.matched_jaccard.to_numpy()[dictionary['order']]
state_names=[f'S{s}'+('*' if jaccard[s]<.5 else '') for s in range(n_states)]
colors=plt.get_cmap('tab20')(np.arange(n_states)) if n_states<=20 else plt.get_cmap('turbo')(np.linspace(0,1,n_states))
fig,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained')
for name,group in state_search.groupby('representation'):
    for ax,metric in zip(axes,['seed_ari','ant_removal_ari']):
        ax.plot(group.n_states,group[metric],'o-',label=name)
        passed=group[group.qualifies]
        ax.scatter(passed.n_states,passed[metric],s=95,facecolors='none',edgecolors='black')
for ax,metric,threshold in zip(axes,['seed_ari','ant_removal_ari'],[.8,.65]):
    ax.axhline(threshold,color='.5',ls=':')
    ax.scatter(n_states,selection[metric],marker='*',s=170,color='black',zorder=5)
    ax.set(xlabel='Number of states',ylabel=metric.replace('_',' '))
axes[0].legend();fig.suptitle(f"2. Selected {selection['representation']}: {n_states} states; circles pass all checks")
show('02_feature_expansion')
# Only display parameters change below. Apparent island separation is not
# evidence for extra states. All panels show exactly the same state labels.
rng=np.random.default_rng(SEED);sample=np.sort(rng.choice(len(dictionary['x']),min(12000,len(dictionary['x'])),replace=False))
fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
for ax,(neighbors,distance,spread) in zip(axes,[(15,.02,2.),(40,.1,2.),(80,.3,1.5)]):
    file=OUTPUT/f'umap_n{neighbors}_d{distance}_s{spread}.npz'
    if file.exists():
        with np.load(file) as z:
            np.testing.assert_array_equal(z['sample'],sample);xy=z['xy']
    else:
        xy=UMAP(n_neighbors=neighbors,min_dist=distance,spread=spread,n_epochs=400,random_state=SEED,n_jobs=1).fit_transform(dictionary['x'][sample])
        np.savez(file,sample=sample,xy=xy)
    for s in range(n_states):
        keep=dictionary['state'][sample]==s
        ax.scatter(*xy[keep].T,s=2,color=colors[s],label=state_names[s],rasterized=True)
    ax.set(title=f'n={neighbors}, min_dist={distance}, spread={spread}',xlabel='UMAP 1',ylabel='UMAP 2')
axes[-1].legend(fontsize=6,markerscale=3,ncol=2,loc='best');fig.suptitle('3. Alternative views of the same behavior states')
show('03_state_umaps')
profiles=np.array([np.median(dictionary['raw'][dictionary['state']==s],axis=0) for s in range(n_states)])
fig,ax=plt.subplots(figsize=(13,max(4,n_states*.26)),layout='constrained')
image=ax.imshow(encode(profiles,normalizer),aspect='auto',cmap='coolwarm',vmin=-2,vmax=2)
feature_columns=[j for group in normalizer for j in group['columns']]
ax.set(xticks=np.arange(len(feature_columns)),xticklabels=[NAMES[j] for j in feature_columns],
    yticks=np.arange(n_states),yticklabels=state_names,title='4. State feature profiles; * = uncertain ant-removal match')
ax.tick_params(axis='x',rotation=90);fig.colorbar(image,ax=ax,label='Median feature / family scale')
show('04_state_profiles')
quiet_states,motion_profiles=select_quiet_states(SOURCE)
motion_profiles.to_csv(OUTPUT/'state_motion_profiles.csv',index=False)
if SLEEP_STATES is not None:quiet_states=np.asarray(SLEEP_STATES,int)
print('Quiet-state candidates:',quiet_states.tolist(),'\n',motion_profiles.to_string(index=False),flush=True)
fig,ax=plt.subplots(figsize=(6,4),layout='constrained')
for row in motion_profiles.itertuples():
    ax.scatter(row.body_peak_median_mm_s,row.antenna_median_mm_s,color=colors[row.state],s=65)
    ax.annotate(state_names[row.state],(row.body_peak_median_mm_s,row.antenna_median_mm_s),fontsize=8)
ax.axvline(.05,color='.5',ls=':');ax.axhline(.15,color='.5',ls=':')
ax.set(xscale='log',yscale='log',xlabel='Median unsigned body peak (mm/s)',
    ylabel='Median unsigned antennal speed (mm/s)',title='Quiet-cluster interpretation after discovery')
show('04_quiet_states')

# %% 4. Count states directly, then normalize; no hourly averaging or extra filter
# Counts are observed 12-Hz samples, so a 250-ms tail receives its actual time
# weight rather than counting as a full second. Missing samples are not a state.
counts=np.zeros((len(ants),n_states))
for ai,ant in enumerate(ants.ant):
    name=ant.replace(':','_')
    state=np.load(SOURCE/'assignments'/name/'state.npy')
    length=np.load(SOURCE/'features'/name/'length.npy')
    counts[ai]=np.bincount(state,weights=length,minlength=n_states)
proportions=counts/counts.sum(axis=1,keepdims=True)
np.testing.assert_allclose(proportions.sum(axis=1),1)
role_x=np.sqrt(proportions)  # Hellinger geometry; ALL state dimensions, no PCA.

# %% 5. Compare ant clustering methods, allowing finer roles when reproducible
# K-means, Ward, tied/diagonal GMM, Leiden, and HDBSCAN; K up to 8. Selection is
# the finest stable partition with >=3 ants/class. Ant-removal and time-split
# checks include the least reproducible class (Jaccard), so two stable big groups
# cannot conceal an unstable tiny group. Hour bins are ONLY validation blocks.
role_search=pd.read_csv(SOURCE/'roles/method_comparison.csv')
role_choice=json.loads((SOURCE/'roles/selection.json').read_text())
k_roles=int(role_choice['k'])
labels=(np.zeros(len(ants),int) if k_roles==1 else fit_labels(role_x,role_choice['method'],role_choice['parameter']))
original=np.unique(labels)
order=np.argsort([np.mean(proportions[labels==s]@np.arange(n_states)) for s in original])
remap={int(original[i]):rank for rank,i in enumerate(order)}
ants['ant_class']=[remap[int(s)] for s in labels]
reference_roles=pd.read_csv(SOURCE/'roles/ant_classes.csv')
np.testing.assert_array_equal(ants.ant,reference_roles.ant)
np.testing.assert_array_equal(ants.ant_class,reference_roles.ant_class)
fig,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained')
for method,group in role_search.groupby('method'):
    axes[0].scatter(group.k,group.ant_removal_ari,label=method)
    axes[1].scatter(group.k,group.temporal_smallest_jaccard,label=method)
axes[0].axhline(.8,color='.5',ls=':');axes[1].axhline(.6,color='.5',ls=':')
axes[0].set(xlabel='Ant classes',ylabel='Ant-removal ARI');axes[1].set(xlabel='Ant classes',ylabel='Weakest-class temporal Jaccard')
axes[0].legend(fontsize=8);fig.suptitle(f"5. Selected {role_choice['model']}: K={k_roles}")
show('05_ant_methods')

# %% 6. Inspect the ant classes, and the coverage available to each ant
ordered=ants.sort_values(['ant_class','track_id']).index.to_numpy()
fig,axes=plt.subplots(1,2,figsize=(14,9),layout='constrained')
im=axes[0].imshow(proportions[ordered],aspect='auto',cmap='viridis',vmin=0)
axes[0].set(xticks=np.arange(n_states),xticklabels=state_names,yticks=np.arange(len(ants)),
    yticklabels=[f"C{ants.loc[a,'ant_class']} / {ants.loc[a,'track_id']:03d}" for a in ordered],title='6. Ant state proportions')
axes[0].tick_params(axis='y',labelsize=7);axes[0].tick_params(axis='x',rotation=90)
fig.colorbar(im,ax=axes[0],label='Fraction of classified observations')
axes[1].barh(np.arange(len(ants)),ants.loc[ordered,'detection_fraction'],label='Detection coverage',alpha=.5)
axes[1].barh(np.arange(len(ants)),counts.sum(axis=1)[ordered]/(ants.n_frames.to_numpy()[ordered]/2),label='State-classified coverage',alpha=.8)
axes[1].set(ylim=(len(ants)-.5,-.5),xlabel='Fraction of full recording',yticks=[],title='Tracking and usable posture coverage')
for boundary in np.flatnonzero(np.diff(ants.loc[ordered,'ant_class']))+.5:
    axes[0].axhline(boundary,color='white',lw=1)
    axes[1].axhline(boundary,color='.6',lw=.7)
axes[1].legend();show('06_ant_classes')
ants.to_csv(OUTPUT/'ant_classes.csv',index=False)
pd.DataFrame(proportions,index=ants.ant,columns=state_names).to_csv(OUTPUT/'ant_state_proportions.csv')
# Compatibility with the downstream utilities: every >60% right ant is eligible.
eligible=np.ones(len(ants),bool);ids=np.arange(len(ants))

# %% 7. Hand the fixed behavioral classes to the grid-occupancy utilities
# This is the FIRST read of spatial information. Require an exact identity AND
# filename match for every eligible ant; do not silently drop or relabel ants.
# The utilities call the numerical class column 'leiden_cluster' for historical
# reasons. Here it contains the selected ANT CLASS.
stitched = BLOCK / "stitched"
grid_root = stitched / "grid_occupancy_histograms_arena"
grid = go.load_grid_tracks(grid_root)
cluster_table = ants[eligible][["side", "track_id", "track_name", "ant_class"]].merge(
    grid, on=["side", "track_id", "track_name"], how="left", validate="one_to_one", indicator=True)
assert cluster_table._merge.eq("both").all(), "Missing grid cache for a classified ant"
cluster_table = cluster_table.drop(columns="_merge").assign(leiden_cluster=lambda d: d.ant_class,
    cluster_id=lambda d: "right_behavior_" + d.ant_class.astype(str))
cluster_ids = cluster_table[["side", "track_id", "track_name", "cluster_id", "leiden_cluster"]]
cluster_ids.to_csv(OUTPUT / "behavior_class_ids.csv", index=False)
# Never overwrite the spatial track_cluster_ids.csv. Reference labels are only
# a post hoc agreement check and cannot alter states, K, or eligibility.
reference_path=SOURCE/'inputs/spatial_reference.csv'
if not reference_path.exists():
    reference=pd.read_csv(grid_root/'track_cluster_ids.csv').rename(columns={'TrackID':'track_id'})
    reference['spatial_cluster']=reference.leiden_cluster_id
    reference_path.parent.mkdir(exist_ok=True)
    reference.to_csv(reference_path,index=False)
reference = pd.read_csv(reference_path)
comparison = ants[eligible].merge(reference[["side", "track_id", "spatial_cluster"]],
                                  on=["side", "track_id"], how="left", validate="one_to_one")
observed = comparison.dropna(subset="spatial_cluster")
spatial_ari = adjusted_rand_score(observed.ant_class, observed.spatial_cluster)
spatial_homogeneity = homogeneity_score(observed.spatial_cluster, observed.ant_class)
print(f"Post hoc spatial agreement: ARI={spatial_ari:.3f}, {len(observed)}/{len(ids)} ants", flush=True)
comparison.to_csv(OUTPUT / "posthoc_spatial_comparison.csv", index=False)
overlap=pd.crosstab(observed.ant_class,observed.spatial_cluster)
overlap.to_csv(OUTPUT/'class_spatial_counts.csv')
overlaps=[(f'Selected {k_roles} classes',overlap,'C')]
# Also describe the broad two-way fit. Select it by temporal/ant stability,
# using the SAME behavior-only ordering as above, never by spatial agreement.
coarse=role_search[(role_search.k==2)&role_search.qualifies].sort_values(
    ['temporal_ari','ant_removal_ari'],ascending=False)
coarse_choice=None;coarse_spatial_ari=None
if len(coarse):
    coarse_choice=coarse.iloc[0].to_dict()
    with np.load(SOURCE/'roles/candidate_labels.npz') as z:broad_labels=z[coarse_choice['model']]
    broad=comparison.assign(broad_ant_class=broad_labels)
    broad.to_csv(OUTPUT/'broad_class_spatial_comparison.csv',index=False)
    coarse_spatial_ari=adjusted_rand_score(broad_labels[observed.index],observed.spatial_cluster)
    table=pd.crosstab(broad.loc[observed.index,'broad_ant_class'],observed.spatial_cluster)
    overlaps.append((f"Broad grouping: {coarse_choice['model']}",table,'B'))
    print(f'Broad two-class spatial ARI={coarse_spatial_ari:.3f}',flush=True)
fig,axes=plt.subplots(1,len(overlaps),figsize=(5*len(overlaps),4),squeeze=False,layout='constrained')
for ax,(title,table,prefix) in zip(axes.flat,overlaps):
    image=ax.imshow(table.div(table.sum(axis=1),axis=0),vmin=0,vmax=1,cmap='Blues',aspect='auto')
    for i in range(len(table)):
        for j in range(len(table.columns)):
            ax.text(j,i,str(table.iloc[i,j]),ha='center',va='center',color='red')
    ax.set(xticks=np.arange(len(table.columns)),xticklabels=table.columns,
        yticks=np.arange(len(table)),yticklabels=[f'{prefix}{s}' for s in table.index],
        xlabel='Independent spatial group',ylabel='Behavioral group',title=title)
fig.colorbar(image,ax=axes.ravel().tolist(),label='Fraction within behavioral group')
fig.suptitle('Post hoc spatial overlap: numbers are ants')
# A finer behavior partition can refine a spatial group, lowering ARI without
# mixing spatial identities; inspect this count table alongside the occupancy.
go.plot_cluster_mean_histograms(grid, cluster_table, title="7. Occupancy emerging from behavioral ant classes")
go.plot_cluster_example_histograms(grid, cluster_table, n_examples=4, title="8. Individual occupancy, grouped by behavioral class")
regions = go.load_panorama_regions(go.panorama_regions_path(BLOCK), x_split_px=None)
region_use = go.compute_region_occupancy(cluster_table, regions)
colony_use = go.summarize_colony_use(region_use)
go.plot_cluster_colony_use(colony_use)
plt.gcf().suptitle("9. Nest occupancy by behavioral class (post hoc)")
colony_use.to_csv(OUTPUT / "ant_colony_use.csv", index=False)
show("04_occupancy")

# %% 8. Full-recording speed and state-derived putative sleep by behavioral class
# Downstream plots use the full recording, as in grid_occupancy.py. Behavioral
# classes use all usable observed segments from the full recording.
# Equal ant weights; unknown sleep/velocity is missing, never zero. Putative
# sleep means quiet-cluster membership; there is no physiological validation.
# Classes were fitted to these state proportions, so differences in total quiet
# time describe the classes and are not an independent validation of them.
context = go.recording_context(BLOCK, grid)
clock = context["start_clock_seconds"]
start, stop = context["frame_min"], context["frame_stop"]
speed, ant_speed = go.plot_cluster_speed_timeseries(cluster_table, stitched / "speed_vectors",
    bin_seconds=600, smooth_seconds=600, start_clock_seconds=clock,
    light_off_hour=LIGHT_OFF, light_on_hour=LIGHT_ON, title="10. Speed by behavioral ant class")
show('05_speed')
sleep_root = build_state_sleep(SOURCE, BLOCK, quiet_states, MIN_SLEEP_SECONDS)
bouts=pd.read_parquet(sleep_root/'quiet_bouts.parquet').merge(
    cluster_ids[['side','track_id','cluster_id']],on=['side','track_id'],how='inner',validate='many_to_one')
fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
for label,group in bouts.groupby('cluster_id'):
    duration=np.sort(group.duration_seconds.to_numpy())
    ax.plot(duration,np.arange(1,len(duration)+1)/len(duration),label=f'{label}: {group.right_censored.mean():.0%} end at a gap')
ax.set(xscale='log',xlabel='Observed contiguous quiet duration (s)',ylabel='Cumulative fraction of runs',
    title='Quiet-run durations; gaps give lower bounds, not awakenings')
ax.legend(fontsize=8)
show('05_quiet_bouts')
sleep_tracks = sma.load_sleep_label_tracks(sleep_root, cluster_ids)
sleep, ant_sleep = sma.cluster_sleep_timeseries(sleep_tracks, bin_seconds=600,
    min_classified_fraction=.5, recording_start_frame=start, recording_stop_frame=stop)
sma.plot_cluster_sleep_timeseries(sleep, start_clock_seconds=clock).suptitle("11. Putative sleep from behavioral states, by ant class")
sma.plot_ant_sleep_heatmap(ant_sleep, start_clock_seconds=clock).suptitle("12. Individual state-derived putative sleep")
speed.to_csv(OUTPUT / "class_speed.csv", index=False)
sleep.to_csv(OUTPUT / "class_sleep.csv", index=False)
ant_sleep.to_parquet(OUTPUT / "ant_sleep_bins.parquet", index=False)
show("05_speed_sleep")

# %% 9. Individual activity/sleep versus time of day
# Use complete 05:30-to-05:30 cycles only when available, not partially observed
# days disguised as repeat cycles. This dataset has limited independent days.
fps = float(sleep_tracks.fps.iloc[0])
offset = LIGHT_ON * 3600 - clock
complete_cycles = max(0, int(np.floor((stop / fps - offset) / 86400) - np.ceil((start / fps - offset) / 86400)))
cycle_bins, clock_profiles, clock_audit = sma.compute_activity_sleep_clock_profiles(
    cluster_ids, go.load_speed_tracks(stitched / "speed_vectors"), sleep_tracks, fps=fps,
    recording_start_frame=start, recording_stop_frame=stop, start_clock_seconds=clock,
    light_on_hour=LIGHT_ON, light_off_hour=LIGHT_OFF, bin_minutes=30, min_bin_coverage=.5,
    min_cycles=min(2, max(1, complete_cycles)), include_partial_cycles=complete_cycles == 0,
    recording_date=context["recording_date"], max_workers=4)
sma.plot_activity_sleep_clock_matrices(cycle_bins, clock_profiles)
clock_profiles.to_csv(OUTPUT / "ant_clock_profiles.csv", index=False)
clock_audit.to_csv(OUTPUT / "clock_coverage.csv", index=False)
show("06_clock_profiles")

# %% 10. Returns and sleeping-recipient responses, retaining the same classes
# Same definitions as grid_occupancy: >=30 s outside, >=5 s colony anchors,
# observed entry crossing within 5 frames; new skeleton contacts <=0.1 mm.
# Retain ALL right ants as social context, including ants without a role. Their
# returns can expose classified recipients, without calling those ants a class.
# Contacts are themselves state features, so contact-triggered changes in quiet
# labels are partly definition dependent. One-second windows also blur event
# timing. Inspect physical motion responses; this is not an independent arousal test.
settings = rs.ReturnSettings()
all_sleep = sma.load_sleep_label_tracks(sleep_root)
all_sleep = all_sleep[all_sleep.side.eq("right")].copy()
cache = SOURCE / "return_context_cache"
chunks, interaction_run = rs.load_published_interaction_chunks(rs.resolve_interaction_root(BLOCK),
    BLOCK / "tracks", fps=settings.fps, start_clock_seconds=clock)
contexts = rs.load_contexts(all_sleep, BLOCK, regions, settings, cache, force=False)
bouts, coverage = rs.load_contact_bouts(chunks, settings, cache, force=False)
rs.attach_contact_context(contexts, bouts, coverage, settings)
all_classes = all_sleep[["side", "track_id"]].merge(cluster_ids[["side", "track_id", "cluster_id"]],
    on=["side", "track_id"], how="left", validate="one_to_one").fillna({"cluster_id": "unclassified_context"})
all_returns = rs.extract_returns(contexts, all_classes, settings)
returns = all_returns[all_returns.cluster_id.isin(cluster_ids.cluster_id)].copy()
return_curves, latency = rs.return_activity_curves(returns, contexts, settings)
contacts = rs.eligible_sleeping_contacts(sleep_tracks, contexts, bouts, all_returns, settings)
triggers, matching = rs.match_sleeping_controls(contacts, contexts, settings)
recipient_curves = rs.trigger_state_curves(triggers, sleep_tracks, contexts, settings)
# The canonical plots pool classes; explicitly subset here so class membership
# survives all the way to the return and recipient comparisons.
for label in sorted(cluster_ids.cluster_id.unique()):
    fig, summary = rs.plot_return_curves(return_curves[return_curves.cluster_id.eq(label)], side="right")
    fig.suptitle(f"Returns: {label} (equal-ant mean; 95% ant bootstrap)")
    summary.to_csv(OUTPUT / f"{label}_return_summary.csv", index=False)
    fig, summary = rs.plot_recipient_curves(recipient_curves[recipient_curves.cluster_id.eq(label)], side="right", xlim=(-30, 30))
    fig.suptitle(f"Quiet-state recipients: {label}\nContacts contribute to state labels; inspect motion responses")
    summary.to_csv(OUTPUT / f"{label}_recipient_summary.csv", index=False)
returns.to_csv(OUTPUT / "returns.csv", index=False)
triggers.to_csv(OUTPUT / "matched_triggers.csv", index=False)
matching.to_csv(OUTPUT / "matching_diagnostics.csv", index=False)
show("07_return_response")
(OUTPUT / "summary.json").write_text(json.dumps(dict(source=str(SOURCE),
    n_right_ants=len(ants), n_states=n_states, n_classes=k_roles,
    class_sizes=ants.ant_class.value_counts().sort_index().to_dict(), spatial_ari=spatial_ari,
    spatial_homogeneity=spatial_homogeneity,
    broad_ant_model=coarse_choice, broad_spatial_ari=coarse_spatial_ari,
    state_model=selection, ant_model=role_choice, quiet_states=quiet_states.tolist(),
    minimum_sleep_seconds=MIN_SLEEP_SECONDS, sleep_source="discovered state membership; no cached sleep labels",
    complete_clock_cycles=complete_cycles),indent=2)+"\n")
print(f"Finished. Figures and tables: {OUTPUT}",flush=True)
