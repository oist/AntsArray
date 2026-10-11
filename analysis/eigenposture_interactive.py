# %% 1. Inputs: both colonies for states, right colony for ant classes
"""Full-body behavior -> pooled state proportions -> ant classes -> sleep/space.

Run these eleven cells in VS Code/Jupyter. Expensive full-track preprocessing and
method comparisons are produced by full_behavior_extract/states/roles.py.
Every eligible ant's usable segments are assigned; no hourly clip sampling.
"""
from pathlib import Path
import os
import sys
import json
import hashlib
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
from analysis import trip_phenotyping_utils as trip_go
from analysis.full_behavior_extract import SEED,HZ,FPS
from analysis.full_behavior_states import NAMES,encode
from analysis.full_behavior_roles import fit_labels,compare_ant_umaps
from analysis.full_behavior_sleep import build as build_state_sleep,select_quiet_states,contact_history_audit
from analysis.full_behavior_profiles import build as build_state_means

BLOCK=Path(os.environ.get('ANTS_DATASET_ROOT','/home/sam-reiter/bucket/ReiterU/Ants/basler/20260724/block01'))
SOURCE=Path(os.environ.get('ANTS_BEHAVIOR_SOURCE',str(BLOCK/'analysis_outputs/full_body_behavior_20261011')))
OUTPUT=Path(os.environ.get('ANTS_ROLE_OUTPUT',str(SOURCE/'report')))
LIGHT_ON,LIGHT_OFF=5.5,19.5
# Explicit POST HOC interpretation: S0 alone is putative sleep. S1 stays a
# separate non-sleep state. None opts into motion-profile candidates instead.
SLEEP_STATES=[0]
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
    yticks=np.arange(n_states),yticklabels=state_names,title='4a. State feature profiles; * = uncertain ant-removal match')
ax.tick_params(axis='x',rotation=90);fig.colorbar(image,ax=ax,label='Median feature / family scale')
show('04_state_profiles')
# Actual observed mean skeletons, not a reconstruction from retained PCs. The
# compact summary pools ALL assigned observations from both colonies. Each 12-Hz
# sample has equal weight: a short tail contributes only its observed duration.
# Build once on local extraction output; published packets carry this summary.
if not (SOURCE/'models/state_means.npz').exists():build_state_means(SOURCE)
mean_metadata=json.loads((SOURCE/'models/state_means.json').read_text())
assert mean_metadata['dictionary_sha256']==hashlib.sha256((SOURCE/'models/dictionary.npz').read_bytes()).hexdigest()
with np.load(SOURCE/'models/state_means.npz') as archive:state_means=dict(archive)
mean_pose,mean_features=state_means['mean_pose'],state_means['mean_features']
assert mean_pose.shape==(n_states,10,2) and mean_features.shape==(n_states,7)
np.testing.assert_array_equal(state_means['ants'],cohort.ant.to_numpy(str))
mean_table=pd.DataFrame(mean_features,columns=state_means['feature_names'])
mean_table.insert(0,'observed_hours',state_means['sample_count']/HZ/3600)
mean_table.insert(0,'state',np.arange(n_states));mean_table.to_csv(OUTPUT/'state_mean_features.csv',index=False)
pd.DataFrame(mean_pose.reshape(n_states,20),index=state_names,
    columns=[f'{node}_{axis}' for node in landmarks for axis in ('anterior','lateral')]).to_csv(OUTPUT/'state_mean_postures.csv',index_label='state')
columns=min(4,n_states);rows=int(np.ceil(n_states/columns))
fig=plt.figure(figsize=(14,2.2*rows+6))
grid=fig.add_gridspec(rows+1,columns,height_ratios=[1]*rows+[2.6],
    left=.08,right=.95,bottom=.18,top=.92,hspace=.35,wspace=.3)
xlim=(mean_pose[:,:,0].min()-.35,mean_pose[:,:,0].max()+.35)
ylim=(mean_pose[:,:,1].min()-.45,mean_pose[:,:,1].max()+.35)
for s in range(n_states):
    ax=fig.add_subplot(grid[s//columns,s%columns]);shape=mean_pose[s]
    for chain,color in [([3,2,0,1],'#176b78'),([1,4,5,6],'#d87b16'),([1,7,8,9],'#d87b16')]:
        ax.plot(*shape[chain].T,'o-',color=color,lw=2,markersize=3)
    fraction=state_means['sample_count'][s]/state_means['sample_count'].sum()
    ax.set(title=f'{state_names[s]}  |  {fraction:.1%} of observed time',xlim=xlim,ylim=ylim,aspect='equal')
    ax.axis('off')
    if s==0:
        x,y=xlim[0]+.15,ylim[0]+.18
        ax.plot([x,x+1],[y,y],color='black',lw=2)
        ax.text(x+.5,y-.12,'1 body-axis length',ha='center',va='top',fontsize=7)
# A common raw color scale would mix mm/s, rad/s and partner counts. Normalize
# each column between its smallest/largest state mean, printing the raw values.
# These are pooled means, unlike the ant-balanced dictionary medians above.
scaled=(mean_features-mean_features.min(axis=0))/np.maximum(np.ptp(mean_features,axis=0),1e-12)
ax=fig.add_subplot(grid[-1,:]);image=ax.imshow(scaled,aspect='auto',cmap='viridis',vmin=0,vmax=1)
feature_labels=['Forward peak\n(mm/s)','Lateral peak\n(mm/s)','Antenna A mean\n(mm/s)',
    'Antenna B mean\n(mm/s)','Turning mean\n(rad/s)','Contact mean\n(partners)','Contact peak\n(partners)']
ax.set(xticks=np.arange(7),xticklabels=feature_labels,yticks=np.arange(n_states),yticklabels=state_names)
for s in range(n_states):
    for j in range(7):ax.text(j,s,f'{mean_features[s,j]:.3g}',ha='center',va='center',color='black' if scaled[s,j]>.5 else 'white')
ax.axvline(4.5,color='white',lw=2)
fig.colorbar(image,ax=ax,pad=.02,label='Within-feature scale: lowest → highest state mean')
fig.suptitle('4b. Mean posture and velocity / interaction features by behavioral state',fontsize=14)
fig.text(.08,.065,'Postures: body in teal, antennae in orange; anterior points right. All panels use the same body-normalized scale.\n'
    'Heatmap numbers: observed-duration-weighted means. Forward/lateral speeds and peak contacts are maxima within ≤1 s windows.\n'
    f"Both colonies: {len(cohort)} ants; {state_means['sample_count'].sum():,} assigned posture observations. Color is scaled separately for each feature.",fontsize=9,linespacing=1.6)
show('04_mean_posture_motion')
quiet_states,motion_profiles=select_quiet_states(SOURCE)
if SLEEP_STATES is not None:quiet_states=np.asarray(SLEEP_STATES,int)
motion_profiles['putative_sleep']=motion_profiles.state.isin(quiet_states)
motion_profiles.to_csv(OUTPUT/'state_motion_profiles.csv',index=False)
print('Putative-sleep states:',quiet_states.tolist(),'\n',motion_profiles.to_string(index=False),flush=True)
fig,ax=plt.subplots(figsize=(6,4),layout='constrained')
for row in motion_profiles.itertuples():
    ax.scatter(row.body_peak_median_mm_s,row.antenna_median_mm_s,color=colors[row.state],s=100 if row.putative_sleep else 65,
        edgecolor='black' if row.putative_sleep else 'none',linewidth=1.5)
    ax.annotate(state_names[row.state],(row.body_peak_median_mm_s,row.antenna_median_mm_s),fontsize=8)
ax.axvline(.05,color='.5',ls=':');ax.axhline(.15,color='.5',ls=':')
ax.set(xscale='log',yscale='log',xlabel='Median unsigned body peak (mm/s)',
    ylabel='Median unsigned antennal speed (mm/s)',title=f'Putative sleep: {quiet_states.tolist()} (black outline); thresholds are descriptive')
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
class_colors=plt.get_cmap('tab10')(np.arange(k_roles))
# Each dot is an ANT represented by its eight state proportions. The selected
# clustering above uses all sqrt(proportion) coordinates; UMAP is only a view.
umap_scores,role_layouts,selected_view=compare_ant_umaps(role_x)
umap_scores.to_csv(OUTPUT/'ant_umap_parameter_comparison.csv',index=False)
np.savez(OUTPUT/'ant_umap_layouts.npz',ants=ants.ant.to_numpy(str),x=role_x,**role_layouts)
role_view=role_layouts[selected_view]
view_choice=umap_scores.set_index('key').loc[selected_view]
class_profiles=np.array([proportions[ants.ant_class.eq(c)].mean(axis=0) for c in range(k_roles)])
pd.DataFrame(class_profiles,index=[f'C{c}' for c in range(k_roles)],columns=state_names).to_csv(OUTPUT/'class_state_profiles.csv',index_label='ant_class')
ants[['ant','ant_class']].assign(umap_1=role_view[:,0],umap_2=role_view[:,1]).to_csv(OUTPUT/'ant_class_umap.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(13,5))
for c in range(k_roles):
    keep=ants.ant_class.eq(c).to_numpy()
    axes[0].scatter(*role_view[keep].T,s=55,color=class_colors[c],label=f'C{c}: {keep.sum()} ants')
for i,row in enumerate(ants.itertuples()):axes[0].annotate(f'{row.track_id:02d}',role_view[i],xytext=(4,3),textcoords='offset points',fontsize=7)
axes[0].set(xlabel='UMAP 1',ylabel='UMAP 2',title='Ants grouped by their state proportions');axes[0].legend(fontsize=8)
image=axes[1].imshow(class_profiles*100,aspect='auto',cmap='viridis',vmin=0,vmax=max(1,class_profiles.max()*100))
for c in range(k_roles):
    for s in range(n_states):axes[1].text(s,c,f'{class_profiles[c,s]:.0%}',ha='center',va='center',fontsize=9,
        color='black' if class_profiles[c,s]>.5*class_profiles.max() else 'white')
axes[1].set(xticks=np.arange(n_states),xticklabels=state_names,yticks=np.arange(k_roles),
    yticklabels=[f'C{c} (n={ants.ant_class.eq(c).sum()})' for c in range(k_roles)],title='Mean state proportions within each ant class')
fig.colorbar(image,ax=axes[1],label='Class mean of ant state proportions (%)')
fig.suptitle(f'6a. {n_states} behavioral states → ant state proportions → {k_roles} behavioral ant classes')
fig.text(.08,.02,f"Fit: {role_choice['model']} in all {n_states} square-root proportion dimensions. UMAP: n={int(view_choice.n_neighbors)}, min_dist={view_choice.min_dist}; display only.",fontsize=9)
fig.tight_layout(rect=(0,.055,1,.95));show('06_state_to_ant_classes')
fig,axes=plt.subplots(3,3,figsize=(12,10))
for ax,((n,d),group) in zip(axes.flat,umap_scores.groupby(['n_neighbors','min_dist'],sort=True)):
    xy=role_layouts[f'n{n}_d{d}_seed{SEED}']
    for c in range(k_roles):
        ax.scatter(*xy[ants.ant_class.eq(c)].T,s=25,color=class_colors[c],label=f'C{c}')
    chosen=int(view_choice.n_neighbors)==n and view_choice.min_dist==d
    ax.set(title=f'n={n}, min_dist={d}'+('  [selected]' if chosen else '')+
        f'\nMedian trust={group.trustworthiness.median():.3f}, 5-NN overlap={group.neighbor_overlap.median():.0%}',
        xticks=[],yticks=[])
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.045),ncol=k_roles)
fig.suptitle('6b. UMAP views of the same behavioral ant classes')
fig.text(.06,.015,'Three seeds per setting; panels show the fixed primary seed. Selection preserves input neighborhoods, without using class/spatial labels.',fontsize=9)
fig.tight_layout(rect=(0,.09,1,.96));show('06_ant_umap_comparison')
ordered=ants.sort_values(['ant_class','track_id']).index.to_numpy()
fig,axes=plt.subplots(1,2,figsize=(14,9),layout='constrained')
im=axes[0].imshow(proportions[ordered],aspect='auto',cmap='viridis',vmin=0)
axes[0].set(xticks=np.arange(n_states),xticklabels=state_names,yticks=np.arange(len(ants)),
    yticklabels=[f"C{ants.loc[a,'ant_class']} / {ants.loc[a,'track_id']:03d}" for a in ordered],title='6c. Ant state proportions')
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
# Audit successive support requirements, with each count divided by its own
# frame grid. 12-Hz sampling therefore does NOT artificially halve coverage.
extraction=pd.DataFrame([dict(ant=row['ant'],**{k:row['extraction'][k] for k in
    ['total_samples','valid_pose_samples','valid_motion_samples']}) for row in mean_metadata['inputs']])
coverage=cohort[['ant','side','n_frames','detection_fraction']].merge(extraction,on='ant',validate='one_to_one').merge(
    pd.read_csv(SOURCE/'feature_coverage.csv'),on='ant',validate='one_to_one')
assert len(coverage)==len(cohort)
stages=['detection_fraction','posture_fraction','motion_fraction','contact_fraction','assigned_fraction']
for target,column in zip(stages[1:],['valid_pose_samples','valid_motion_samples','n_observed_samples','n_assigned_samples']):
    coverage[target]=coverage[column]/coverage.total_samples
assert (np.diff(coverage[stages[1:]].to_numpy(),axis=1)<=0).all()
np.testing.assert_array_equal(coverage.set_index('ant').loc[ants.ant,'n_assigned_samples'],counts.sum(axis=1))
coverage.to_csv(OUTPUT/'state_coverage_stages.csv',index=False)
coverage_summary=coverage.groupby('side')[stages].mean()
coverage_summary.to_csv(OUTPUT/'state_coverage_summary.csv')
fig,ax=plt.subplots(figsize=(11,5))
for i,(side,values) in enumerate(coverage_summary.iterrows()):
    bars=ax.bar(np.arange(len(stages))+(i-.5)*.36,values*100,width=.36,label=f'{side}: {cohort.side.eq(side).sum()} ants')
    ax.bar_label(bars,fmt='%.1f%%',fontsize=9)
ax.set(xticks=np.arange(len(stages)),xticklabels=['Detected position','Valid full-body\nposture','+ Valid motion','+ Contacts\navailable','State assigned\n(≥250 ms segments)'],
    ylabel='Mean ant coverage (% of full recording)',ylim=(0,100),title='6d. Where coverage is lost before state assignment')
ax.legend()
fig.text(.08,.025,'Posture: all 10 landmarks, geometry checks, five valid 24-Hz frames, one camera.\n'
    'Motion: finite derivatives, matching position/pose camera, speed ≤20 mm/s. Short segments are removed last.\n'
    'All stages are normalized to full recording time; missing observations never count as a behavioral state.',fontsize=9)
fig.tight_layout(rect=(0,.16,1,1));show('06_state_coverage')
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
# Direct full-recording ant summaries: count all classified samples for sleep,
# all finite trajectory-cache samples for speed, then give each ant equal weight.
# Never average hourly fractions to estimate an ant's overall sleep fraction.
ant_summary=ants[['ant','side','track_id','track_name','ant_class']].merge(
    pd.read_csv(sleep_root/'ant_quiet_summary.csv')[['side','track_id','classified_seconds','quiet_seconds']],
    on=['side','track_id'],validate='one_to_one')
ant_summary['sleep_fraction']=ant_summary.quiet_seconds/ant_summary.classified_seconds
if MIN_SLEEP_SECONDS==0:np.testing.assert_allclose(ant_summary.sleep_fraction,proportions[:,quiet_states].sum(axis=1))
speed_means=[]
for row in ant_summary.itertuples():
    folder=stitched/'speed_vectors/per_track'/Path(row.track_name).stem
    vector=np.load(folder/'speed_mm_s.npy',mmap_mode='r');finite=np.isfinite(vector)
    speed_means.append(dict(ant=row.ant,mean_speed_mm_s=float(np.mean(vector[finite],dtype=np.float64)),
        valid_speed_seconds=int(finite.sum())/FPS))
ant_summary=ant_summary.merge(pd.DataFrame(speed_means),on='ant',validate='one_to_one')
ant_summary.to_csv(OUTPUT/'ant_sleep_speed_summary.csv',index=False)
class_summary=ant_summary.groupby('ant_class').agg(n_ants=('ant','size'),
    sleep_fraction=('sleep_fraction','mean'),mean_speed_mm_s=('mean_speed_mm_s','mean'))
class_summary.to_csv(OUTPUT/'class_sleep_speed_summary.csv')
fig,axes=plt.subplots(2,2,figsize=(13,8));rng=np.random.default_rng(SEED)
for c in range(k_roles):
    group=ant_summary[ant_summary.ant_class.eq(c)];color=class_colors[c]
    for ax,column,factor in [(axes[0,0],'sleep_fraction',100),(axes[0,1],'mean_speed_mm_s',1)]:
        value=group[column].to_numpy()*factor
        ax.scatter(c+rng.uniform(-.13,.13,len(value)),value,color=color,s=28,alpha=.8)
        ax.plot([c-.23,c+.23],[value.mean()]*2,color='black',lw=2)
    curve=sleep[sleep.cluster_id.eq(f'right_behavior_{c}')]
    axes[1,0].plot(curve.time_h,curve.mean_sleep_fraction*100,color=color,label=f'C{c}')
    curve=speed[speed.leiden_cluster.eq(c)]
    axes[1,1].plot(curve.time_h,curve.mean_speed_mm_s,color=color,label=f'C{c}')
for ax in axes[0]:ax.set(xticks=np.arange(k_roles),xticklabels=[f'C{c}\nn={len(ant_summary[ant_summary.ant_class.eq(c)])}' for c in range(k_roles)])
axes[0,0].set(ylabel='Putative sleep (% of classified time)',ylim=(0,100),title='Each point is one ant; bars are equal-ant means')
axes[0,1].set(ylabel='Mean trajectory speed (mm/s)',ylim=(0,None),title='Full-recording speed by behavioral ant class')
axes[1,0].set(ylabel='Putative sleep (%)',ylim=(0,100));axes[1,1].set(ylabel='Mean trajectory speed (mm/s)',ylim=(0,None))
for ax in axes[1]:
    go.add_light_shading(ax,clock,stop/FPS/3600,light_on_hour=LIGHT_ON,light_off_hour=LIGHT_OFF)
    ax.set(xlabel=f'Elapsed hours from {go.format_clock_time(clock)}');ax.legend(ncol=k_roles,fontsize=8)
fig.suptitle('11a. Putative sleep and speed by behavioral ant class')
fig.text(.08,.025,f"Sleep = states {quiet_states.tolist()}; unknown time is excluded. This describes classes fitted from the same state proportions.\n"
    'Curves: 10-min equal-ant means; sleep bins require ≥50% classified coverage. Points pool all valid samples for each ant.\n'
    'Speed uses the existing trajectory cache (not within-window peaks); dark intervals are shaded.',fontsize=9)
fig.tight_layout(rect=(0,.13,1,.96));show('05_class_sleep_speed')
sma.plot_cluster_sleep_timeseries(sleep, start_clock_seconds=clock).suptitle("11b. Putative sleep from behavioral states, by ant class")
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

# %% 10. Grid-occupancy trip investment, retaining all behavioral ant classes
# Reuse its trip rules: 1-s position bins, bridge state gaps <=30 s, remove
# <5-s border flicker, >=5-s colony anchors, >=30-s trips, >=20% observed bins.
# These position-gap rules apply ONLY to trips; sleep gaps remain unknown.
# Include zero-trip ants in rates/time totals; their median duration is undefined.
trip_tracks=cluster_table.assign(frame_min=start,frame_max=stop-1)
completed_trips,trip_positions,trip_diagnostics=trip_go.load_or_extract_completed_trips(
    trip_tracks,regions,stitched/'per_track',SOURCE/'trips',fps=FPS,start_clock_seconds=clock,
    bodypoint=0,position_bin_seconds=1,max_state_gap_seconds=30,min_state_run_seconds=5,
    min_colony_anchor_seconds=5,min_trip_seconds=30,min_trip_coverage=.2,max_path_gap_seconds=3,max_workers=4)
trip_summary=trip_go.summarize_trip_candidates(trip_tracks,completed_trips,trip_positions,fps=FPS)
investment=trip_go.compute_trip_investment_confidence(trip_summary,completed_trips,fps=FPS,n_bootstrap=2000,random_state=SEED)
investment=investment.merge(ants[['side','track_id','ant_class']],on=['side','track_id'],validate='one_to_one').merge(
    trip_summary[['track_name','total_trip_minutes','observed_trip_hours']],on='track_name',validate='one_to_one')
investment['trip_minutes_per_day']=investment.total_trip_minutes/investment.recording_days
investment.to_csv(OUTPUT/'ant_trip_investment.csv',index=False)
trip_summary.to_csv(OUTPUT/'trip_summary.csv',index=False)
completed_trips.to_csv(OUTPUT/'completed_trips.csv',index=False)
trip_diagnostics.to_csv(OUTPUT/'trip_extraction_diagnostics.csv',index=False)
investment.groupby('ant_class').agg(n_ants=('track_id','size'),n_completed_trips=('n_completed_trips','sum'),
    completed_trips_per_day=('completed_trips_per_day','mean'),trip_minutes_per_day=('trip_minutes_per_day','mean'),
    mean_ant_median_trip_minutes=('median_trip_minutes','mean')).to_csv(OUTPUT/'class_trip_investment.csv')
fig,axes=plt.subplots(1,3,figsize=(16,5));rng=np.random.default_rng(SEED)
for c in range(k_roles):
    group=investment[investment.ant_class.eq(c)];valid=group[group.n_completed_trips>0]
    x,y=valid.completed_trips_per_day,valid.median_trip_minutes
    axes[0].errorbar(x,y,xerr=np.vstack([x-valid.completed_trips_per_day_ci_low,valid.completed_trips_per_day_ci_high-x]),
        yerr=np.vstack([y-valid.median_trip_minutes_ci_low,valid.median_trip_minutes_ci_high-y]),fmt='o',
        color=class_colors[c],alpha=.75,ms=4,elinewidth=.7,label=f'C{c}')
    for ax,column in zip(axes[1:],['completed_trips_per_day','trip_minutes_per_day']):
        values=group[column].to_numpy();ax.scatter(c+rng.uniform(-.12,.12,len(values)),values,color=class_colors[c],s=25)
        ax.plot([c-.23,c+.23],[values.mean()]*2,color='black',lw=2)
axes[0].set(xlabel='Completed returns / recording day',ylabel='Median time per trip (min)',yscale='log',
    title='Grid-occupancy trip frequency versus duration');axes[0].legend()
for ax,label in zip(axes[1:],['Completed returns / recording day','Time on completed trips (min / recording day)']):
    ax.set(xticks=np.arange(k_roles),xticklabels=[f'C{c}\nn={ants.ant_class.eq(c).sum()}' for c in range(k_roles)],ylabel=label,ylim=(0,None))
axes[1].set_title('Each point is one ant');axes[2].set_title('Black bars: equal-ant class means')
fig.suptitle('13. Return rate and time spent on trips by behavioral ant class')
fig.text(.06,.025,f'Completed colony → outside → colony trips; ≥30 s outside, ≥5 s colony anchors. All {len(ants)} classified ants enter rates and time totals.\n'
    'Left: Poisson rate / bootstrap median 95% intervals; zero-trip ants have no duration estimate. Trip time includes bridged position gaps ≤30 s.\n'
    'Rates use full recording time, including missing observations; these trip rules are more permissive than the precise return-response event rules.',fontsize=9)
fig.tight_layout(rect=(0,.17,1,.95));show('07_trip_investment')

# %% 11. Returns and sleeping-recipient responses, retaining the same classes
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
contact_history_audit(sleep_tracks,bouts,settings.fps,settings.prior_sleep_seconds).to_csv(OUTPUT/'recipient_sleep_support.csv',index=False)
rs.attach_contact_context(contexts, bouts, coverage, settings)
all_classes = all_sleep[["side", "track_id"]].merge(cluster_ids[["side", "track_id", "cluster_id"]],
    on=["side", "track_id"], how="left", validate="one_to_one").fillna({"cluster_id": "unclassified_context"})
all_returns = rs.extract_returns(contexts, all_classes, settings)
returns = all_returns[all_returns.cluster_id.isin(cluster_ids.cluster_id)].copy()
return_curves, latency = rs.return_activity_curves(returns, contexts, settings)
contacts = rs.eligible_sleeping_contacts(sleep_tracks, contexts, bouts, all_returns, settings)
triggers, matching = rs.match_sleeping_controls(contacts, contexts, settings)
recipient_curves = rs.trigger_state_curves(triggers, sleep_tracks, contexts, settings)
# Retain the pooled return view so class splitting can be distinguished from
# losses caused by recipient eligibility. Return events do not depend on sleep.
fig,summary=rs.plot_return_curves(return_curves,side='right')
fig.suptitle(f'All classified right ants: {len(returns)} returns from {returns.track_id.nunique()} ants\n'
    'Equal-ant means; same return criteria as the class panels')
summary.to_csv(OUTPUT/'pooled_return_summary.csv',index=False)
# Pooled recipients make the available evidence visible before class splitting.
# Counts are matched return-contact events; contributors can fall with lag.
exposed=triggers[triggers.condition.eq('Recent return contact')]
fig,summary=rs.plot_recipient_curves(recipient_curves,side='right',xlim=(-30,30))
fig.suptitle(f'All classified right ants: {len(exposed)} matched return contacts in {exposed.track_id.nunique()} recipients\n'
    f'Putative sleep = {quiet_states.tolist()}; contact features contribute to labels; 95% intervals require ≥5 contributing ants')
summary.to_csv(OUTPUT/'pooled_recipient_summary.csv',index=False)
# The canonical plots pool classes; explicitly subset here so class membership
# survives all the way to the return and recipient comparisons.
for label in sorted(cluster_ids.cluster_id.unique()):
    fig, summary = rs.plot_return_curves(return_curves[return_curves.cluster_id.eq(label)], side="right")
    events=returns[returns.cluster_id.eq(label)]
    fig.suptitle(f"Returns: {label} — {len(events)} returns, {events.track_id.nunique()} ants (equal-ant mean; 95% ant bootstrap)")
    summary.to_csv(OUTPUT / f"{label}_return_summary.csv", index=False)
    fig, summary = rs.plot_recipient_curves(recipient_curves[recipient_curves.cluster_id.eq(label)], side="right", xlim=(-30, 30))
    events=exposed[exposed.cluster_id.eq(label)]
    fig.suptitle(f"Quiet-state recipients: {label} — {len(events)} matched contacts, {events.track_id.nunique()} ants\n"
        "Contacts contribute to state labels; 95% intervals require ≥5 contributing ants")
    summary.to_csv(OUTPUT / f"{label}_recipient_summary.csv", index=False)
returns.to_csv(OUTPUT / "returns.csv", index=False)
triggers.to_csv(OUTPUT / "matched_triggers.csv", index=False)
matching.to_csv(OUTPUT / "matching_diagnostics.csv", index=False)
sample_index=pd.MultiIndex.from_product([sorted(cluster_ids.cluster_id.unique()),
    ['Recent return contact','No contact control','Other contact']],names=['cluster_id','condition'])
triggers.groupby(['cluster_id','condition']).agg(n_events=('track_id','size'),n_ants=('track_id','nunique')).reindex(
    sample_index,fill_value=0).to_csv(OUTPUT/'recipient_sample_counts.csv')
show("07_return_response")
(OUTPUT / "summary.json").write_text(json.dumps(dict(source=str(SOURCE),
    n_right_ants=len(ants), n_states=n_states, n_classes=k_roles,
    class_sizes=ants.ant_class.value_counts().sort_index().to_dict(), spatial_ari=spatial_ari,
    spatial_homogeneity=spatial_homogeneity,
    broad_ant_model=coarse_choice, broad_spatial_ari=coarse_spatial_ari,
    ant_umap_view=selected_view, n_completed_trips=len(completed_trips),
    state_model=selection, ant_model=role_choice, quiet_states=quiet_states.tolist(),
    minimum_sleep_seconds=MIN_SLEEP_SECONDS, sleep_source="discovered state membership; no cached sleep labels",
    complete_clock_cycles=complete_cycles),indent=2)+"\n")
print(f"Finished. Figures and tables: {OUTPUT}",flush=True)
