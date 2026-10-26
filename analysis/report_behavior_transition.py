"""Summarize matched-clock, common-reference behavior and excursion changes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.compare_behavior_blocks import (
    centroids, predict_profiles, probabilities, role_coordinate, fingerprint, SEED)


def bootstrap_pair(before, after, reference, labels, nurse, forager, draws=1000, seed=SEED):
    """Resample paired clock hours and baseline ants within their existing classes.

    These intervals describe sampling sensitivity within two recordings, not
    independent experimental replication or uncertainty in the state dictionary.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    centers = np.empty((draws, len(classes), reference.shape[1]))
    for i, label in enumerate(classes):
        members = np.sqrt(reference[labels == label])
        centers[:, i] = members[rng.integers(len(members), size=(draws, len(members)))].mean(axis=1)
    hours = rng.integers(len(before), size=(draws, len(before)))
    ni, fi = np.where(classes == nurse)[0][0], np.where(classes == forager)[0][0]
    direction = centers[:, fi] - centers[:, ni]
    coordinates = []
    for counts in (before, after):
        p = probabilities(counts[hours].sum(axis=1))
        coordinate = ((np.sqrt(p)-centers[:, ni])*direction).sum(axis=1)/(direction**2).sum(axis=1)
        coordinates.append(coordinate)
    return np.column_stack(coordinates)


def spatial_bootstrap(before, after, draws=1000, seed=SEED):
    """Paired-hour changes in outside share, ignoring unobserved time."""
    rng = np.random.default_rng(seed)
    hours = rng.integers(len(before), size=(draws, len(before)))
    b, a = before[hours].sum(axis=1), after[hours].sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        delta = a[:, 0]/a[:, 1] - b[:, 0]/b[:, 1]
    return np.nanquantile(delta, [.025, .975])


def class_models(profiles, baseline, positions, variant):
    models, audit, calibration = {}, [], []
    for side in ('left', 'right'):
        train = baseline[baseline.side.eq(side)][['ant', 'ant_class']].merge(
            profiles.query('phase == "before" and scope == "full" and variant == @variant'), on='ant')
        state_columns = [c for c in train if c.startswith('S') and c[1:].isdigit()]
        p, labels = train[state_columns].to_numpy(), train.ant_class.to_numpy(int)
        classes, centers = centroids(p, labels)
        fullpos = positions.query('phase == "before" and scope == "full"')[['ant', 'outside_fraction', 'n_completed_trips']]
        calibration_table = train[['ant','ant_class']].merge(fullpos, on='ant')
        ranking = calibration_table.groupby('ant_class').outside_fraction.median().sort_values()
        nurse, forager = (int(ranking.index[0]), int(ranking.index[-1])) if len(classes) > 1 else (None, None)
        models[side] = dict(classes=classes, centers=centers, profiles=p, labels=labels,
                            nurse=nurse, forager=forager)
        for index, row in train.iterrows():
            included = train.ant.ne(row.ant).to_numpy()
            leave_classes, leave_centers = centroids(p[included], labels[included])
            label, _ = predict_profiles(p[index], leave_classes, leave_centers)
            audit.append(dict(variant=variant, ant=row.ant, side=side,
                              published_class=int(row.ant_class), leave_one_ant_out_class=int(label[0])))
        calibration_table['variant'] = variant
        calibration.extend(calibration_table.to_dict('records'))
    return models, audit, calibration


def generate_tables(root, draws):
    manifest = json.loads((root/'comparison_manifest.json').read_text())
    profiles = pd.read_csv(root/'state_profiles.csv')
    positions = pd.read_csv(root/'position_and_trip_metrics.csv')
    baseline = pd.read_csv(root/'baseline_ant_classes.csv')
    identity = pd.read_csv(root/'identity_and_coverage.csv')
    state_columns = [c for c in profiles if c.startswith('S') and c[1:].isdigit()]
    count_columns = ['count_'+c for c in state_columns]
    n_hours = profiles.scope.str.startswith('hour_').groupby(profiles.ant).sum().max()
    hour_names = sorted(profiles.loc[profiles.scope.str.startswith('hour_'),'scope'].unique())
    rows, audits, calibrations = [], [], []
    model_description = {}
    for variant in profiles.variant.unique():
        models, audit, calibration = class_models(profiles, baseline, positions, variant)
        audits.extend(audit); calibrations.extend(calibration)
        model_description[variant] = {
            side:dict(classes=m['classes'].tolist(), centers_sqrt_state_proportion=m['centers'].tolist(),
                      nurse_like_class=m['nurse'], forager_like_class=m['forager']) for side,m in models.items()}
        for ant, group in profiles[profiles.variant.eq(variant)].groupby('ant'):
            side = group.side.iloc[0]; model = models[side]
            row = dict(ant=ant, side=side, track_id=int(group.track_id.iloc[0]), variant=variant)
            for scope in ('full','matched'):
                for phase in ('before','after'):
                    subset = group[group.scope.eq(scope)&group.phase.eq(phase)]
                    if subset.empty: continue
                    record = subset.iloc[0]
                    label, distances = predict_profiles(record[state_columns].to_numpy(float), model['classes'], model['centers'])
                    row[f'{phase}_{scope}_class'] = int(label[0])
                    row[f'{phase}_{scope}_behavior_coverage'] = record.behavior_coverage
                    row[f'{phase}_{scope}_usable_seconds'] = record.usable_seconds
                    if len(model['classes']) == 2:
                        ni = np.where(model['classes']==model['nurse'])[0][0]
                        fi = np.where(model['classes']==model['forager'])[0][0]
                        row[f'{phase}_{scope}_forager_coordinate'] = float(role_coordinate(
                            record[state_columns].to_numpy(float), model['centers'], ni, fi))
            if set(group.phase)=={'before','after'} and len(model['classes']) == 2:
                hours = [group[group.phase.eq(phase)].set_index('scope').reindex(hour_names)[count_columns].fillna(0).to_numpy()
                         for phase in ('before','after')]
                boot = bootstrap_pair(*hours, model['profiles'], model['labels'], model['nurse'], model['forager'],
                                      draws=draws, seed=SEED+row['track_id'])
                for i,phase in enumerate(('before','after')):
                    row[f'{phase}_forager_coordinate_low'], row[f'{phase}_forager_coordinate_high'] = np.nanquantile(boot[:,i],[.025,.975])
                    row[f'{phase}_forager_assignment_fraction'] = float(np.mean(boot[:,i] > .5))
                row['forager_coordinate_change_low'], row['forager_coordinate_change_high'] = np.nanquantile(boot[:,1]-boot[:,0],[.025,.975])
                row['nurse_to_forager_assignment_fraction'] = float(np.mean((boot[:,0]<.5)&(boot[:,1]>.5)))
            rows.append(row)
    roles = pd.DataFrame(rows)
    roles.to_csv(root/'role_assignments_all_variants.csv',index=False)
    pd.DataFrame(audits).to_csv(root/'baseline_class_transfer_validation.csv',index=False)
    pd.DataFrame(calibrations).to_csv(root/'baseline_class_spatial_calibration.csv',index=False)
    (root/'role_classifier.json').write_text(json.dumps(model_description,indent=2)+'\n')

    # One row per identity includes original classes and the eligibility reason.
    result = identity[identity.in_comparison].merge(baseline[['ant','ant_class']].rename(columns={'ant_class':'published_before_class'}),on='ant',how='left')
    main = roles[roles.variant.eq('standard')].drop(columns=['side','track_id','variant'])
    result = result.merge(main,on='ant',how='left')
    sensitivity = roles[roles.variant.eq('without_contacts')].drop(columns=['side','track_id','variant'])
    sensitivity = sensitivity.rename(columns={c:'no_contact_'+c for c in sensitivity if c!='ant'})
    result = result.merge(sensitivity,on='ant',how='left')
    for phase in ('before','after'):
        subset = positions[positions.phase.eq(phase)&positions.scope.eq('matched')].drop(columns=['phase','scope','side','track_id'])
        result = result.merge(subset.rename(columns={c:phase+'_'+c for c in subset if c!='ant'}),on='ant',how='left')
    result['paired_high_coverage'] = (
        result.selected_before.eq(True)&result.selected_after.eq(True)
        &result.before_position_coverage.ge(.6)&result.after_position_coverage.ge(.6)
        &result.before_matched_behavior_coverage.ge(.2)&result.after_matched_behavior_coverage.ge(.2))
    result['has_paired_measurements'] = result.before_matched_usable_seconds.gt(0)&result.after_matched_usable_seconds.gt(0)
    result['outside_fraction_change'] = result.after_outside_fraction-result.before_outside_fraction
    result['aruco_outside_fraction_change'] = result.after_aruco_outside_fraction-result.before_aruco_outside_fraction
    result['strict_outside_fraction_change'] = result.after_strict_outside_fraction-result.before_strict_outside_fraction
    left_model = model_description['standard']['left']
    nurse, forager = left_model['nurse_like_class'], left_model['forager_like_class']
    result['baseline_nurse_like'] = result.side.eq('left')&result.published_before_class.eq(nurse)
    result['matched_nurse_to_forager'] = result.baseline_nurse_like & result.before_matched_class.eq(nurse) & result.after_matched_class.eq(forager)
    result['no_contact_matched_nurse_to_forager'] = (result.baseline_nurse_like
        &result.no_contact_before_matched_class.eq(nurse)&result.no_contact_after_matched_class.eq(forager))
    intervals=[]
    for row in result[result.has_paired_measurements].itertuples():
        hours = [positions[positions.ant.eq(row.ant)&positions.phase.eq(phase)].set_index('scope').reindex(hour_names)
                 for phase in ('before','after')]
        b,a = [table[['outside_seconds','observed_seconds']].fillna(0).to_numpy() for table in hours]
        lo, hi = spatial_bootstrap(b,a,draws=draws,seed=SEED+row.track_id)
        intervals.append(dict(ant=row.ant,outside_fraction_change_low=lo,outside_fraction_change_high=hi))
    result = result.merge(pd.DataFrame(intervals),on='ant',how='left')
    result['corroborated_outside_increase'] = (result.outside_fraction_change_low.gt(0)
        &result.aruco_outside_fraction_change.gt(0)&result.strict_outside_fraction_change.gt(0))
    result.sort_values(['side','track_id']).to_csv(root/'paired_ant_comparison.csv',index=False)

    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    joint_rows, joint_summary = [], []
    for side in ('left','right'):
        table = profiles.query('variant == "standard" and scope == "matched" and side == @side').copy()
        table = table.merge(identity[['ant','selected_before','selected_after']],on='ant')
        selected = np.where(table.phase.eq('before'),table.selected_before,table.selected_after)
        table = table[np.asarray(selected,dtype=bool)&table.behavior_coverage.ge(.2)].copy()
        k=len(model_description['standard'][side]['classes'])
        x=np.sqrt(table[state_columns].to_numpy())
        labels=AgglomerativeClustering(n_clusters=k,linkage='ward').fit_predict(x) if k>1 else np.zeros(len(table),int)
        table['joint_cluster']=labels
        table=table.merge(baseline[['ant','ant_class']],on='ant',how='left')
        mapping={}
        for label in np.unique(labels):
            existing=table[table.phase.eq('before')&table.joint_cluster.eq(label)].ant_class.dropna()
            mapping[int(label)] = int(existing.mode().iloc[0]) if len(existing) else -1
        table['joint_cluster_baseline_majority_class']=table.joint_cluster.map(mapping)
        original=table[table.phase.eq('before')&table.ant_class.notna()]
        joint_summary.append(dict(side=side,k=k,n_ant_days=len(table),
            baseline_agreement_ari=float(adjusted_rand_score(original.ant_class,original.joint_cluster)),
            silhouette=float(silhouette_score(x,labels)) if k>1 else None,mapping=mapping))
        joint_rows.extend(table.to_dict('records'))
    pd.DataFrame(joint_rows).to_csv(root/'joint_clusters.csv',index=False)
    (root/'joint_cluster_summary.json').write_text(json.dumps(joint_summary,indent=2)+'\n')
    return result, profiles, positions, manifest, model_description


def plots(root, result, profiles, positions, manifest, models):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import Normalize
    from datetime import datetime, timedelta
    from scipy.cluster.hierarchy import linkage, leaves_list
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    colors={'before':'#3d6eaa','after':'#d66a38'}
    directory=root/'figures';directory.mkdir(exist_ok=True)
    pdf=PdfPages(root/'role_transition_report.pdf')
    def save(fig,name):
        fig.savefig(directory/(name+'.png'),bbox_inches='tight')
        fig.savefig(directory/(name+'.svg'),bbox_inches='tight')
        pdf.savefig(fig,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,8),gridspec_kw={'width_ratios':[1,1]})
    for side,ax in zip(('left','right'),axes):
        subset=result[result.side.eq(side)&result.published_before_class.notna()&result.has_paired_measurements].sort_values('outside_fraction_change')
        y=np.arange(len(subset))
        for i,row in enumerate(subset.itertuples()):
            ax.plot([row.before_outside_fraction*100,row.after_outside_fraction*100],[i,i],color='#b8bdc3',lw=2,
                    ls='-' if row.paired_high_coverage else '--')
            for phase in ('before','after'):
                value=getattr(row,phase+'_outside_fraction')*100
                ax.scatter(value,i,c=colors[phase],s=45,marker='o' if row.paired_high_coverage else 'x',
                           label=phase if i==0 else None,zorder=3)
        labels=[f'{r.track_id:02d}'+(' •' if r.baseline_nurse_like else '')+(' †' if not r.paired_high_coverage else '') for r in subset.itertuples()]
        ax.set(yticks=y,yticklabels=labels,xlabel='Outside colony (% of observed seconds)',ylabel='Ant ID',
               title=side.capitalize()+(' colony: • baseline nurse-like' if side=='left' else ' colony: one baseline class'))
        ax.grid(axis='x',alpha=.2);ax.legend(title='Matched 12 h')
    fig.suptitle('After forager removal: did the same ants spend more time outside?',fontsize=15)
    fig.text(.5,.015,'9/15 → 9/16; matched 22:54–10:54. † lower-coverage pair; gaps excluded. Outside activity is not proof of food collection.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.96));save(fig,'01_outside_activity_by_ant')

    subset=result[result.side.eq('left')&result.published_before_class.notna()&result.has_paired_measurements].sort_values(['published_before_class','track_id'])
    fig,axes=plt.subplots(1,2,figsize=(13,8),sharey=True)
    for ax,prefix,title in zip(axes,('','no_contact_'),('Full reference model','Sensitivity: omit contact features')):
        for i,row in enumerate(subset.to_dict('records')):
            b,a=[row[prefix+phase+'_matched_forager_coordinate'] for phase in ('before','after')]
            ax.plot([b,a],[i,i],color='#abb1b8',ls='-' if row['paired_high_coverage'] else '--')
            for phase,value,shift in [('before',b,-.09),('after',a,.09)]:
                lo,hi=[row[prefix+phase+'_forager_coordinate_'+bound] for bound in ('low','high')]
                ax.plot([lo,hi],[i+shift,i+shift],color=colors[phase],lw=1.8,alpha=.65)
                ax.scatter(value,i+shift,c=colors[phase],s=30,zorder=3)
        ax.axvline(.5,c='#555',ls=':',lw=1)
        ax.set(title=title,xlabel='Position between baseline nurse (0) and forager (1) centers',
               yticks=np.arange(len(subset)),yticklabels=[f'{r.track_id:02d} — C{int(r.published_before_class)}'+(' †' if not r.paired_high_coverage else '') for r in subset.itertuples()])
        ax.grid(axis='x',alpha=.2)
    axes[0].set_ylabel('Left-colony ant ID — published 9/15 class')
    fig.suptitle('Role assignments in the fixed 9/15 class reference',fontsize=15)
    fig.text(.5,.015,'Blue: before; orange: after. Bars: 95% paired-hour / baseline-ant resampling intervals. Dotted line: assignment boundary.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.96));save(fig,'02_left_role_assignments')

    columns=[c for c in profiles if c.startswith('S') and c[1:].isdigit()]
    for side in ('left','right'):
        subset=result[result.side.eq(side)].sort_values('track_id')
        ants=subset.ant.tolist()
        frame=profiles.query('variant == "standard" and scope == "matched" and side == @side')
        pooled=frame.groupby('ant')[columns].mean().reindex(ants).to_numpy()
        order=leaves_list(linkage(np.sqrt(pooled),method='ward')) if len(ants)>1 else np.arange(len(ants))
        ants=[ants[i] for i in order]
        fig,axes=plt.subplots(1,2,figsize=(10,max(6,len(ants)*.24)),sharey=True)
        for phase,ax in zip(('before','after'),axes):
            values=frame[frame.phase.eq(phase)].set_index('ant').reindex(ants)[columns].to_numpy()
            cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#dedede')
            im=ax.imshow(values,aspect='auto',vmin=0,vmax=1,cmap=cmap)
            ax.set(xticks=np.arange(len(columns)),xticklabels=columns,yticks=np.arange(len(ants)),
                   yticklabels=[ant.split(':')[1] for ant in ants],title=phase.capitalize(),xlabel='9/15 behavior state')
        axes[0].set_ylabel('Ant ID (joint profile ordering)')
        fig.colorbar(im,ax=axes.ravel().tolist(),label='Proportion of classified time',shrink=.65)
        fig.suptitle(side.capitalize()+' colony — same behavior dictionary on both days')
        save(fig,'03_'+side+'_common_state_profiles')

    for side in ('left','right'):
        selected=result[result.side.eq(side)&result.published_before_class.notna()&result.has_paired_measurements].sort_values('outside_fraction_change',ascending=False)
        ants=selected.ant.tolist()
        hourly=positions[positions.scope.str.startswith('hour_')]
        hours=sorted(hourly.scope.unique())
        fig,axes=plt.subplots(1,2,figsize=(13,max(5,len(ants)*.3)),sharey=True)
        beginning=datetime.fromisoformat(manifest['matched_windows']['before']['start'])
        for phase,ax in zip(('before','after'),axes):
            table=hourly[hourly.phase.eq(phase)].pivot(index='ant',columns='scope',values='outside_fraction').reindex(index=ants,columns=hours)
            coverage=hourly[hourly.phase.eq(phase)].pivot(index='ant',columns='scope',values='position_coverage').reindex(index=ants,columns=hours)
            # Low observation exposure is marked, never recolored as zero activity.
            cmap=plt.get_cmap('magma').copy();cmap.set_bad('#ddd')
            im=ax.imshow(table.to_numpy()*100,aspect='auto',vmin=0,vmax=100,cmap=cmap)
            weak=np.where(coverage.to_numpy()<.6);ax.scatter(weak[1],weak[0],marker='x',c='cyan',s=15)
            labels=[(beginning+timedelta(hours=i)).strftime('%H:%M') for i in range(len(hours))]
            ax.set(xticks=np.arange(len(hours)),xticklabels=labels,yticks=np.arange(len(ants)),
                   yticklabels=[a.split(':')[1] for a in ants],title=phase.capitalize(),xlabel='Clock time (hour starts)')
            ax.tick_params(axis='x',rotation=60)
            light_on=beginning.replace(hour=5,minute=30,second=0)+timedelta(days=1)
            light_offset=(light_on-beginning).total_seconds()/3600-.5
            ax.axvline(light_offset,color='#66dfff',ls='--',lw=1.3)
        axes[0].set_ylabel('Ant ID')
        fig.suptitle(side.capitalize()+' colony — hourly outside activity, matched by clock')
        fig.text(.5,.015,'Color: outside % of observed seconds. Cyan ×: <60% position coverage. Dashed line: nominal lights on at 05:30 (unverified).',ha='center',fontsize=9)
        fig.subplots_adjust(bottom=.2,top=.89,right=.87,wspace=.1)
        cax=fig.add_axes([.89,.23,.018,.55]);fig.colorbar(im,cax=cax,label='Outside (%)')
        save(fig,'04_'+side+'_hourly_outside')

    fig,axes=plt.subplots(1,2,figsize=(11,5))
    for side,ax in zip(('left','right'),axes):
        frame=result[result.side.eq(side)]
        ax.scatter(frame.detection_fraction_before*100,frame.detection_fraction_after*100,
                   c=np.where(frame.paired_high_coverage,'#287e8e','#aaa'),s=35)
        for row in frame[frame.baseline_nurse_like].itertuples():
            ax.annotate(str(row.track_id),(row.detection_fraction_before*100,row.detection_fraction_after*100),fontsize=8)
        ax.axvline(60,c='#777',ls=':');ax.axhline(60,c='#777',ls=':')
        ax.set(xlim=(0,100),ylim=(0,100),xlabel='9/15 tracking coverage (%)',ylabel='9/16 tracking coverage (%)',title=side.capitalize())
    fig.suptitle('Tracking coverage and paired-cohort eligibility')
    fig.text(.5,.015,'Teal: primary paired cohort. Gray: lower coverage. Unobserved IDs are not confirmed removals.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.95));save(fig,'05_tracking_coverage')

    from matplotlib.colors import ListedColormap
    joint=pd.read_csv(root/'joint_clusters.csv')
    subset=result[result.side.eq('left')&result.published_before_class.notna()].sort_values(['published_before_class','track_id'])
    data=subset[['published_before_class','before_matched_class','after_matched_class']].to_numpy()
    joint_pair=joint.pivot(index='ant',columns='phase',values='joint_cluster_baseline_majority_class').reindex(subset.ant)
    data=np.column_stack([data,joint_pair[['before','after']].to_numpy()])
    cmap=ListedColormap(['#dc874e','#39748b']);cmap.set_bad('#ddd')
    fig,ax=plt.subplots(figsize=(10,7))
    ax.imshow(data,aspect='auto',vmin=0,vmax=1,cmap=cmap)
    for i in range(len(data)):
        for j in range(data.shape[1]):
            if np.isfinite(data[i,j]):ax.text(j,i,'F' if data[i,j]==0 else 'N',ha='center',va='center',color='white',fontsize=11)
    ax.set(xticks=np.arange(5),xticklabels=['Published 9/15','Reference: 9/15','Reference: 9/16','Joint: 9/15','Joint: 9/16'],
        yticks=np.arange(len(subset)),yticklabels=[f'{r.track_id:02d}'+(' †' if not r.paired_high_coverage else '') for r in subset.itertuples()],
        ylabel='Left-colony ant ID',title='Do fixed-reference assignment and joint clustering agree?')
    ax.tick_params(axis='x',rotation=18)
    fig.text(.5,.015,'N: nurse-like; F: forager-like. Gray: unavailable / excluded from joint clustering. † lower-coverage pair. See uncertainty in role plot.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.05,1,1));save(fig,'06_left_joint_cluster_comparison')
    pdf.close()


def write_report(root, result, manifest, models, draws):
    primary=result[result.paired_high_coverage]
    nurses=primary[primary.baseline_nurse_like]
    fields=['ant','paired_high_coverage','published_before_class','before_matched_class','after_matched_class',
            'before_outside_fraction','after_outside_fraction','outside_fraction_change',
            'outside_fraction_change_low','outside_fraction_change_high',
            'before_n_completed_trips','after_n_completed_trips','before_aruco_outside_fraction','after_aruco_outside_fraction',
            'no_contact_matched_nurse_to_forager','nurse_to_forager_assignment_fraction']
    summary=dict(primary_paired_by_colony=primary.groupby('side').size().to_dict(),
        primary_baseline_nurse_like_ants=int(len(nurses)),
        primary_matched_nurse_to_forager=int(nurses.matched_nurse_to_forager.sum()),
        primary_no_contact_matched_nurse_to_forager=int(nurses.no_contact_matched_nurse_to_forager.sum()),
        primary_nurses_with_corroborated_outside_increase=int(nurses.corroborated_outside_increase.sum()),
        primary_nurse_details=json.loads(nurses[fields].to_json(orient='records')),
        bootstrap_draws=draws,code=fingerprint(Path(__file__)))
    supported=nurses[nurses.matched_nurse_to_forager&nurses.no_contact_matched_nurse_to_forager&nurses.corroborated_outside_increase]
    summary['primary_nurses_with_concordant_role_and_spatial_change']=supported.ant.tolist()
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Behavior changes after forager removal — 9/15 to 9/16', '',
        'Both colonies had foragers removed. Removed-ID lists and the exact intervention time were not supplied. '
        'IDs are matched by colony side plus ArUco-derived TrackID; missing IDs are unobserved, not confirmed removals.', '',
        '## Comparison', '',
        f"The follow-up contains {manifest['duration_seconds']['after']/3600:.1f} tracked hours. "
        'Both days use the same 22:54:36–10:54:36 clock window. The 9/15 reference classes use its full 27.8-hour recording. '
        'Nominal lights on 05:30, off 19:30 follow existing report settings; actual controller transitions were not verified.', '',
        'Follow-up body-aligned postures are projected into the 9/15 PCA and classified with the frozen 9/15 normalizer '
        'and seven-state dictionary. The original 9/15 states are retained. Ant-day state proportions are assigned to '
        'the nearest original class centroid in square-root proportion space. Spatial measurements do not enter role assignment. '
        'A separate Ward clustering pools both days at the baseline number of classes. Contacts are omitted in a sensitivity analysis.', '',
        f"Primary paired cohort: {summary['primary_paired_by_colony']}. This requires original tracking coverage >60% on both days, "
        '≥60% observed-second position coverage and ≥20% usable behavior coverage in each matched interval. '
        'Other ants are retained as secondary results and marked with † in plots.', '',
        '## Nurse-like ants', '',
        'The left colony supports two baseline classes: nurse-like/nest-associated and forager-like, named by independent '
        'outside activity. This does not verify brood care or food collection. The right colony has only one supported '
        'baseline class, so a right-colony nurse-to-forager class transition is not identifiable.', '',
        f"Among {len(nurses)} primary paired left-colony baseline nurse-like ants, "
        f"{int(nurses.matched_nurse_to_forager.sum())} change from nurse-like to forager-like in matched-clock profiles; "
        f"{int(nurses.no_contact_matched_nurse_to_forager.sum())} do so when contact features are omitted. "
        f"{int(nurses.corroborated_outside_increase.sum())} show a positive outside-share interval with concordant direct-ArUco "
        'and stricter observation-coverage changes.', '',
        ]
    for row in supported.itertuples():
        lines += [f'**{row.ant} is the clearest concordant candidate:** observed outside time rises from '
            f'{row.before_outside_fraction*100:.2f}% to {row.after_outside_fraction*100:.2f}%, '
            f'with {row.before_n_completed_trips:.0f} → {row.after_n_completed_trips:.0f} completed trips. '
            f'The outside-share increase is {row.outside_fraction_change*100:.1f} percentage points '
            f'(paired-hour 95% interval {row.outside_fraction_change_low*100:.1f}–{row.outside_fraction_change_high*100:.1f}). '
            f'Direct-tag outside share rises from {row.before_aruco_outside_fraction*100:.2f}% '
            f'to {row.after_aruco_outside_fraction*100:.2f}%. The nurse-to-forager assignment occurs in '
            f'{row.nurse_to_forager_assignment_fraction*100:.1f}% of reference-model resamples and '
            f'{row.no_contact_nurse_to_forager_assignment_fraction*100:.1f}% when contact features are omitted. '
            'The contact-free point estimate crosses the boundary, but its interval should be read from the role plot.', '']
    lines += ['| Ant | Outside before → after | Completed trips | Role before → after | Primary pair |',
              '|---|---:|---:|---|---|']
    for row in result[result.baseline_nurse_like].sort_values('outside_fraction_change',ascending=False).itertuples():
        lines.append(f'| {row.ant} | {row.before_outside_fraction*100:.1f}% → {row.after_outside_fraction*100:.1f}% | '
                     f'{row.before_n_completed_trips:.0f} → {row.after_n_completed_trips:.0f} | '
                     f'C{row.before_matched_class:.0f} → C{row.after_matched_class:.0f} | {row.paired_high_coverage} |')
    lines += ['', '## Right colony and validation', '',
        'Right-colony increases mostly involve ants already making excursions at baseline. '
        'The largest increases among primary paired ants are:']
    for row in primary[primary.side.eq('right')].nlargest(3,'outside_fraction_change').itertuples():
        lines.append(f'- {row.ant}: {row.before_outside_fraction*100:.1f}% → {row.after_outside_fraction*100:.1f}% observed time outside.')
    audit=pd.read_csv(root/'baseline_class_transfer_validation.csv')
    standard=audit[audit.variant.eq('standard')&audit.side.eq('left')]
    agreement=pd.read_csv(root/'state_assignment_audit.csv').exact_vs_published_agreement.dropna()
    lines += ['', f'The nearest-centroid extension reproduces {int(standard.published_class.eq(standard.leave_one_ant_out_class).sum())}'
        f'/{len(standard)} baseline left-colony labels in leave-one-ant-out validation. Exact state transfer agrees with '
        f'the published baseline assignments on {agreement.mean()*100:.2f}% of windows on average; original baseline labels '
        'are retained. Joint clustering uses both days and is aligned by baseline class majority; it is a separate sensitivity '
        'analysis and can reassign some baseline ants. See `06_left_joint_cluster_comparison.png`.',
        '', '## Interpretation limits', '',
        'Outside time uses observed one-second bins, without filling missing positions. Completed trips require an exit and '
        'return within the comparison window, ≥30 seconds outside, ≥5-second colony anchors, and the existing pipeline gap/coverage '
        'rules. Long ongoing excursions can raise outside time while lowering the number of completed trips. The stricter position '
        'sensitivity requires ≥12 observed frames per second. Direct ArUco locations provide an independent tag-detection check.', '',
        f'Intervals resample matched one-hour blocks and (for class assignment) baseline ants within each class ({draws} draws). '
        'They measure sensitivity within these recordings, not independent colony replication or uncertainty in the state dictionary. '
        'The baseline state dictionary itself failed some automated stability criteria; these behavioral labels are exploratory. '
        'Both colonies were manipulated, there is no untreated control, and the follow-up is only 12 hours. '
        'Changes are consistent with role plasticity only when class and spatial evidence agree; they do not establish a causal treatment effect.', '',
        '## Files', '',
        '- `role_transition_report.pdf`: all figures.',
        '- `paired_ant_comparison.csv`: identity-matched effects, coverage, role changes, uncertainty, direct-tag checks.',
        '- `role_assignments_all_variants.csv`: matched/full-period transfer and contact sensitivity.',
        '- `joint_clusters.csv`: pooled clustering of ant-day profiles using the baseline number of classes.',
        '- `state_profiles.csv`, `position_and_trip_metrics.csv`: full, matched, and hourly measurements.',
        '- `identity_and_coverage.csv`: all tracked IDs, including unobserved identities and selection flags.',
        '- `followup_all_left_ants/ALL_18_LEFT_ANTS_OCCUPANCY.pdf`: matched-window spatial occupancy maps for all 18 baseline left-class ants.',
        '- `followup_all_left_ants/ALL_18_LEFT_ANTS.md`: all-cohort explanation, counts, and per-ant role table.',
        '- `comparison_manifest.json`, `role_classifier.json`, `reference_model/`: reproducible timing and model provenance.', '']
    (root/'REPORT.md').write_text('\n'.join(lines))
    (root/'report_complete.ok').write_text('completed tables, uncertainty, and plots\n')
    print(json.dumps(summary,indent=2))


def run(root,draws=1000):
    if not (root/'measurement_complete.ok').is_file():
        raise RuntimeError('Measurements are incomplete')
    result,profiles,positions,manifest,models=generate_tables(root,draws)
    plots(root,result,profiles,positions,manifest,models)
    write_report(root,result,manifest,models,draws)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--bootstrap',type=int,default=1000)
    args=parser.parse_args()
    run(args.root,args.bootstrap)
