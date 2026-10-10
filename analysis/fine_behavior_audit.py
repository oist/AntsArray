"""Additional ant-removal checks of the frozen dictionary; never reselect it."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import leidenalg as la
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits
from analysis.fine_behavior_features import encode,WIDTHS
from analysis.fine_behavior_states import SEED,neighbor_weights,vote,matched_jaccard
from analysis.eigenposture_leiden import graph_from_features,fit_role_model


def audit(features,models,report,repeats=5,wavelet_features=None,plots_only=False):
    frozen=models/'selection_frozen.json';before=hashlib.sha256(frozen.read_bytes()).hexdigest()
    selected=json.loads(frozen.read_text());wi=WIDTHS.index(selected['width'])
    with np.load(features/'features.npz') as z:data=dict(z)
    with np.load(models/'split_rows.npz') as z:train,test=z['train'],z['test']
    with np.load(models/'states.npz') as z:reference=z['state'][test]
    ants=pd.read_csv(models/'ant_split.csv')
    rows=pd.read_csv(models/'stability_audit.csv').to_dict('records') if plots_only else []
    if plots_only and len(rows)!=repeats:raise ValueError('Incomplete stability audit')
    for repeat in range(0 if plots_only else repeats):
        rng=np.random.default_rng(SEED+500+repeat);retained=[]
        for side in ('left','right'):
            ids=np.flatnonzero(ants.side.eq(side).to_numpy()&ants.split.eq('train').to_numpy())
            retained.extend(rng.choice(ids,max(1,int(.8*len(ids))),replace=False))
        nodes=train[np.isin(data['ant'][train],retained)]
        x,_=encode(data,wi,selected['representation'],nodes)
        graph,_,_=graph_from_features(x[nodes],SEED+100+repeat)
        labels=np.asarray(la.find_partition(graph,la.RBConfigurationVertexPartition,weights='weight',
            resolution_parameter=selected['resolution'],n_iterations=-1,seed=SEED+100+repeat).membership)
        ix,w=neighbor_weights(x[nodes],x[test]);predicted,_=vote(labels,ix,w)
        row=dict(repeat=repeat,n_states=int(labels.max()+1),ari=float(adjusted_rand_score(reference,predicted)),
                 **{f'S{i}_jaccard':float(v) for i,v in enumerate(matched_jaccard(reference,predicted))})
        rows.append(row);print('AUDIT',row,flush=True)
        pd.DataFrame(rows).to_csv(models/'stability_audit.csv',index=False)
    # Keep the original eligible ant cohort; ask whether the role split depends
    # on the state with a weak match in the original test comparison.
    summary=json.loads((report/'summary.json').read_text());weak=summary['weak_test_match_states']
    role_ants=pd.read_csv(report/'ant_roles.csv');p=pd.read_csv(report/'ant_state_proportions.csv',index_col=0)
    np.testing.assert_array_equal(p.index.to_numpy(),role_ants.ant)
    role_checks=[]
    for side in ('left','right'):
        ids=np.flatnonzero(role_ants.side.eq(side).to_numpy()&role_ants.eligible_role.to_numpy())
        k=summary['roles'][side]['k']
        if k is None or len(ids)<10:continue
        values=p.to_numpy()[ids].copy();values[:,weak]=0;values/=values.sum(axis=1,keepdims=True)
        x=np.sqrt(values);fit=fit_role_model(x,k,SEED);one=fit_role_model(x,1,SEED)
        role_checks.append(dict(side=side,k=k,n=len(ids),dropped_states=weak,
            original_role_agreement=float(adjusted_rand_score(role_ants.role.to_numpy()[ids],fit.predict(x))),
            bic_difference_from_k1=float(fit.bic(x)-one.bic(x))))
    assert hashlib.sha256(frozen.read_bytes()).hexdigest()==before
    (report/'role_weak_state_sensitivity.json').write_text(json.dumps(role_checks,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    if wavelet_features is not None:
        with np.load(wavelet_features/'features.npz') as z:slow=z['base'][wi]
        coverage=pd.read_csv(features/'coverage_audit.csv')
        totals=coverage[~coverage.common].groupby('width').n_windows.sum()
        axes[0,0].bar(range(4),[*totals.values,len(slow)],color=['#4288aa']*3+['#aab8bf'])
        axes[0,0].set(xticks=range(4),xticklabels=['0.5 s','1 s','2 s','wavelet\ncommon'],ylabel='Available windows',title='9. Slow wavelets exclude many observations')
        for label,values in [('two-second support',data['base'][wi,:,8]),('slow-wavelet support',slow[:,8])]:
            values=np.sort(values);axes[0,1].plot(values,np.arange(1,len(values)+1)/len(values),label=label)
        axes[0,1].set(xscale='symlog',xlabel='Forward peak speed (mm/s)',ylabel='Cumulative fraction',title='The retained wavelet sample favors stillness');axes[0,1].legend()
        (report/'coverage_comparison.json').write_text(json.dumps(dict(short_windows=len(data['ant']),wavelet_windows=len(slow),
            short_mean_forward=float(data['base'][wi,:,8].mean()),wavelet_mean_forward=float(slow[:,8].mean()),
            short_median_forward=float(np.median(data['base'][wi,:,8])),wavelet_median_forward=float(np.median(slow[:,8]))),indent=2)+'\n')
    results=pd.DataFrame(rows)
    axes[1,0].plot(results.repeat+1,results.ari,'o-');axes[1,0].axhline(.65,color='.6',ls='--')
    for row in results.itertuples():axes[1,0].annotate(f'K={row.n_states}',(row.repeat+1,row.ari),xytext=(0,6),textcoords='offset points',ha='center',fontsize=8)
    axes[1,0].set(xlabel='Additional training-ant omission',ylabel='Agreement on held-out test ants (ARI)',ylim=(0,1),title='Dictionary frozen before these five checks')
    columns=[c for c in results if c.endswith('_jaccard')]
    axes[1,1].boxplot([results[c] for c in columns],tick_labels=[c.split('_')[0] for c in columns],showfliers=False)
    for i,c in enumerate(columns):axes[1,1].scatter(np.full(repeats,i+1),results[c],s=12,color='#206080')
    axes[1,1].axhline(.5,color='.6',ls='--');axes[1,1].set(ylabel='Matched-state Jaccard',ylim=(0,1.05),title='State-level consistency can differ from overall ARI')
    fig.savefig(report/'09_coverage_stability_audit.png',dpi=180,bbox_inches='tight')
    fig.savefig(report/'fine_behavior_audit.pdf',bbox_inches='tight');plt.close(fig)
    (report/'audit_summary.json').write_text(json.dumps(dict(repeats=repeats,
        median_ari=float(results.ari.median()),minimum_ari=float(results.ari.min()),maximum_ari=float(results.ari.max()),
        state_counts=results.n_states.tolist(),state_median_jaccard={c.split('_')[0]:float(results[c].median()) for c in columns},
        selection_changed=False),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('features','models','report'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--repeats',type=int,default=5);p.add_argument('--wavelet-features',type=Path)
    p.add_argument('--plots-only',action='store_true',help='Render already completed audit results without refitting')
    a=p.parse_args()
    with threadpool_limits(limits=4):audit(a.features,a.models,a.report,a.repeats,a.wavelet_features,a.plots_only)
