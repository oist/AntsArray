"""Ant roles from pooled state counts; compare alternatives without spatial labels."""
import argparse
import json
from pathlib import Path

import igraph as ig
import leidenalg as la
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist,squareform
from sklearn.cluster import KMeans, AgglomerativeClustering, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score
from threadpoolctl import threadpool_limits

from analysis.full_behavior_extract import SEED,HZ,FPS


def compare_ant_umaps(x,seed=SEED):
    """Compare display settings by neighborhood preservation, without labels.

    Use three seeds per setting. Select by median five-neighbor trustworthiness,
    then neighbor overlap. This chooses a view, never a clustering or class K.
    """
    from sklearn.manifold import trustworthiness
    from umap import UMAP
    k=min(5,(len(x)-1)//2)
    if k<1:raise ValueError('Need at least three ants for a UMAP comparison')
    def neighbors(values):
        distance=squareform(pdist(values));np.fill_diagonal(distance,np.inf)
        return np.argsort(distance,axis=1)[:,:k]
    reference=neighbors(x);rows=[];layouts={}
    for n in sorted(set(min(value,len(x)-1) for value in (4,8,20))):
        for d in (.02,.15,.5):
            for repeat in range(3):
                xy=UMAP(n_neighbors=n,min_dist=d,n_epochs=500,random_state=seed+repeat,n_jobs=1).fit_transform(x)
                key=f'n{n}_d{d}_seed{seed+repeat}';layouts[key]=xy
                overlap=np.mean([len(set(a)&set(b))/k for a,b in zip(reference,neighbors(xy))])
                rows.append(dict(key=key,n_neighbors=n,min_dist=d,seed=seed+repeat,
                    trustworthiness=trustworthiness(x,xy,n_neighbors=k),neighbor_overlap=overlap))
    scores=pd.DataFrame(rows)
    ranking=scores.groupby(['n_neighbors','min_dist'],as_index=False)[['trustworthiness','neighbor_overlap']].median().sort_values(
        ['trustworthiness','neighbor_overlap','n_neighbors','min_dist'],ascending=[False,False,True,True])
    best=ranking.iloc[0];key=f"n{int(best.n_neighbors)}_d{best.min_dist}_seed{seed}"
    return scores,layouts,key


def fit_labels(x, method, parameter, seed=SEED):
    if method=='kmeans':return KMeans(int(parameter),n_init=20,random_state=seed).fit_predict(x)
    if method=='ward':return AgglomerativeClustering(int(parameter),linkage='ward').fit_predict(x)
    if method in ('gmm_tied','gmm_diag'):
        model=GaussianMixture(int(parameter),covariance_type=method.split('_')[1],reg_covar=.0001,
                              n_init=10,max_iter=500,random_state=seed).fit(x)
        if not model.converged_:raise ValueError('Nonconverged role GMM')
        return model.predict(x)
    if method=='hdbscan':return HDBSCAN(min_cluster_size=int(parameter),min_samples=2,cluster_selection_method='leaf').fit_predict(x)
    if method=='leiden':
        distances=squareform(pdist(x));n=len(x);neighbors=min(8,n-1)
        ix=np.argsort(distances,axis=1)[:,1:neighbors+1]
        scale=np.maximum(np.take_along_axis(distances,ix,axis=1)[:,-1],1e-6)
        weights=np.zeros((n,n))
        for i in range(n):weights[i,ix[i]]=np.exp(-distances[i,ix[i]]**2/(scale[i]*scale[ix[i]]))
        weights=np.maximum(weights,weights.T);a,b=np.where(np.triu(weights,1)>0)
        g=ig.Graph(n=n,edges=np.column_stack([a,b]));g.es['weight']=weights[a,b]
        return np.asarray(la.find_partition(g,la.RBConfigurationVertexPartition,weights='weight',
            resolution_parameter=float(parameter),n_iterations=-1,seed=seed).membership)
    raise ValueError(method)


def min_jaccard(a,b):
    ua=np.unique(a[a>=0]);ub=np.unique(b[b>=0])
    if not len(ua) or not len(ub):return 0.
    table=np.array([[np.sum((a==i)&(b==j)) for j in ub] for i in ua])
    union=np.array([np.sum(a==i) for i in ua])[:,None]+np.array([np.sum(b==j) for j in ub])[None]-table
    score=table/np.maximum(union,1);i,j=linear_sum_assignment(-score)
    return float(score[i,j].min()) if len(i)==len(ua)==len(ub) else 0.


def pooled_counts(root, ants, k):
    total=np.zeros((len(ants),k));hours=int(np.ceil(ants.n_frames.max()/(3600*FPS)))
    by_hour=np.zeros((len(ants),hours,k))
    for ai,ant in enumerate(ants.ant):
        name=ant.replace(':','_');state=np.load(root/'assignments'/name/'state.npy')
        length=np.load(root/'features'/name/'length.npy');start=np.load(root/'features'/name/'start.npy')
        np.add.at(total[ai],state,length)
        h=(start+length/2)//(3600*HZ)
        np.add.at(by_hour[ai],(h.astype(int),state),length)
    return total,by_hour


def proportions(counts):
    total=counts.sum(axis=-1,keepdims=True)
    return np.divide(counts,total,out=np.full(counts.shape,np.nan,dtype=float),where=total>0)


def select_model(table):
    """Prefer the model reproducible under the weakest tested perturbation.

    Higher K no longer wins just by passing a permissive threshold. A one-class
    fallback is used only if no nontrivial candidate meets the existing checks.
    All comparisons use behavior alone; no spatial agreement enters selection.
    """
    passing=table[table.qualifies]
    if not len(passing):return dict(model='one_class',k=1,qualifies=False)
    return passing.sort_values(['stability_score','k','silhouette'],
                               ascending=[False,True,False]).iloc[0].to_dict()


def evaluate_models(x, hourly, candidates=None, repeats=30):
    """Return candidate scores, labels, and each ant-removal/time-split result."""
    if candidates is None:
        candidates=[(method,k) for method in ('kmeans','ward','gmm_tied','gmm_diag') for k in range(2,9)]
        candidates += [('leiden',r) for r in (.3,.5,.8,1.,1.5,2.)]+[('hdbscan',n) for n in (3,4,5)]
    rows=[];assignments={};details=[]
    for method,param in candidates:
        name=f'{method}_{param:g}';labels=fit_labels(x,method,param)
        assigned=labels>=0;classes=np.unique(labels[assigned]);sizes=[int(np.sum(labels==s)) for s in classes]
        ari=[];jac=[];temporal=[];tjac=[]
        if len(classes)>1:
            rng=np.random.default_rng(SEED)
            for repeat in range(repeats):
                sample=np.sort(rng.choice(len(x),round(.8*len(x)),replace=False))
                alternative=fit_labels(x[sample],method,param,SEED+repeat)
                ari.append(adjusted_rand_score(labels[sample],alternative));jac.append(min_jaccard(labels[sample],alternative))
                details.append(dict(model=name,kind='ant_removal',repeat=repeat,width_hours=np.nan,
                                    ari=ari[-1],minimum_jaccard=jac[-1]))
            for width in (1,3,6,24):
                groups=[]
                for parity in (0,1):
                    selected=(np.arange(hourly.shape[1])//width)%2==parity
                    representation=np.sqrt(proportions(hourly[:,selected].sum(axis=1)))
                    if not np.isfinite(representation).all():groups=[];break
                    groups.append(fit_labels(representation,method,param))
                if len(groups)==2:
                    temporal.append(adjusted_rand_score(*groups));tjac.append(min_jaccard(*groups))
                    details.append(dict(model=name,kind='temporal',repeat=0,width_hours=width,
                                        ari=temporal[-1],minimum_jaccard=tjac[-1]))
        row=dict(model=name,method=method,parameter=param,k=len(classes),minimum_ants=min(sizes,default=0),
            noise_fraction=float(np.mean(~assigned)),ant_removal_ari=float(np.median(ari)) if ari else 0,
            smallest_class_jaccard=float(np.median(jac)) if jac else 0,
            temporal_ari=float(np.median(temporal)) if temporal else 0,
            temporal_smallest_jaccard=float(np.median(tjac)) if tjac else 0,
            ant_removal_ari_p10=float(np.quantile(ari,.1)) if ari else 0,
            ant_removal_jaccard_p10=float(np.quantile(jac,.1)) if jac else 0,
            temporal_min_ari=float(np.min(temporal)) if temporal else 0,
            temporal_min_jaccard=float(np.min(tjac)) if tjac else 0,
            temporal_splits=len(temporal),
            silhouette=float(silhouette_score(x[assigned],labels[assigned])) if 1<len(classes)<assigned.sum() else np.nan)
        row['qualifies']=(row['k']>1 and row['minimum_ants']>=3 and row['noise_fraction']==0
            and row['ant_removal_ari']>=.8 and row['smallest_class_jaccard']>=.6
            and row['temporal_ari']>=.65 and row['temporal_smallest_jaccard']>=.6)
        # A median across timescales can hide a failed day-scale split. The
        # primary score is the worst of lower-tail ant-removal agreement and
        # the least reproducible time split, including the weakest class.
        row['stability_score']=min(row[key] for key in ('ant_removal_ari_p10',
            'ant_removal_jaccard_p10','temporal_min_ari','temporal_min_jaccard'))
        rows.append(row);assignments[name]=labels
        print('ANT MODEL',json.dumps(row),flush=True)
    return pd.DataFrame(rows),assignments,pd.DataFrame(details)


def merge_counts(counts, mapping):
    """Pool the last state axis while preserving ant/time axes and duration."""
    counts=np.asarray(counts);mapping=np.asarray(mapping,int)
    if counts.shape[-1]!=len(mapping) or not np.array_equal(np.unique(mapping),np.arange(mapping.max()+1)):
        raise ValueError('Expected one contiguous merged ID per input state')
    merged=np.zeros(counts.shape[:-1]+(int(mapping.max()+1),),dtype=counts.dtype)
    for old,new in enumerate(mapping):merged[...,new]+=counts[...,old]
    np.testing.assert_allclose(merged.sum(axis=-1),counts.sum(axis=-1))
    return merged


def compare_reductions(root, output, weighting='state'):
    """Audit fewer ant classes for each state merge, reusing observed counts."""
    from analysis.full_behavior_states import merge_map
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    import hashlib
    signature=dict(dictionary_sha256=hashlib.sha256((root/'models/dictionary.npz').read_bytes()).hexdigest(),
        counts_sha256=hashlib.sha256((root/'roles/state_counts.npz').read_bytes()).hexdigest(),
        weighting=weighting,repeats=30,methods=['kmeans','ward','gmm_tied','gmm_diag'],class_counts=[2,3,4])
    path=output/'role_reduction_inputs.json'
    if path.exists() and json.loads(path.read_text())!=signature:
        raise ValueError('Role audit belongs to different state counts or settings')
    path.write_text(json.dumps(signature,indent=2)+'\n')
    with np.load(root/'models/dictionary.npz') as z:dictionary=dict(z)
    with np.load(root/'roles/state_counts.npz') as z:source=dict(z)
    inventory=pd.read_csv(root/'inventory.csv').query('selected').reset_index(drop=True)
    np.testing.assert_array_equal(source['ants'],inventory.ant)
    right=inventory.side.eq('right').to_numpy();all_rows=[];all_details=[]
    candidates=[(method,k) for method in ('kmeans','ward','gmm_tied','gmm_diag') for k in (2,3,4)]
    for k in range(2,int(dictionary['state'].max())+2):
        destination=output/f'roles_{k}_states';destination.mkdir(exist_ok=True)
        mapping=merge_map(dictionary['x'],dictionary['state'],k,weighting)
        counts=merge_counts(source['counts'][right],mapping)
        hourly=merge_counts(source['hourly'][right],mapping)
        if all((destination/name).exists() for name in ('comparison.csv','validation_repeats.csv','labels.npz')):
            table=pd.read_csv(destination/'comparison.csv');detail=pd.read_csv(destination/'validation_repeats.csv')
        else:
            print('ANT CLASSES AFTER MERGING TO',k,'STATES',mapping.tolist(),flush=True)
            table,labels,detail=evaluate_models(np.sqrt(proportions(counts)),hourly,candidates)
            table.to_csv(destination/'comparison.csv',index=False)
            detail.to_csv(destination/'validation_repeats.csv',index=False)
            np.savez_compressed(destination/'labels.npz',**labels)
        all_rows.append(table.assign(n_states=k));all_details.append(detail.assign(n_states=k))
    table=pd.concat(all_rows,ignore_index=True);table.to_csv(output/'role_reduction_search.csv',index=False)
    pd.concat(all_details,ignore_index=True).to_csv(output/'role_repeats.csv',index=False)
    return table


def run(root,side='right',output=None):
    output=Path(output) if output is not None else root/'roles';output.mkdir(exist_ok=True)
    inventory=pd.read_csv(root/'inventory.csv');ants=inventory[inventory.selected].reset_index(drop=True)
    choice=json.loads((root/'models/selection.json').read_text());k=choice['n_states']
    counts,hourly=pooled_counts(root,ants,k)
    p=proportions(counts);assert np.isfinite(p).all(), 'A selected ant has no usable posture/motion observations'
    pd.DataFrame(p,index=ants.ant,columns=[f'S{s}' for s in range(k)]).to_csv(output/'all_ant_state_proportions.csv')
    np.savez_compressed(output/'state_counts.npz',counts=counts,hourly=hourly,ants=ants.ant.to_numpy(str))
    right=ants.side.eq(side).to_numpy();ants=ants[right].reset_index(drop=True);p=p[right];hourly=hourly[right]
    x=np.sqrt(p);np.testing.assert_allclose((x*x).sum(axis=1),1)
    table,assignments,details=evaluate_models(x,hourly)
    table.to_csv(output/'method_comparison.csv',index=False)
    details.to_csv(output/'validation_repeats.csv',index=False)
    np.savez_compressed(output/'candidate_labels.npz',**assignments)
    selected=select_model(table)
    labels=assignments[selected['model']] if selected['k']>1 else np.zeros(len(ants),int)
    original=np.unique(labels)
    ranking=np.argsort([np.mean(p[labels==s]@np.arange(k)) for s in original])
    remap={int(original[i]):rank for rank,i in enumerate(ranking)}
    labels=np.asarray([remap[int(s)] for s in labels])
    ants['ant_class']=labels;ants['classified_seconds']=counts[right].sum(axis=1)/HZ
    ants.to_csv(output/'ant_classes.csv',index=False)
    pd.DataFrame(p,index=ants.ant,columns=[f'S{s}' for s in range(k)]).to_csv(output/'ant_state_proportions.csv')
    selected['side']=side
    selected['rule']='Among passing nontrivial models, maximize the minimum of ant-removal ARI/Jaccard 10th percentiles and worst temporal ARI/Jaccard over 1/3/6/24-hour splits. Break ties by fewer classes, then silhouette. All state proportions, no spatial inputs.'
    (output/'selection.json').write_text(json.dumps(selected,indent=2)+'\n')
    print('SELECTED ANT CLASSES',json.dumps(selected), 'sizes',np.bincount(labels).tolist(),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--compare-reductions',type=Path,help='Audit smaller state/class models in a separate directory')
    parser.add_argument('--merge-weighting',choices=['state','observation'],default='observation');parser.add_argument('--side',choices=['left','right'],default='right');parser.add_argument('--output',type=Path);args=parser.parse_args()
    with threadpool_limits(limits=4):
        if args.compare_reductions:compare_reductions(args.root,args.compare_reductions,args.merge_weighting)
        else:run(args.root,args.side,args.output)
