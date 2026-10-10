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


def run(root):
    output=root/'roles';output.mkdir(exist_ok=True)
    inventory=pd.read_csv(root/'inventory.csv');ants=inventory[inventory.selected].reset_index(drop=True)
    choice=json.loads((root/'models/selection.json').read_text());k=choice['n_states']
    counts,hourly=pooled_counts(root,ants,k)
    p=proportions(counts);assert np.isfinite(p).all(), 'A selected ant has no usable posture/motion observations'
    pd.DataFrame(p,index=ants.ant,columns=[f'S{s}' for s in range(k)]).to_csv(output/'all_ant_state_proportions.csv')
    np.savez_compressed(output/'state_counts.npz',counts=counts,hourly=hourly,ants=ants.ant.to_numpy(str))
    right=ants.side.eq('right').to_numpy();ants=ants[right].reset_index(drop=True);p=p[right];hourly=hourly[right]
    x=np.sqrt(p);np.testing.assert_allclose((x*x).sum(axis=1),1)
    candidates=[(method,k) for method in ('kmeans','ward','gmm_tied','gmm_diag') for k in range(2,9)]
    candidates += [('leiden',r) for r in (.3,.5,.8,1.,1.5,2.)]+[('hdbscan',n) for n in (3,4,5)]
    rows=[];assignments={}
    for method,param in candidates:
        name=f'{method}_{param:g}';labels=fit_labels(x,method,param)
        assigned=labels>=0;classes=np.unique(labels[assigned]);sizes=[int(np.sum(labels==s)) for s in classes]
        ari=[];jac=[];temporal=[];tjac=[]
        if len(classes)>1:
            rng=np.random.default_rng(SEED)
            for repeat in range(30):
                sample=np.sort(rng.choice(len(x),round(.8*len(x)),replace=False))
                alternative=fit_labels(x[sample],method,param,SEED+repeat)
                ari.append(adjusted_rand_score(labels[sample],alternative));jac.append(min_jaccard(labels[sample],alternative))
            for width in (1,3,6,24):
                groups=[]
                for parity in (0,1):
                    selected=(np.arange(hourly.shape[1])//width)%2==parity
                    representation=np.sqrt(proportions(hourly[:,selected].sum(axis=1)))
                    if not np.isfinite(representation).all():groups=[];break
                    groups.append(fit_labels(representation,method,param))
                if len(groups)==2:
                    temporal.append(adjusted_rand_score(*groups));tjac.append(min_jaccard(*groups))
        row=dict(model=name,method=method,parameter=param,k=len(classes),minimum_ants=min(sizes,default=0),
            noise_fraction=float(np.mean(~assigned)),ant_removal_ari=float(np.median(ari)) if ari else 0,
            smallest_class_jaccard=float(np.median(jac)) if jac else 0,
            temporal_ari=float(np.median(temporal)) if temporal else 0,
            temporal_smallest_jaccard=float(np.median(tjac)) if tjac else 0,
            silhouette=float(silhouette_score(x[assigned],labels[assigned])) if 1<len(classes)<assigned.sum() else np.nan)
        row['qualifies']=(row['k']>1 and row['minimum_ants']>=3 and row['noise_fraction']==0
            and row['ant_removal_ari']>=.8 and row['smallest_class_jaccard']>=.6
            and row['temporal_ari']>=.65 and row['temporal_smallest_jaccard']>=.6)
        rows.append(row);assignments[name]=labels
        print('ANT MODEL',json.dumps(row),flush=True)
    table=pd.DataFrame(rows);table.to_csv(output/'method_comparison.csv',index=False)
    np.savez_compressed(output/'candidate_labels.npz',**assignments)
    passing=table[table.qualifies]
    if len(passing):
        selected=passing.sort_values(['k','temporal_ari','ant_removal_ari'],ascending=False).iloc[0].to_dict()
        labels=assignments[selected['model']]
    else:
        selected=dict(model='one_class',k=1,qualifies=False);labels=np.zeros(len(ants),int)
    # Numerical IDs ordered by average state-ID rank; no spatial comparison yet.
    original=np.unique(labels)
    ranking=np.argsort([np.mean(p[labels==s]@np.arange(k)) for s in original])
    remap={int(original[i]):rank for rank,i in enumerate(ranking)}
    labels=np.asarray([remap[int(s)] for s in labels])
    ants['ant_class']=labels;ants['classified_seconds']=counts[right].sum(axis=1)/HZ
    ants.to_csv(output/'ant_classes.csv',index=False)
    pd.DataFrame(p,index=ants.ant,columns=[f'S{s}' for s in range(k)]).to_csv(output/'ant_state_proportions.csv')
    selected['rule']='Finest partition with >=3 ants/class, no noise, ant-removal ARI>=.8/min-Jaccard>=.6, temporal ARI>=.65/min-Jaccard>=.6; all state axes, no spatial inputs.'
    (output/'selection.json').write_text(json.dumps(selected,indent=2)+'\n')
    print('SELECTED ANT CLASSES',json.dumps(selected), 'sizes',np.bincount(labels).tolist(),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    with threadpool_limits(limits=4):run(args.root)
