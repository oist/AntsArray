"""Ant-disjoint evaluation of fine behavior dictionaries in full feature space.

No spatial labels, sleep labels, or desired colony roles enter model selection.
Leiden resolution controls granularity; it does not estimate a biological K.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

from analysis.fine_behavior_features import WIDTHS, encode
from analysis.eigenposture_leiden import graph_from_features, partition

SEED = 7241011
REPRESENTATIONS = ('kinematics', 'trajectory', 'spectrum')
RESOLUTIONS = (.5, 1., 2.)
PROTOCOL = dict(seed=SEED, widths_seconds=WIDTHS, representations=REPRESENTATIONS,
    resolutions=RESOLUTIONS, graph_neighbors=30, seed_repeats=3,
    dictionary='Random ant-balanced maximum 240 windows/ant on a 4-second center grid; training ants only',
    split='Eligible ants >=100 common windows: within each colony random 60% train, 20% validation, remainder test',
    perturbation='Refit after omitting 20% of training ants; compare predictions on different, validation ants',
    rule='Finest tested partition passing minimum seed ARI >=0.75, validation ant-removal ARI >=0.65, minimum dictionary fraction >=0.005, >=5 training ants per state; then validation ARI, seed ARI',
    fallback='If none pass, highest validation ARI then seed ARI; label provisional',
    test='Test ants never select representation/width/resolution. Report ant-removal agreement on test after freezing choice.',
    normalization='Training-only centers and one total-variance scale per feature family, all coordinates retained',
    assignment='5-nearest dictionary neighbors, inverse-distance vote; exact dictionary membership on dictionary rows',
    no_labels_used=True, interpretation='Exploratory behavioral vocabulary, not a count of discrete natural behaviors')


def split_and_sample(data, ants):
    rng = np.random.default_rng(SEED)
    counts = np.bincount(data['ant'], minlength=len(ants))
    split = np.full(len(ants), 'insufficient', dtype='<U12')
    for side in ('left','right'):
        ids = np.flatnonzero(ants.side.eq(side).to_numpy() & (counts >= 100))
        rng.shuffle(ids)
        a,b = int(.6*len(ids)),int(.8*len(ids))
        split[ids[:a]]='train'; split[ids[a:b]]='validation'; split[ids[b:]]='test'
    # All widths are compared at the same nonadjacent centers. Wavelet contexts
    # may still overlap; ant-disjoint evaluation avoids treating them as replicas.
    candidates = np.flatnonzero(data['sample'] % 48 == 0)
    samples = {}
    for group in ('train','validation','test'):
        rows=[]
        for ant in np.flatnonzero(split==group):
            choices=candidates[data['ant'][candidates]==ant]
            rows.extend(rng.choice(choices,min(len(choices),240),replace=False))
        samples[group]=np.sort(np.asarray(rows,dtype=int))
    retained=[]
    for side in ('left','right'):
        ids=np.flatnonzero((split=='train') & ants.side.eq(side).to_numpy())
        retained.extend(rng.choice(ids,max(1,int(.8*len(ids))),replace=False))
    samples['reduced_train']=samples['train'][np.isin(data['ant'][samples['train']],retained)]
    return split,samples


def neighbor_weights(training, queries):
    model=NearestNeighbors(n_neighbors=5,n_jobs=4).fit(training)
    distances,indices=model.kneighbors(queries)
    weights=1/np.maximum(distances,1e-6)
    return indices,weights


def vote(labels, indices, weights):
    k=int(labels.max()+1)
    scores=np.zeros((len(indices),k),dtype=np.float32)
    for j in range(indices.shape[1]):
        np.add.at(scores,(np.arange(len(indices)),labels[indices[:,j]]),weights[:,j])
    result=scores.argmax(axis=1)
    confidence=scores.max(axis=1)/scores.sum(axis=1)
    return result,confidence


def matched_jaccard(a,b):
    table=np.zeros((a.max()+1,b.max()+1),dtype=int)
    np.add.at(table,(a,b),1)
    union=table.sum(axis=1)[:,None]+table.sum(axis=0)[None,:]-table
    j=np.divide(table,union,out=np.zeros(table.shape),where=union>0)
    rows,cols=linear_sum_assignment(-j)
    matches=np.zeros(len(table));matches[rows]=j[rows,cols]
    return matches


def run(features, source, output):
    output.mkdir(parents=True,exist_ok=True)
    protocol=output/'protocol.json'
    if protocol.exists() and json.loads(protocol.read_text()) != json.loads(json.dumps(PROTOCOL)):
        raise ValueError('Protocol changed; use a new output directory')
    protocol.write_text(json.dumps(PROTOCOL,indent=2)+'\n')
    fingerprints=dict(features_sha256=hashlib.sha256((features/'features.npz').read_bytes()).hexdigest(),
        ant_inventory_sha256=hashlib.sha256((source/'all_ant_coverage.csv').read_bytes()).hexdigest())
    signature=output/'input_fingerprints.json'
    if signature.exists() and json.loads(signature.read_text())!=fingerprints:
        raise ValueError('Inputs changed; use a new output directory')
    signature.write_text(json.dumps(fingerprints,indent=2)+'\n')
    with np.load(features/'features.npz') as z:data=dict(z)
    ants=pd.read_csv(source/'all_ant_coverage.csv')
    split,samples=split_and_sample(data,ants)
    ants=ants.assign(split=split,common_windows=np.bincount(data['ant'],minlength=len(ants)))
    ants.to_csv(output/'ant_split.csv',index=False)
    np.savez_compressed(output/'split_rows.npz',**samples)
    train,small,validation=samples['train'],samples['reduced_train'],samples['validation']
    print('SPLIT',ants.groupby(['side','split']).size().to_dict(),{k:len(v) for k,v in samples.items()},flush=True)
    all_rows=[]
    for wi,width in enumerate(WIDTHS):
        for representation in REPRESENTATIONS:
            name=f'{width:g}s_{representation}'
            saved=output/(name+'.npz');table_file=output/(name+'.csv')
            if saved.exists() and table_file.exists():
                all_rows.extend(pd.read_csv(table_file).to_dict('records'));continue
            print('FIT',name,flush=True)
            x,normalizer=encode(data,wi,representation,train)
            xs,small_normalizer=encode(data,wi,representation,small)
            graph,_,_=graph_from_features(x[train],SEED)
            small_graph,_,_=graph_from_features(xs[small],SEED+1)
            vi,vw=neighbor_weights(x[train],x[validation])
            si,sw=neighbor_weights(xs[small],xs[validation])
            full_labels=[];small_labels=[];rows=[]
            for resolution in RESOLUTIONS:
                labels=[partition(graph,resolution,SEED+i) for i in range(3)]
                lab=labels[0]; perturb=partition(small_graph,resolution,SEED)
                seed_ari=min(adjusted_rand_score(labels[i],labels[j]) for i in range(3) for j in range(i))
                predicted,_=vote(lab,vi,vw);changed,_=vote(perturb,si,sw)
                k=int(lab.max()+1);counts=np.bincount(lab)
                support=min(len(np.unique(data['ant'][train][lab==s])) for s in range(k))
                ari=adjusted_rand_score(predicted,changed)
                row=dict(model=name,width=width,representation=representation,resolution=resolution,
                    n_states=k,seed_ari=seed_ari,validation_ari=ari,
                    smallest_fraction=float(counts.min()/len(lab)),minimum_ants=support,
                    validation_min_jaccard=float(matched_jaccard(predicted,changed).min()),
                    qualifies=bool(k>1 and seed_ari>=.75 and ari>=.65 and counts.min()/len(lab)>=.005 and support>=5))
                rows.append(row);full_labels.append(lab);small_labels.append(perturb)
                print('RESULT',json.dumps(row),flush=True)
            # Each resolution can have different K but always the same node count.
            np.savez_compressed(saved,labels=np.asarray(full_labels),small_labels=np.asarray(small_labels))
            (output/(name+'_normalizer.json')).write_text(json.dumps(dict(full=normalizer,reduced=small_normalizer),indent=2)+'\n')
            pd.DataFrame(rows).to_csv(table_file,index=False)
            all_rows.extend(rows)
            pd.DataFrame(all_rows).to_csv(output/'search.csv',index=False)
    results=pd.DataFrame(all_rows)
    accepted=results[results.qualifies]
    if len(accepted):
        best=accepted.sort_values(['n_states','validation_ari','seed_ari'],ascending=False).iloc[0]
    else:
        best=results.sort_values(['validation_ari','seed_ari'],ascending=False).iloc[0]
    selection=dict(best.to_dict(),supported=bool(len(accepted)))
    # Freeze before touching test predictions, sleep, occupancy or roles.
    (output/'selection_frozen.json').write_text(json.dumps(selection,indent=2)+'\n')
    results.to_csv(output/'search.csv',index=False)
    print('SELECTED',json.dumps(selection),flush=True)
    name=best.model;wi=WIDTHS.index(float(best.width));ri=RESOLUTIONS.index(float(best.resolution))
    with np.load(output/(name+'.npz')) as z:
        labels=z['labels'][ri];small_labels=z['small_labels'][ri]
    normalizers=json.loads((output/(name+'_normalizer.json')).read_text())
    x,_=encode(data,wi,best.representation,train,normalizers['full'])
    xs,_=encode(data,wi,best.representation,small,normalizers['reduced'])
    assigned=[];confidence=[]
    for first in range(0,len(x),20000):
        ix,w=neighbor_weights(x[train],x[first:first+20000]);a,c=vote(labels,ix,w)
        assigned.append(a);confidence.append(c)
    assigned=np.concatenate(assigned);confidence=np.concatenate(confidence)
    assigned[train]=labels;confidence[train]=1.
    test=samples['test'];ix,w=neighbor_weights(xs[small],xs[test]);alternative,_=vote(small_labels,ix,w)
    test_scores=dict(ant_removal_ari=float(adjusted_rand_score(assigned[test],alternative)),
        matched_jaccard=matched_jaccard(assigned[test],alternative).tolist(),n_windows=len(test),
        n_ants=int(len(np.unique(data['ant'][test]))),
        silhouette=float(silhouette_score(x[test],assigned[test],sample_size=min(2000,len(test)),random_state=SEED)))
    (output/'test_evaluation.json').write_text(json.dumps(test_scores,indent=2)+'\n')
    # Order IDs by body speed, purely for legibility, after all decisions.
    raw=data['base'][wi];order=np.argsort([np.median(raw[assigned==s,8]) for s in range(assigned.max()+1)])
    inverse=np.argsort(order);assigned=inverse[assigned];labels=inverse[labels]
    np.savez_compressed(output/'states.npz',state=assigned,confidence=confidence,dictionary=train,
        dictionary_state=labels,x=x,raw=raw,order=order,**{k:data[k] for k in ('ant','hour','sample','frame','camera','interpolation','residual')})
    print('TEST',json.dumps(test_scores),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('features','source','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    with threadpool_limits(limits=4):run(a.features,a.source,a.output)
