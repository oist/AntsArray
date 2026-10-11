"""Segment-wise full-body features, Leiden dictionaries, and feature ablation.

Every usable detected segment is assigned. The graph uses an ant-balanced
landmark dictionary drawn across the complete recording for tractable scaling.
Spatial coordinates and sleep labels are never model inputs or selectors.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import igraph as ig
import leidenalg as la
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.cluster.hierarchy import linkage, cut_tree
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits
from umap.umap_ import nearest_neighbors, fuzzy_simplicial_set
from pynndescent import NNDescent

from analysis.full_behavior_extract import SEED, HZ, FPS

NAMES = ([f'PC{i+1}' for i in range(8)] + [f'PC{i+1}_sd' for i in range(8)]
         + [f'PC{i+1}_rate' for i in range(8)] + ['forward_peak','lateral_peak',
         'antenna_A_mean','antenna_B_mean','turn_mean','exact_contact_mean','exact_contact_peak'])
SETS = dict(posture4=[list(range(4)),list(range(24,29)),[29,30]],
            posture8=[list(range(8)),list(range(24,29)),[29,30]],
            variation=[list(range(8)),list(range(24,29)),[29,30],list(range(8,16))],
            dynamics=[list(range(8)),list(range(24,29)),[29,30],list(range(8,16)),list(range(16,24))])


def segment_windows(valid, camera, width=12, minimum=3):
    """Nonoverlapping <=1-s pieces; never bridge gaps/cameras, keep short tails."""
    valid=np.asarray(valid,bool)
    boundary=np.r_[True, (np.diff(camera)!=0)|~valid[1:]|~valid[:-1], True]
    cuts=np.flatnonzero(boundary)
    starts=[];lengths=[]
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        if not valid[lo]:continue
        positions=np.arange(lo,hi,width)
        sizes=np.minimum(width,hi-positions)
        keep=sizes>=minimum
        starts.extend(positions[keep]);lengths.extend(sizes[keep])
    return np.asarray(starts,np.int32),np.asarray(lengths,np.int16)


def make_features(task):
    root, ant=task; threadpool_limits(limits=1)
    output=root/'features'/ant.replace(':','_');output.mkdir(parents=True,exist_ok=True)
    if (output/'metadata.json').exists():return json.loads((output/'metadata.json').read_text())
    source=root/'sequences'/ant.replace(':','_')
    pose=np.load(source/'pose.npy',mmap_mode='r');motion=np.load(source/'motion.npy',mmap_mode='r')
    camera=np.load(source/'camera.npy',mmap_mode='r')
    contact=np.load(root/'contacts'/(ant.replace(':','_')+'.npy'),mmap_mode='r')
    coverage=np.load(root/'contacts'/(ant.split(':')[0]+'_covered.npy'),mmap_mode='r')
    with np.load(root/'full_body_pca.npz') as z:pc=((pose-z['center'])@z['modes'][:8].T).astype(np.float32)
    valid=np.isfinite(pc).all(axis=1)&np.isfinite(motion).all(axis=1)&coverage[::2][:len(pc)]
    # At least 250 ms is necessary for a variation/rate estimate. This is a
    # feature-support rule, not hourly/ant filtering. Audit the omitted tails.
    starts,lengths=segment_windows(valid,camera)
    n=len(starts);features=np.empty((n,len(NAMES)),np.float32)
    for lo in range(0,n,20000):
        hi=min(n,lo+20000);s=starts[lo:hi];length=lengths[lo:hi]
        offsets=np.arange(12)[None,:];mask=offsets<length[:,None]
        ix=np.minimum(s[:,None]+offsets,s[:,None]+length[:,None]-1)
        values=pc[ix];mean=(values*mask[:,:,None]).sum(axis=1)/length[:,None]
        sd=np.sqrt(((values-mean[:,None])**2*mask[:,:,None]).sum(axis=1)/length[:,None])
        delta=np.diff(values,axis=1)*HZ
        rate=np.sqrt((delta**2*mask[:,1:,None]).sum(axis=1)/(length[:,None]-1))
        speeds=motion[ix]
        peaks=np.where(mask[:,:,None],speeds[:,:,:2],-np.inf).max(axis=1)
        averages=(speeds[:,:,2:]*mask[:,:,None]).sum(axis=1)/length[:,None]
        fr=s[:,None]*2+np.arange(24)[None,:]
        cmask=np.arange(24)[None,:]<length[:,None]*2
        fr=np.minimum(fr,len(contact)-1)
        hits=contact[fr]
        exact=np.column_stack([(hits*cmask).sum(axis=1)/(length*2),np.where(cmask,hits,0).max(axis=1)])
        features[lo:hi]=np.column_stack([mean,sd,rate,peaks,averages,exact])
    assert np.isfinite(features).all()
    np.save(output/'raw.npy',features);np.save(output/'start.npy',starts);np.save(output/'length.npy',lengths)
    info=dict(ant=ant,n_windows=n,n_observed_samples=int(valid.sum()),n_assigned_samples=int(lengths.sum()),
              excluded_short_samples=int(valid.sum()-lengths.sum()),median_window_seconds=float(np.median(lengths)/HZ),
              mean_window_seconds=float(np.mean(lengths)/HZ))
    (output/'metadata.json').write_text(json.dumps(info,indent=2)+'\n')
    return info


def fit_normalizer(raw, representation):
    values=transform(raw)
    normalizer=[]
    for columns in SETS[representation]:
        data=values[:,columns]
        normalizer.append(dict(columns=columns,center=data.mean(axis=0).tolist(),scale=max(float(np.sqrt(data.var(axis=0).sum())),1e-6)))
    return normalizer


def transform(raw):
    values=np.asarray(raw,float).copy()
    values[:,8:24]=np.log1p(values[:,8:24]/.1)
    values[:,24:29]=np.log1p(values[:,24:29]/.1)
    values[:,29:]=np.log1p(values[:,29:])
    return values


def encode(raw, normalizer):
    values=transform(raw)
    return np.column_stack([(values[:,n['columns']]-n['center'])/n['scale'] for n in normalizer]).astype(np.float32)


def graph(x, seed=SEED):
    rng=np.random.RandomState(seed)
    ix,d,_=nearest_neighbors(x,30,'euclidean',{},False,rng,n_jobs=4)
    adj,_,_=fuzzy_simplicial_set(x,30,rng,'euclidean',knn_indices=ix,knn_dists=d)
    coo=sparse.triu(adj,k=1).tocoo()
    g=ig.Graph(n=len(x),edges=np.column_stack([coo.row,coo.col]));g.es['weight']=coo.data
    return g


def partition(g,resolution,seed=SEED):
    return np.asarray(la.find_partition(g,la.RBConfigurationVertexPartition,weights='weight',
        resolution_parameter=resolution,n_iterations=-1,seed=seed).membership,int)


def vote(labels, ix, distance):
    weight=1/np.maximum(distance,1e-6);k=int(labels.max()+1)
    scores=np.zeros((len(ix),k),np.float32)
    for j in range(ix.shape[1]):np.add.at(scores,(np.arange(len(ix)),labels[ix[:,j]]),weight[:,j])
    return scores.argmax(axis=1), scores.max(axis=1)/scores.sum(axis=1)


def matched_jaccard(a,b):
    table=np.zeros((a.max()+1,b.max()+1),dtype=int)
    np.add.at(table,(a,b),1)
    union=table.sum(axis=1)[:,None]+table.sum(axis=0)[None,:]-table
    score=np.divide(table,union,out=np.zeros(table.shape),where=union>0)
    rows,columns=linear_sum_assignment(-score)
    matches=np.zeros(len(table));matches[rows]=score[rows,columns]
    return matches


def merge_map(x, labels, n_states, weighting='state'):
    """Ward merges of state centroids in the fitted, balanced feature space.

    ``state`` gives each existing centroid equal mass. ``observation`` uses
    dictionary sample counts and minimizes the increase in within-state squared
    error at each merge. No ant classes or spatial labels enter either hierarchy.
    """
    labels=np.asarray(labels,int);ids=np.unique(labels)
    if not np.array_equal(ids,np.arange(len(ids))):
        raise ValueError('State IDs must be contiguous and nonnegative')
    if not 1 <= n_states <= len(ids):raise ValueError('Invalid merged state count')
    if n_states==len(ids):return ids
    centers=np.array([x[labels==s].mean(axis=0) for s in ids],float)
    if weighting=='state':
        return cut_tree(linkage(centers,method='ward'),n_clusters=n_states).ravel()
    if weighting!='observation':raise ValueError('Unknown merge weighting')
    mass=np.bincount(labels).astype(float);mapping=ids.copy()
    while len(mass)>n_states:
        cost=((centers[:,None]-centers[None,:])**2).sum(axis=2)
        cost*=mass[:,None]*mass[None,:]/(mass[:,None]+mass[None,:])
        cost[np.tril_indices(len(mass))]=np.inf
        a,b=np.unravel_index(cost.argmin(),cost.shape)
        centers[a]=(centers[a]*mass[a]+centers[b]*mass[b])/(mass[a]+mass[b])
        mass[a]+=mass[b];centers=np.delete(centers,b,axis=0);mass=np.delete(mass,b)
        mapping[mapping==b]=a;mapping[mapping>b]-=1
    return mapping


def between_state_variance(x, labels):
    """Dictionary-mass-weighted feature variation explained by state means."""
    center=x.mean(axis=0)
    return float(sum(np.sum(labels==s)*np.sum((x[labels==s].mean(axis=0)-center)**2)
                     for s in np.unique(labels)))


def resolution_scan(root, output):
    """Check lower Leiden resolutions in the unchanged feature space.

    This exploratory screen compares two new partition seeds to seed zero.
    It does not claim held-out-ant validation of these alternative dictionaries.
    """
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with np.load(root/'models/dictionary.npz') as z:
        x,original=z['x'],z['state']
    g=graph(x);denominator=between_state_variance(x,original)
    rows=[];labels={}
    for resolution in (.01,.025,.05,.075,.1,.15,.2,.25,.3,.35,.4):
        fits=[partition(g,resolution,SEED+s) for s in range(3)]
        row=dict(resolution=resolution,n_states=int(fits[0].max()+1),
                 seed_ari=float(np.median([adjusted_rand_score(fits[0],v) for v in fits[1:]])),
                 retained_between_state_variance=between_state_variance(x,fits[0])/denominator)
        rows.append(row)
        for repeat,fit in enumerate(fits):labels[f'r{resolution:g}_seed{repeat}']=fit
        print('RESOLUTION',row,flush=True)
    pd.DataFrame(rows).to_csv(output/'resolution_search.csv',index=False)
    np.savez_compressed(output/'resolution_labels.npz',**labels)


def reduction_scan(root, output, repeats=5, seed_repeats=3, weighting='state'):
    """Audit exact state merges against fresh Leiden fits, then select a scale.

    Refit the original Leiden model before merging each perturbation: comparing
    a fixed dictionary with itself would exaggerate the stability of a merge.
    Preserve >=90% of the original between-state feature variation, and require
    seed/ant-removal stability at least as high as the original dictionary.
    The 85%, 90%, and 95% retention sensitivity is saved rather than concealed.
    """
    import hashlib
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256((root/'models/dictionary.npz').read_bytes()).hexdigest()
    signature=dict(dictionary_sha256=digest,repeats=repeats,seed_repeats=seed_repeats,weighting=weighting)
    manifest=output/'reduction_inputs.json'
    if manifest.exists() and json.loads(manifest.read_text())!=signature:
        raise ValueError('Reduction output belongs to a different dictionary or repeat count')
    manifest.write_text(json.dumps(signature,indent=2)+'\n')
    with np.load(root/'models/dictionary.npz') as z:data=dict(z)
    choice=json.loads((root/'models/selection.json').read_text())
    x,original,ant=data['x'],data['state'],data['ant'];fine_k=int(original.max()+1)
    maps={k:merge_map(x,original,k,weighting) for k in range(2,fine_k+1)}
    references={k:m[original] for k,m in maps.items()}
    denominator=between_state_variance(x,original)
    records=[];state_records=[]
    def evaluate(kind,repeat,fit_x,fitted,test,neighbors=None,distance=None):
        fine_prediction=fitted if neighbors is None else vote(fitted,neighbors,distance)[0]
        for k in maps:
            # The unmerged baseline keeps the native count of the repeat fit.
            if k==fine_k:m=np.arange(int(fitted.max()+1))
            else:m=merge_map(fit_x,fitted,min(k,int(fitted.max()+1)),weighting)
            prediction=m[fine_prediction];reference=references[k][test]
            jac=matched_jaccard(reference,prediction)
            records.append(dict(kind=kind,repeat=repeat,n_states=k,
                native_repeat_states=int(fitted.max()+1),predicted_states=int(prediction.max()+1),
                ari=adjusted_rand_score(reference,prediction),minimum_jaccard=float(jac.min())))
            state_records.extend(dict(kind=kind,repeat=repeat,n_states=k,state=s,jaccard=float(v))
                                 for s,v in enumerate(jac))
    graph_path=output/'graph.pickle'
    if graph_path.exists():g=ig.Graph.Read_Pickle(str(graph_path))
    else:g=graph(x);g.write_pickle(str(graph_path))
    # Exclude seed zero: it is the reference fit, not independent evidence.
    for repeat in range(1,seed_repeats+1):
        path=output/f'seed_fit_{repeat}.npz'
        if path.exists():
            with np.load(path) as z:fitted=z['labels']
        else:
            fitted=partition(g,choice['resolution'],SEED+repeat)
            np.savez_compressed(path,labels=fitted)
        evaluate('seed',repeat,x,fitted,np.ones(len(x),bool))
        print('REDUCTION seed repeat',repeat,flush=True)
    for repeat in range(repeats):
        path=output/f'ant_removal_fit_{repeat}.npz'
        if path.exists():
            with np.load(path) as z:part=dict(z)
        else:
            rng=np.random.default_rng(SEED+100+repeat)
            removed=rng.choice(np.unique(ant),round(.2*len(np.unique(ant))),replace=False)
            keep=~np.isin(ant,removed)
            small=graph(x[keep],SEED+100+repeat)
            fitted=partition(small,choice['resolution'],SEED+repeat)
            distance,neighbors=NearestNeighbors(n_neighbors=5,n_jobs=4).fit(x[keep]).kneighbors(x[~keep])
            part=dict(keep=keep,labels=fitted,distance=distance,neighbors=neighbors)
            np.savez_compressed(path,**part)
        evaluate('ant_removal',repeat,x[part['keep']],part['labels'],~part['keep'],part['neighbors'],part['distance'])
        print('REDUCTION ant-removal repeat',repeat,flush=True)
    detail=pd.DataFrame(records);detail.to_csv(output/'state_repeats.csv',index=False)
    state_detail=pd.DataFrame(state_records);state_detail.to_csv(output/'state_repeat_jaccards.csv',index=False)
    rows=[]
    for k,mapping in maps.items():
        sub=detail[detail.n_states.eq(k)];seed=sub[sub.kind.eq('seed')];removed=sub[sub.kind.eq('ant_removal')]
        rows.append(dict(n_states=k,state_mapping=json.dumps(mapping.tolist()),
            retained_between_state_variance=between_state_variance(x,references[k])/denominator,
            seed_ari=float(seed.ari.median()),ant_removal_ari=float(removed.ari.median()),
            ant_removal_min_ari=float(removed.ari.min()),
            minimum_state_jaccard=float(removed.minimum_jaccard.median()),
            minimum_fraction=float(np.bincount(references[k]).min()/len(x))))
    table=pd.DataFrame(rows);baseline=table[table.n_states.eq(fine_k)].iloc[0]
    table['stable']=(table.seed_ari>=max(.8,baseline.seed_ari)-1e-9)&(
        table.ant_removal_ari>=max(.65,baseline.ant_removal_ari)-1e-9)&(
        table.minimum_state_jaccard>=baseline.minimum_state_jaccard-1e-9)
    table['qualifies']=table.stable & table.retained_between_state_variance.ge(.9)
    table.to_csv(output/'state_reduction_search.csv',index=False)
    sensitivity=[]
    for retention in (.85,.9,.95):
        passed=table[table.stable & table.retained_between_state_variance.ge(retention)]
        selected=int(passed.n_states.min()) if len(passed) else fine_k
        sensitivity.append(dict(retention_requirement=retention,n_states=selected))
    selected=sensitivity[1]['n_states'];result=table[table.n_states.eq(selected)].iloc[0].to_dict()
    result['state_mapping']=maps[selected].tolist();result['original_n_states']=fine_k
    result['parent_dictionary_sha256']=digest;result['retention_sensitivity']=sensitivity;result['merge_weighting']=weighting
    result['supported']=bool(result['qualifies'])
    result['decision']=('supported_reduction' if selected<fine_k else
                        'retain_reference_no_supported_reduction')
    result['reference_seed_ari']=float(baseline.seed_ari)
    result['reference_ant_removal_ari']=float(baseline.ant_removal_ari)
    result['rule']='Coarsest Ward merge retaining >=90% of between-state feature variation, with seed ARI>=.8 and ant-removal ARI>=.65 and no decrease from the original in median seed ARI, ant-removal ARI, or weakest-state Jaccard; refit Leiden before merging every repeat. No ant-class or spatial target.'
    (output/'state_reduction_selection.json').write_text(json.dumps(result,indent=2)+'\n')
    print('SELECTED STATE REDUCTION',json.dumps(result),flush=True)
    return result


def learn(root):
    ants=pd.read_csv(root/'inventory.csv');ants=ants[ants.selected].reset_index(drop=True)
    output=root/'models';output.mkdir(exist_ok=True)
    rng=np.random.default_rng(SEED)
    raw=[];ant_ids=[];window_ids=[]
    for ai,ant in enumerate(ants.ant):
        data=np.load(root/'features'/ant.replace(':','_')/'raw.npy',mmap_mode='r')
        length=np.load(root/'features'/ant.replace(':','_')/'length.npy')
        # Equal ant contribution, duration-weighted within ant: fragmented
        # short windows must not dominate the landmark graph.
        rows=np.sort(rng.choice(len(data),min(300,len(data)),replace=False,p=length/length.sum()))
        raw.append(data[rows]);ant_ids.extend([ai]*len(rows));window_ids.extend(rows)
    raw=np.concatenate(raw);ant_ids=np.asarray(ant_ids);window_ids=np.asarray(window_ids)
    removed=rng.choice(len(ants),max(1,round(.2*len(ants))),replace=False)
    keep=~np.isin(ant_ids,removed);evaluate=np.isin(ant_ids,removed)
    search=[]
    for name in SETS:
        path=output/(name+'.npz')
        if path.exists():
            search.extend(pd.read_csv(path.with_suffix('.csv')).to_dict('records'));continue
        norm=fit_normalizer(raw,name);x=encode(raw,norm)
        g=graph(x);small=graph(x[keep],SEED+1)
        neighbors=NearestNeighbors(n_neighbors=5,n_jobs=4).fit(x[keep])
        d,ix=neighbors.kneighbors(x[evaluate])
        labels=[];rows=[]
        for resolution in (.4,.8,1.2,1.6):
            lab=partition(g,resolution);other=partition(g,resolution,SEED+1)
            perturb=partition(small,resolution);pred,_=vote(perturb,ix,d)
            k=int(lab.max()+1);counts=np.bincount(lab)
            observed_matches=matched_jaccard(lab[evaluate],pred)
            matches=np.zeros(k);matches[:len(observed_matches)]=observed_matches
            seed=adjusted_rand_score(lab,other);stability=adjusted_rand_score(lab[evaluate],pred)
            support=min(len(np.unique(ant_ids[lab==s])) for s in range(k))
            row=dict(representation=name,dimensions=x.shape[1],resolution=resolution,n_states=k,seed_ari=seed,
                     ant_removal_ari=stability,minimum_state_jaccard=float(matches.min()),minimum_ants=support,minimum_fraction=counts.min()/len(lab),
                     qualifies=bool(seed>=.8 and stability>=.65 and support>=5 and counts.min()/len(lab)>=.002))
            print('STATE MODEL',json.dumps(row),flush=True);rows.append(row);labels.append(lab)
            pd.DataFrame(dict(state=np.arange(k),matched_jaccard=matches)).to_csv(output/f'{name}_{resolution:g}_state_stability.csv',index=False)
        np.savez(path,labels=labels,x=x,raw=raw,ant=ant_ids,window=window_ids)
        (output/(name+'_normalizer.json')).write_text(json.dumps(norm,indent=2)+'\n')
        pd.DataFrame(rows).to_csv(path.with_suffix('.csv'),index=False);search.extend(rows)
    table=pd.DataFrame(search);table.to_csv(output/'search.csv',index=False)
    passing=table[table.qualifies]
    ordering=['n_states','ant_removal_ari','seed_ari'] if len(passing) else ['ant_removal_ari','seed_ari','n_states']
    choice=(passing if len(passing) else table).sort_values(ordering,ascending=False).iloc[0].to_dict()
    choice['supported']=bool(len(passing));choice['rule']='Finest passing partition; seed ARI>=.8, ant-removal ARI>=.65, >=5 ants/state, minimum fraction .002; no spatial/sleep/role target.'
    (output/'selection.json').write_text(json.dumps(choice,indent=2)+'\n')
    print('SELECTED',json.dumps(choice),flush=True)
    return choice


def assign(root):
    output=root/'models';choice=json.loads((output/'selection.json').read_text())
    norm=json.loads((output/(choice['representation']+'_normalizer.json')).read_text())
    with np.load(output/(choice['representation']+'.npz')) as z:data=dict(z)
    labels=data['labels'][list((.4,.8,1.2,1.6)).index(choice['resolution'])]
    # Sort IDs by combined median body and antennal movement, for interpretation.
    order=np.argsort([np.median(data['raw'][labels==s,24:28],axis=0).sum() for s in range(labels.max()+1)])
    labels=np.argsort(order)[labels]
    index=NNDescent(data['x'],n_neighbors=30,random_state=SEED,n_jobs=4);index.prepare()
    ants=pd.read_csv(root/'inventory.csv');ants=ants[ants.selected].reset_index(drop=True)
    total=0;audits=[]
    for ai,ant in enumerate(ants.ant):
        folder=root/'features'/ant.replace(':','_');raw=np.load(folder/'raw.npy',mmap_mode='r')
        out=root/'assignments'/ant.replace(':','_');out.mkdir(parents=True,exist_ok=True)
        state=np.empty(len(raw),np.int16);confidence=np.empty(len(raw),np.float32)
        for lo in range(0,len(raw),30000):
            hi=min(len(raw),lo+30000);x=encode(raw[lo:hi],norm)
            ix,d=index.query(x,k=5,epsilon=.15);state[lo:hi],confidence[lo:hi]=vote(labels,ix,d)
        exact=data['ant']==ai;state[data['window'][exact]]=labels[exact];confidence[data['window'][exact]]=1
        # Audit approximate-neighbor assignment against exact neighbors.
        rng=np.random.default_rng(SEED+ai);sample=rng.choice(len(raw),min(100,len(raw)),replace=False)
        d,ix=NearestNeighbors(n_neighbors=5,n_jobs=4).fit(data['x']).kneighbors(encode(raw[sample],norm))
        truth,_=vote(labels,ix,d)
        audits.append(dict(ant=ant,exact_agreement=float(np.mean(truth==state[sample]))))
        np.save(out/'state.npy',state);np.save(out/'confidence.npy',confidence)
        total+=len(raw);print('ASSIGNED',ant,len(raw),'exact audit',audits[-1]['exact_agreement'],flush=True)
    np.savez(output/'dictionary.npz',x=data['x'],raw=data['raw'],state=labels,ant=data['ant'],window=data['window'],order=order)
    pd.DataFrame(audits).to_csv(output/'assignment_audit.csv',index=False)
    print('ALL SEGMENT WINDOWS',total,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['features','learn','assign','reduce','resolutions'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--output',type=Path,help='Separate audit directory for reduce')
    p.add_argument('--merge-weighting',choices=['state','observation'],default='observation');a=p.parse_args()
    with threadpool_limits(limits=4):
        if a.action=='features':
            ants=pd.read_csv(a.root/'inventory.csv');ants=ants[ants.selected]
            with ProcessPoolExecutor(max_workers=a.workers) as pool:
                results=list(pool.map(make_features,[(a.root,ant) for ant in ants.ant]))
            pd.DataFrame(results).to_csv(a.root/'feature_coverage.csv',index=False)
        elif a.action=='learn':learn(a.root)
        elif a.action=='resolutions':resolution_scan(a.root,a.output or a.root/'reduction')
        elif a.action=='reduce':reduction_scan(a.root,a.output or a.root/'reduction',weighting=a.merge_weighting)
        else:assign(a.root)


if __name__=='__main__':main()
