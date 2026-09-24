"""Focused activity landscape: one frozen method, K evidence and density-aware UMAP."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
from analysis.activity_discovery import ActivityTransform

UMAP_PARAMETERS=dict(n_neighbors=15,min_dist=.01,n_components=2,metric='euclidean',densmap=True,dens_lambda=2.,dens_frac=.3,n_epochs=700,random_state=724,transform_seed=724,n_jobs=1)
INPUTS=('ACTIVITY_ONLY_FROZEN.json','activity_models.joblib','activity_feature_bank.npz','activity_cohort_audit.csv','activity_spatial_comparison.csv','all_candidate_tests.csv','motif_umap_input.npz','method_comparison_with_controls.csv')


def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(source,output):
    output.mkdir(parents=True,exist_ok=True)
    frozen=json.loads((source/'ACTIVITY_ONLY_FROZEN.json').read_text())
    if (frozen['selection']['representation'],frozen['selection']['algorithm'])!=('locomotion_levels','kmeans'):raise ValueError('Unexpected primary activity method')
    for name,key in [('activity_models.joblib','model_sha256'),('activity_feature_bank.npz','feature_sha256')]:
        if sha256(source/name)!=frozen[key]:raise ValueError(f'Changed frozen input: {name}')
    for name in INPUTS:shutil.copy2(source/name,output/name)
    table=pd.read_csv(source/'method_comparison_with_controls.csv')
    primary=table[table.scope.eq('common')&table.representation.eq('locomotion_levels')&table.algorithm.eq('kmeans')]
    if len(primary)!=2 or not primary.selected_k.eq(2).all():raise ValueError('Expected existing primary K=2 fits')
    primary.to_csv(output/'primary_activity_metrics.csv',index=False)
    candidates=pd.read_csv(source/'all_candidate_tests.csv');candidates[candidates.key.isin(primary.key)].to_csv(output/'primary_k_selection.csv',index=False)
    manifest=dict(source=str(source),created_utc=datetime.now(timezone.utc).isoformat(),primary_method='Full-trajectory locomotor levels / KMeans',activity_models_refitted=False,selection=frozen['selection'],input_sha256={n:sha256(source/n) for n in INPUTS},umap_parameters=UMAP_PARAMETERS,software={p:importlib.metadata.version(p) for p in ('numpy','pandas','scipy','scikit-learn','matplotlib','umap-learn')})
    (output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


def log_neighbor_radius(x,k=15):
    distances=NearestNeighbors(n_neighbors=k+1).fit(x).kneighbors(x,return_distance=True)[0][:,1:]
    return np.log(np.maximum(np.sqrt(np.mean(distances**2,axis=1)),1e-12))


def fit_umap(source,output):
    import umap
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with np.load(output/'motif_umap_input.npz') as z:data={k:z[k] for k in z.files}
    model=umap.UMAP(**UMAP_PARAMETERS);embedding=model.fit_transform(data['histories'])
    with np.load(source/'motif_umap.npz') as z:
        if not np.array_equal(z['histories'],data['histories']):raise ValueError('Different UMAP samples')
        old_embedding=z['embedding']
    original_radius=log_neighbor_radius(data['histories']);new_radius=log_neighbor_radius(embedding);old_radius=log_neighbor_radius(old_embedding)
    rng=np.random.default_rng(724);ix=rng.choice(len(embedding),min(1500,len(embedding)),replace=False)
    metadata=dict(parameters=UMAP_PARAMETERS,n_histories=len(embedding),n_ants=len(np.unique(data['ant'])),n_features=data['histories'].shape[1],original_radius_spearman_old=float(spearmanr(original_radius,old_radius).statistic),original_radius_spearman_new=float(spearmanr(original_radius,new_radius).statistic),trustworthiness_15nn_on1500=float(trustworthiness(data['histories'][ix],embedding[ix],n_neighbors=15)),definition='Spearman correlation of log RMS distance to 15 nearest neighbors in original 143D vs embedding; identical 6600 histories',embedding_role='Display only. Motif labels and individual activity clusters remain unchanged.',reference='https://umap-learn.readthedocs.io/en/latest/densmap_demo.html')
    np.savez_compressed(output/'motif_densmap.npz',**data,embedding=embedding,original_log_radius=original_radius,embedding_log_radius=new_radius)
    (output/'umap_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n');joblib.dump(model,output/'densmap_model.joblib',compress=3)
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    for ax,xy,name in zip(axs,[old_embedding,embedding],['Previous UMAP','Density-preserving UMAP']):
        im=ax.scatter(xy[:,0],xy[:,1],c=-original_radius,s=2,cmap='magma',alpha=.6);ax.set_title(name);ax.set_aspect('equal')
    fig.colorbar(im,ax=axs,label='Original local density proxy (−log neighbor radius)');fig.savefig(output/'umap_density_validation.png',dpi=160);plt.close(fig)
    print('DENSMAP_COMPLETE',json.dumps(metadata),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--block',type=Path);p.add_argument('--dictionary',type=Path)
    p.add_argument('--stage',choices=['umap','render','all'],default='all');a=p.parse_args()
    if a.stage in ('umap','all'):prepare(a.source,a.output);fit_umap(a.source,a.output)
    if a.stage in ('render','all'):
        if a.block is None or a.dictionary is None:p.error('--block and --dictionary are required for rendering')
        from analysis.activity_landscape_plots import render
        render(a.output,a.block,a.dictionary)

if __name__=='__main__':main()
