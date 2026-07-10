import scanpy as sc
import pandas as pd
import numpy as np
from datasets import load_from_disk
import os

adata = sc.read_h5ad('tripso_reproducibility/02.1_benchmarking_repeat/perturbseq/data/Jiang.h5ad')
gpdb = pd.read_csv('tripso_reproducibility/02.1_benchmarking_repeat/perturbseq/gpdb_progeny_tnfa_tgfb.csv')

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

for gp in gpdb.columns:
    sc.tl.score_genes(adata, gene_list=gpdb[gp].dropna().tolist(), score_name=gp, use_raw=False)
    
# create anndata object
# with scores genes in .X

adata_scores = sc.AnnData(
    X = np.array(adata.obs[gpdb.columns].values, dtype=np.float32),
    obs = adata.obs,
    var = pd.DataFrame(index=gpdb.columns),
)

# split in train/val/test
# get reference indices

root_dir = 'tripso_reproducibility/02.1_benchmarking_repeat/perturbseq'
output_dir = os.path.join(root_dir, 'evaluation_metrics')

emb = load_from_disk(os.path.join(root_dir, 'run_1/output_global/embeddings/train_set'))
train_idx = emb['idx']

emb = load_from_disk(os.path.join(root_dir, 'run_1/output_global/embeddings/test_set'))
test_idx = emb['idx']

# save 
adata_scores[train_idx].write_h5ad('score_genes_train.h5ad')
adata_scores[test_idx].write_h5ad('score_genes_test.h5ad')

