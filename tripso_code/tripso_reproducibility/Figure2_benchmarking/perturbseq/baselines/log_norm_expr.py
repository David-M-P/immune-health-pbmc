import scanpy as sc
import pandas as pd
import numpy as np
from datasets import load_from_disk
import os

# ==============================================
# Reference indices
# split in train/val/test
# get reference indices

root_dir = 'tripso_reproducibility/02.1_benchmarking_repeat/perturbseq'
output_dir = os.path.join(root_dir, 'evaluation_metrics')

emb = load_from_disk(os.path.join(root_dir, 'run_1/output_global/embeddings/train_set'))
train_idx = emb['idx']

emb = load_from_disk(os.path.join(root_dir, 'run_1/output_global/embeddings/test_set'))
test_idx = emb['idx']


# ==============================================
# Gene expression

adata = sc.read_h5ad('tripso_reproducibility/02.1_benchmarking_repeat/perturbseq/data/Jiang.h5ad')
gpdb = pd.read_csv('tripso_reproducibility/02.1_benchmarking_repeat/perturbseq/gpdb_progeny_tnfa_tgfb.csv')

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# ==============================================
# Set up dataset class


train_dataset_dict = {}
test_dataset_dict = {}

for gp in gpdb.columns:
    train_dataset_dict[gp] = adata[train_idx, adata.var_names.isin(gpdb[gp].dropna().tolist())].X.toarray()
    test_dataset_dict[gp] = adata[test_idx, adata.var_names.isin(gpdb[gp].dropna().tolist())].X.toarray()

# add metadata
train_dataset_dict['cell_type'] = adata.obs['cell_type'][train_idx].values
test_dataset_dict['cell_type'] = adata.obs['cell_type'][test_idx].values    

train_dataset_dict['target_pathway'] = adata.obs['target_pathway'][train_idx].values
test_dataset_dict['target_pathway'] = adata.obs['target_pathway'][test_idx].values

train_dataset_dict['gene'] = adata.obs['gene'][train_idx].values
test_dataset_dict['gene'] = adata.obs['gene'][test_idx].values

try:
    train_dataset_dict['Batch_info'] = adata.obs['Batch_info'][train_idx].values
    test_dataset_dict['Batch_info'] = adata.obs['Batch_info'][test_idx].values
except KeyError:
    # If 'Batch_info' is not present, we can skip it or handle it differently
    print("Batch_info not found in adata.obs, skipping this field.")
    train_dataset_dict['Batch_info'] = None
    test_dataset_dict['Batch_info'] = None

train_dataset_dict['idx'] = adata[train_idx].obs.index.values
test_dataset_dict['idx'] = adata[test_idx].obs.index.values

# convert to huggingface datasets
from datasets import Dataset
train_dataset = Dataset.from_dict(train_dataset_dict)
test_dataset = Dataset.from_dict(test_dataset_dict) 

# save to disk
train_dataset.save_to_disk('log_expr/embeddings/train_set')
test_dataset.save_to_disk('log_expr/embeddings/test_set')