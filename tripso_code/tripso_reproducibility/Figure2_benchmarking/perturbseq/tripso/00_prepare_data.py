import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad 

tgfb = sc.read_h5ad('data/raw/TGFB_Perturbseq.h5ad')
tgfb.obs['target_pathway'] = 'TGFb'
print('tgfb', tgfb.X.max())
print(type(tgfb.X))

tnfa = sc.read_h5ad('data/raw/TNFA_Perturbseq.h5ad')
tnfa.obs['target_pathway'] = 'TNFa'
print('tnfa', tnfa.X.max())
print(type(tnfa.X))

adata = ad.concat([tgfb, tnfa])

# add n_counts for Geneformer
adata.obs['n_counts'] = adata.X.sum(axis = 1)

sc.pp.highly_variable_genes(adata, flavor = 'seurat_v3', batch_key = 'Batch_info')

# add ensembl gene names
from geneformer import ENSEMBL_MAPPING_FILE
name_to_ens = pd.read_pickle(ENSEMBL_MAPPING_FILE)

adata.var['ensembl_id'] = adata.var.index.map(name_to_ens)

# remove genes with no ensembl id
n1 = adata.shape[1]
adata = adata[:, ~adata.var['ensembl_id'].isna()]
n2 = adata.shape[1]
print(f"Removed {n1-n2} genes with no ensembl id")

adata.write_h5ad('data/Jiang.h5ad')
