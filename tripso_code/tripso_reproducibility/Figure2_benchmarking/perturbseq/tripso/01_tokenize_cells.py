import pickle
import scanpy as sc
from geneformer import ENSEMBL_MAPPING_FILE, GENE_MEDIAN_FILE, TOKEN_DICTIONARY_FILE
import os
import pandas as pd


# -------------------------------------------------------
# Prepare tripso
# -------------------------------------------------------

import tripso 

# Directory paths for loading/saving 
root_dir = 'tripso_reproducibility/02.1_benchmarking_repeat/perturbseq' 

gpdb = pd.read_csv('../gpdb_progeny_200.csv')

gpdb[['TNFa', 'TGFb']].to_csv('gpdb_progeny_tnfa_tgfb.csv', index=False)

gp_genes = set()
for g in gpdb.columns:
    if (g == 'TNFa') | (g == 'TGFb'):
        gp_genes.update(set(gpdb[g].dropna().values))

adata = sc.read_h5ad('data/Jiang.h5ad')
hvg = adata[:, adata.var['highly_variable']].var_names

genes_to_keep = set(hvg) | gp_genes

# load data and preprocess
tripso.pp_and_tokenize(root_dir=root_dir,
                          adata_path = 'data/Jiang.h5ad',
                          vars_to_keep= ['target_pathway', 'cell_type', 'gene', 'guide', 'Batch_info', 'mixscale_score', 'sample_ID'],
                          batch_keys = 'Batch_info',
                          calculate_hvg = False,
                          subsample_by = None,
                          cov_to_encode = ['target_pathway', 'cell_type', 'gene'],
                          save_gp_genes_object = True, 
                          name_tag='progeny_tnfa_tgfb',
                          input_size = 4096,
                          gp_genes_union = genes_to_keep,
                          use_gp_tokenizer = True,
                         )
