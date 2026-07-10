#import packages
import scanpy as sc
import numpy as np
import pandas as pd
import os

#spectra imports 
# from Spectra import Spectra_gpu as spc
import Spectra as spc
from Spectra import Spectra_util as spc_tl

# from datasets import load_from_disk
# from tripso.Evaluate.downstream import gpEval

SEED = 0

# set seed
np.random.seed(SEED)

import torch
torch.manual_seed(SEED)

    
################################################
# Load gene set dictionary
################################################

root_dir = f'tripso_reproducibility/05_gpfinder/spectra_repeat/recover_tnfa/seed_{SEED}'

gpdb = pd.read_csv(
    'tripso_reproducibility/02.1_benchmarking_repeat/perturbseq/gpdb_progeny_tnfa_tgfb.csv'
    )

################################################
# Load adata
################################################

# define data paths
obs_key = 'target_pathway' #indicat the column name for the dataframe in adata.obs where to find the cell type lab
# nb if using cell-type specific markers these need to match those in gene set dictionary

# load adata
adata = sc.read_h5ad('tripso_reproducibility/02.1_benchmarking_repeat/perturbseq/data/Jiang.h5ad')
adata.obs_names_make_unique()

# Normalize and log transform
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# keep union of HVG and GP genes 
adata_ctrl = sc.read_h5ad('tripso_reproducibility/02.1_benchmarking_repeat/perturbseq/data/processed/input_h5ad/perturbseq_gp_genes.h5ad')
adata_ctrl.obs_names_make_unique()

# Only keep TGFb and TNFa cells
adata = adata[adata_ctrl.obs_names]

# Keep TNFa genes and HVG
adata = adata[:, adata.var_names.isin(gpdb['TNFa']) | adata.var['highly_variable']]

print('adata', adata.shape)
print(adata)

gene_set_dict = {
    'TNFa': {
        'TNFa_from_HVG' : adata.var_names[adata.var['highly_variable']].tolist(),
    },
    'TGFb': {'TGFb' : gpdb['TGFb'].dropna().tolist()},  
    'global': {
        'HVG' : adata.var_names[adata.var['highly_variable']].tolist()
        }  
}


print(gene_set_dict)
#filter gene set annotation dict for genes contained in adata
annotations = spc_tl.check_gene_set_dictionary(
    adata,
    gene_set_dict,
    obs_key='target_pathway',
    global_key='global')

# Vocab is a boolean array that is True for genes that were used while fitting the model 
# note that this quantity is only added to the AnnData when highly_variable is set to True:
adata.var['spectra_vocab']= True

# ################################################
# # Fit the model
# ################################################

# Training
# **Returns**: SPECTRA_Model object [after training]
# **In place**: adds 1. factors, 2. cell scores, 3. vocabulary, and 4. markers as attributes in .obsm, .var, .uns

# Recommended 10_000 epochs
model = spc.est_spectra(adata=adata,
    gene_set_dictionary=annotations,
    use_highly_variable=False,
    cell_type_key="target_pathway",
    use_weights=True,
    lam=0.1, # varies depending on data and gene sets, try between 0.5 and 0.001
    delta=0.001,
    kappa=None,
    rho=0.001,
    use_cell_types=True,
    n_top_vals=50,
    label_factors=True, # absent in GPU mode
    overlap_threshold=0.2,
    clean_gs = True,
    min_gs_num = 3,
    num_epochs=5_000 
                       )

adata.write_h5ad(os.path.join(root_dir, "adata_spectra.h5ad"))

#this way needs less storage but requires the original adata, annotations and cell type annotations to load the model again
model.save(os.path.join(root_dir, 'spectra_model_compact'))
