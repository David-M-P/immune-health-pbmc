#import packages
from datasets import load_from_disk
import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import pickle
import torch
import random


import Spectra as spc
from Spectra import Spectra_util as spc_tl

# set seed
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

    
################################################
# Load gene set dictionary
################################################

root_dir = 'tripso_reproducibility/02.1_benchmarking_repeat/endometrium/spectra/spectra_1' 
os.chdir(root_dir)

gpdb = pd.read_csv(
    'tripso_reproducibility/02.1_benchmarking_repeat/endometrium/gpdb_progeny_200.csv'
    )

gpdb = gpdb.drop(columns = ['Trail'])

all_genes = set()

for col in gpdb.columns:
    all_genes.update(gpdb[col].dropna().tolist())

all_genes = list(all_genes)


gene_set_dict = {
    'global' : {col: gpdb[col].dropna().tolist() for col in gpdb.columns},
}


################################################
# Load adata
################################################

# define data paths
obs_key = 'cell_type' #indicat the column name for the dataframe in adata.obs where to find the cell type lab
# nb if using cell-type specific markers these need to match those in gene set dictionary

adata_ctrl = sc.read_h5ad(
    'tripso_reproducibility/02.1_benchmarking_repeat/endometrium/data/processed/input_h5ad/endometrium_gp_genes.h5ad'
)

genes_to_keep = adata_ctrl.var_names

# load adata
adata = sc.read_h5ad(
    '/nfs/team292/lg18/endometriosis/cellxgene_objects/endometriumAtlasV2_cells.h5ad'
)

# select GP genes
adata = adata[:, genes_to_keep]

adata.obs = adata.obs.rename(columns={'celltype': 'cell_type'})

# add dummy cell-type specific gene sets 
for cell_type in adata.obs[obs_key].unique():
    if cell_type not in gene_set_dict:
        gene_set_dict[cell_type] = {}

#filter gene set annotation dict for genes contained in adata
annotations = spc_tl.check_gene_set_dictionary(
    adata,
    gene_set_dict,
    obs_key='cell_type',
    global_key='global')

# Vocab is a boolean array that is True for genes that were used while fitting the model 
# note that this quantity is only added to the AnnData when highly_variable is set to True:
adata.var['spectra_vocab']= True

################################################
# Fit the model
################################################

# Training
# **Returns**: SPECTRA_Model object [after training]
# **In place**: adds 1. factors, 2. cell scores, 3. vocabulary, and 4. markers as attributes in .obsm, .var, .uns

# Recommended 10_000 epochs
model = spc.est_spectra(adata=adata,
    gene_set_dictionary=annotations,
    use_highly_variable=False,
    cell_type_key="cell_type",
    use_weights=True,
    lam=0.001, # varies depending on data and gene sets, try between 0.5 and 0.001, default =0.1
    delta=0.001,
    kappa=None,
    rho=0.001,
    use_cell_types=True,
    n_top_vals=50,
    label_factors=True, # absent in GPU mode
    overlap_threshold=0.2,
    clean_gs = False,
    min_gs_num = 3,
    num_epochs=6_000,
                       )

adata.write_h5ad("adata_spectra.h5ad")

# #this way needs less storage but requires the original adata, annotations and cell type annotations to load the model again
model.save('spectra_model_compact')
