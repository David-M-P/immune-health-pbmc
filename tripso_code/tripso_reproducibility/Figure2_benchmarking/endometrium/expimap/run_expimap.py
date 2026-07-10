import scanpy as sc
import scarches as sca
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from datasets import load_from_disk

# set seed
import torch
import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

######################################
# Load the data
######################################

data_dir = 'tripso_reproducibility/02.1_benchmarking_repeat/endometrium'
root_dir = 'tripso_reproducibility/02.1_benchmarking_repeat/endometrium/expimap/expimap_1'

# GP genes and HVG
adata_full = sc.read_h5ad('tripso_reproducibility/02.1_benchmarking_repeat/endometrium/data/processed/input_h5ad/endometrium_gp_genes.h5ad')

######################################
# Load the GP
######################################

# convert database to .gmt format
gpdb = pd.read_csv(os.path.join(data_dir, 'gpdb_progeny_200.csv'))
gpdb = gpdb.drop(columns = ['Trail'])

# remove spaces in columns for annotation function
gpdb.columns = [c.replace(" ", "_") for c in gpdb.columns]

gp_gmt = gpdb.T

gp_gmt.to_csv('gpdb_progeny_T.txt', sep = "\t")

# Read the Reactome annotations, make a binary matrix where rows represent gene symbols and columns represent the terms, and add the annotations matrix to the reference dataset. 
# for comparison with other methods, min_genes is decreased to 1 
sca.utils.add_annotations(adata_full, 'gpdb_progeny_T.txt', min_genes=1, clean=False)

# Remove all genes not present in annotation
adata_full._inplace_subset_var(adata_full.varm['I'].sum(1)>0)


# Split into train and test set
# Split into train and test set 
from datasets import load_from_disk
x_train = load_from_disk('tripso_reproducibility/02.1_benchmarking_repeat/endometrium/run_1/output_global/embeddings/train_set')
x_test = load_from_disk('tripso_reproducibility/02.1_benchmarking_repeat/endometrium/run_1/output_global/embeddings/test_set')

adata = adata_full[adata_full.obs.index.isin(x_train['idx'])]
adata_test = adata_full[adata_full.obs.index.isin(x_test['idx'])]


intr_cvae = sca.models.EXPIMAP(
    adata=adata,
    condition_key='batch_key',
    hidden_layer_sizes=[256, 128, 64],
    recon_loss='nb'
)

ALPHA = 0.1 # 0.2

early_stopping_kwargs = {
    "early_stopping_metric": "val_unweighted_loss", # val_unweighted_loss
    "threshold": 0,
    "patience": 50,
    "reduce_lr": True,
    "lr_patience": 13,
    "lr_factor": 0.1,
}

intr_cvae.train(
    n_epochs=100,
    alpha_epoch_anneal=50,
    alpha=ALPHA,
    alpha_kl=0.001, # 0.5,
    weight_decay=0.,
    early_stopping_kwargs=early_stopping_kwargs,
    use_early_stopping=True,
    monitor_only_val=False,
    seed=0,
)

intr_cvae.save(os.path.join(root_dir, "model_alphakl001/"), overwrite = True)

# intr_cvae = intr_cvae.load('model', adata_train)

MEAN = False

adata.obsm['X_cvae'] = intr_cvae.get_latent(adata.X, 
                                            adata.obs['batch_key'], 
                                            mean=MEAN, 
                                            only_active=False)

latent = sc.AnnData(X = adata.obsm['X_cvae'],
                    obs = adata.obs)

latent.var.index = adata.uns['terms']

sc.pp.neighbors(latent)
sc.tl.umap(latent)

latent.write_h5ad(os.path.join(root_dir, 'train_latent.h5ad'))

# for test set
adata_test.obsm['X_cvae'] = intr_cvae.get_latent(adata_test.X, 
                                            adata_test.obs['batch_key'], 
                                            mean=MEAN, 
                                            only_active=False)

latent = sc.AnnData(X = adata_test.obsm['X_cvae'],
                    obs = adata_test.obs)

latent.var.index = adata.uns['terms']

sc.pp.neighbors(latent)
sc.tl.umap(latent)

latent.write_h5ad(os.path.join(root_dir, 'test_latent.h5ad'))
