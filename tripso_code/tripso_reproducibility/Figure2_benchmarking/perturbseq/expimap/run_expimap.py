import scanpy as sc
import scarches as sca
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# Change the default figure size (width, height) in inches
plt.rcParams['figure.figsize'] = (5, 6)  # Change the values (width, height) as needed

# set seed
import torch
import random
random.seed(2020)
np.random.seed(2020)
torch.manual_seed(2020)

######################################
# Load the data
######################################

root_dir = 'tripso_reproducibility/02.1_benchmarking_repeat/perturbseq/expimap/run_1'

# counts: union of HVG and GP genes 
adata_full = sc.read_h5ad('tripso_reproducibility/02.1_benchmarking_repeat/perturbseq/data/processed/input_h5ad/perturbseq_gp_genes.h5ad')
# convert to float32 for pytorch
adata_full.X = adata_full.X.astype(np.float32)

######################################
# Load the GP
######################################

# convert database to .gmt format
gpdb = pd.read_csv('tripso_reproducibility/02.1_benchmarking_repeat/perturbseq/gpdb_progeny_tnfa_tgfb.csv')

# remove spaces in columns for annotation function
gpdb.columns = [c.replace(" ", "_") for c in gpdb.columns]

gp_gmt = gpdb.T

gp_gmt.to_csv('gpdb_manual_T.txt', sep = "\t")

# Read the Reactome annotations, make a binary matrix where rows represent gene symbols and columns represent the terms, and add the annotations matrix to the reference dataset. 
# for comparison with other methods, min_genes is decreased to 1 
sca.utils.add_annotations(adata_full, 'gpdb_manual_T.txt', min_genes=1, clean=False)

# Remove all genes not present in annotation
adata_full._inplace_subset_var(adata_full.varm['I'].sum(1)>0)

# Split into train and test set 
from datasets import load_from_disk
x_train = load_from_disk('tripso_reproducibility/02.1_benchmarking_repeat/perturbseq/run_1/output_global/embeddings/train_set')
x_test = load_from_disk('tripso_reproducibility/02.1_benchmarking_repeat/perturbseq/run_1/output_global/embeddings/test_set')

adata = adata_full[adata_full.obs.index.isin(x_train['idx'])]
adata_test = adata_full[adata_full.obs.index.isin(x_test['idx'])]

print('Training data', adata.shape)

intr_cvae = sca.models.EXPIMAP(
    adata=adata,
    condition_key='Batch_info',
    hidden_layer_sizes=[256, 128, 32],
    recon_loss='nb'
)

ALPHA = 0.2

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
    alpha_kl=0.01, # 0.5,
    weight_decay=0.,
    early_stopping_kwargs=early_stopping_kwargs,
    use_early_stopping=True,
    monitor_only_val=False,
    seed=2020,
)

intr_cvae.save(os.path.join(root_dir, "run_1/model/"), overwrite = True)

MEAN = False

## for training set
adata.obsm['X_cvae'] = intr_cvae.get_latent(adata.X, 
                                            adata.obs['Batch_info'], 
                                            mean=MEAN, 
                                            only_active=False)

latent = sc.AnnData(X = adata.obsm['X_cvae'],
                    obs = adata.obs)

latent.var.index = adata.uns['terms']

latent.write_h5ad(os.path.join(root_dir, 'run_1/train_latent.h5ad'))

# for test set 
adata_test.obsm['X_cvae'] = intr_cvae.get_latent(adata_test.X, 
                                            adata_test.obs['Batch_info'], 
                                            mean=MEAN, 
                                            only_active=False)

latent = sc.AnnData(X = adata_test.obsm['X_cvae'],
                    obs = adata_test.obs)

latent.var.index = adata.uns['terms']

latent.write_h5ad(os.path.join(root_dir, 'run_1/test_latent.h5ad'))
