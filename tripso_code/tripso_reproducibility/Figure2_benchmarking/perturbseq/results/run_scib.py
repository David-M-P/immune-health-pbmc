
# Fix error with jax.numpy.clip
import jax.numpy as jnp
# Patch before importing anything else
_original_clip = jnp.clip
def fixed_clip(a, *args, **kwargs):
    if 'min' in kwargs:
        kwargs['a_min'] = kwargs.pop('min')
    if 'max' in kwargs:
        kwargs['a_max'] = kwargs.pop('max')
    return _original_clip(a, *args, **kwargs)
jnp.clip = fixed_clip

# Running scIB metrics 

import numpy as np
import scanpy as sc
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection
# from datasets import load_from_disk, concatenate_datasets
import os

# =====================================================
# Set up object
# =====================================================

# # libra
# libra_dir = 'tripso_reproducibility/02.1_benchmarking_repeat/perturbseq/run_1/output_global/embeddings'
# # x1 = load_from_disk(os.path.join(libra_dir, 'train_set'))
# x2 = load_from_disk(os.path.join(libra_dir, 'test_set'))
# # x = concatenate_datasets([x1, x2])
# x = x2

# adata = sc.AnnData(
#     X = np.zeros((x.shape[0], 1)),  # Placeholder for expression data
#     obs = x.select_columns(['Batch_info', 'target_pathway', 'cell_type', 'idx']).to_pandas(),
# )

# adata.obs = adata.obs.set_index('idx')
# adata.obsm['X_libra_TGFb'] = np.array(x['TGFb'])
# adata.obsm['X_libra_TNFa'] = np.array(x['TNFa'])


# expr_dir = 'tripso_reproducibility/02.1_benchmarking_repeat/perturbseq/baselines/log_expr/embeddings'
# # expr1 = load_from_disk(os.path.join(expr_dir, 'train_set'))
# expr2 = load_from_disk(os.path.join(expr_dir, 'test_set'))
# # expr = concatenate_datasets([expr1, expr2])
# expr = expr2

# # convet to anndata first to check indices are aligned 
# expr_tgfb = sc.AnnData(
#     X = np.array(expr['TGFb']),
#     obs = expr.select_columns(['idx']).to_pandas(),
# )

# expr_tgfb.obs = expr_tgfb.obs.set_index('idx')
# expr_tgfb = expr_tgfb[adata.obs_names]

# # Add expression data to adata
# adata.obsm['X_expr_TGFb'] = expr_tgfb.X

# expr_tnfa = sc.AnnData(
#     X = np.array(expr['TNFa']),
#     obs = expr.select_columns(['idx']).to_pandas(),
# )

# expr_tnfa.obs = expr_tnfa.obs.set_index('idx')
# expr_tnfa = expr_tnfa[adata.obs_names]

# # Add expression data to adata
# adata.obsm['X_expr_TNFa'] = expr_tnfa.X

# # Save to disk
# adata.write("gp_emb_obsm_test_set.h5ad")

# =====================================================
# Perform the benchmark
# =====================================================

adata = sc.read_h5ad("gp_emb_obsm_test_set.h5ad")

bdata = sc.AnnData(
    X = np.random.rand(adata.shape[0], 256), 
    obs = adata.obs.copy(),
)

for x in adata.obsm.keys():
    bdata.obsm[x] = adata.obsm[x]

for col in ['target_pathway', 'cell_type']:
    bm = Benchmarker(
        bdata,
        batch_key="Batch_info",
        label_key=col,
        bio_conservation_metrics=BioConservation(),
        batch_correction_metrics=BatchCorrection(),
        embedding_obsm_keys=list(adata.obsm.keys()),
        n_jobs=8,
    )
    bm.benchmark()

    df = bm.get_results(min_max_scale=False)
    df.to_csv(f'scib_results_{col}.csv')




