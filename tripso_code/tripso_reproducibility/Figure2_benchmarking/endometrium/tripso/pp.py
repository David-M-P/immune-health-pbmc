import pickle
import scanpy as sc
from geneformer import ENSEMBL_MAPPING_FILE, GENE_MEDIAN_FILE, TOKEN_DICTIONARY_FILE
import os
import pandas as pd

adata = sc.read_h5ad('cellxgene_objects/endometriumAtlasV2_cells_with_counts.h5ad')
adata.X = adata.layers['counts']

# for converting between gene formats
# load gene token dict
with open(
    TOKEN_DICTIONARY_FILE,
    'rb',
) as f:
    token_dictionary = pickle.load(f)


# load gene name to ensembl dict
with open(
    ENSEMBL_MAPPING_FILE,
    'rb',
) as f:
    name_dictionary = pickle.load(f)

ensembl_to_name = {v: k for k, v in name_dictionary.items()}
token_to_gene = {v: k for k, v in token_dictionary.items()}


adata.var = adata.var.drop(columns = adata.var.columns)

adata.var['ensembl_id'] = adata.var.index.map(name_dictionary)

# remove genes with no ensembl id
n1 = adata.shape[1]
adata = adata[:, ~adata.var['ensembl_id'].isna()]
n2 = adata.shape[1]
print(f"Removed {n1-n2} genes with no ensembl id")

del adata.obsm
del adata.layers['counts']
del adata.obsp
del adata.uns

for c in ["celltype",'Binary Stage', 'Group', 'Endometriosis_stage', 'Hormonal treatment', 'lineage']:
    print(f'{c}: {len(adata.obs[c].unique())}')

# set up directory
import os
os.makedirs('data', exist_ok=True)

adata.write_h5ad('data/processed/heca.h5ad')

# -------------------------------------------------------
# Prepare tripso
# -------------------------------------------------------

import tripso 

# Directory paths for loading/saving 
root_dir = 'tripso_reproducibility/02.1_benchmarking_repeat/endometrium' 

gpdb = pd.read_csv('../gpdb_progeny_200.csv')

gp_genes = set()
for g in gpdb.columns:
    if g != 'Trail':
        gp_genes.update(set(gpdb[g].dropna().values))

adata = sc.read_h5ad('cellxgene_objects/endometriumAtlasV2_cells.h5ad')
hvg = adata[:, adata.var['highly_variable']].var_names

genes_to_keep = set(hvg) | gp_genes

# load data and preprocess
tripso.pp_and_tokenize(root_dir=root_dir,
                          adata_path = os.path.join(root_dir, 'data/processed/heca.h5ad'),
                          vars_to_keep = ["celltype", "n_counts", 'Binary Stage', 'Group', 'Endometriosis_stage', 'Hormonal treatment', 'lineage'],
                          cov_to_encode = ["celltype", 'Binary Stage', 'Endometriosis_stage', 'lineage'],
                          batch_keys = 'dataset',
                          hvg_batch_key = 'dataset',
                          subsample_by = None,
                          name_tag='progeny_200',
                          save_gp_genes_object = True,
                          calculate_hvg=False,
                          input_size = 4096,
                          gp_genes_union = genes_to_keep,
                          use_gp_tokenizer = True,
                         )
