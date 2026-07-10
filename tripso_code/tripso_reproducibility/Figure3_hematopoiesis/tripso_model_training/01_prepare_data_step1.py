import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd

from geneformer import ENSEMBL_DICTIONARY_FILE
ensembl_dict = pd.read_pickle(ENSEMBL_DICTIONARY_FILE)



##############################################
print(' ------- Wrangle MNC ------- ')
##############################################

mnc = sc.read_h5ad(
    '/lustre/scratch126/cellgen/team361/data/trace/tomo/HCA_share_20240828/data/MNC_RNA_80393cells_to_share.h5ad'
)

# Reset raw counts
mnc.X = mnc.layers['counts']

# Rename ensembl ID 
# and set as index for merging
mnc.var = mnc.var.rename(columns = {'gene_ids' : 'ensembl_id'})

mnc = mnc[:, mnc.var['ensembl_id'].notnull()]
duplicate_genes = mnc.var[mnc.var['ensembl_id'].duplicated()]['ensembl_id'].values
mnc = mnc[:, ~mnc.var['ensembl_id'].isin(duplicate_genes)]

mnc.var = mnc.var.reset_index()
mnc.var = mnc.var.set_index('ensembl_id', drop = False)

# Wrangle obs columns

mnc_cols = [
    'runid_mrna_sample', 'sorting', 'biological_replicate_labID',
    'age', 'sex', 'tissue', 'age_general',
     'phase',
     'celltype_v2', 'donor_tissue'
    # 'assignment_id', 'mrna_samples', 'runid',  'runid_prot_samples', 'prot_samples',  'n_genes_by_counts', 'log1p_n_genes_by_counts', 'total_counts', 'log1p_total_counts', 'pct_counts_in_top_20_genes', 'total_counts_mt', 'log1p_total_counts_mt', 'pct_counts_mt', 'S_score', 'G2M_score', 'celltype_v1', 'leiden',
       ]

mnc.obs = mnc.obs[mnc_cols]
mnc.obs = mnc.obs.rename(columns = {'celltype_v2' : 'cell_type'})
mnc.obs['tissue'] = mnc.obs['donor_tissue'].str.split('_', n=2).str[1]
mnc.obs['source'] = 'in vivo'
mnc.obs['tissue_source'] = mnc.obs['source'].astype(str) + '_' + mnc.obs['tissue'].astype(str)

# Recalculate total counts
mnc.obs['n_counts'] = mnc.X.sum(axis = 1)

# Drop extra fields
del mnc.uns
del mnc.obsm
del mnc.obsp

# Save
mnc.obs['study'] = 'Isobe_MNC'
mnc.write_h5ad('data/processed/mnc.h5ad')



##############################################
# print(' ------- Wrangle Wrangle CD34+ in vivo dataset ------- ')
##############################################


cd34 = sc.read_h5ad('/lustre/scratch126/cellgen/team361/data/trace/tomo/HCA_share_20240828/data/CD34_RNA_98266cells_to_share.h5ad')

# Reset raw counts
cd34.X = cd34.layers['counts']

# Rename ensembl id
cd34.var = cd34.var.reset_index()
cd34.var['ensembl_id'] = cd34.var['gene_ids']
cd34.var = cd34.var.set_index('gene_ids')


# Wrangle obs columns
cd34_cols = [
    'runid_mrna_sample', # 'assignment_id', 
    'age', 'sorting', 'sex', 'tissue', 'age_general', 'phase', 'S_score', 'G2M_score',
    'leiden', 'celltype_v2', 'donor_tissue'
]

cd34.obs = cd34.obs[cd34_cols]
cd34.obs = cd34.obs.rename(columns = {'celltype_v2' : 'cell_type'})

cd34.obs['source'] = 'in vivo'
cd34.obs['tissue_source'] = cd34.obs['source'].astype(str) + '_' + cd34.obs['tissue'].astype(str)

# Recalculate total counts
cd34.obs['n_counts'] = cd34.X.sum(axis = 1)

# Drop extra fields
del cd34.obsm
del cd34.varm
del cd34.obsp

# Save
cd34.obs['study'] = 'Isobe_CD34'
cd34.write_h5ad('data/processed/cd34.h5ad')



##############################################
print(' ------- Wrangle cord blood dataset ------- ')
##############################################


cd_full =sc.read_h5ad('/lustre/scratch126/cellgen/team361/data/trace/qi/adata_cord_blood_as_submitted_with_RawCounts.h5ad')

# Reset raw counts
raw_data = cd_full.raw
adata_from_raw = ad.AnnData(X=cd_full.uns['raw_counts'], var=raw_data.var, obs=cd_full.obs)
cb = adata_from_raw

# Map ensembl ID 
cb.var['ensembl_id'] = cb.var.index.map(ensembl_dict)
cb = cb[:, cb.var['ensembl_id'].notnull()]

duplicate_genes = cb.var[cb.var['ensembl_id'].duplicated()]['ensembl_id'].values
cb = cb[:, ~cb.var['ensembl_id'].isin(duplicate_genes)]

cb.var = cb.var.reset_index()
cb.var['ens'] = cb.var['ensembl_id'].values.copy()
cb.var = cb.var.set_index('ens')

# Wrangle obs columns
cb_cols =  [
    'Timepoint', 'GFP', 'Tissue', 'Batch', 
    'phase', 'clones', 'Meta clones', 'leiden', 'def_lab'
]

cb.obs = cb.obs[cb_cols]

cb.obs = cb.obs.rename(columns = {'def_lab' : 'cell_type'})

cb.obs['source'] = 'in vitro'

# Recalculate total counts
cb.obs['n_counts'] = cb.X.sum(axis = 1)


# Save
cb.obs['study'] = 'Gao_CB'
cb.write_h5ad('data/processed/cb.h5ad')

##############################################
print(' ------- Wrangle in vitro HSCs ------- ')
##############################################

hsc = sc.read_h5ad('data/raw/GSE192519_Integrated_hHSC.h5ad')

hsc.var['ensembl_id'] = hsc.var.index.map(ensembl_dict)
hsc = hsc[:, hsc.var['ensembl_id'].notnull()]

duplicate_genes = hsc.var[hsc.var['ensembl_id'].duplicated()]['ensembl_id'].values
hsc = hsc[:, ~hsc.var['ensembl_id'].isin(duplicate_genes)]

hsc.var['ens'] = hsc.var['ensembl_id'].values.copy()
hsc.var = hsc.var.set_index('ens')

# Wrangle obs columns
hsc_cols =  [
   'orig.ident', 'Phase', 'seurat_clusters'
]

hsc.obs = hsc.obs[hsc_cols]
hsc.obs = hsc.obs.rename(columns = {'orig.ident' : 'condition'})

# Add cell type labels
sakurai_cluster = pd.read_csv('sakurai_cell_types.csv')

leiden_to_cell_type = dict(zip(
    sakurai_cluster['seurat_clusters'].astype(int), sakurai_cluster['cell_type']
))

hsc.obs['cell_type'] = hsc.obs['seurat_clusters'].astype(int).map(leiden_to_cell_type)


hsc.obs['source'] = 'in vitro'

# Recalculate total counts
hsc.obs['n_counts'] = hsc.X.sum(axis = 1)


# Save
hsc.obs['study'] = 'Sakurai_HSC'
hsc.write_h5ad('data/processed/sakurai_hsc.h5ad')

# ##############################################
# print(' ------- Wrangle Zeng dataset ------- ')
# ##############################################


zeng = sc.read_h5ad(
    '/lustre/scratch126/cellgen/team361/am74/Adib/TRACE/cohort/Processed/TRACE_Approach/Zeng_et_al_Bone_marrow_healthy_harmonized_qc.h5ad'
)

# drop duplicate ensembl ids
duplicated_vars = zeng.var_names[zeng.var_names.duplicated()]
zeng = zeng[:, ~zeng.var.index.isin(duplicated_vars)]

zeng_cols = [
    'Study', 'donor_id', 'Sorting',
    'ExactAge', 'S.Score', 'G2M.Score', 'CyclePhase',
    'cell_type', 'AuthorCellType', 
]

zeng.obs = zeng.obs[zeng_cols]
zeng.obs = zeng.obs.rename(columns = {'Study' : 'original_study',
                                      'donor_id' : 'donor'
                                     })

zeng.var = zeng.var.drop(columns = zeng.var.columns)
zeng.var['ensembl_id'] = zeng.var.index.tolist()

# add back counts
zeng.obs['n_counts'] = zeng.X.sum(axis = 1)

# add study
zeng.obs['source'] = 'in vivo'
zeng.obs['study'] = 'Zeng'
zeng.write_h5ad('data/processed/zeng.h5ad')


##############################################
print(' ------- Wrangle Hojun dataset ------- ')
##############################################


hojun = sc.read_h5ad(
    '/lustre/scratch126/cellgen/team361/am74/Adib/TRACE/cohort/Hojun_LI_HPSC_Atlas/Hojun_et_al_HPSC.h5ad'
)

# drop duplicated genes (these are pseudogenes)
duplicated_vars = hojun.var_names[hojun.var_names.duplicated()]
hojun = hojun[:, ~hojun.var.index.isin(duplicated_vars)]

hojun_cols = [
    'seurat_clusters', 'donor_id', 'tissue', 'development_stage', 
    'cell_type'
]

hojun.obs = hojun.obs[hojun_cols]
hojun.obs = hojun.obs.rename(columns = {'donor_id' : 'donor'})
hojun.var['ensembl_id'] = hojun.var.index.tolist()
hojun.var['ens'] = hojun.var['ensembl_id'].values.copy()
hojun.var = hojun.var.set_index('ens')

# add back counts
hojun.obs['n_counts'] = hojun.X.sum(axis = 1)

# add study
hojun.obs['source'] = 'in vivo'
hojun.obs['study'] = 'Hojun'
hojun.write_h5ad('data/processed/hojun.h5ad')



##############################################
print(' ------- Merge and calculate HVGs ------- ')
##############################################

# Merge
mnc = sc.read_h5ad('data/processed/mnc.h5ad')
cd34 = sc.read_h5ad('data/processed/cd34.h5ad')
cb = sc.read_h5ad('data/processed/cb.h5ad')
hsc = sc.read_h5ad('data/processed/sakurai_hsc.h5ad')
zeng = sc.read_h5ad('data/processed/zeng.h5ad')
hojun = sc.read_h5ad('data/processed/hojun.h5ad')

adata = ad.concat([mnc, cd34, cb, hsc, zeng, hojun])

print('Merged adata shape', adata.shape)

sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes = 2000, batch_key='study')

adata.var.to_csv('data/processed/merged_adata_hvg.csv')

