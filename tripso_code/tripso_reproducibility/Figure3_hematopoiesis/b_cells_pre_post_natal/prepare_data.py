import scanpy as sc
import pandas as pd
import numpy as np
import os
from datasets import load_from_disk, concatenate_datasets

train = load_from_disk('../../embeddings/train_set')
val = load_from_disk('../../embeddings/val_set')
test = load_from_disk('../../embeddings/test_set')

emb = concatenate_datasets([train, val, test])

adata = sc.AnnData(X=np.array(emb['GP_IKZF1']),
                   obs = emb.select_columns(['study', 'tissue',
                                             'donor_tissue',
                                             'GP_IKZF1_num_genes',
                                             'idx'
                                            ]).to_pandas(),
)


adata.obs = adata.obs.set_index('idx')

adata = adata[adata.obs['tissue'] != 'CB']

adata = adata[adata.obs['study'].str.contains('Isobe')]

adata.obs['donor'] = adata.obs['donor_tissue'].str.extract(r'^([^_]+)')

# use updated labels
cd34 = pd.read_csv('tripso_reproducibility/04.5_HSC_post_qc/data/raw/CD34_annotations.csv', index_col = 0)
mnc = pd.read_csv('tripso_reproducibility/04.5_HSC_post_qc/data/raw/MNC_cell_types.csv', index_col = 0)
cell_labels = pd.concat([cd34, mnc])

adata.obs = adata.obs.join(cell_labels)

lymphoid_cd34 = [
    '1_LT-HSC', '2_ST-HSC', '3_MPP',
    '9_LMPP', '10_CLP', '11_PreProB', '12_Cycling_Pro_B', 
     '13_Pro_B', '14_Pre-B/B', 
     '15_ILC_pre', 
     '17_T_NK_prog', 
     '16_CD4_T', '17_CD8_T', '18_NK',
    ]

lymphoid_mnc = [
    '1_HSC_MPP', 
    '10_CLP', 
    '11_Cycling_Pro_B', '12_Pro_B',
    '13_Large_Pre_B', 
    '14_Small_Pre_B', '15_Immature_B', '16_Mature_B', '17_Plasma', 
    '18_CD4_T', '19_CD8_T',  '20_NK',]

adata = adata[adata.obs['cell_type'].isin(lymphoid_cd34 + lymphoid_mnc)]

adata.obs['cell_label'] = adata.obs['cell_type'].str.replace(r'^\d+_', '', regex=True)

adata.obs['cell_label'] = adata.obs['cell_label'].replace({'T_NK_prog' : 'ILC_pre'})

print('Number of cells:', adata.shape[0])
print('Number of cell types:', adata.obs['cell_label'].nunique())
print(adata.obs['cell_label'].value_counts())
print('\n\n')
print('As a list:', sorted(list(adata.obs['cell_label'].unique())))

# Save
adata.write_h5ad('ikzf1_lymphoid.h5ad')
