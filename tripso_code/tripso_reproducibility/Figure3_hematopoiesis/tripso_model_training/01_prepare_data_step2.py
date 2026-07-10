import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd

from geneformer import ENSEMBL_DICTIONARY_FILE
ensembl_dict = pd.read_pickle(ENSEMBL_DICTIONARY_FILE)



##############################################
print(' ------- Wrangle MNC ------- ')
##############################################
mnc = sc.read_h5ad('/nfs/team361/mm58/tripso_reproducibility_old/HSC/data/processed/mnc.h5ad')

print('mnc.X max', mnc.X.max())

mnc.obs['donor'] = mnc.obs['donor_tissue'].str.extract(r'^([^_]+)_')


# update obs to match Tomo
mnc.obs['tissue'] = np.where(
    mnc.obs['tissue'] == 'EL',
    'FL', 
    mnc.obs['tissue']
)

# separate young and aged in adult BM
# categorise age groups { end with PCW : 'Fetal, 0-15: Pediatric, 16-30 : Young Adult, 31-50:  Middle Age, 50+: Aged}
mnc.obs['age_group'] = None
mnc.obs.loc[(mnc.obs['age'].str.contains('PCW')) , 'age_group'] = 'Fetal'
mnc.obs.loc[(mnc.obs['age']=='0') , 'age_group'] = 'Cord Blood'
# replace all PCW rows with empty string, e.g. 14PCW -> ''
mnc.obs['age'] = mnc.obs['age'].str.replace(r'\d+PCW', '', regex=True)
mnc.obs['age'] = mnc.obs['age'].replace('', np.nan)
mnc.obs['age'] = mnc.obs['age'].astype(float)

# distinguish between aged bone marrow Aged (60+) and young (<60)
mnc.obs['tissue'] = pd.Categorical(mnc.obs['tissue'])
mnc.obs['tissue'] = mnc.obs['tissue'].cat.add_categories(['ABM_+60y', 'ABM_29-50y']) # 'PBM'
mnc.obs.loc[mnc.obs['age'] >= 60, 'tissue'] = 'ABM_+60y'
mnc.obs.loc[(mnc.obs['age'] < 60) & (mnc.obs['age'] >= 17), 'tissue'] = 'ABM_29-50y'

mnc.obs['tissue'] = mnc.obs['tissue'].cat.remove_unused_categories()


mnc.obs['tissue'] = mnc.obs['tissue'].cat.reorder_categories(
    ['YS', 'FL', 'FBM', 'CB', 'ABM_29-50y', 'ABM_+60y']
    )

print('In MNC dataset, the tissue categories are:')
print(mnc.obs['tissue'].value_counts(dropna = False))

ct_to_stage = {
    '1_HSC_MPP' : 'stem', 
    '2_MEMP' : 'intermediate', 
    '3_MK' : 'terminal', 
    '4_Ery_prog': 'intermediate', 
    '5_Early_Ery': 'terminal', 
    '6_Mid_Ery': 'terminal', 
    '7_Late_Ery': 'terminal', 
    '8_HBE+_embryonic_Ery' : 'terminal', 
    '9_BaEoMa': 'terminal', 
    '10_CLP':'intermediate',
    '11_Pro_B': 'terminal', 
    '12_Cycling_Pro_B': 'terminal', 
    '13_Large_Pre_B': 'terminal', 
    '14_Small_Pre_B': 'terminal', 
    '15_Immature_B': 'terminal', 
    '16_Mature_B': 'terminal', 
    '17_Plasma': 'terminal', 
    '18_CD4_T': 'terminal',
    '19_CD8_T': 'terminal', 
    '20_NK': 'terminal', 
    '21_GMP': 'terminal', 
    '22_pDC': 'terminal',  
    '23_Mono_Pre': 'terminal',  
    '24_CD14_Mono': 'terminal',  
    '25_CD16_Mono': 'terminal',  
    '26_DC_pre': 'terminal', 
    '27_cDC1': 'terminal',  
    '28_cDC2': 'terminal',  
    '29_Mac' : 'terminal', 
}


mnc.obs['cell_type_stage'] = mnc.obs['cell_type'].map(ct_to_stage)


mnc.obs['tissue_cell_cat'] = mnc.obs['tissue'].astype(str) + '_' + mnc.obs['cell_type_stage'].astype(str)


mnc.write_h5ad('data/processed/mnc.h5ad')



##############################################
# print(' ------- Wrangle Wrangle CD34+ in vivo dataset ------- ')
##############################################

cd34 = sc.read_h5ad('/nfs/team361/mm58/tripso_reproducibility_old/HSC/data/processed/cd34.h5ad')


print('cd34.X max', cd34.X.max())

cd34.obs['donor'] = cd34.obs['donor_tissue'].str.extract(r'^([^_]+)_')


# update obs to match Tomo
cd34.obs['tissue'] = np.where(
    cd34.obs['tissue'] == 'EL',
    'FL', 
    cd34.obs['tissue']
)

# separate young and aged in adult BM
# categorise age groups { end with PCW : 'Fetal, 0-15: Pediatric, 16-30 : Young Adult, 31-50:  Middle Age, 50+: Aged}
cd34.obs['age_group'] = None
cd34.obs.loc[(cd34.obs['age'].str.contains('PCW')) , 'age_group'] = 'Fetal'
cd34.obs.loc[(cd34.obs['age']=='0') , 'age_group'] = 'Cord Blood'
# replace all PCW rows with empty string, e.g. 14PCW -> ''
cd34.obs['age'] = cd34.obs['age'].str.replace(r'\d+PCW', '', regex=True)
cd34.obs['age'] = cd34.obs['age'].replace('', np.nan)
cd34.obs['age'] = cd34.obs['age'].astype(float)

# distinguish between aged bone marrow Aged (60+) and young (<60)
cd34.obs['tissue'] = pd.Categorical(cd34.obs['tissue'])
cd34.obs['tissue'] = cd34.obs['tissue'].cat.add_categories(['ABM_+60y', 'ABM_29-50y']) # 'PBM'
cd34.obs.loc[cd34.obs['age'] >= 60, 'tissue'] = 'ABM_+60y'
cd34.obs.loc[(cd34.obs['age'] < 60) & (cd34.obs['age'] >= 17), 'tissue'] = 'ABM_29-50y'
cd34.obs.loc[(cd34.obs['age'] < 17) & (cd34.obs['age'] >= 1), 'tissue'] = 'PBM'

cd34.obs['tissue'] = cd34.obs['tissue'].cat.remove_unused_categories()


cd34.obs['tissue'] = cd34.obs['tissue'].cat.reorder_categories(
    ['YS', 'FL', 'FBM', 'CB', 'PBM', 'ABM_29-50y', 'ABM_+60y']
    )

print('In CD34 dataset, the tissue categories are:')
print(cd34.obs['tissue'].value_counts(dropna = False))

ct_to_stage = {
    '1_LT-HSC' : 'stem', 
    '2_ST-HSC' : 'stem',
    '3_MPP' : 'stem', 
    '4_MEMP' : 'intermediate', 
    '5_MK' : 'terminal', 
    '6_Early_Ery' : 'terminal', 
    '7_Late_Ery' : 'terminal', 
    '8_BaEoMa' : 'terminal', 
    '9_LMPP' : 'intermediate', 
    '10_CLP' : 'intermediate', 
    '11_PreProB'  : 'terminal', 
    '12_Pro_B'  : 'terminal', 
    '13_Cycling_Pro_B'  : 'terminal', 
    '14_Large_Pre_B'  : 'terminal', 
    '15_Small_Pre_B'  : 'terminal',  
    '16_Immature_B' : 'terminal', 
    '17_T_NK_prog'  : 'terminal', 
    '18_CD4_T'  : 'terminal', 
    '19_CD8_T' : 'terminal', 
    '20_NK'  : 'terminal', 
    '21_GMP' : 'intermediate', 
    '22_pDC' : 'terminal', 
    '23_DC_pre'  : 'terminal',  
    '24_Mono_pre' : 'terminal',
    '25_Macrophage'  : 'terminal', 
}


cd34.obs['cell_type_stage'] = cd34.obs['cell_type'].map(ct_to_stage)


cd34.obs['cell_type_stage'].value_counts()


cd34.obs['tissue_cell_cat'] = cd34.obs['tissue'].astype(str) + '_' + cd34.obs['cell_type_stage'].astype(str)


cd34.write_h5ad('data/processed/cd34.h5ad')


##############################################
print(' ------- Wrangle cord blood dataset ------- ')
##############################################

cb = sc.read_h5ad('/nfs/team361/mm58/tripso_reproducibility_old/HSC/data/processed/cb.h5ad')

print('cb.X max', cb.X.max())

cb.obs['tissue'] = 'CB_in vitro_1'

cb.write_h5ad('data/processed/cb.h5ad')


##############################################
print(' ------- Wrangle in vitro HSCs ------- ')
##############################################

hsc = sc.read_h5ad('/nfs/team361/mm58/tripso_reproducibility_old/HSC/data/processed/sakurai_hsc.h5ad')

hsc.obs['tissue'] = 'CB_in vitro_2'

hsc.write_h5ad('data/processed/sakurai_hsc.h5ad')


# ##############################################
print(' ------- Wrangle Zeng dataset ------- ')
# ##############################################

zeng = sc.read_h5ad('/nfs/team361/mm58/tripso_reproducibility_old/HSC/data/processed/zeng.h5ad')


zeng.obs['tissue'] = np.where(
    zeng.obs['ExactAge'] < 60,
    'ABM_29-50y',
    'ABM_+60y'
)

zeng = zeng[zeng.obs['cell_type'] != 'stromal cell of bone marrow']

ct_to_stage = {
    'hematopoietic stem cell' : 'stem', 
    'erythroid progenitor cell' : 'intermediate',
    'common myeloid progenitor' : 'intermediate', 
    'megakaryocyte-erythroid progenitor cell' : 'intermediate',
    'common lymphoid progenitor': 'intermediate', 
    'T cell' : 'terminal',
    'dendritic cell' : 'terminal',
    'basophilic erythroblast' : 'terminal', 
    'polychromatophilic erythroblast'  : 'terminal',
    'orthochromatic erythroblast' : 'terminal', 
    'megakaryocyte progenitor cell'  : 'intermediate',
    'megakaryocyte' : 'terminal', 
    'granulocyte monocyte progenitor cell'  : 'intermediate', 
    'promonocyte'  : 'terminal',
    'natural killer cell' : 'terminal', 
    'mature B cell' : 'terminal', 
    'plasma cell' : 'terminal',
    'CD4-positive, CD25-positive, alpha-beta regulatory T cell'  : 'terminal',
    'immature B cell' : 'terminal', 
    'pro-B cell'  : 'terminal', 
    'hematopoietic multipotent progenitor cell' : 'stem',
    'naive thymus-derived CD4-positive, alpha-beta T cell'  : 'terminal',
    'naive thymus-derived CD8-positive, alpha-beta T cell' : 'terminal',
    'central memory CD4-positive, alpha-beta T cell' : 'terminal',
    'effector memory CD4-positive, alpha-beta T cell' : 'terminal',
    'central memory CD8-positive, alpha-beta T cell' : 'terminal',
    'CD8-positive, alpha-beta memory T cell' : 'terminal',
    'effector memory CD8-positive, alpha-beta T cell' : 'terminal',
    'CD16-negative, CD56-bright natural killer cell, human' : 'terminal',
    'small pre-B-II cell'  : 'terminal', 
    'large pre-B-II cell' : 'terminal',
    'conventional dendritic cell' : 'terminal', 
    'CD14-positive monocyte' : 'terminal',
    'plasmacytoid dendritic cell, human' : 'terminal', 
    'pre-conventional dendritic cell' : 'terminal',
    'basophil mast progenitor cell' : 'terminal',
    'hematopoietic oligopotent progenitor cell' : 'stem', 
    'late pro-B cell'  : 'terminal',
    'CD14-positive, CD16-positive monocyte' : 'terminal', 
}


zeng.obs['cell_type_stage'] = zeng.obs['cell_type'].map(ct_to_stage)

zeng.obs['tissue_cell_cat'] = zeng.obs['tissue'].astype(str) + '_' + zeng.obs['cell_type_stage'].astype(str)

zeng.write_h5ad('data/processed/zeng.h5ad')


##############################################
print(' ------- Wrangle Li dataset ------- ')
##############################################


li = sc.read_h5ad('/nfs/team361/mm58/tripso_reproducibility_old/HSC/data/processed/hojun.h5ad')

li.obs['study'] = 'Li'


li.obs['tissue'].value_counts()

li.obs['development_stage'].value_counts()


# separate young and aged in adult BM
li.obs['age'] = li.obs['development_stage'].copy()

# allow conversion to float
li.obs['age'] = li.obs['age'].str.replace('cord blood', '')
li.obs['age'] = li.obs['age'].str.replace(r'\d+w', '', regex=True)
li.obs['age'] = li.obs['age'].str.replace('y', '')
li.obs['age'] = li.obs['age'].replace('', np.nan)
li.obs['age'] = li.obs['age'].astype(float)

# distinguish between aged bone marrow Aged (60+) and young (<60)
li.obs['tissue'] = pd.Categorical(li.obs['tissue'])
li.obs['tissue'] = li.obs['tissue'].cat.add_categories(['ABM_+60y', 'ABM_29-50y', 'PBM'])
li.obs.loc[li.obs['age'] >= 60, 'tissue'] = 'ABM_+60y'
li.obs.loc[(li.obs['age'] < 60) & (li.obs['age'] >= 17), 'tissue'] = 'ABM_29-50y'
li.obs.loc[(li.obs['age'] < 17) & (li.obs['age'] >= 1), 'tissue'] = 'PBM'

li.obs['tissue'] = li.obs['tissue'].cat.remove_unused_categories()


li.obs['tissue'] = li.obs['tissue'].replace(
    {'fetal liver' : 'FL',
     'cord blood' : 'CB'
    }
)

li.obs['tissue'] = li.obs['tissue'].cat.reorder_categories(
    ['FL', 'CB', 'PBM', 'ABM_29-50y', 'ABM_+60y']
    )


ct_to_stage = {
    'Baso/Mast-Prog' : 'intermediate', 
    'E-Prog-1' : 'intermediate', 
    'E-Prog-2' : 'intermediate', 
    'E-Prog-3' : 'intermediate', 
    'G-Prog-1' : 'intermediate',
    'G-Prog-2' : 'intermediate', 
    'G-Prog-3' : 'intermediate', 
    'G/Mono-Prog'  : 'intermediate', 
    'HSC' : 'stem', 
    'Ly-Prog-1'  : 'intermediate', 
    'Ly-Prog-2'  : 'intermediate',
    'Ly-Prog-3' : 'intermediate', 
    'Ly-Prog-4' : 'intermediate', 
    'Ly-Prog-5' : 'intermediate', 
    'MPP-1' : 'stem', 
    'MPP-2' : 'stem', 
    'MPP-3' : 'stem',
    'Mk-Prog'  : 'intermediate', 
    'Mk/E-MPP' : 'intermediate', 
    'Mono/cDC-Prog' : 'intermediate', 
    'My-MPP' : 'intermediate',
    'pDC-Prog'  : 'intermediate'
}


li.obs['cell_type_stage'] = li.obs['cell_type'].map(ct_to_stage)


li.obs['cell_type_stage'].value_counts()


li.obs['tissue_cell_cat'] = li.obs['tissue'].astype(str) + '_' + li.obs['cell_type_stage'].astype(str)


li.write_h5ad('data/processed/li.h5ad')


##############################################
print(' ------- Merge and calculate HVGs ------- ')
##############################################

# Merge
mnc = sc.read_h5ad('data/processed/mnc.h5ad')
cd34 = sc.read_h5ad('data/processed/cd34.h5ad')
cb = sc.read_h5ad('data/processed/cb.h5ad')
hsc = sc.read_h5ad('data/processed/sakurai_hsc.h5ad')
zeng = sc.read_h5ad('data/processed/zeng.h5ad')
li = sc.read_h5ad('data/processed/li.h5ad')

adata = ad.concat([mnc, cd34, cb, hsc, zeng, li])

print('Merged adata shape', adata.shape)

sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes = 2000, batch_key='study')

adata.var.to_csv('data/processed/merged_adata_hvg.csv')

