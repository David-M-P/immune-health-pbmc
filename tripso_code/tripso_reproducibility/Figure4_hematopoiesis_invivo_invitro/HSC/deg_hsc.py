

import scanpy as sc
import decoupler as dc

# Only needed for processing
import numpy as np
import pandas as pd

# Needed for some plotting
import matplotlib.pyplot as plt

# Plotting options, change to your liking
sc.settings.set_figure_params(dpi=200, frameon=False)
sc.set_figure_params(dpi=200)
sc.set_figure_params(figsize=(4, 4))

import anndata as ad
import gc

# gpdb = pd.read_csv('../gpdb_nw.csv')

# Set up dataset

import scanpy as sc
import anndata as ad

cd34 = sc.read_h5ad(
    'tripso_reproducibility/04.4_HSC_fix_hvg/data/raw/CD34_RNA_98266cells_to_share.h5ad'
)

cd34.X = cd34.layers['log_counts']

cd34

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

cd34.obs['dataset'] = 'Isobe CD34'

cd34.obs['donor'] = cd34.obs['donor_tissue'].str.extract(r'^([^_]+)')


hsc = cd34[cd34.obs['celltype_v2'].isin(
    ['1_LT-HSC', '2_ST-HSC', '3_MPP', '4_MEMP',
                           '9_LMPP', '10_CLP', '21_GMP',
                          ]
)]

hsc.obs['tissue'] = hsc.obs['tissue'].cat.reorder_categories(
    ['YS', 'FL', 'FBM', 'CB', 'PBM', 'ABM_29-50y', 'ABM_+60y']
    )

hsc.obs['CellTypeBinary'] = np.where(
    hsc.obs['celltype_v2'] == '1_LT-HSC',
    'LTHSC',
    'other'
)


print(hsc.obs['CellTypeBinary'].value_counts())

import decoupler as dc
print(dc.__version__)    

# Get pseudo-bulk profile
pdata = dc.pp.pseudobulk(
    hsc,
    sample_col='donor',
    groups_col='celltype_v2',
    layer='counts',
    mode='sum',
)


## DEG

# Import DESeq2
from pydeseq2.dds import DeseqDataSet, DefaultInference
from pydeseq2.ds import DeseqStats


# Build DESeq2 object
inference = DefaultInference(n_cpus=1)

dds = DeseqDataSet(
    adata=pdata,
    design_factors=['CellTypeBinary', 'donor'],
    ref_level=['CellTypeBinary', 'other'],
    refit_cooks=True,
    inference=inference,
)

# Compute LFCs
dds.deseq2()


# Extract contrast 
stat_res = DeseqStats(
    dds,
    contrast=["CellTypeBinary", 'LTHSC', 'other'],
    inference=inference,
)

# Compute Wald test
stat_res.summary()

# Extract results
results_df = stat_res.results_df

results_df.to_csv('cd34_hsc_deg.csv')