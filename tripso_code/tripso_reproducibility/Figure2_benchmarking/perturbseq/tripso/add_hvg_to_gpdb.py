import pandas as pd
import scanpy as sc
import numpy as np

gpdb = pd.read_csv(
    '../gpdb_progeny_200.csv'
)

adata = sc.read_h5ad(
    'data/Jiang.h5ad'
)

var_df = adata[:, adata.var['highly_variable']].var

gpdb_with_hvg = {}

for col in ['TGFb', 'TNFa']:
    gpdb_with_hvg[col] = gpdb[col].tolist() + [np.nan for _ in range(len(var_df) - len(gpdb[col]))]
    
gpdb_with_hvg['HVG'] = var_df.index.tolist()

gpdb_with_hvg = pd.DataFrame(gpdb_with_hvg)

gpdb_with_hvg.to_csv(
    'gpdb_with_hvg.csv',
    index=False
)