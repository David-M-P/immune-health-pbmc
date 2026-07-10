
import pandas as pd
import scanpy as sc
import numpy as np


adata = sc.read_h5ad('/nfs/team292/lg18/endometriosis/cellxgene_objects/endometriumAtlasV2_cells.h5ad')
hvg = adata[:, adata.var['highly_variable']].var_names


gpdb = pd.read_csv('gpdb_progeny_200.csv')
gpdb = gpdb[[c for c in gpdb.columns if c != 'Trail']]


gp_dict = {}

for c in gpdb.columns:
    gp_dict[c] = list(gpdb[c].values) + [np.nan for _ in range(len(hvg) - len(gpdb))]
    
gp_dict['HVG'] = hvg
                                                  
                                                


gpdb2 = pd.DataFrame(gp_dict)


gpdb2.to_csv('gpdb_with_hvg.csv', index = False)


gpdb2.head()





