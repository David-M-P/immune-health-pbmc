from typing import Union, Optional

import os
import re
import click

import numpy as np
import pandas as pd
import scanpy as sc

from geneformer import ENSEMBL_MAPPING_FILE
# ENSEMBL_MAPPING_FILE = '/nfs/users/nfs_c/cs60/dictionary_files_jul_2024/gene_name_id_dict_gc95M.pkl' # HACK

def prepare_adata(
    adata_fn: Union[str, os.PathLike],
    gpdb_fn: Union[str, os.PathLike],
    output_dir: Union[str, os.PathLike],
    output_name: Optional[str] = None,
    batch_key: str = "Batch_info",
    filtered_obs: dict = {},
    batch_key_min_cells: Optional[int] = None,
) -> None:
    """
    Prepares the input adata file, ensuring that it contains the necessary
    variables, and the highly variable genes.
    
    This functions outputs two files; one with
    
    This function also filters to only a subset of the genes 
    (union of HVG and known GPs).
    
    Returns two adata files; one with all cells (for GPFinder),
    one with the target class removed (for pre-train + fine-tune).
    
    Also saves the HVG file.
    
    Parameters
    ----------
    adata_fn : str or path-like
        File path for adata file.
    gpdb_fn : str or path-like
        File path for gene program database
        (containing only known GPs).
    output_dir : str or path-like
        Folder in which to output the files.
    output_name : str
        Output name for processed adata file.
    batch_key : str
        Batch key to use for HVG calculation.
    filtered_obs : dict
        Dictionary of obs values to filter from the adata file.
    batch_key_min_cells : int
        Minimum number of cells that a batch can have
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Check that the gpdb filename follows the correct format
    gpdb_dir, gpdb_name = os.path.split(gpdb_fn)
    gpdb_name_tag = re.match(r"gpdb_(.+)", gpdb_name)
    if gpdb_name_tag is None:
        raise ValueError('The gpdb filename must be of the form "gpdb_X" where X is a string.')
    
    if output_name is None:
        output_name = os.path.splitext(os.path.basename(adata_fn))[0]
    
    adata = sc.read_h5ad(adata_fn)
    print(f'adata has {len(adata)} cells')
    
    # Filter adata
    for obs_col, excluded_obs in filtered_obs.items():
        adata = adata[~adata.obs[obs_col].isin(excluded_obs)].copy()
        print(f'Filtered {obs_col} values {excluded_obs}')
        print(f'adata has {len(adata)} cells')
        
    # Filter batches by num of cells
    if batch_key_min_cells:
        batch_counts = adata.obs[batch_key].value_counts()
        print(f'adata has {len(batch_counts)} batches')
        small_batches = batch_counts[batch_counts < batch_key_min_cells].index
        adata = adata[~adata.obs[batch_key].isin(small_batches)].copy()
        print(f'adata has {len(adata)} cells in {len(batch_counts) - len(small_batches)} batches.')
    
    # Assign highly variable genes (HVG)
    if 'highly_variable' not in adata.var.columns:
        print('Calculating HVG...')
        sc.pp.highly_variable_genes(adata, flavor='seurat_v3', batch_key=batch_key, n_top_genes=2000)
    hvg = list(adata[:, adata.var['highly_variable']].var_names)
    hvg_df = pd.DataFrame({'HVG': hvg})
    print('Saving HVG...')
    hvg_df.to_csv(os.path.join(output_dir, 'hvg.csv'), index=False)
    
    # Assign necessary variables
    adata.obs['idx'] = adata.obs.index
    adata.obs['batch_key'] = adata.obs[batch_key]
    
    # Remove genes with no ensembl id (and therefore no token id)
    print('Removing genes with no ensembl id...')
    num_genes_initial = adata.shape[1]
    name_to_ens = pd.read_pickle(ENSEMBL_MAPPING_FILE)
    adata.var['ensembl_id'] = adata.var.index.map(name_to_ens)
    adata = adata[:, ~adata.var['ensembl_id'].isna()]
    print(f'{num_genes_initial} genes before removal; {adata.shape[1]} genes after removal.')
    
    # Filter genes to include only HVG + known GP genes
    print('Filtering genes...')
    genes_to_keep = set()
    gpdb = pd.read_csv(gpdb_fn)
    
    genes_to_keep.update(hvg)
    for gp in gpdb.columns:
        genes_to_keep.update(gpdb[gp].dropna().tolist())
    genes_to_keep = [gene for gene in list(genes_to_keep) if gene in adata.var_names]
    adata = adata[:, genes_to_keep]
    
    genes_to_keep_df = pd.DataFrame({'genes_to_keep': genes_to_keep})
    genes_to_keep_df.to_csv(
        os.path.join(output_dir, f'genes_to_keep.csv'),
        index=False,
    )
    
    # Remove zero-gene cells
    cell_sums = np.array(adata.X.sum(axis=1)).ravel()
    nonzero_mask = cell_sums > 0
    adata = adata[nonzero_mask]

    # Save processed adata file
    print('Saving h5ad file...')
    os.makedirs(os.path.join(output_dir, 'data/processed'), exist_ok=True)
    adata.write_h5ad(
        os.path.join(output_dir, f'data/processed/{output_name}_processed.h5ad')
    )
        
@click.command()
@click.option(
    "--adata_fn", 
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True)
)
@click.option(
    "--gpdb_fn", 
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True)
)
@click.option(
    "--output_dir", 
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, resolve_path=True)
)
@click.option("--output_name", type=str, default=None)
@click.option("--batch_key", type=str, default='Batch_info')
def main(
    adata_fn,
    gpdb_fn,
    output_dir,
    output_name,
    batch_key,
):
    prepare_adata(
        adata_fn,
        gpdb_fn,
        output_dir,
        output_name,
        batch_key,
    )

if __name__ == "__main__":
    main()