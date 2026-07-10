from typing import List, Union, Optional

import os
import click

from clean_adata import prepare_adata
from filter_hvg import filter_hvg
from tokenize_adata import tokenize_dataset

def prepare_and_tokenize(
    adata_fn: Union[str, os.PathLike],
    gpdb_fn: Union[str, os.PathLike],
    output_dir: Union[str, os.PathLike],
    target_label: str,
    batch_key: str = 'Batch_info',
    excluded_gps: List[str] = [],
    seq_len: int = 4096,
    extra_vars_to_keep: List[str] = [],
    filtered_obs: dict = {},
    batch_key_min_cells: Optional[int] = None,
):
    """Create a tokenized (huggingface) dataset from a h5ad object.
    
    Parameters
    ----------
    adata_fn : str or path-like
        Path to the input adata h5ad.
    gpdb_fn : str or path-like
        Path to gene program database.
    output_dir : str or path-like
        Directory to output the processed files. 
    target_label : str
        GPFinder label.
    batch_key : str
        Batch key for HVG calculation.
    excluded_gps : list of str
        GPs to exclude from HVG, based on the gpdb given.
    seq_len : int
        Max length of the tokenized sequence.
    extra_vars_to_keep : list of str
        Extra variables to keep in the dataset.
    filtered_obs : dict
        Dictionary of obs values to filter from the adata file.
    batch_key_min_cells : int
        Minimum number of cells that a batch can have
    """
    # Filenames
    adata_name = os.path.splitext(os.path.basename(adata_fn))[0]

    # Prepare adata
    prepare_adata(
        adata_fn=adata_fn,
        gpdb_fn=gpdb_fn,
        output_dir=output_dir,
        output_name=adata_name,
        batch_key=batch_key,
        filtered_obs=filtered_obs,
        batch_key_min_cells=batch_key_min_cells,
    )
    
    # Filter HVG
    if excluded_gps:
        filter_hvg(
            gpdb_fn=gpdb_fn,
            hvg_fn=os.path.join(output_dir, 'hvg.csv'),
            excluded_gps=excluded_gps,
        )
        
    # Tokenize
    tokenize_dataset(
        adata_fn=os.path.join(
            output_dir, 
            f'data/processed/{adata_name}_processed.h5ad'
        ),
        gpdb_fn=gpdb_fn,
        output_dir=output_dir,
        target_label=target_label,
        batch_key=batch_key,
        input_size=seq_len,
        extra_vars_to_keep=extra_vars_to_keep,
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
    type=click.Path(file_okay=False, dir_okay=True, readable=True, resolve_path=True)
)
@click.option("--target_label", type=str, default=None)
@click.option("--batch_key", type=str, default='Batch_info')
@click.option("--excluded_gps", type=str, default=None)
@click.option("--seq_len", type=int, default=4096)
@click.option("--extra_vars_to_keep", type=str, default=None)
@click.option("--filtered_obs_column", type=str, default=None)
@click.option("--filtered_obs_values", type=str, default=None)
@click.option("--batch_key_min_cells", type=int, default=None)
def main(
    adata_fn,
    gpdb_fn,
    output_dir,
    target_label,
    batch_key,
    excluded_gps,
    seq_len,
    extra_vars_to_keep,
    filtered_obs_column,
    filtered_obs_values,
    batch_key_min_cells,
):
    if excluded_gps is None:
        excluded_gps = []
    else:
        excluded_gps = excluded_gps.split(',')
        
    if extra_vars_to_keep is None:
        extra_vars_to_keep = []
    else:
        extra_vars_to_keep = extra_vars_to_keep.split(',')
        
    print('excluded_gps = ',excluded_gps)
    print('extra_vars_to_keep = ',extra_vars_to_keep)
    
    if filtered_obs_column is not None and filtered_obs_values is not None:
        filtered_obs = {filtered_obs_column: filtered_obs_values.split(',')}
    else:
        filtered_obs = {}
    
    prepare_and_tokenize(
        adata_fn=adata_fn,
        gpdb_fn=gpdb_fn,
        output_dir=output_dir,
        target_label=target_label,
        batch_key=batch_key,
        excluded_gps=excluded_gps,
        seq_len=seq_len,
        extra_vars_to_keep=extra_vars_to_keep,
        filtered_obs=filtered_obs,
        batch_key_min_cells=batch_key_min_cells,
    )

if __name__ == "__main__":
    main()