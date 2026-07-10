from typing import List, Union, Optional

import os
import re
import click
import shutil
import tripso 

import pandas as pd

def tokenize_dataset(
    adata_fn: Union[str, os.PathLike],
    gpdb_fn: Union[str, os.PathLike],
    output_dir: Union[str, os.PathLike],
    target_label: Optional[str] = None,
    batch_key: str = "Batch_info",
    input_size: int = 4096,
    extra_vars_to_keep: List[str] = []
):
    """
    Tokenizes an h5ad dataset.
    
    Parameters
    ----------
    adata_fn : str or path-like
        File path of processed adata file
        (with all cells included).
    gpdb_fn : str or path-like
        File path of gene program database.
    output_dir : str or path-like
        Folder in which to output the files.
    target_label : str or None
        Obs name of target class, if specified.
    batch_key : str
        Batch key to use.
    input_size : int
        Input length of token sequences.
    extra_vars_to_keep : list of str or None
        Extra variable names to keep.
    """
    genes_to_keep_df = pd.read_csv(
        os.path.join(output_dir, 'genes_to_keep.csv')
    )
    
    # Check that the gpdb filename follows the correct format
    gpdb_name_tag = re.match(r"gpdb_(.+)", os.path.basename(gpdb_fn))
    if gpdb_name_tag is None:
        raise ValueError('The gpdb filename must be of the form "gpdb_X" where X is a string.')
    
    vars_to_keep = extra_vars_to_keep + [target_label]
    print('name_tag = ',os.path.splitext(gpdb_name_tag.group(1))[0])
    
    if not os.path.exists(os.path.join(output_dir, gpdb_name_tag.group(1))):
        shutil.copyfile(
            gpdb_fn,
            os.path.join(output_dir, gpdb_name_tag.group(1)),
        )
    
    # Tokenize
    tripso.pp_and_tokenize(
        root_dir=output_dir,
        adata_path=adata_fn,
        vars_to_keep=vars_to_keep + ['batch_key'],
        batch_keys=batch_key,
        calculate_hvg=False,
        subsample_by=None,
        cov_to_encode=vars_to_keep,
        save_gp_genes_object=False, 
        name_tag=os.path.splitext(gpdb_name_tag.group(1))[0],
        input_size=input_size,
        gp_genes_union=genes_to_keep_df['genes_to_keep'],
        use_gp_tokenizer=True,
        pp_cellxgene=True,
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
@click.option("--target_label", type=str, default=None)
@click.option("--batch_key", type=str, default='Batch_info')
@click.option("--input_size", type=int, default=4096)
@click.option("--extra_vars_to_keep", type=List[str], default=[])
def main(
    adata_fn,
    gpdb_fn,
    output_dir,
    target_label,
    batch_key,
    input_size,
    extra_vars_to_keep,
):
    tokenize_dataset(
        adata_fn,
        gpdb_fn,
        output_dir,
        target_label,
        batch_key,
        input_size,
        extra_vars_to_keep,
    )

if __name__ == "__main__":
    main()