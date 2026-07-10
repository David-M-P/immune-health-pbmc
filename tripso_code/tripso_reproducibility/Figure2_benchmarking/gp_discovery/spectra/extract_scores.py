import os
import argparse

import scanpy as sc
import pandas as pd

def pp_spectra(adata, gp):
    index_labels = adata.uns['SPECTRA_overlap'].index
    gene_weights = pd.DataFrame(adata.uns['SPECTRA_factors'], 
                            index= index_labels,
                            columns=adata.var[adata.var['spectra_vocab']].index)
    
    gene_weights = gene_weights.T
    
    print('gene weights ')
    print(gene_weights.head())
    print('learned factors:', gene_weights.columns)
                
    # Find the relevant column 
    col_of_interest = [c for c in gene_weights.columns if (gp in c) and ('HVG' in c)]
    output = gene_weights[col_of_interest]
    output = output.reset_index().rename(columns={'index': 'gene'})
    output = output.sort_values(by=col_of_interest[0], ascending=False)
    
    print('long format gene weights ')
    print(output.head())

    return output

def export_scores(target_gp: str, seed: int, output_folder: str):
    adata_spectra = sc.read_h5ad(os.path.join(output_folder, 'adata_spectra.h5ad'))
    
    print('=============================')
    print(f'For {target_gp} with seed {seed}')
    print(adata_spectra)
    print('=============================')
    
    gene_weights = pp_spectra(adata_spectra, target_gp)
    
    gene_weights.to_csv(os.path.join(output_folder, f'{target_gp}_gene_weights.csv'), index=False)
    
    return gene_weights

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract gene weights from Spectra results.')
    parser.add_argument('target_gp', type=str, help='Target gene program')
    parser.add_argument('seed', type=int, help='Random seed')
    parser.add_argument('output_folder', type=str, help='Output folder path')
    
    args = parser.parse_args()
    
    export_scores(args.target_gp, args.seed, args.output_folder)
    
    


    