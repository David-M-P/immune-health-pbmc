from tripso.Metrics.metrics import evaluate_emd_ref_vs_query
from tripso.Utils.utils import sample_cells

import scanpy as sc
import anndata as ad
import pandas as pd
from datasets import load_from_disk, concatenate_datasets
import numpy as np
import os
import argparse

import jax.numpy as jnp
from tripso.Metrics.metrics import compute_sinkhorn


# ===========================================
# Define functions
# ===========================================

def sinkhorn_experiment(gp):
    '''
    How does sinkhorn divergence change across media conditions
    With bootstrap sampling for categories with >= 500 cells
    BIDIRECTIONAL: bootstraps both reference AND query cells
    '''
    
    adata = sc.AnnData(
        X=np.array(emb[gp]),
        obs=emb.select_columns([
            'idx', 'source', 'cell_type', 'tissue', 'study',
            'condition', # media condition
            'age_general', 'age'
        ]).to_pandas()
    )
    
    order_source =[
        '1_LT-HSC',
        '2_ST-HSC', 
        # '3_MPP'
    ]
    
    order_target = [
        'HSPC-HLF', 'HSPC', #'HSPC-Cycling', 'CD34+GATA2+ Prog',
        ]

    
    adata = adata[adata.obs['study'].isin(['Isobe_CD34', 'Sakurai_HSC'])]
    adata = adata[adata.obs['cell_type'].isin(order_source + order_target)]
    
    # only keep ST HSC in young adult
    adata = adata[~((adata.obs['cell_type'] == '2_ST-HSC') & (adata.obs['tissue'] != 'ABM_29-50y'))]
    
    adata.obs['mapping_category'] = np.where(
        adata.obs['source'] == 'in vitro',
        adata.obs['condition'], 
        adata.obs['tissue'], 
    )
    
    adata.obs['cell_type_by_tissue'] = adata.obs['cell_type'].astype(str) + '_' + adata.obs['mapping_category'].astype(str)

    adata.obs['cell_type_by_tissue'] = adata.obs['cell_type_by_tissue'].str.rstrip('_')
    
    # convert to pca
    sc.tl.pca(adata, n_comps = 50)
    pdata = sc.AnnData(
        X = adata.obsm['X_pca'],
        obs = adata.obs
    )
    
    # balance cell numbers
    # pdata = sample_cells(pdata, 'cell_type_by_tissue', 500)
    
    # now we've done the sampling, group together HSC and HSC-HLF
    pdata.obs['cell_type_by_tissue'] = pdata.obs['cell_type_by_tissue'].str.replace('HSPC-HLF', 'HSPC', regex=False)
    
    # Compute sinkhorn divergence
    cond2 = pdata[pdata.obs['tissue'] != 'ABM_29-50y'].obs['cell_type_by_tissue'].unique().tolist() + ['2_ST-HSC_ABM_29-50y']
    
    # Create directory for individual comparison files
    individual_dir = f'output_bootstrap_bidir/individual_{gp}'
    if not os.path.exists(individual_dir):
        os.makedirs(individual_dir)
    
    processed_conditions = []
    
    for c2 in cond2:
        # Create safe filename from c2 (replace problematic characters)
        c2_safe = c2.replace('/', '_').replace('\\', '_').replace(':', '_')
        individual_csv = f'{individual_dir}/{c2_safe}.csv'
        
        # Check if this comparison already exists
        if os.path.exists(individual_csv):
            print(f"Skipping {c2}: already computed")
            processed_conditions.append(c2_safe)
            continue
        
        adata1 = pdata[(pdata.obs['tissue'] == 'ABM_29-50y') & (pdata.obs['cell_type'] == '1_LT-HSC')]    
        adata2 = pdata[pdata.obs['cell_type_by_tissue'] == c2]
        
        n_ref_cells = adata1.shape[0]
        n_query_cells = adata2.shape[0]
        
        # Skip if query has < 200 cells
        if n_query_cells < 200:
            print(f"Skipping {c2}: only {n_query_cells} cells (< 200)")
            continue
        
        # Determine if we need to bootstrap
        bootstrap_ref = n_ref_cells >= 500
        bootstrap_query = n_query_cells >= 500
        
        # If neither needs bootstrap, compute once
        if not bootstrap_ref and not bootstrap_query:
            sink = compute_sinkhorn(adata1, adata2, epsilons = [1e-3])
            sink['reference cells'] = 'ABM_29-50y'
            sink['query cells'] = c2
            
            sink['n reference cells'] = n_ref_cells
            sink['n query cells'] = n_query_cells
            sink['bootstrap'] = False
            sink['bootstrap_ref'] = False
            sink['bootstrap_query'] = False
            sink['n_bootstrap_samples'] = 1
        
        # If at least one needs bootstrap
        else:
            print(f"Bootstrap sampling - Reference: {n_ref_cells} cells (bootstrap={bootstrap_ref}), Query {c2}: {n_query_cells} cells (bootstrap={bootstrap_query})")
            n_bootstrap = 50
            n_sample = 750
            bootstrap_results = []
            
            for i in range(n_bootstrap):
                # Sample reference cells (without replacement)
                if bootstrap_ref:
                    ref_sample_idx = np.random.choice(adata1.shape[0], size=n_sample, replace=False)
                    adata1_sampled = adata1[ref_sample_idx].copy()
                else:
                    adata1_sampled = adata1.copy()
                
                # Sample query cells (without replacement)
                if bootstrap_query:
                    query_sample_idx = np.random.choice(adata2.shape[0], size=n_sample, replace=False)
                    adata2_sampled = adata2[query_sample_idx].copy()
                else:
                    adata2_sampled = adata2.copy()
                
                sink_bootstrap = compute_sinkhorn(adata1_sampled, adata2_sampled, epsilons = [1e-3])
                bootstrap_results.append(sink_bootstrap)
            
            # Concatenate all bootstrap results
            bootstrap_df = pd.concat(bootstrap_results)
            
            # Calculate mean and std for sinkhorn_divergence only
            mean_sinkhorn = bootstrap_df['sinkhorn_divergence'].mean()
            std_sinkhorn = bootstrap_df['sinkhorn_divergence'].std()
            
            # Create summary dataframe
            # Take epsilon from first bootstrap (it's constant across all samples)
            sink = pd.DataFrame({
                'epsilon': [bootstrap_df['epsilon'].iloc[0]],
                'sinkhorn_divergence': [mean_sinkhorn],
                'sinkhorn_divergence_std': [std_sinkhorn]
            })
            
            sink['reference cells'] = 'ABM_29-50y'
            sink['query cells'] = c2
            
            sink['n reference cells'] = n_ref_cells
            sink['n query cells'] = n_query_cells
            sink['bootstrap'] = True
            sink['bootstrap_ref'] = bootstrap_ref
            sink['bootstrap_query'] = bootstrap_query
            sink['n_bootstrap_samples'] = n_bootstrap
        
        # Save individual comparison immediately
        sink.to_csv(individual_csv, index=False)
        print(f"Saved {c2} to {individual_csv}")
        processed_conditions.append(c2_safe)
    
    # Compile all individual CSVs into one final output
    all_individual_csvs = [f'{individual_dir}/{c2_safe}.csv' for c2_safe in processed_conditions]
    
    if len(all_individual_csvs) == 0:
        print(f"No valid conditions for {gp}")
        return
    
    # Read and concatenate all individual files
    holder = []
    for csv_file in all_individual_csvs:
        if os.path.exists(csv_file):
            holder.append(pd.read_csv(csv_file))
    
    if len(holder) == 0:
        print(f"No valid conditions for {gp}")
        return
    
    output = pd.concat(holder, ignore_index=True)
    output['gp'] = gp
    
    # if not os.path.exists('output_distances'):
    #     os.makedirs('output_distances')
    
    output.to_csv(f'output_bootstrap_bidir/{gp}_sinkhorn.csv', index=False)
    print(f"Final compiled output saved to output_bootstrap_bidir/{gp}_sinkhorn.csv")
    
# ===========================================
# Run
# ===========================================  

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Calculate Sinkhorn divergence for a specific gene program')
    parser.add_argument('--gp', type=str, required=True, 
                        help='Gene program to analyze (e.g., WNT, GP_ZBTB7A, GP_JUNB, GP_ATF3, cell_token, PI3K, Hypoxia, GP_HOXA9)')
    args = parser.parse_args()
    
    # Load embeddings
    emb1 = load_from_disk('../../../embeddings/test_set')  
    emb2 = load_from_disk('../../../embeddings/train_set')
    emb3 = load_from_disk('../../../embeddings/val_set')
    emb = concatenate_datasets([emb1, emb2, emb3])
    
    # Create output directory if needed
    if not os.path.exists('output_bootstrap_bidir'):
        os.makedirs('output_bootstrap_bidir')
    
    # Run for the specified gene program
    gp = args.gp
    
    # Skip if already exists
    if os.path.exists(f'output_bootstrap_bidir/{gp}_sinkhorn.csv'):
        print(f"Output for {gp} already exists. Skipping.")
    else:
        print(f"Running Sinkhorn experiment for gene program: {gp}")
        sinkhorn_experiment(gp)
