import argparse
import os
import pandas as pd
import numpy as np
import scanpy as sc
from tqdm import tqdm

# set seed for reproducibility
np.random.seed(0)

def rank_genes_by_attention(
    gpfinder_dir: str,
    target_class: str,
    target_label: str = "target_pathway",
    min_nonzero_frac: float = 0.10,
    min_attn_var: float = 1e-6,
):
    """
    Rank genes based on the difference in attention between target_class and other classes.
    
    Simple approach:
      - For each gene, compute mean attention in target_class
      - For each gene, compute mean attention in other classes
      - Rank genes by (target_attention - mean_other_attention)
    """

    # ============ SETUP AND DATA LOADING ============
    output_dir = gpfinder_dir
    target_class = 'TNFa' if target_class.lower() == 'tnfa' else 'TGFb'

    os.makedirs(os.path.join(output_dir, "attention_simple_by_gene"), exist_ok=True)
    print("Output dir:", output_dir)

    # clean up - remove any existing output files
    attn_out_dir = os.path.join(output_dir, "attention_simple_by_gene")
    for f in os.listdir(attn_out_dir):
        file_path = os.path.join(attn_out_dir, f)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Load attention weights
    attn = sc.read_h5ad(os.path.join(gpfinder_dir, "attention/HVG_attention_test_set.h5ad"))
    # Drop CLS token if present
    attn = attn[:, attn.var.index != "cls"]
    print("Initial attention shape:", attn.shape)

    # Load expression data with cell annotations
    expr = sc.read_h5ad(
        'tripso_reproducibility/05_gpfinder/benchmark_repeat/Jiang_log_norm.h5ad'
    )
    # Subset expression data to match cells in attention data
    expr = expr[attn.obs["idx"].values, :]

    print("expr shape:", expr.shape)
    print(expr.obs[target_label].value_counts())
    print("\n\n")

    # Work with a dense copy for filtering
    X_dense = attn.X.todense()  # cells x genes
    n_cells = attn.n_obs

    # Filter genes by attention sparsity and variance
    gene_nonzero_frac = np.array((X_dense > 0).sum(axis=0)).ravel() / n_cells
    gene_var = np.array(np.var(X_dense, axis=0)).ravel()

    keep_mask = (gene_nonzero_frac > min_nonzero_frac) & (gene_var > min_attn_var)
    genes_to_keep = np.array(attn.var.index)[keep_mask]

    attn = attn[:, genes_to_keep]
    genes = np.array(attn.var.index)
    print("Attention shape after filtering:", attn.shape)
    
    # check number of ground truth genes that were dropped
    dropped_genes = [g for g in attn.var_names if g not in genes_to_keep]
    gt_genes = pd.read_csv('/lustre/scratch126/cellgen/lotfollahi/cs60/proj_gpfinder/benchmark/gpdb_perturbseq_extended.csv')[target_class].tolist()
    print(f"Dropped {len(set(dropped_genes) & set(gt_genes))} ground truth genes out of {len(gt_genes)} for {target_class}")

    # ============ COMPUTE MEAN ATTENTION PER CLASS ============
    all_classes = expr.obs[target_label].unique()
    other_classes = [c for c in all_classes if c != target_class]
    
    print(f"\nTarget class: {target_class}")
    print(f"Other classes: {other_classes}")

    mean_attn_per_class = {}
    
    for cls in tqdm(all_classes, desc="Computing mean attention per class"):
        mask_cls = expr.obs[target_label] == cls
        attn_subset = attn[mask_cls, :]
        
        # Mean attention across cells in this class
        mean_attn_vals = attn_subset.X.mean(axis=0)
        if hasattr(mean_attn_vals, "A1"):
            mean_attn_per_class[cls] = mean_attn_vals.A1
        else:
            mean_attn_per_class[cls] = np.ravel(mean_attn_vals)

    # Convert to DataFrame for easier manipulation
    mean_attn_df = pd.DataFrame(mean_attn_per_class, index=genes)
    
    # Save mean attention per class
    mean_attn_df.to_csv(
        os.path.join(output_dir, "attention_simple", "mean_attention_per_class.csv")
    )

    # ============ RANK GENES BY ATTENTION DIFFERENCE ============
    target_attn = mean_attn_df[target_class]
    
    if other_classes:
        other_attn_mean = mean_attn_df[other_classes].mean(axis=1)
    else:
        other_attn_mean = pd.Series(0.0, index=genes)
    
    # Compute attention difference
    attn_diff = target_attn - other_attn_mean
    
    # Create ranking DataFrame
    gene_ranking = pd.DataFrame({
        'gene': genes,
        'target_attention': target_attn.values,
        'other_attention_mean': other_attn_mean.values,
        'attention_diff': attn_diff.values,
    })
    
    # Add individual class attention values
    for cls in all_classes:
        gene_ranking[f'{cls}_attention'] = mean_attn_df[cls].values
    
    # Sort by attention difference (higher is better)
    gene_ranking = gene_ranking.sort_values('attention_diff', ascending=False).reset_index(drop=True)
    
    # Add rank column
    gene_ranking.insert(0, 'rank', range(1, len(gene_ranking) + 1))
    
    # Save ranked genes
    gene_ranking.to_csv(
        os.path.join(output_dir, "attention_simple", "genes_ranked_by_attention.csv"),
        index=False,
    )
    
    print(f"\nTop 10 genes for {target_class}:")
    print(gene_ranking.head(10)[['rank', 'gene', 'attention_diff', 'target_attention', 'other_attention_mean']])
    
    # ============ COMPUTE STATISTICS ============
    stats = {
        'target_class': target_class,
        'n_genes_total': len(genes),
        'n_genes_positive_diff': (attn_diff > 0).sum(),
        'n_genes_negative_diff': (attn_diff < 0).sum(),
        'mean_attention_diff': attn_diff.mean(),
        'median_attention_diff': attn_diff.median(),
        'max_attention_diff': attn_diff.max(),
        'min_attention_diff': attn_diff.min(),
    }
    
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(
        os.path.join(output_dir, "attention_simple", "ranking_statistics.csv"),
        index=False,
    )
    
    print("\nRanking statistics:")
    for key, val in stats.items():
        print(f"  {key}: {val}")


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="output", help="Main directory where all subfolders are located")
    parser.add_argument("--gp_of_interest", type=str)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    ######################################
    # Run gene ranking function
    ######################################
    
    GPFINDER_DIR = os.path.join(args.root_dir, args.gp_of_interest, f'output_gpfinder_{args.seed}')
    
    rank_genes_by_attention(
        gpfinder_dir=GPFINDER_DIR,
        target_class=args.gp_of_interest,
    ) 

    
if __name__ == "__main__":
    main()
