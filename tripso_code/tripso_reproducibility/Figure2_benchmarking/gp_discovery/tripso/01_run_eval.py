import argparse
import os
import pandas as pd
import tripso
from tripso.Train.training_flexi import run_training_from_select_gps

# Model training parameters
TISSUE = "pert"
MODEL_TYPE = "Base"
N_HEADS = 8
N_BLOCKS = 2
WEIGHT_DECAY = 1e-4
MGM = 0.25
N_EPOCHS = 20
LR_SCHEDULER = "ReduceLROnPlateau"
LR = 1e-4


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="output", help="Main directory where all subfolders are located")
    parser.add_argument("--gp_of_interest", type=str)
    parser.add_argument("--gp_known")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    DATA_DIR = f"/lustre/scratch126/cellgen/lotfollahi/cs60/proj_gpfinder/benchmark_marie_hyperparams/{args.gp_of_interest}/data/processed"
    dataset_fn = os.path.join(DATA_DIR, "input_dataset")
    
    genes_to_keep = pd.read_csv(f'/lustre/scratch126/cellgen/lotfollahi/cs60/proj_gpfinder/benchmark_global/{args.gp_of_interest}/genes_to_keep.csv')
    ALL_GENES = genes_to_keep['genes_to_keep'].tolist()
    GPDB_PATH = f'/lustre/scratch126/cellgen/lotfollahi/cs60/proj_gpfinder/benchmark_global/{args.gp_of_interest}/gpdb_{args.gp_known}.csv'
        
    
    ######################################
    # Train GPFinder
    ######################################
    
    GPDB_OLD = GPDB_PATH
    GPDB_NEW = f'/lustre/scratch126/cellgen/lotfollahi/cs60/proj_gpfinder/benchmark_global/{args.gp_of_interest}/gpdb_{args.gp_known}_with_hvg.csv'
    gpdb = pd.read_csv(GPDB_NEW)
        
    # GP inputs
    gp_inputs_old = list(pd.read_csv(GPDB_OLD).columns)
    
    GPFINDER_DIR = os.path.join(args.root_dir, args.gp_of_interest, f'output_gpfinder_{args.seed}')
    
    # only run eval if ckpt exists
    if os.path.exists(os.path.join(GPFINDER_DIR, "checkpoints/last.ckpt")):
        gp_downstream = tripso.gpEval(
            dataset_path=dataset_fn,
            gpdb_path=GPDB_NEW,
            output_dir=GPFINDER_DIR,
            tissue='pert',
            model_type='Global',
            batch_size = 128
        )
        
        for t in ['train', 'test', 'val']:
            gp_downstream.generate_attention_matrix(
                gp_for_forward='HVG',
                gp_for_downstream='HVG',
                genes_to_keep=gpdb['HVG'].dropna().tolist(),
                precision = '16-mixed',
                split = t
            )
            
        for t in ['train', 'test', 'val']:
            gp_downstream.generate_gene_embeddings(
                gp_for_forward='HVG',
                gp_for_downstream='HVG',
                split=t,
                genes_to_keep=gpdb['HVG'].dropna().tolist(),
                precision = '16-mixed',
                return_gene_cosim='gene_to_gp',
                )
            
        
    
    
if __name__ == "__main__":
    main()
