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
    adata_fn = os.path.join(DATA_DIR, "Jiang_processed.h5ad")
    dataset_fn = os.path.join(DATA_DIR, "input_dataset")
    
    genes_to_keep = pd.read_csv(f'/lustre/scratch126/cellgen/lotfollahi/cs60/proj_gpfinder/benchmark_global/{args.gp_of_interest}/genes_to_keep.csv')
    ALL_GENES = genes_to_keep['genes_to_keep'].tolist()
    GPDB_PATH = f'/lustre/scratch126/cellgen/lotfollahi/cs60/proj_gpfinder/benchmark_global/{args.gp_of_interest}/gpdb_{args.gp_known}.csv'
    
    OUTPUT_DIR = os.path.join(args.root_dir, args.gp_of_interest, f'output_base_{args.seed}')
    
    # Model configuration dictionary
    config_dict = {
        "hidden_size": 512,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "tokenization_input_size": 4096,
        "mlm_masking_prob": 0.25,
        "use_pos_emb": "sin_cos",
        "use_l2_norm": False,
        "tokenization_vocab_size": 20275,
        "torch_dtype": 'bf16',
        "use_flash": True,
        "max_seq_len": len(ALL_GENES),
    }
    
    # check if checkpoint exists or train base model
    if os.path.exists(os.path.join(OUTPUT_DIR, "checkpoints/last.ckpt")):
        print(f"Checkpoint found in {OUTPUT_DIR}. Skipping base model training.")
    else:
        # Train Base Model
        tripso.train(
            dataset_path=dataset_fn,
            gpdb_path=GPDB_PATH,
            output_dir=OUTPUT_DIR,
            batch_size=128,
            mgm=MGM,
            tissue=TISSUE,
            model_type=MODEL_TYPE,
            n_heads=8,
            n_epochs=20,
            n_blocks=N_BLOCKS,
            weight_decay=WEIGHT_DECAY,
            lr_scheduler=LR_SCHEDULER,
            precision='bf16-mixed',
            fm_encoder_pkg="from_scratch",
            bert_config=config_dict,
            lr=LR,
            use_l2_norm=False,
            seed=args.seed,
            data_seed=0,
            all_genes=ALL_GENES,
            gp_latent_size=256,
            use_gene_embeddings="gf-12L-95M-i4096",
            use_flash=True,
            use_pos_emb="sin_cos",
            warmup=10,
            attn_dropout=0.0,
            sampler='weighted',
            sample_by = 'target_pathway'
        )
    
    
    ######################################
    # Train GPFinder
    ######################################
    
    GPDB_OLD = GPDB_PATH
    GPDB_NEW = f'/lustre/scratch126/cellgen/lotfollahi/cs60/proj_gpfinder/benchmark_global/{args.gp_of_interest}/gpdb_{args.gp_known}_with_hvg.csv'
        
    # GP inputs
    gp_inputs_old = list(pd.read_csv(GPDB_OLD).columns)
    
    GPFINDER_DIR = os.path.join(args.root_dir, args.gp_of_interest, f'output_gpfinder_{args.seed}')
    
    # train model if checkpoint does not exist
    if os.path.exists(os.path.join(GPFINDER_DIR, "checkpoints/last.ckpt")):
        print(f"Checkpoint found in {GPFINDER_DIR}. Skipping GPFinder training.")
        return
    else:   
        run_training_from_select_gps(
                adata_path=adata_fn,
                dataset_path=dataset_fn,
                gpdb_path=GPDB_NEW,
                gpdb_old=GPDB_OLD,
                output_dir=GPFINDER_DIR,
                tissue='pert',
                model_type_old='Base',
                model_type='Global',
                global_loss='reconstruction', # 'supervised' or 'reconstruction'
                reconstruction_loss='nb',
                global_training = 'finetune',
                path_to_base_model=OUTPUT_DIR,
                gp_inputs_new=None, # use all GPs
                gp_inputs_old=gp_inputs_old,
                fm_encoder_pkg='from_scratch', 
                bert_config=config_dict, 
                sampler='weighted',
                sample_by='target_pathway',
                seed=args.seed,
                all_genes=ALL_GENES,
                lr=LR,
                n_epochs=10,
                batch_size=128,
                gp_latent_size=256,
                n_heads=8,
                calc_gp_loss=False,
                use_gene_embeddings="gf-12L-95M-i4096", # from scratch
                precision='bf16-mixed',
            )
    
    
if __name__ == "__main__":
    main()
