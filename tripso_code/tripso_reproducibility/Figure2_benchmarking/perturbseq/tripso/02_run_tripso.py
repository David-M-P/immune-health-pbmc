import argparse
import os
import pandas as pd
import tripso

# Constants for directory paths
ROOT_DIR = (
    "tripso_reproducibility/02.1_benchmarking_repeat/perturbseq/run_1"
)
DATA_DIR = (
    "tripso_reproducibility/02.1_benchmarking_repeat/perturbseq/data/processed/input_dataset"
)

# GPDB settings
GPDB_TAG = "progeny_tnfa_tgfb"
GPDB_PATH = os.path.join(ROOT_DIR, f"../gpdb_{GPDB_TAG}.csv")

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

def load_all_genes(gpdb_file: str) -> list:
    """Loads all unique gene names from the provided GPDB file."""
    all_genes_df = pd.read_csv(gpdb_file)
    all_genes = set()
    for column in all_genes_df.columns:
        all_genes.update(set(all_genes_df[column].dropna().tolist()))
    return list(all_genes)

# Load genes globally to avoid reloading during execution
ALL_GENES = load_all_genes(os.path.join(ROOT_DIR, "../gpdb_with_hvg.csv"))

def flexible_type(value):
    """Custom argument type function to accept either 'bf16-mixed' or integer 32."""
    if isinstance(value, str):
        return value
    elif isinstance(value, int):
        return value
    else:
        raise argparse.ArgumentTypeError("Allowed values are 'bf16-mixed' or 32.")

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="A script for training gene prediction models with configurable hyperparameters."
    )
    
    parser.add_argument("--depth", type=int, default=3, help="Number of blocks for gene encoder block")
    parser.add_argument("--precision", type=flexible_type, default='bf16-mixed', help="Model precision")
    parser.add_argument("--batch", type=int, default=256, help="Batch size")
    parser.add_argument("--attn_dropout", type=float, default=0.0, help="Dropout for attention layers")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument('--gp_latent_size', type=int, default=128, help='Size of latent space for GP')
    parser.add_argument('--n_epochs', type=int, default=20, help='Number of epochs')

    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Model configuration dictionary
    config_dict = {
        "hidden_size": 512,
        "num_hidden_layers": args.depth,
        "num_attention_heads": 8,
        "tokenization_input_size": 4096,
        "mlm_masking_prob": 0.25,
        "use_pos_emb": "sin_cos",
        "use_l2_norm": False,
        "tokenization_vocab_size": 20275,
        "torch_dtype": args.precision,
        "use_flash": True,
        "max_seq_len": len(ALL_GENES),
    }

    # Train the model
    tripso.train(
        dataset_path=DATA_DIR,
        gpdb_path=GPDB_PATH,
        output_dir=os.path.join(ROOT_DIR, args.output_dir),
        batch_size=args.batch,
        mgm=MGM,
        tissue=TISSUE,
        model_type=MODEL_TYPE,
        n_heads=8,
        n_epochs=args.n_epochs,
        n_blocks=N_BLOCKS,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler=LR_SCHEDULER,
        precision=args.precision,
        fm_encoder_pkg="from_scratch",
        bert_config=config_dict,
        lr=LR,
        use_l2_norm=False,
        seed=0,
        data_seed=0,
        all_genes=ALL_GENES,
        gp_latent_size=args.gp_latent_size,
        use_gene_embeddings="gf-12L-95M-i4096",
        use_flash=True,
        condition_on_length=False,
        use_pos_emb="sin_cos",
        warmup=10,
        attn_dropout=args.attn_dropout,
    )
    
    
    # --------------- Global model ----------------
    
    # train model
    tripso.train(
        dataset_path= DATA_DIR,
        gpdb_path= GPDB_PATH,
        output_dir= os.path.join(ROOT_DIR, 'output_global'),
        batch_size= args.batch,
        mgm= MGM,
        tissue= TISSUE,
        model_type='Global',
        n_heads= N_HEADS,
        n_epochs=5,
        n_blocks= args.depth,
        global_loss = 'reconstruction',
        reconstruction_loss = 'nb',
        global_training = 'sequential',
        global_attn_heads = 8,
        adata_path = os.path.join(ROOT_DIR, '../data/processed/input_h5ad/perturbseq_gp_genes.h5ad'),
        path_to_base_model = os.path.join(ROOT_DIR, 'output_base'),
        lr = 1e-3,
        weight_decay = WEIGHT_DECAY,
        precision = 'bf16-mixed',
        fm_encoder_pkg = 'from_scratch',
        bert_config = config_dict,
        use_gene_embeddings = 'gf-12L-95M-i4096',
        seed = 0, 
        data_seed = 0,
        all_genes = ALL_GENES,
        gp_latent_size = 256,
        global_attn_dropout = 0.2,
        # resume_training = True
    )
    
    # raise ValueError('Training complete. Exiting script.')
    
    # ---------------- Evaluation ----------------
    
    gp_downstream = tripso.gpEval(
        dataset_path=DATA_DIR,
        gpdb_path=GPDB_PATH,
        output_dir=os.path.join(ROOT_DIR, 'output_global'),
        tissue=TISSUE,
        model_type='Global',
        seed = 0
    )

    gp_downstream.generate_embeddings(split = 'test', precision = args.precision)
    gp_downstream.generate_embeddings(split = 'train', precision = args.precision)
        

if __name__ == "__main__":
    main()
