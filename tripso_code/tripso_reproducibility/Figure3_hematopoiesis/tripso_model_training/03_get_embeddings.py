import tripso
import os
import pandas as pd

# Directory paths for loading/saving 
root_dir = 'tripso_reproducibility/04.4_HSC_fix_hvg/run_1_by_study'
data_dir = 'tripso_reproducibility/04.4_HSC_fix_hvg/data/processed/input_dataset'

output_dir = os.path.join(root_dir, "output_base") 
gpdb_path = os.path.join(root_dir, '../gpdb_tf.csv')


# define list of genes to use    
var_df = pd.read_csv(os.path.join(root_dir, '../data/processed/merged_adata_hvg.csv'), index_col = 0)
var_df = var_df[var_df['highly_variable']]
hvg = var_df.index.tolist()

gpdb = pd.read_csv(gpdb_path)

all_genes = set()

for i in gpdb.columns:
    all_genes.update(gpdb[i].dropna().values)
    
all_genes.update(hvg)
all_genes = list(all_genes)

# define model training arguments
tissue = "HSC"
model_type = "Base"
n_heads = 8
n_blocks = 2
weight_decay = 1e-4
mgm = 0.25
n_epochs = 20 
batch_size = 128 
lr_scheduler = 'ReduceLROnPlateau'
lr = 1e-4


########################################################
# Base model
########################################################

# Configuration from the provided config file
config_dict = {
    'hidden_size' : 512,
    'num_hidden_layers' : 2,
    'num_attention_heads' : 8,
    'tokenization_input_size' : 4096,
    'mlm_masking_prob' : 0.25,
    'use_pos_emb' : 'sin_cos',
    'use_l2_norm' : False,
    'tokenization_vocab_size': 20275,
    'torch_dtype' : 'bf16',
    'use_flash' : True,
    'max_seq_len': len(all_genes),
}


########################################################
# Global model
########################################################

model_type = "Global"
global_loss = 'reconstruction'
reconstruction_loss = 'nb'
n_epochs = 5
batch_size = 64
global_attn_heads = 8 

path_to_base_model = os.path.join(root_dir, 'output_base')

output_dir = os.path.join(root_dir, "output_global")

########################################################
# Downstream evaluation
########################################################


gp_downstream = tripso.gpEval(
    dataset_path=data_dir,
    gpdb_path=gpdb_path,
    output_dir=output_dir,
    tissue=tissue,
    model_type=model_type,
    seed = 2,
)

gp_downstream.test_random_baseline('tripso_reproducibility/04.4_HSC_fix_hvg/data/processed/joint_gp_genes.h5ad')