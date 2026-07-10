import tripso
import os
import pandas as pd

# Directory paths for loading/saving 
root_dir = 'tripso_reproducibility/04.5_HSC_post_qc/run_1_by_study'
data_dir = 'tripso_reproducibility/04.5_HSC_post_qc/data/processed/input_dataset'

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

# # train model
# tripso.train(
#     dataset_path=data_dir,
#     gpdb_path=gpdb_path,
#     output_dir=output_dir,
#     batch_size=batch_size,
#     mgm=mgm,
#     tissue=tissue,
#     model_type=model_type,
#     n_heads=n_heads,
#     n_epochs=n_epochs,
#     # use_flash = True,
#     n_blocks = n_blocks,
#     weight_decay = weight_decay,
#     lr_scheduler = lr_scheduler,
#     sampler = 'weighted',
#     sample_by = 'tissue_study',
#     precision = 'bf16-mixed',
#     fm_encoder_pkg = 'from_scratch',
#     bert_config = config_dict,
#     lr = lr,
#     use_l2_norm = False,
#     seed = 0, 
#     data_seed = 2,
#     all_genes = all_genes,
#     gp_latent_size = 256,
#     use_gene_embeddings = 'gf-12L-95M-i4096',
#     use_flash = True,
#     condition_on_length = False,
#     use_pos_emb = 'sin_cos',  
#     warmup = 10,    
#     init_sparsity = 0,
#     gene_format = "ensembl"
# )

# raise ValueError('done')


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

# # train model
# tripso.train(
#     dataset_path=data_dir,
#     gpdb_path=gpdb_path,
#     output_dir=output_dir,
#     batch_size=batch_size,
#     mgm=mgm,
#     tissue=tissue,
#     model_type=model_type,
#     n_heads=n_heads,
#     n_epochs=n_epochs,
#     n_blocks=n_blocks,
#     global_loss = global_loss,
#     reconstruction_loss = reconstruction_loss,
#     global_training = 'sequential',
#     global_attn_heads = global_attn_heads,
#     adata_path = os.path.join(root_dir, '../data/processed/joint_gp_genes.h5ad'),
#     path_to_base_model = path_to_base_model,
#     lr = 1e-3,
#     sampler = 'weighted',
#     sample_by = 'tissue_study',
#     weight_decay = weight_decay,
#     precision = 'bf16-mixed',
#     fm_encoder_pkg = 'from_scratch',
#     bert_config = config_dict,
#     use_gene_embeddings = 'gf-12L-95M-i4096',
#     seed = 0, 
#     data_seed = 2,
#     all_genes = all_genes,
#     gp_latent_size = 256,
#     gene_format = "ensembl",
#     global_attn_dropout = 0.2
#     # resume_training = True
# )

# raise ValueError('done training global model')

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


# for t in ['train', 'val', 'test']:
#     gp_downstream.generate_embeddings(split = t, precision = '16-mixed')
#     # gp_downstream.generate_attention_matrix(gp = 'cell_token', precision = '16-mixed', split = t)


########################################################
# generate GP importance scores
#######################################################

from tripso.Evaluate.downstream import gpAblationEval

# Directory paths for loading/saving 
output_dir = os.path.join(root_dir, "output_global/ablation")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# downstream evaluation
gp_downstream = gpAblationEval(
    dataset_path=data_dir,
    gpdb_path=gpdb_path,
    output_dir=output_dir,
    model_type = 'Global',
    tissue = 'HSC',
    main_ckpt_dir = os.path.join(root_dir, 'output_global'),
    seed = 2, 
    compute_cosine = True
)

# Generate embeddings for train and test set
# gp_downstream.generate_embeddings(split = 'test', precision = '16-mixed')
# gp_downstream.generate_embeddings(split = 'val', precision = '16-mixed')
gp_downstream.generate_embeddings(split = 'train', precision = '16-mixed')
