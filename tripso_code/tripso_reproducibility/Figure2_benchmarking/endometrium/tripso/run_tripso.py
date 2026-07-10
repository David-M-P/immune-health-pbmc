import tripso
import os
import pandas as pd

# Directory paths for loading/saving 
root_dir = 'tripso_reproducibility/02.1_benchmarking_repeat/endometrium/run_1' 
data_dir = 'tripso_reproducibility/02.1_benchmarking_repeat/endometrium/data/processed/input_dataset' 

output_dir = os.path.join(root_dir, "output_base") 
gpdb_path = 'tripso_reproducibility/02.1_benchmarking_repeat/endometrium/gpdb_progeny_200.csv'

gpdb = pd.read_csv(gpdb_path)
gp_inputs = [g for g in list(gpdb.columns) if g != 'Trail']

gpdb_with_hvg = pd.read_csv('tripso_reproducibility/02.1_benchmarking_repeat/endometrium/gpdb_with_hvg.csv')
all_gp_genes = set()
for gp in gp_inputs:
    all_gp_genes.update(set(gpdb_with_hvg[gp].dropna().tolist()))

all_gp_genes.update(set(gpdb_with_hvg['HVG'].dropna().tolist()))
all_genes = list(all_gp_genes)

# define model training arguments
tissue = "HECA"
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

# train model
tripso.train(
    dataset_path=data_dir,
    gpdb_path=gpdb_path,
    output_dir=output_dir,
    batch_size=batch_size,
    mgm=mgm,
    tissue=tissue,
    model_type=model_type,
    n_heads=n_heads,
    n_epochs=n_epochs,
    # use_flash = True,
    n_blocks = n_blocks,
    weight_decay = weight_decay,
    lr_scheduler = lr_scheduler,
    sampler = 'weighted',
    sample_by = 'celltype',
    precision = 'bf16-mixed',
    fm_encoder_pkg = 'from_scratch',
    bert_config = config_dict,
    lr = lr,
    use_l2_norm = False,
    seed = 0, 
    data_seed = 2,
    all_genes = all_genes,
    gp_latent_size = 256,
    use_gene_embeddings = 'gf-12L-95M-i4096',
    gp_inputs = gp_inputs,
    use_flash = True,
    condition_on_length = False,
    use_pos_emb = 'sin_cos',  
    warmup = 10,    
    init_sparsity = 0
)

# raise ValueError('done')


########################################################
# Global model
########################################################

model_type = "Global"
global_loss = 'reconstruction'
reconstruction_loss = 'nb'
n_epochs = 5
batch_size = 128
global_attn_heads = 8

path_to_base_model = os.path.join(root_dir, 'output_base')

output_dir = os.path.join(root_dir, "output_global")

# train model
tripso.train(
    dataset_path=data_dir,
    gpdb_path=gpdb_path,
    output_dir=output_dir,
    batch_size=batch_size,
    mgm=mgm,
    tissue=tissue,
    model_type=model_type,
    n_heads=n_heads,
    n_epochs=n_epochs,
    n_blocks=n_blocks,
    global_loss = global_loss,
    reconstruction_loss = reconstruction_loss,
    global_training = 'sequential',
    global_attn_heads = global_attn_heads,
    adata_path = 'tripso_reproducibility/02.1_benchmarking_repeat/endometrium/data/processed/input_h5ad/endometrium_gp_genes.h5ad',
    path_to_base_model = path_to_base_model,
    lr = 1e-3,
    sampler = 'weighted',
    sample_by = 'celltype',
    weight_decay = weight_decay,
    gp_inputs = gp_inputs,
    precision = 'bf16-mixed',
    fm_encoder_pkg = 'from_scratch',
    bert_config = config_dict,
    use_gene_embeddings = 'gf-12L-95M-i4096',
    seed = 0, 
    data_seed = 2,
    all_genes = all_genes,
    gp_latent_size = 256,
    global_attn_dropout = 0.2,
    # resume_training = True
)

# raise ValueError('done training')


########################################################
# Downstream evaluation
########################################################

gp_downstream = tripso.gpEval(
    dataset_path=data_dir,
    gpdb_path=gpdb_path,
    output_dir=output_dir,
    tissue=tissue,
    model_type=model_type,
    seed = 2
)


for t in ['test', 'train', 'val']:
    gp_downstream.generate_embeddings(split = t, precision = '16-mixed')
