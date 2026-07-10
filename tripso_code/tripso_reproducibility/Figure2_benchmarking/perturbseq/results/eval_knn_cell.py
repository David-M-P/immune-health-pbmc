##############################################
# Run kNN evaluation
##############################################

from tripso.Utils.utils import encode_labels_h5ad
from tripso.Evaluate.downstream import calc_eval_metrics
import os
import scanpy as sc
from datasets import load_from_disk
import pandas as pd
import numpy as np
from sklearn import preprocessing


# set seed
np.random.seed(0)

def pp_tripso(data, obs_cols, filter_var = None, str_to_match = None):
    adata = sc.AnnData(
        X = np.array(data['cell_token']),
        obs = data.select_columns(obs_cols).to_pandas()

    )
    
    # filter to cells of interest
    if filter_var is not None:
        adata = adata[adata.obs[filter_var].str.contains(str_to_match)]
    
    adata.var.index = [f'cls_{i}' for i in range(len(adata.var.index))]
    
    return adata

def pp_scalar(adata, idx_to_keep = None, filter_var = None, str_to_match = None):    
    # filter to cells of interest
    if filter_var is not None:
        adata = adata[adata.obs[filter_var].str.contains(str_to_match)]
    
    # filter relevant split
    if idx_to_keep is not None:
        if 'idx' in adata.obs.columns:
            adata = adata[adata.obs['idx'].isin(idx_to_keep)]
        else:
            adata = adata[adata.obs.index.isin(idx_to_keep)]
    
    return adata


def run_satija_evaluation(embedding_dir, out_dir_label, model_family = 'tripso', train_idx = None, test_idx = None):
    if model_family == 'tripso':
        train_set = load_from_disk(os.path.join(embedding_dir, 'embeddings/train_set')).shuffle(seed = 0) 
        test_set = load_from_disk(os.path.join(embedding_dir, 'embeddings/test_set')).shuffle(seed = 0) 
        
        cell_train = pp_tripso(train_set,  
                                obs_cols = ['idx', 'target_pathway', 'gene']
                                )
         
        # reencode y label
        cell_train = encode_labels_h5ad(cell_train, 'target_pathway', 'target_pathway_id')
        conversion_df = cell_train.obs[['target_pathway', 'target_pathway_id']].drop_duplicates()
        conversion_dict = {k: v for k, v in zip(conversion_df['target_pathway'], conversion_df['target_pathway_id'])}
        
        cell_test = pp_tripso(test_set,
                               obs_cols = ['idx', 'target_pathway', 'gene']
        )
        
        cell_test.obs['target_pathway_id'] = cell_test.obs['target_pathway'].map(conversion_dict)
        
    elif model_family == 'expimap':
        latent_train = sc.read_h5ad(os.path.join(embedding_dir, 'train_latent.h5ad'))
        latent_test = sc.read_h5ad(os.path.join(embedding_dir, 'test_latent.h5ad'))
        
        cell_train = pp_scalar(latent_train)
        cell_test = pp_scalar(latent_test)
        
        cell_train = encode_labels_h5ad(cell_train, 'target_pathway', 'target_pathway_id')
        conversion_df = cell_train.obs[['target_pathway', 'target_pathway_id']].drop_duplicates()
        conversion_dict = {k: v for k, v in zip(conversion_df['target_pathway'], conversion_df['target_pathway_id'])}
        
        cell_test.obs['target_pathway_id'] = cell_test.obs['target_pathway'].map(conversion_dict)
        
    else:
        latent = sc.read_h5ad(os.path.join(embedding_dir))
        
        cell_train = pp_scalar(latent, idx_to_keep=train_idx)
        
        cell_train = encode_labels_h5ad(cell_train, 'target_pathway', 'target_pathway_id')
        conversion_df = cell_train.obs[['target_pathway', 'target_pathway_id']].drop_duplicates()
        conversion_dict = {k: v for k, v in zip(conversion_df['target_pathway'], conversion_df['target_pathway_id'])}
        
        cell_test = pp_scalar(latent, idx_to_keep=test_idx)
        
        cell_test.obs['target_pathway_id'] = cell_test.obs['target_pathway'].map(conversion_dict)

    
    print('===========================')
    print('Predicting target pathway from cell token')
    print('===========================')

    # KNN evaluation
    calc_eval_metrics(cell_train, 
                  cell_test, 
                  'cell_token', 
                  'target_pathway_id', 
                  output_dir = os.path.join(output_dir, 
                                            out_dir_label,
                                            'knn'), 
                  k = 30,
                  data_type = 'h5ad',
                 )

    
    
    # Scale for logistic regression
    print('')
    print('*** Scaling data for logistic regression ***')
    print('')
    scaler = preprocessing.StandardScaler().fit(cell_train.X)
    cell_train.X = scaler.transform(cell_train.X)
    cell_test.X = scaler.transform(cell_test.X)
    
    calc_eval_metrics(cell_train, 
                cell_test, 
                'cell_token', 
                'target_pathway_id', 
                output_dir = os.path.join(output_dir,
                                          out_dir_label,
                                          'logistic_regression'
                                          ), 
                data_type = 'h5ad',
                model_type = 'logistic' 
                )


##############################################


root_dir = 'tripso_reproducibility/02.1_benchmarking_repeat/perturbseq'
output_dir = os.path.join(root_dir, 'evaluation_metrics')

emb = load_from_disk(os.path.join(root_dir, 'run_1/output_global/embeddings/train_set'))
train_idx = emb['idx']

emb = load_from_disk(os.path.join(root_dir, 'run_1/output_global/embeddings/test_set'))
test_idx = emb['idx']

topk_genes = pd.read_pickle('tripso_reproducibility/02.1_benchmarking_repeat/perturbseq/top_k_genes.pkl')



##############################################
# GPformer
##############################################

run_satija_evaluation(
    os.path.join(root_dir, 'run_1/output_global'),
    'GPformer/run_1'
)

run_satija_evaluation(
    os.path.join(root_dir, 'run_2/output_global'),
    'GPformer/run_2'
)

run_satija_evaluation(
    os.path.join(root_dir, 'run_3/output_global'),
    'GPformer/run_3'
)



##############################################
# Expimap
##############################################

run_satija_evaluation(
    os.path.join(root_dir, 'expimap/run_1/run_1'),
    'expimap/run_1',
    model_family = 'expimap',
    train_idx = train_idx,
    test_idx = test_idx,
)

run_satija_evaluation(
    os.path.join(root_dir, 'expimap/run_2/run_2'),
    'expimap/run_2',
    model_family = 'expimap',
    train_idx = train_idx,
    test_idx = test_idx,
)

run_satija_evaluation(
    os.path.join(root_dir, 'expimap/run_3/run_3'),
    'expimap/run_3',
    model_family = 'expimap',
    train_idx = train_idx,
    test_idx = test_idx,
)


##############################################
# Spectra
##############################################

run_satija_evaluation(
    os.path.join(root_dir, 'spectra/run_1/spectra_latent.h5ad'),
    'spectra/run_1',
    model_family = 'spectra',
    train_idx = train_idx,
    test_idx = test_idx,
)

run_satija_evaluation(
    os.path.join(root_dir, 'spectra/run_2/spectra_latent.h5ad'),
    'spectra/run_2',
    model_family = 'spectra',
    train_idx = train_idx,
    test_idx = test_idx,
)

run_satija_evaluation(
    os.path.join(root_dir, 'spectra/run_3/spectra_latent.h5ad'),
    'spectra/run_3',
    model_family = 'spectra',
    train_idx = train_idx,
    test_idx = test_idx,
)
