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

def pp_tripso(data, gp, y, obs_cols, filter_var = None, str_to_match = None):
    adata = sc.AnnData(
        X = np.array(data[gp]),
        obs = data.select_columns(obs_cols).to_pandas()

    )
    
    # filter to cells of interest
    if filter_var is not None:
        adata = adata[adata.obs[filter_var].str.contains(str_to_match)]
    
    adata.var.index = [f'{gp}_{i}' for i in range(len(adata.var.index))]
    
    return adata

def pp_scalar(adata, gp, idx_to_keep = None, filter_var = None, str_to_match = None):
    adata = adata[:, adata.var.index.str.contains(gp, case = False)].copy()
    
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


def do_gene_eval(topk_genes, adata_train_in, adata_test_in, gp, out_dir_label, k):
    adata_train = adata_train_in[adata_train_in.obs['gene'].isin(topk_genes)].copy()
    adata_train = adata_train[adata_train.obs['target_pathway'] == gp].copy()
    
    adata_test = adata_test_in[adata_test_in.obs['gene'].isin(topk_genes)]
    adata_test = adata_test[adata_test.obs['target_pathway'] == gp]
    
    # encode gene variable
    adata_train = encode_labels_h5ad(adata_train, 'gene', 'gene_id')
    
    encoding_df = adata_train.obs[['gene', 'gene_id']].drop_duplicates()
    encoding_dict = {k : v for k, v in zip(encoding_df['gene'], encoding_df['gene_id'])}
    
    adata_test.obs['gene_id'] = adata_test.obs['gene'].map(encoding_dict)
    
    calc_eval_metrics(adata_train, 
                      adata_test, 
                      gp, 
                      'gene_id', 
                      output_dir = os.path.join(
                          output_dir,
                            out_dir_label,
                            f'knn_top_{k}'
                          ), 
                      k = 30,
                      data_type = 'h5ad',
                 )
    
    # scale for logistic regression
    print('')
    print('*** Scaling data for logistic regression ***')
    print('')
    scaler = preprocessing.StandardScaler().fit(adata_train.X)
    adata_train.X = scaler.transform(adata_train.X)
    adata_test.X = scaler.transform(adata_test.X)
    
    calc_eval_metrics(adata_train,
                        adata_test,
                        gp,
                        'gene_id',
                        output_dir = os.path.join(
                            output_dir,
                            out_dir_label,
                            f'logistic_regression_top_{k}'
                                                  ),
                        data_type = 'h5ad',
                        model_type = 'logistic'
                        )
    



def run_satija_evaluation(embedding_dir, out_dir_label, model_family = 'tripso', train_idx = None, test_idx = None):
    if model_family == 'tripso':
        train_set = load_from_disk(os.path.join(embedding_dir, 'embeddings/train_set')).shuffle(seed = 0) 
        test_set = load_from_disk(os.path.join(embedding_dir, 'embeddings/test_set')).shuffle(seed = 0) 
        
        tnf_train = pp_tripso(train_set, 
                                'TNFa', 
                                'target_pathway', 
                                ['idx', 'target_pathway', 'gene']
                                )
         
        # reencode y label
        tnf_train = encode_labels_h5ad(tnf_train, 'target_pathway', 'target_pathway_id')
        conversion_df = tnf_train.obs[['target_pathway', 'target_pathway_id']].drop_duplicates()
        conversion_dict = {k: v for k, v in zip(conversion_df['target_pathway'], conversion_df['target_pathway_id'])}
        
        tnf_test = pp_tripso(test_set,
                               'TNFa',
                               'target_pathway',
                               ['idx', 'target_pathway', 'gene']
        )
        
        tnf_test.obs['target_pathway_id'] = tnf_test.obs['target_pathway'].map(conversion_dict)
        
        tgf_train = pp_tripso(train_set,
                                'TGFb',
                                'target_pathway',
                                ['idx',  'target_pathway', 'gene']
        )
        
        tgf_train = encode_labels_h5ad(tgf_train, 'target_pathway', 'target_pathway_id')
        conversion_df = tgf_train.obs[['target_pathway', 'target_pathway_id']].drop_duplicates()
        conversion_dict = {k: v for k, v in zip(conversion_df['target_pathway'], conversion_df['target_pathway_id'])}
        
        tgf_test = pp_tripso(test_set,
                               'TGFb',
                               'target_pathway',
                               ['idx', 'target_pathway', 'gene']
        )
        
        tgf_test.obs['target_pathway_id'] = tgf_test.obs['target_pathway'].map(conversion_dict)
        
    elif model_family == 'expimap':
        latent_train = sc.read_h5ad(os.path.join(embedding_dir, 'train_latent.h5ad'))
        latent_test = sc.read_h5ad(os.path.join(embedding_dir, 'test_latent.h5ad'))
        
        tnf_train = pp_scalar(latent_train, gp = 'TNFa')
        
        tnf_train = encode_labels_h5ad(tnf_train, 'target_pathway', 'target_pathway_id')
        conversion_df = tnf_train.obs[['target_pathway', 'target_pathway_id']].drop_duplicates()
        conversion_dict = {k: v for k, v in zip(conversion_df['target_pathway'], conversion_df['target_pathway_id'])}
        
        tnf_test = pp_scalar(latent_test, gp = 'TNFa')
        
        tnf_test.obs['target_pathway_id'] = tnf_test.obs['target_pathway'].map(conversion_dict)
        
        tgf_train = pp_scalar(latent_train, gp = 'TGFb')
        
        tgf_train = encode_labels_h5ad(tgf_train, 'target_pathway', 'target_pathway_id')
        
        conversion_df = tgf_train.obs[['target_pathway', 'target_pathway_id']].drop_duplicates()
        conversion_dict = {k: v for k, v in zip(conversion_df['target_pathway'], conversion_df['target_pathway_id'])}
        
        tgf_test = pp_scalar(latent_test, gp = 'TGFb')
        
        tgf_test.obs['target_pathway_id'] = tgf_test.obs['target_pathway'].map(conversion_dict)
        
    
    else:
        latent = sc.read_h5ad(os.path.join(embedding_dir))
        
        tnf_train = pp_scalar(latent, 
                              gp = 'TNFa', 
                              idx_to_keep = train_idx)
        
        tnf_train = encode_labels_h5ad(tnf_train, 'target_pathway', 'target_pathway_id')
        conversion_df = tnf_train.obs[['target_pathway', 'target_pathway_id']].drop_duplicates()
        conversion_dict = {k: v for k, v in zip(conversion_df['target_pathway'], conversion_df['target_pathway_id'])}
        
        tnf_test = pp_scalar(latent,
                             gp = 'TNFa',
                             idx_to_keep = test_idx
                             )
        
        tnf_test.obs['target_pathway_id'] = tnf_test.obs['target_pathway'].map(conversion_dict)

        tgf_train = pp_scalar(latent, gp = 'TGFb', idx_to_keep=train_idx)
        
        tgf_train = encode_labels_h5ad(tgf_train, 'target_pathway', 'target_pathway_id')
        conversion_df = tgf_train.obs[['target_pathway', 'target_pathway_id']].drop_duplicates()
        conversion_dict = {k: v for k, v in zip(conversion_df['target_pathway'], conversion_df['target_pathway_id'])}
        
        tgf_test = pp_scalar(latent, gp = 'TGFb', idx_to_keep=test_idx)
        
        tgf_test.obs['target_pathway_id'] = tgf_test.obs['target_pathway'].map(conversion_dict)
        
    
    print('===========================')
    print('Predicting target pathway from TNFa')
    print('===========================')

    # KNN evaluation
    calc_eval_metrics(tnf_train, 
                  tnf_test, 
                  'TNFa', 
                  'target_pathway_id', 
                  output_dir = os.path.join(output_dir, 
                                            out_dir_label,
                                            'knn'), 
                  k = 30,
                  data_type = 'h5ad',
                 )
    
    
    for k in [1, 2, 3, 4, 5, 10]:
        do_gene_eval(topk_genes[f'TNFa_top_{k}'], tnf_train, tnf_test, 'TNFa', out_dir_label, k)
    
    
    # Scale for logistic regression
    print('')
    print('*** Scaling data for logistic regression ***')
    print('')
    scaler = preprocessing.StandardScaler().fit(tnf_train.X)
    tnf_train.X = scaler.transform(tnf_train.X)
    tnf_test.X = scaler.transform(tnf_test.X)
    
    calc_eval_metrics(tnf_train, 
                tnf_test, 
                'TNFa', 
                'target_pathway_id', 
                output_dir = os.path.join(output_dir,
                                          out_dir_label,
                                          'logistic_regression'
                                          ), 
                data_type = 'h5ad',
                model_type = 'logistic' 
                )


    print('===========================')
    print('Predicting target pathway from TGFb')
    print('===========================')

    calc_eval_metrics(tgf_train, 
                  tgf_test, 
                  'TGFb', 
                  'target_pathway_id', 
                  output_dir = os.path.join(
                        output_dir,
                        out_dir_label,
                        'knn'
                      ),
                  k = 30,
                  data_type = 'h5ad',
                 )
    
    
    for k in [1, 2, 3, 4, 5, 10]:
        do_gene_eval(topk_genes[f'TGFb_top_{k}'], tgf_train, tgf_test, 'TGFb', out_dir_label, k)
    
    # Scale for logistic regression
    print('')
    print('*** Scaling data for logistic regression ***')
    print('')
    scaler = preprocessing.StandardScaler().fit(tgf_train.X)
    tgf_train.X = scaler.transform(tgf_train.X)
    tgf_test.X = scaler.transform(tgf_test.X)
    
    calc_eval_metrics(tgf_train, 
                  tgf_test, 
                  'TGFb', 
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
