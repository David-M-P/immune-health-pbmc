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

def pp_tripso(data, gp, y, obs_cols, filter_var, str_to_match,):
    adata = sc.AnnData(
        X = np.array(data[gp]),
        obs = data.select_columns(obs_cols).to_pandas()

    )

    # filter to cells of interest
    adata = adata[adata.obs[filter_var].str.contains(str_to_match)]

    adata.var.index = [f'{gp}_{i}' for i in range(len(adata.var.index))]
    
    return adata

def pp_scalar(adata, gp, filter_var, str_to_match, idx_to_keep = None):
    adata = adata[:, adata.var.index.str.contains(gp, case = False)]
    
    # filter to cells of interest
    adata = adata[adata.obs[filter_var].str.contains(str_to_match)]
    
    # filter relevant split
    if idx_to_keep:
        if 'idx' in adata.obs.columns:
            adata = adata[adata.obs['idx'].isin(idx_to_keep)]
        else:
            adata = adata[adata.obs.index.isin(idx_to_keep)]
    
    return adata

    

def run_endometrium_evaluation(embedding_dir, out_dir_label, model_family = 'tripso', train_idx = None, test_idx = None):
    if model_family == 'tripso':
        train_set = load_from_disk(os.path.join(embedding_dir, 'embeddings/train_set')).shuffle(seed = 0) 
        test_set = load_from_disk(os.path.join(embedding_dir, 'embeddings/test_set')).shuffle(seed = 0) 
        
        wnt_train = pp_tripso(train_set, 
                                'WNT', 
                                'celltype', 
                                ['idx', 'celltype', 'lineage', 'Binary Stage', 'TGFb_num_genes', 'Endometriosis_stage'], 
                                'celltype',
                                'Glandular|Luminal|Ciliated')
        
        wnt_test = pp_tripso(test_set,
                               'WNT',
                               'celltype',
                               ['idx', 'celltype', 'lineage', 'Binary Stage', 'TGFb_num_genes', 'Endometriosis_stage'],
                               'celltype',
                               'Glandular|Luminal|Ciliated'
        )
        
        tgf_train = pp_tripso(train_set,
                                'TGFb',
                                'Binary Stage',
                                ['idx', 'celltype', 'lineage', 'Binary Stage', 'TGFb_num_genes', 'Endometriosis_stage'],
                                'lineage',
                                'Mesenchymal'
        )
        
        tgf_test = pp_tripso(test_set,
                               'TGFb',
                               'Binary Stage',
                               ['idx', 'celltype', 'lineage', 'Binary Stage', 'TGFb_num_genes', 'Endometriosis_stage'],
                               'lineage',
                               'Mesenchymal'
        )
        
    elif model_family == 'expimap':
        latent_train = sc.read_h5ad(os.path.join(embedding_dir, 'train_latent.h5ad'))
        latent_test = sc.read_h5ad(os.path.join(embedding_dir, 'test_latent.h5ad'))
        
        wnt_train = pp_scalar(latent_train, 
                              'WNT',
                              'celltype',
                              'Glandular|Luminal|Ciliated',
                               )
        
        wnt_test = pp_scalar(latent_test,
                             'WNT',
                              'celltype',
                              'Glandular|Luminal|Ciliated',
                               )
        
        tgf_train = pp_scalar(latent_train,
                                 'TGFb',
                                 'lineage',
                                 'Mesenchymal'
          )
        
        tgf_test = pp_scalar(latent_test,
                                'TGFb',
                                'lineage',
                                'Mesenchymal'
          )
        
        
    else:
        latent = sc.read_h5ad(os.path.join(embedding_dir))
        
        # for spectra, rename cell type column
        if model_family == 'spectra':
            latent.obs = latent.obs.rename(columns = {'cell_type' : 'celltype'})
        
        wnt_train = pp_scalar(latent, 
                              'WNT', 
                              'celltype',
                              'Glandular|Luminal|Ciliated',
                              train_idx)
        
        wnt_test = pp_scalar(latent,
                             'WNT',
                             'celltype',
                             'Glandular|Luminal|Ciliated',
                             test_idx
        )
        
        tgf_train = pp_scalar(latent,
                               'TGFb',
                               'lineage',
                               'Mesenchymal',
                               train_idx
        )
        
        tgf_test = pp_scalar(latent,
                               'TGFb',
                               'lineage',
                               'Mesenchymal',
                               test_idx
        )
        
    
    print('===========================')
    print('Predicting cell type from WNT')
    print('===========================')

    # broad cell type labels
    wnt_train.obs['ct_broad'] = np.where(
        wnt_train.obs['celltype'] == 'Ciliated', 
        'Ciliated', 
        'Secretory'
    )

    wnt_test.obs['ct_broad'] = np.where(
        wnt_test.obs['celltype'] == 'Ciliated', 
        'Ciliated', 
        'Secretory'
    )

    wnt_train = encode_labels_h5ad(wnt_train, 'ct_broad', 'ct_broad_id')
    conversion_df = wnt_train.obs[['ct_broad', 'ct_broad_id']].drop_duplicates()
    conversion_dict = {k: v for k, v in zip(conversion_df['ct_broad'], conversion_df['ct_broad_id'])}
    wnt_test.obs['ct_broad_id'] = wnt_test.obs['ct_broad'].map(conversion_dict)
    
    wnt_train = encode_labels_h5ad(wnt_train, 'celltype', 'celltype_id')
    conversion_df = wnt_train.obs[['celltype', 'celltype_id']].drop_duplicates()
    conversion_dict = {k: v for k, v in zip(conversion_df['celltype'], conversion_df['celltype_id'])}
    wnt_test.obs['celltype_id'] = wnt_test.obs['celltype'].map(conversion_dict)
    
    
    # KNN evaluation

    calc_eval_metrics(wnt_train, 
                      wnt_test, 
                      'WNT', 
                      'celltype_id', 
                      output_dir = os.path.join(root_dir, 
                                                out_dir_label,
                                                'knn'), 
                      k = 15,
                      data_type = 'h5ad',
                      )

    calc_eval_metrics(wnt_train, 
                      wnt_test, 
                      'WNT', 
                      'ct_broad_id', 
                      output_dir = os.path.join(root_dir, 
                                                out_dir_label,
                                                'knn'), 
                      k = 15,
                      data_type = 'h5ad',
                 )
    
    
    # Scale for logistic regression
    print('')
    print('*** Scaling data for logistic regression ***')
    print('')
    scaler = preprocessing.StandardScaler().fit(wnt_train.X)
    wnt_train.X = scaler.transform(wnt_train.X)
    wnt_test.X = scaler.transform(wnt_test.X)
    
    calc_eval_metrics(wnt_train, 
                      wnt_test, 
                      'WNT', 
                      'celltype_id', 
                      output_dir = os.path.join(root_dir, 
                                                out_dir_label,
                                                'logistic_regression'), 
                      data_type = 'h5ad',
                      model_type = 'logistic' 
                      )

    calc_eval_metrics(wnt_train, 
                      wnt_test, 
                      'WNT', 
                      'ct_broad_id', 
                      output_dir = os.path.join(root_dir, 
                                                out_dir_label,
                                                'logistic_regression'), 
                      data_type = 'h5ad',
                      model_type = 'logistic' 
                      )


    print('===========================')
    print('Predicting stage from TGFb')
    print('===========================')


    tgf_train.obs = tgf_train.obs.rename(columns = {'Binary Stage' : 'stage'})
    tgf_test.obs = tgf_test.obs.rename(columns = {'Binary Stage' : 'stage'})
    
    tgf_train = tgf_train[tgf_train.obs['stage'].isin(['Proliferative', 'Secretory'])]
    tgf_test = tgf_test[tgf_test.obs['stage'].isin(['Proliferative', 'Secretory'])]
    
    # rencode stage
    tgf_train = encode_labels_h5ad(tgf_train, 'stage', 'stage_id')
    
    conversion_df = tgf_train.obs[['stage', 'stage_id']].drop_duplicates()
    conversion_dict = {k: v for k, v in zip(conversion_df['stage'], conversion_df['stage_id'])}
    tgf_test.obs['stage_id'] = tgf_test.obs['stage'].map(conversion_dict)

    calc_eval_metrics(tgf_train, 
                  tgf_test, 
                  'TGFb', 
                  'stage_id', 
                  output_dir = os.path.join(root_dir, 
                                                out_dir_label,
                                                'knn'), 
                  k = 15,
                  data_type = 'h5ad',
                 )
    
    
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
                  'stage_id', 
                  output_dir = os.path.join(root_dir, 
                                                out_dir_label,
                                                'logistic_regression'), 
                  data_type = 'h5ad',
                  model_type = 'logistic'
                 )
    
    print('===========================')
    print('Predicting granular stage from TGFb')
    print('===========================')
    
    # load back granular stage labels
    heca = sc.read_h5ad('/nfs/team292/lg18/endometriosis/cellxgene_objects/endometriumAtlasV2_cells.h5ad')
    idx_to_stage = dict(zip(heca.obs.index, heca.obs['Stage']))
    
    if 'idx' in tgf_train.obs.columns:
        # to do --> check what happens in spectra latent object

        tgf_train.obs['stage_granular'] = tgf_train.obs['idx'].map(idx_to_stage)
        tgf_test.obs['stage_granular'] = tgf_test.obs['idx'].map(idx_to_stage)

        tgf_train = tgf_train[tgf_train.obs['stage_granular'].isin([
            'Proliferative Early', 'Proliferative Late', 
            'Secretory Early', 'Secretory Early-Mid', 'Secretory Mid', 'Secretory Late'])]

        tgf_test = tgf_test[tgf_test.obs['stage_granular'].isin([
            'Proliferative Early', 'Proliferative Late',
            'Secretory Early', 'Secretory Early-Mid', 'Secretory Mid', 'Secretory Late'])]

        # rencode stage
        tgf_train = encode_labels_h5ad(tgf_train, 'stage_granular', 'stage_granular_id')

        conversion_df = tgf_train.obs[['stage_granular', 'stage_granular_id']].drop_duplicates()
        conversion_dict = {k: v for k, v in zip(conversion_df['stage_granular'], conversion_df['stage_granular_id'])}
        tgf_test.obs['stage_granular_id'] = tgf_test.obs['stage_granular'].map(conversion_dict)

        calc_eval_metrics(tgf_train, 
                      tgf_test, 
                      'TGFb', 
                      'stage_granular_id', 
                      output_dir = os.path.join(root_dir, 
                                                    out_dir_label,
                                                    'knn'), 
                      k = 15,
                      data_type = 'h5ad',
                     )


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
                      'stage_granular_id', 
                      output_dir = os.path.join(root_dir, 
                                                    out_dir_label,
                                                    'logistic_regression'), 
                      data_type = 'h5ad',
                      model_type = 'logistic'
                     )


##############################################


root_dir = 'tripso_reproducibility/02.1_benchmarking_repeat/endometrium/evaluation_metrics'

emb = load_from_disk(os.path.join(root_dir, '../run_1/output_global/embeddings/train_set'))
train_idx = emb['idx']

emb = load_from_disk(os.path.join(root_dir, '../run_1/output_global/embeddings/test_set'))
test_idx = emb['idx']


##############################################
# GPformer 
##############################################

# run_endometrium_evaluation(
#     os.path.join(root_dir, '../run_1/output_global'),
#     'GPformer/run_1',
# )

# run_endometrium_evaluation(
#     os.path.join(root_dir, '../run_2/output_global'),
#     'GPformer/run_2',
# )

# run_endometrium_evaluation(
#     os.path.join(root_dir, '../run_3/output_global'),
#     'GPformer/run_3',
# )



##############################################
# Expimap
##############################################

run_endometrium_evaluation(
    os.path.join(root_dir, '../expimap/expimap_1'),
    'expimap/run_1',
    model_family = 'expimap',
    train_idx = train_idx,
    test_idx = test_idx,
)

run_endometrium_evaluation(
    os.path.join(root_dir, '../expimap/expimap_2'),
    'expimap/run_2',
    model_family = 'expimap',
    train_idx = train_idx,
    test_idx = test_idx,
)

run_endometrium_evaluation(
    os.path.join(root_dir, '../expimap/expimap_3'),
    'expimap/run_3',
    model_family = 'expimap',
    train_idx = train_idx,
    test_idx = test_idx,
)



##############################################
# Spectra
##############################################

run_endometrium_evaluation( 
                           os.path.join(root_dir, '../spectra/spectra_1/spectra_latent.h5ad'),
                           'spectra/run_1',
                           model_family = 'spectra',
                           train_idx = train_idx,
                           test_idx = test_idx
                           )

run_endometrium_evaluation( 
                           os.path.join(root_dir, '../spectra/spectra_2/spectra_latent.h5ad'),
                           'spectra/run_2',
                           model_family = 'spectra',
                           train_idx = train_idx,
                           test_idx = test_idx
                           )

run_endometrium_evaluation( 
                           os.path.join(root_dir, '../spectra/spectra_3/spectra_latent.h5ad'),
                           'spectra/run_3',
                           model_family = 'spectra',
                           train_idx = train_idx,
                           test_idx = test_idx
                           )