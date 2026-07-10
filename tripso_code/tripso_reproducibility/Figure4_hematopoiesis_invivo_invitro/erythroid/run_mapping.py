import argparse
import os
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import anndata as ad
from datasets import load_from_disk, concatenate_datasets
from tripso.Evaluate.optimal_transport import (
    compute_centroid_mapping,
    make_contingency_table,
    plot_umap_with_centroids
)

from tripso.Utils.utils import sample_cells

def compute_unbalanced_ot(gp, epsilon, cluster_algo, output_dir, resolution=1, threshold=1e-12, num_clusters = 50, tau = 0.999, load_train = True, num_cells = 100):
    if load_train:
        emb1 = load_from_disk('../../../embeddings/train_set')
    
    emb2 = load_from_disk('../../../embeddings/val_set')
    emb3 = load_from_disk('../../../embeddings/test_set')
    
    if load_train:
        emb = concatenate_datasets([emb1, emb2, emb3])
    else:
        emb = concatenate_datasets([emb2, emb3])
        
    order_source1 = [
        # MNC
        '1_HSC_MPP','2_MEMP', '4_BFU-E/CFU-E', '5_Early_erythroblast', '6_Mid_erythroblast', 
        '7_Late_erythroblast', '8_HBE+_embryonic_erythrocyte',
    ]
    
    order_source2 = [# CD34
        '1_LT-HSC', '2_ST-HSC', '3_MPP', '4_MEMP', 
        '6_Early_Ery', '7_Late_Ery', 
         ]
    
    order_target = [
        'HSC/MPP 1', 'HSC/MPP 2', 'MEMP', 'Early Erythroid', 'Mid Erythroid',
        'Late Erythroid',
        #'NMP', 'Mono precur', 'Monocyte', 'Mast cell', 'DC precursor', 'DC'
    ]

    
    adata = sc.AnnData(
        X=np.array(emb[gp]),
        obs=emb.select_columns([
            'idx', 'source', 'cell_type', 'tissue', 'study',
            'age_general'
        ]).to_pandas()
    )
    
    adata = adata[adata.obs['study'].isin(['Isobe_CD34', 'Isobe_MNC', 'Gao_CB'])]
    
    adata = adata[adata.obs['cell_type'].isin(order_source1 + order_source2 + order_target)]
    
    # wrangle in vivo cell types
    invitro = adata[adata.obs['source'] == 'in vitro'].copy()
    invivo = adata[adata.obs['source'] == 'in vivo'].copy()
        
    invivo.obs['cell_type'] = invivo.obs['cell_type'].replace(
        {
            '6_Early_Ery' : '6_BFU-E/CFU-E',
            '7_Late_Ery' : '7_Early_erythroblast',        
        }
    )

    invivo.obs['cell_type'] = invivo.obs['cell_type'].str.replace(r'^\d+_', '', regex=True)

    invivo = invivo[invivo.obs['cell_type'] != 'HSC_MPP']
    
    order_source = ['LT-HSC', 'ST-HSC', 'MPP', 'MEMP',
                  'BFU-E/CFU-E', 
                  'Early_erythroblast', 'Mid_erythroblast',
                  'Late_erythroblast',
                   'HBE+_embryonic_erythrocyte',
                 ]

    invivo.obs['cell_type'] = pd.Categorical(
        invivo.obs['cell_type'],
        categories = order_source,
        ordered = True
    )
    
    # rejoin
    adata = ad.concat([invitro, invivo])
    
    adata.obs['mapping_tissue'] = np.where(
        adata.obs['source'] == 'in vivo', 
        adata.obs['tissue'],
        ''
    )
        
    adata.obs['mapping_label'] = adata.obs['cell_type'].astype(str) #+ '_' + adata.obs['mapping_tissue'].astype(str)
    
    # drop categories with less than 10 observations
    adata = adata[adata.obs['mapping_label'].map(adata.obs['mapping_label'].value_counts()) > 10]

    if gp == 'gene_encoder_cls':
        print('\n\n\n Before downsampling, number of cells per category')
        print(adata.obs['mapping_label'].value_counts())
        print('\n\n\n')
    
    # balance cell numbers
    adata = sample_cells(adata, 'mapping_label', num_cells)
    
    if gp == 'gene_encoder_cls':
        print('\n\n\n Number of cells per category')
        print(adata.obs['mapping_label'].value_counts())
        print(adata.obs['study'].value_counts())
        print('\n\n\n')
    
    sc.pp.neighbors(adata, use_rep='X')
    sc.tl.umap(adata)
    sc.tl.pca(adata, n_comps = 50)
    
    print(f'Computing mapping between centroids for {gp}...')
    
    # convert to PCA instead 
    pdata = sc.AnnData(
        X=adata.obsm['X_pca'],
        obs=adata.obs
    )
    
    # transfer obsm
    pdata.obsm['X_umap'] = adata.obsm['X_umap']
    
    # transfer obsp
    pdata.obsp['distances'] = adata.obsp['distances']
    pdata.obsp['connectivities'] = adata.obsp['connectivities']
    pdata.uns['neighbors'] = adata.uns['neighbors']
    
    point_map_centroid, closest_ref_idx, closest_query_idx, mapping_df, ot_out = compute_centroid_mapping(
        adata=pdata,
        col='source',
        ref='in vivo',
        target='in vitro',
        label_col='mapping_label',
        return_mapping=True,
        cluster_algo=cluster_algo,
        threshold=threshold,
        cluster_col='mapping_label',
        epsilon=epsilon,
        resolution=resolution,
        num_clusters=int(num_clusters),
        tau_a = tau,
        tau_b = tau
    )
    
    mappings_dir = os.path.join(output_dir, 'mappings')
    heatmaps_dir = os.path.join(output_dir, 'heatmaps')
    umaps_dir = os.path.join(output_dir, 'umaps')
    
    os.makedirs(mappings_dir, exist_ok=True)
    os.makedirs(heatmaps_dir, exist_ok=True)
    os.makedirs(umaps_dir, exist_ok=True)
    
    mapping_df.to_csv(os.path.join(mappings_dir, f'{gp}.csv'), index=False)
    
    print(f'Visualizing results for {gp}...')
    
    make_contingency_table(
        mapping_df,
        labels_source=order_source,
        labels_target=order_target,
        by='target',
        fig_size=(8, 8),
        save_to=os.path.join(heatmaps_dir, f'{gp}.pdf')
    )
    
    plot_umap_with_centroids(
        adata=pdata,
        col='source',
        ref='in vivo',
        target='in vitro',
        ref_label='mapping_label',
        ref_label_order=order_source,
        target_label='mapping_label',
        target_label_order=order_target,
        point_map=point_map_centroid,
        closest_ref_idx=closest_ref_idx,
        closest_query_idx=closest_query_idx,
        show_legend=False,
        set_alpha=False,
        fig_size=(14, 12),
        save_path=os.path.join(umaps_dir, f'{gp}.pdf')
    )
    

def main():
    parser = argparse.ArgumentParser(description='Compute unbalanced OT mapping between centroids.')
    parser.add_argument('--epsilon', type=float, required=True, help='Epsilon value for Sinkhorn algorithm')
    parser.add_argument('--cluster_algo', type=str, default='precomputed', help='Clustering algorithm used')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to store output files')
    parser.add_argument('--resolution', type=float, default=1, help='Resolution for leiden clustering')
    parser.add_argument('--threshold', type=float, default=1e-12, help='Threshold for pairings')
    parser.add_argument('--num_clusters', type=float, default=20, help='Number of clusters for knn')
    parser.add_argument('--tau', type=float, default=0.999, help='tau value for Sinkhorn algorithm')
    parser.add_argument('--load_train', type=bool, default=False, help='Load training set')
    parser.add_argument('--num_cells', type=int, default=200, help='Number of cells per category')
    
    args = parser.parse_args()
    
    gpdb = pd.read_csv('tripso_reproducibility/04.4_HSC_fix_hvg/gpdb_tf.csv')
    gpx = ['gene_encoder_cls', 'cell_token'] + gpdb.columns.tolist()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    # print args
    training_args = pd.DataFrame(
        {
            'argument' : ['epsilon', 'cluster_algo', 'resolution', 'threshold', 'num_clusters', 'tau', 'load_train', 'num_cells'],
            'value' : [args.epsilon, args.cluster_algo, args.resolution, args.threshold, args.num_clusters, args.tau, args.load_train, args.num_cells]
        }
    )
    
    training_args.to_csv(os.path.join(args.output_dir, 'hyperparameters.csv'), index = False)
    
    cluster_algo = args.cluster_algo if args.cluster_algo != 'None' else None
    load_train = True if (args.load_train == 'True') | (args.load_train == True) else False
    
    for gp in gpx:
        if os.path.exists(os.path.join(args.output_dir, 'umaps', f'{gp}.pdf')):
            print(f'Skipping {gp}, already computed.')
            continue
        
        print(f'Running GP: {gp}...')
        try:
            compute_unbalanced_ot(gp, 
                                args.epsilon, 
                                cluster_algo, 
                                args.output_dir, 
                                args.resolution, 
                                args.threshold, 
                                args.num_clusters,
                                args.tau,
                                load_train,
                                args.num_cells
                                )
        except Exception as e:
            print(f'Error processing {gp}: {e}')
            continue
    
if __name__ == '__main__':
    main()
