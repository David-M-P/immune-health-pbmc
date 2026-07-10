import tripso
import os
import pandas as pd
import numpy as np
from datasets import load_from_disk
from tripso.Utils.utils import dataset_pivot_longer
import shutil

# Directory paths for loading/saving 
root_dir = 'tripso_reproducibility/04.5_HSC_post_qc/run_1_by_study' 
output_dir = os.path.join(root_dir, "output_global")
target_dir = os.path.join(output_dir, 'DCA_lymphoid/IKZF1_repeat')
gpdb_path = os.path.join(root_dir, '../gpdb_tf.csv')

gpdb = pd.read_csv(gpdb_path)

def write_gene_embeddings(gp):
    gp_genes = gpdb[gp].dropna().tolist()

    # (1) in CD34 atlas
    data_dir = os.path.join(root_dir, '../data/processed/cd34')

    gp_downstream = tripso.gpEval(
        dataset_path=data_dir,
        gpdb_path=gpdb_path,
        output_dir=output_dir,
        tissue='HSC',
        model_type='Global',
    )
    
    cd34_ct = ['14_Large_Pre_B', '13_Cycling_Pro_B',
               '12_Pro_B', '11_PreProB',
               '10_CLP'
              ]
    
    for split in ['train', 'val', 'test']:
        gp_downstream.generate_gene_embeddings(
            gp_for_forward=gp,
            gp_for_downstream=gp,
            split=split,
            obs_key='cell_type',
            obs_value=cd34_ct,
            genes_to_keep=gp_genes,
            precision = '16-mixed',
            do_ensembl_conversion=False,
            return_gene_cosim='gene_to_gp',
            )
    
    # move saved embeddings to target directory
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    emb_file_name = f"{gp}_gene_embeddings_{'_'.join(cd34_ct)}"
        
    shutil.move(os.path.join(output_dir, emb_file_name), 
                os.path.join(target_dir, "cd34_immature_b_large_pre_b"))
    
    # (2) in MNC
    data_dir = os.path.join(root_dir, '../data/processed/mnc')

    gp_downstream = tripso.gpEval(
        dataset_path=data_dir,
        gpdb_path=gpdb_path,
        output_dir=output_dir,
        tissue='HSC',
        model_type='Global',
    )

    mnc_ct = ['11_Pro_B', '12_Cycling_Pro_B', '10_CLP']
    
    for split in ['train', 'val', 'test']:
        gp_downstream.generate_gene_embeddings(
            gp_for_forward=gp,
            gp_for_downstream=gp,
            split=split,
            obs_key='cell_type',
            obs_value=mnc_ct,
            genes_to_keep=gp_genes,
            precision = '16-mixed',
            do_ensembl_conversion=False,
            return_gene_cosim='gene_to_gp',
            )
    
    # move saved embeddings to target directory
    emb_file_name = f"{gp}_gene_embeddings_{'_'.join(mnc_ct)}"
    shutil.move(os.path.join(output_dir, emb_file_name),
                os.path.join(target_dir, "mnc_immature_b_large_pre_b"))


    
for gp in ['GP_IKZF1']: 
    write_gene_embeddings(gp)