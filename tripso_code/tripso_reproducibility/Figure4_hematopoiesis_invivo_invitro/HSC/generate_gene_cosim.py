import tripso
import os
import pandas as pd
import numpy as np
import shutil

# Directory paths for loading/saving 
root_dir = 'tripso_reproducibility/04.4_HSC_fix_hvg/run_1_by_study'
data_dir = 'tripso_reproducibility/04.4_HSC_fix_hvg/data/processed/input_dataset'

output_dir = os.path.join(root_dir, "output_global") 
gpdb_path = os.path.join(root_dir, '../gpdb_tf.csv')


gpdb = pd.read_csv(gpdb_path)

target_dir = 'tripso_reproducibility/04.4_HSC_fix_hvg/run_1_by_study/output_global/gene_embedding_analysis'


for gp in ['PI3K']:
    gp_genes = gpdb[gp].dropna().tolist()
    
    
    # # (1) in CD34 atlas
    # data_dir = os.path.join(root_dir, '../data/processed/cd34')
    
    # gp_downstream = tripso.gpEval(
    #     dataset_path=data_dir,
    #     gpdb_path=gpdb_path,
    #     output_dir=output_dir,
    #     tissue='HSC',
    #     model_type='Global',
    # )
    
    
    # cd34_ct = ['1_LT-HSC', '2_ST-HSC', '3_MPP', '4_MEMP', '9_LMPP', '10_CLP', '21_GMP']

    # for t in ['train', 'val', 'test']:
    #     gp_downstream.generate_gene_embeddings(
    #         pathway = gp,
    #         split=t,
    #         genes_to_keep=gp_genes,
    #         obs_key='cell_type',
    #         obs_value = cd34_ct,
    #         precision = '16-mixed',
    #         return_gene_cosim='gene_to_gp',
    #         do_ensembl_conversion=False,
    #     )
    
    # # move saved embeddings to target directory
    # if not os.path.exists(target_dir):
    #     os.makedirs(target_dir)
        
    # emb_file_name = f"{gp}_gene_embeddings_{'_'.join(cd34_ct)}"
        
    # shutil.move(os.path.join(output_dir, emb_file_name), 
    #             os.path.join(target_dir, f'{gp}_cd34_hspc'))
    
    
    # (2) in vitro
    data_dir = os.path.join(root_dir, '../data/processed/sakurai_hsc')
    
    gp_downstream = tripso.gpEval(
        dataset_path=data_dir,
        gpdb_path=gpdb_path,
        output_dir=output_dir,
        tissue='HSC',
        model_type='Global',
    )
    
    sakurai_ct = ['HSPC-HLF',  'HSPC-Cycling', 'HSPC']

    for t in ['train', 'val', 'test']:
        gp_downstream.generate_gene_embeddings(
            pathway = gp,
            split=t,
            obs_key='cell_type',
            obs_value=sakurai_ct,
            genes_to_keep=gp_genes,
            precision = '16-mixed',
            do_ensembl_conversion=False,
            return_gene_cosim='gene_to_gp',
            )

    emb_file_name = f"{gp}_gene_embeddings_{'_'.join(sakurai_ct)}"

    shutil.move(os.path.join(output_dir, emb_file_name),
                os.path.join(target_dir, f'{gp}_sakurai_hspc'))
