
import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42


models = [
    'expimap', 
    'GPformer',
    'spectra',
]


def find_and_load_csvs(directories):
    dataframes = []
    
    # Loop through the list of directories
    eval_types = ['knn', 'logistic_regression']
    runs = ['run_1', 'run_2', 'run_3']
    
    tasks = ['', '_top_1', '_top_2', '_top_3', '_top_4', '_top_5', ] # '_top_10'
    
    for m in models:
        for r in runs:
            for e in eval_types:
                for t in tasks:
                    directory = os.path.join(m, r, f'{e}{t}')

                    # now load csv files:
                    for filename in os.listdir(directory):

                        # Check if the file has a .csv extension
                        if filename.endswith('.csv'):
                            file_path = os.path.join(directory, filename)

                            label_type, gp = filename.replace('.csv', '').replace('from_', '').split('id_')

                            df = pd.read_csv(file_path)
                            df['model'] = m
                            df['eval_type'] = e
                            df['task'] = 'target_pathway' if t == '' else f'gene{t}'
                            df['label_type'] = label_type
                            df['gp'] = gp
                            df['run'] = r

                            dataframes.append(df)
        
    dataframes = pd.concat(dataframes)
    
    return dataframes



results = find_and_load_csvs(models)

results['model'] = results['model'].replace(
    {'expimap': 'Expimap', 'spectra': 'Spectra', 'GPformer': 'Tripso'}
)

results['model'] = pd.Categorical(
    results['model'],
    categories=['Tripso', 'Expimap', 'Spectra'],
    ordered=True
)

# Logistic regression

df = results 
df = df[df['eval_type'] == 'logistic_regression']


# Filter for metric == 'f1-score', group and aggregate
filtered = df[df['metric'] == 'f1-score']

grouped = filtered.groupby([
    'output_class', 'model', 'eval_type', 'task', 'label_type', 'gp'
])['value'].agg(['mean', 'std']).reset_index()


# drop if null
grouped = grouped.dropna(subset=['mean', 'std'])

# round to 3 decimals
grouped['mean'] = grouped['mean'].round(3)
grouped['std'] = grouped['std'].round(3)

# Keep metrics we care about

tnfa = grouped[(grouped['output_class'] == 'TNFa') & 
               (grouped['gp'] == 'TNFa')
               ]

tgfb = grouped[(grouped['output_class'] == 'TGFb') &
                (grouped['gp'] == 'TGFb')
                ]


genes_tnfa = grouped[(grouped['task'] != 'target_pathway') &
                     (grouped['output_class'] == 'NT') &
                     (grouped['gp'] == 'TNFa')
                     ]

genes_tgfb = grouped[(grouped['task'] != 'target_pathway') &
                        (grouped['output_class'] == 'NT') &
                        (grouped['gp'] == 'TGFb')
                        ]

df = pd.concat([tnfa, tgfb, genes_tnfa, genes_tgfb])
df = df[['task', 'output_class', 'model', 'gp', 'mean', 'std']]
df = df.sort_values(
    ['task', 'output_class', 'model', 'gp',]
)

df.to_csv('perturbseq_suppl_table_tripso.csv', index=False)
