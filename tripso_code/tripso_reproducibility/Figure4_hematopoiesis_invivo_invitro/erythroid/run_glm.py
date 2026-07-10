
import scanpy as sc
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.sparse import issparse
import anndata as ad
from tqdm.notebook import tqdm
import os
import random


def set_seed(seed=42):
    """
    Sets random seed across random, numpy, and PYTHONHASHSEED environment for reproducibility.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    
set_seed(0)



# -----------------------------------------------------------
# Data loading and preprocessing
# -----------------------------------------------------------

def get_path(training_split, run, output_name = 'global'):
    base_path = 'tripso_reproducibility/04.5_HSC_post_qc'
    return f'{base_path}/{run}_by_study/output_{output_name}/ablation/with_gp_ablation/{training_split}_set.h5ad'


train1 = sc.read_h5ad(get_path('train', 'run_1'))
train1 = train1[train1.obs['study'].isin(['Isobe_CD34', 'Isobe_MNC'])]

val1 = sc.read_h5ad(get_path('val', 'run_1'))
val1 = val1[val1.obs['study'].isin(['Isobe_CD34', 'Isobe_MNC'])]

test1 = sc.read_h5ad(get_path('test', 'run_1'))
test1 = test1[test1.obs['study'].isin(['Isobe_CD34', 'Isobe_MNC'])]


pert_data1 = ad.concat([train1, val1, test1])


train2 = sc.read_h5ad(get_path('train', 'run_2'))
train2 = train2[train2.obs['study'].isin(['Isobe_CD34', 'Isobe_MNC'])]

val2 = sc.read_h5ad(get_path('val', 'run_2'))
val2 = val2[val2.obs['study'].isin(['Isobe_CD34', 'Isobe_MNC'])]

test2 = sc.read_h5ad(get_path('test', 'run_2'))
test2 = test2[test2.obs['study'].isin(['Isobe_CD34', 'Isobe_MNC'])]

pert_data2 = ad.concat([train2, val2, test2])


train3 = sc.read_h5ad(get_path('train', 'run_3'))
train3 = train3[train3.obs['study'].isin(['Isobe_CD34', 'Isobe_MNC'])]

val3 = sc.read_h5ad(get_path('val', 'run_3'))
val3 = val3[val3.obs['study'].isin(['Isobe_CD34', 'Isobe_MNC'])]

test3 = sc.read_h5ad(get_path('test', 'run_3'))
test3 = test3[test3.obs['study'].isin(['Isobe_CD34', 'Isobe_MNC'])]

pert_data3 = ad.concat([train3, val3, test3])


pert_data = sc.AnnData(
    X = (pert_data1.X + pert_data2.X + pert_data3.X)/3,
    obs = pert_data1.obs,
    var = pert_data1.var
)


pert_data.obs['tissue'] = pd.Categorical(
    pert_data.obs['tissue'],
    categories = ['YS', 'FL', 'FBM', 'CB', 'PBM', 'ABM_29-50y', 'ABM_+60y'],
    ordered = True
)


ery_ct = [
    # MNC
    # '1_HSC_MPP',
    '2_MEMP', '4_BFU-E/CFU-E', '5_Early_erythroblast', '6_Mid_erythroblast', 
    '7_Late_erythroblast', '8_HBE+_embryonic_erythrocyte',
    
    # CD34
    '1_LT-HSC', '2_ST-HSC', '3_MPP', '4_MEMP', 
    '6_Early_Ery', '7_Late_Ery', 
         ]


pert_data = pert_data[pert_data.obs['cell_type'].isin(ery_ct)]


pd.crosstab(pert_data.obs['cell_type'], pert_data.obs['study'])

# Update CD34 labels to most up to date version from Tomo
pert_data.obs['cell_type'] = pert_data.obs['cell_type'].replace(
    {
        '6_Early_Ery' : '6_BFU-E/CFU-E',
        '7_Late_Ery' : '7_Early_erythroblast',        
    }
)

pert_data.obs['cell_label'] = pert_data.obs['cell_type'].str.replace(r'^\d+_', '', regex=True)

pert_data = pert_data[pert_data.obs['cell_label'] != 'HSC_MPP']

pert_data.obs['cell_label'] = pd.Categorical(
    pert_data.obs['cell_label'],
    categories = ['LT-HSC', 'ST-HSC', 'MPP', 'MEMP',
                  'BFU-E/CFU-E', 
                  'Early_erythroblast', 'Mid_erythroblast',
                  'Late_erythroblast', 'HBE+_embryonic_erythrocyte',
                 ],
    ordered = True
)


# -----------------------------------------------------------
# Helper functions
# -----------------------------------------------------------

def min_max_normalization(df, axis=0):
    """
    Applies min-max normalization along specified axis.

    Parameters
    ----------
    df : pandas.DataFrame or array-like
        The input DataFrame to be normalized.

    axis : int, optional (default: 0)
        The axis along which to normalize. Use 0 to normalize
        each column or 1 to normalize each row using their
        cognate min and max values.

    Returns
    -------
    df_scaled : pandas.DataFrame
        A DataFrame containing the normalized values. Minimum and maximum values
        are calculated along the specified axis. Minimum and maximum values are
        0 and 1, respectively. NaN values are filled with 0.
    """
    if isinstance(df, pd.DataFrame):
        df = df.copy()
    else:
        df = pd.DataFrame(df)
    min_vals = df.min(axis=axis)
    max_vals = df.max(axis=axis)
    df_scaled = df.sub(min_vals, axis=1 - axis).div(max_vals - min_vals, axis=1 - axis).fillna(0)
    return df_scaled


def get_matrix_gene_expression(matrix, var_names, gene, normalize=False):
    """
    Safely extracts expression values for a gene from any matrix type.

    Parameters
    ----------
    matrix : numpy.ndarray
        The matrix containing the expression data. Rows correspond to cells and columns to genes.

    var_names : list or pandas.Index
        The index or array containing the gene names.

    gene : str
        The gene name to extract.

    normalize : bool, optional (default: False)
        If True, apply min-max normalization to the expression values.

    Returns
    -------
    expression : numpy.ndarray
        An array containing the expression values for the specified gene.
    """
    # Find gene index
    if isinstance(var_names, pd.Index):
        gene_idx = var_names.get_loc(gene)
    elif isinstance(var_names, list):
        gene_idx = var_names.index(gene)
    else:
        gene_idx = np.where(var_names == gene)[0][0]

    # Handle different matrix types
    if issparse(matrix):
        expression = matrix[:, gene_idx].toarray().flatten()
    elif isinstance(matrix, np.ndarray):
        expression = matrix[:, gene_idx].flatten()
    elif isinstance(matrix, pd.DataFrame):
        expression = matrix[gene].values
    else:
        raise ValueError(f"Unsupported matrix type: {type(matrix)}")

    expression = expression.astype(np.float64)

    # Apply normalization if requested
    if normalize:
        expression = min_max_normalization(expression).values

    return expression


import statsmodels.api as sm
import statsmodels.formula.api as smf

def fit_glm_model(adata, cell_type_key, cell_type_order=None, continuous_key=None, genes=None,
                  layer=None, use_raw=False, normalize=False, 
                  use_pseudobulk=False, **kwargs):
    """
    Fits Generalized Linear Models (GLMs) to single-cell data for each gene.

    Parameters are similar to `fit_gam_model`, with the GAM-specific arguments removed.
    """

    adata_use = adata

    # Get the expression matrix
    if use_raw and adata_use.raw is not None:
        matrix = adata_use.raw.X
        var_names = adata_use.raw.var_names
    else:
        var_names = adata_use.var_names
        if layer is not None:
            matrix = adata_use.layers[layer]
        else:
            matrix = adata_use.X

    # Filter and order cell types
    preserve_order = False
    if cell_type_order is not None:
        cell_filter = adata_use.obs[cell_type_key].isin(cell_type_order)
        matrix = matrix[cell_filter, :]
        obs_df = adata_use.obs.loc[cell_filter].copy()
        obs_df[cell_type_key] = pd.Categorical(obs_df[cell_type_key], categories=cell_type_order, ordered=True)
        preserve_order = True
    else:
        obs_df = adata_use.obs.copy()

    # Define predictor variable
    if continuous_key is not None:
        obs_df['_predictor'] = obs_df[continuous_key].astype(float)
    elif preserve_order:
        obs_df['_predictor'] = obs_df[cell_type_key].cat.codes.astype(float)
    else:
        obs_df['_predictor'] = pd.Categorical(obs_df[cell_type_key]).codes.astype(float)

    # Prepare gene list
    if genes is None:
        genes = var_names.tolist()
    else:
        genes = [g for g in genes if g in var_names]

    models = {}
    scores = {}

    for gene in tqdm(genes, desc='Fitting GLMs for each gene'):
        try:
            y = get_matrix_gene_expression(matrix, var_names, gene, normalize=normalize)
            df = obs_df.copy()
            df['expression'] = y

            # Fit GLM using ordinary least squares (can adapt to other families if needed)
            glm_model = smf.glm('expression ~ _predictor', data=df, family=sm.families.Gaussian()).fit()

            models[gene] = glm_model
            scores[gene] = {
                'n_samples': int(glm_model.nobs),
                'aic': glm_model.aic,
                'bic': glm_model.bic,
                'deviance': glm_model.deviance,
                'llf': glm_model.llf,
                'p_value': glm_model.pvalues.get('_predictor', np.nan),
                'coef': glm_model.params.get('_predictor', np.nan),
                'r_squared': 1 - glm_model.deviance / glm_model.null_deviance
            }

        except Exception as e:
            print(f"Failed to fit GLM for gene {gene}: {e}")
            continue

    result = {
        'models': models,
        'scores': pd.DataFrame(scores).T
    }

    return result

def analyze_glm_results(glm_results, significance_threshold=0.05, fdr_level=0.05):
    """
    Analyzes GLM model results with FDR correction using statsmodels.

    Parameters
    ----------
    glm_results : dict
        A dictionary containing the results of the GLM analysis. It should
        contain the 'scores' key with a DataFrame of model scores for each gene.

    significance_threshold : float, optional (default: 0.05)
        The p-value threshold to consider a gene significant.

    fdr_level : float, optional (default: 0.05)
        The False Discovery Rate (FDR) level for multiple testing correction.

    Returns
    -------
    results_df : pandas.DataFrame
        A DataFrame containing GLM scores, adjusted p-values, and significance flags.
    """
    results_df = glm_results['scores'].copy()
    results_df['gene'] = results_df.index
    results_df['significant'] = results_df['p_value'] < significance_threshold

    # Filter out rows with NaNs in p_value
    nan_results = results_df[results_df['p_value'].isna()]
    results_df = results_df[~results_df['p_value'].isna()]

    # Apply FDR correction
    _, adj_pvals, _, _ = multipletests(
        results_df['p_value'],
        alpha=fdr_level,
        method='fdr_bh'
    )
    results_df['adj_p_value'] = adj_pvals
    results_df['significant_fdr'] = results_df['adj_p_value'] < fdr_level

    # Append NaN rows back with NaN adjusted p-values
    nan_results['adj_p_value'] = np.nan
    nan_results['significant_fdr'] = False
    results_df = pd.concat([results_df, nan_results], axis=0)

    # Sort by R-squared if available, otherwise deviance
    sort_col = 'r_squared' if 'r_squared' in results_df.columns else 'deviance'
    return results_df.sort_values(sort_col, ascending=False)


# -----------------------------------------------------------
# Run GLM analysis
# -----------------------------------------------------------


glm_res = fit_glm_model(pert_data, 'cell_label', cell_type_order=pert_data.obs['cell_label'].cat.categories)


output = analyze_glm_results(glm_res)

output.to_csv('glm_results.csv', index=False)






