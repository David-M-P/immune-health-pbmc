# Data provenance: from source H5ADs to merged lineage datasets

## Scope and terminology

The starting point was five previously assembled single-cell RNA-seq AnnData/H5AD cohorts:

1. AIDA v2
2. ImmuneIndonesia
3. ImmunobiologyAging
4. OneK1K
5. Terekhova healthy ageing

Here, **raw dataset** means the cohort-level H5AD supplied to this project. It does not mean FASTQ files, and this pipeline did not run Cell Ranger, read alignment, barcode calling, or the original cohort-specific cell QC.

The relevant processing sequence was:

```text
Source cohort H5AD
-> recover and preserve raw counts
-> harmonize genes and metadata
-> reannotate cells using CellTypist
-> split each cohort into lineages
-> merge the same lineage across the five cohorts
-> unbalanced merged lineage H5AD
```

The scope described here stops at the unbalanced merged lineage files. No maximum-cells-per-donor cap, donor balancing, sampling weights, highly variable gene selection, gene-program restriction, Tripso processing, or model training is included.

## 1. Common preparation applied to every cohort

Each source H5AD was processed independently using a shared preparation pipeline.

### Raw-count recovery

The pipeline searched for integer raw counts in this order:

1. An existing `layers["counts"]`
2. `.X`, if it contained non-negative integer-like values
3. `raw.X`, if that contained integer-like counts and could be aligned to the current genes

The recovered counts were stored in `layers["counts"]`.

The original `.X` was preserved. Consequently, the standardized cohort files did not initially have an identical `.X` interpretation:

- AIDA v2, ImmuneIndonesia and ImmunobiologyAging retained their existing transformed/log-scale `.X`, while raw counts were recovered from `raw.X`.
- OneK1K and Terekhova already had raw integer counts in `.X`; these were copied to `layers["counts"]`.

This difference was removed later during the cross-dataset merge, where raw counts were explicitly moved into `.X`.

### Gene harmonization

For every feature, the pipeline created:

- `orig_var_names`: the original feature identifier
- `unified_ensembl`: a standardized Ensembl gene ID
- `unified_gname`: a standardized gene symbol

Ensembl version suffixes were removed. Gene symbols, previous symbols and aliases were mapped using a project-local HGNC table, with a GTF file as fallback.

If multiple original features mapped to the same `unified_ensembl`, they were not summed. The pipeline retained one representative: the feature with the highest mean raw-count expression, with deterministic tie-breaking. The dropped features were recorded in a `collapsed_genes` audit table.

Genes without a usable Ensembl identifier could remain in the standardized cohort file, but they were excluded during the final merge because the merge key was `unified_ensembl`.

### Metadata harmonization

The following common cell-level fields were created:

- `dataset`
- `donor_id`
- `sample_id`
- `age`
- `sex`
- `chemistry`

Strings were stripped and common missing-value strings were converted to missing values. Age was converted to a number; when age was embedded in text, the first numeric value was extracted. Sex was normalized to `female`, `male`, or `unknown`.

The original values were generally preserved in corresponding `raw_*` fields.

The actual source-to-standard-field mappings were:

| Dataset | `donor_id` | `sample_id` | `age` | `sex` | `chemistry` |
|---|---|---|---|---|---|
| AIDA v2 | `donor_id` | `sample_uuid` | `development_stage` | `sex` | `assay` |
| ImmuneIndonesia | `donor_id` | `sample_id` | `development_stage` | `sex` | `batch` |
| ImmunobiologyAging | `subject.subjectGuid` | `sample.sampleKitGuid` | `sample.subjectAgeAtDraw` | `subject.biologicalSex` | `batch_id` |
| OneK1K | `donor_id` | `pool_number` | `age` | `sex` | `assay` |
| Terekhova | `Donor_id` | `Tube_id` | `Age` | `Sex` | `Batch` |

The field called `chemistry` is therefore sometimes a sequencing assay and sometimes a batch-like source field; it should be interpreted as a harmonized technical provenance field rather than a perfectly uniform laboratory chemistry variable.

### Cell-type reannotation with CellTypist

All cells were reannotated rather than relying solely on the annotations supplied by the original studies.

Two CellTypist models were used:

- `Immune_All_High.pkl` for broad immune-cell classes
- `Immune_All_Low.pkl` for detailed immune-cell classes

CellTypist was run with majority voting and seed 0.

For annotation only, raw counts were placed in a temporary matrix, normalized to 10,000 counts per cell, and `log1p` transformed. This temporary normalization did not overwrite the stored raw counts or the original `.X`.

Four standard annotation fields were added:

- `ctype_high`
- `ctype_high_conf`
- `ctype_low`
- `ctype_low_conf`

No cells were removed during this general preparation stage based on mitochondrial percentage, UMI count, detected genes, doublet score, or cells per donor. Those fields could be retained as metadata, but they were not used as filters here.

## 2. Dataset-specific inputs and preparation outcomes

### AIDA v2

Source H5AD:

```text
AIDAv2/AIDAv2.h5ad
```

Standardized result:

- 1,265,624 cells
- 625 donors
- 36,406 genes
- Raw counts recovered from `raw.X`
- No duplicated genes removed during gene harmonization
- All standardized genes had a usable `unified_ensembl`

No cohort-specific health filter was applied.

### ImmuneIndonesia

Source H5AD:

```text
immuneIndonesia/immuneIndonesia.h5ad
```

Standardized result:

- 462,034 cells
- 199 donors
- 36,030 genes
- Raw counts recovered from `raw.X`
- No duplicated genes removed during gene harmonization
- All genes had a usable `unified_ensembl`, although some lacked a standardized symbol

ImmuneIndonesia was the only reference dataset with an explicit health filter. This filter was applied after general standardization but before lineage assignment.

A cell was retained only if all of the following were true:

- `reported_diseases == "none"`
- `Pregnant_Breastfeeding == "no"`
- `Chronic_Diseases_Medicine == "no"`
- `Fever == "no"`
- `Malaria` was either `"neg (-)"` or the literal `"na"`

Missing or unexpected values failed the corresponding condition.

This removed 41,937 cells, leaving:

- 420,097 cells
- 174 donors

The failure counts for individual health variables overlap and must not be added together.

No filter was applied using dengue status, vaccination status, autoimmune status or the provided doublet annotation.

### ImmunobiologyAging

ImmunobiologyAging was supplied as seven separate H5AD partitions:

```text
ImmunoAging_b-plasma_cells.h5ad
ImmunoAging_cd4-memory-treg_cells.h5ad
ImmunoAging_cd4-naive_cells.h5ad
ImmunoAging_cd8-gdt-mait-dnt_cells.h5ad
ImmunoAging_dc-monocyte_cells.h5ad
ImmunoAging_nk-ilc_cells.h5ad
ImmunoAging_other_cells.h5ad
```

Together these contained:

- 3,758,514 cells
- 234 donors
- 18,082 original features per partition
- 18,080 standardized features per partition after removing two duplicate Ensembl mappings
- Raw counts recovered from `raw.X`

Each partition was standardized and annotated independently. The supplied partition name was not treated as a definitive lineage label. Every cell was reannotated with CellTypist.

Therefore, small numbers of cells could move to an unexpected lineage. For example, a small number of confident B cells were identified in the CD8/gamma-delta/MAIT/DNT partition. Such cells were retained according to their CellTypist result.

During the later merge, every ImmunobiologyAging partition containing the requested lineage was included. The seven partitions were not first collapsed into one single ImmunobiologyAging H5AD.

No additional health filter was applied.

### OneK1K

Source H5AD:

```text
OneK1K/processedData/OneK1K.h5ad
```

Standardized result:

- 1,248,980 cells
- 981 donors
- 36,571 genes
- `.X` already contained integer counts and was copied to `layers["counts"]`
- No duplicated genes removed
- All genes had a usable `unified_ensembl`

No additional health filter was applied.

### Terekhova healthy ageing

Source H5AD:

```text
HealthyAging/processedData/david/pbmc_rawcounts_plusmeta.h5ad
```

Standardized result:

- 1,916,367 cells
- 166 donors
- 36,601 original features
- 125 duplicate Ensembl mappings removed
- 36,476 standardized features
- 36,464 features with a usable `unified_ensembl`
- `.X` already contained raw integer counts and was copied to `layers["counts"]`

No additional health filter was applied.

## 3. Lineage splitting

Each standardized dataset, or each standardized ImmunobiologyAging partition, was split independently.

### Broad-lineage confidence rule

The actual split jobs used a minimum broad CellTypist confidence of 0.9.

A cell was assigned to `Ambiguous` if:

- `ctype_high` was missing,
- `ctype_high_conf` was missing, or
- `ctype_high_conf < 0.9`.

Confident non-T cells retained their broad `ctype_high` label.

### T-cell subdivision

Cells with a confident broad label of `T cells` were divided using `ctype_low` and `ctype_low_conf`.

Detailed helper, regulatory, follicular-helper and related CD4 labels with confidence at least 0.7 were assigned to:

```text
CD4_like
```

Detailed cytotoxic T-cell labels with confidence at least 0.7 were assigned to:

```text
CD8_like
```

The following went to `T_others`:

- MAIT cells
- Gamma-delta T cells
- NKT cells
- Other explicitly forced-special T labels
- Fine labels with confidence below 0.7
- Missing or unmapped fine T-cell labels

Thus, a low-confidence fine T-cell prediction did not make the cell broadly ambiguous. If its broad T-cell prediction had confidence at least 0.9, it remained a confident T cell and was assigned to `T_others`.

### Dataset-level lineage results

| Dataset | Cells before split | Health-filtered | Ambiguous | Confident |
|---|---:|---:|---:|---:|
| AIDA v2 | 1,265,624 | 0 | 33,256 | 1,232,368 |
| ImmuneIndonesia | 462,034 | 41,937 | 47,204 | 372,893 |
| ImmunobiologyAging | 3,758,514 | 0 | 125,523 | 3,632,991 |
| OneK1K | 1,248,980 | 0 | 38,807 | 1,210,173 |
| Terekhova | 1,916,367 | 0 | 39,913 | 1,876,454 |

The confident total includes some lineages that were not selected for the final reference, such as plasma cells, HSC/MPP and megakaryocyte-related cells.

Each lineage file retained:

- Raw counts in `layers["counts"]`
- The source `.X`
- Standardized donor/sample/age/sex/technical metadata
- CellTypist broad and fine annotations
- `lineage`
- `batch`, initially set to the dataset name
- Harmonized gene identifiers

## 4. Lineages selected for the reference merge

The following lineages were requested:

```text
B cells
CD4_like
CD8_like
T_others
NK_ILC
Monocytes
DC
pDC
```

CellTypist's broad label `ILC` was renamed to the canonical merged label `NK_ILC`.

The following were not included in the reference merge:

- `Ambiguous`
- Plasma cells
- HSC/MPP
- Megakaryocytes/platelets
- Macrophages
- Mast cells
- Other minor lineages

B cells, CD4_like, CD8_like, T_others, NK_ILC and Monocytes were considered the main reference lineages. DC and pDC were retained as QC-only lineages.

SoundLife and Galsky lineage files existed elsewhere in the project, but they were not included in this five-dataset reference merge.

## 5. Cross-dataset lineage merge

For each requested lineage, the pipeline searched all lineage split manifests and collected every matching source file from:

```text
aidav2
immuneindonesia
immunobiologyaging
onek1k
terekhova
```

For ImmunobiologyAging, this could mean collecting the lineage from multiple original partitions.

### Common gene space

For a lineage to be merged, a gene had to have a nonmissing `unified_ensembl` identifier in every contributing source file.

The intersection was based on gene-identifier presence, not whether the gene was expressed. Therefore, genes with zero counts in one or more datasets were retained and documented in separate QC reports.

All eight merged lineages ended with the same intersection of:

```text
18,035 Ensembl genes
```

### Expression matrix used in the merge

Before concatenating cells, each source lineage was:

1. Subset to the 18,035 common Ensembl genes
2. Reordered into the same gene order
3. Converted so that `.X` contained the raw counts from `layers["counts"]`
4. Stripped of its layers, `raw`, embeddings, graphs and dataset-specific `uns`

Consequently, the final merged lineage H5ADs have:

- Raw integer counts in `.X`
- No separate `layers["counts"]`
- No log-normalized expression matrix
- No batch correction
- No donor downsampling
- No maximum-cells-per-donor restriction

### Common metadata in the merged files

The merged files retained:

- `donor_id`
- `age`
- `sex`
- `ctype_high`
- `ctype_high_conf`
- `ctype_low`
- `ctype_low_conf`
- `dataset`
- `sample_id`
- `chemistry`
- `lineage`
- `batch`
- `pct_mt`

`batch` was set to the broad dataset name.

Mitochondrial percentage was harmonized from:

- AIDA v2: `pMito`
- ImmunobiologyAging: `pct_counts_mito`
- ImmuneIndonesia, OneK1K and Terekhova: `percent.mt`

Cell identifiers were prefixed with their source directory, for example:

```text
aidav2_std__<original_cell_id>
ImmunoAging_cd4-naive_cells_standardized__<original_cell_id>
```

This prevented barcode collisions between cohorts and between ImmunobiologyAging partitions.

Donor IDs were not prefixed. Instead, the merge explicitly checked that the same donor ID did not occur in more than one broad dataset.

The actual merged `var` currently retains only `unified_ensembl`. Source-specific gene symbols and original feature names differed across datasets and were dropped during AnnData concatenation.

## 6. Final unbalanced lineage datasets

Every file below contains all five reference datasets and 18,035 common genes:

| Lineage | Cells | Unique donors |
|---|---:|---:|
| B cells | 780,051 | 2,180 |
| CD4_like | 3,147,275 | 2,180 |
| CD8_like | 1,252,814 | 2,180 |
| T_others | 918,729 | 2,180 |
| NK_ILC | 1,088,930 | 2,179 |
| Monocytes | 1,046,535 | 2,176 |
| DC | 33,149 | 1,995 |
| pDC | 30,023 | 1,651 |

These are the starting unbalanced reference datasets:

```text
intermediate_data/reference_lineages/merged/<lineage>/merged.h5ad
```

Each lineage directory also contains:

- `cell_manifest.tsv.gz`
- `merge_manifest.json`
- `merge_qc.tsv`
- Gene zero-expression reports

In summary, **standardized** at this stage means that cell metadata, cell-type annotations and gene identifiers were harmonized across cohorts. It does not mean that expression was normalized across datasets or batch corrected. The final merged matrices contain raw counts on a shared Ensembl-gene space, with all qualifying cells retained regardless of donor size.

## Reproducibility notes

- The archived raw-input paths used during the April 2026 runs are no longer present at those original paths. The preparation logs and standardized H5AD provenance are therefore the available record of those inputs.
- ImmuneIndonesia, OneK1K and ImmunobiologyAging were annotated before an April 21 optimization of the CellTypist preparation implementation. Their stored annotations are valid inputs to the lineage split, but rerunning the current code would not reproduce the original annotation mechanics bit-for-bit.
- The preparation QC files used fixed filenames such as `dataset_summary.tsv` and were written into the same `intermediate_data` directory. Later preparation jobs overwrote earlier versions. The archived preparation logs and split/merge manifests are the reliable dataset-specific provenance records.
