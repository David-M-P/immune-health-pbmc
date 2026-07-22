# PBMC reference data-structure audit

## Evidence boundary

**Directly observed in this audit:** file discovery; merge and split manifests; QC tables; HDF5/AnnData keys, shapes and dtypes; observation metadata; gene identifiers; sampled sparse values; donor/sample/fine-label counts; missingness; and SoundLife/Galsky path presence. All H5AD files were opened read-only with `h5py.File(..., mode='r')`; count matrices were not densified.

**Provenance-document only:** upstream raw-count recovery order, CellTypist execution details, health-filter intent, historical source paths, and the claim that merge-time `.X` was populated from `layers['counts']`. Source: `/faststorage/project/CancerEvolution_shared/Projects/David/phd/scih/docs/raw_to_lineage_split_and_merge.md`. The current files directly establish CSR matrices whose stored `float64` values are integer-like in the audit sample, but historical transformations cannot be reconstructed from HDF5 alone.

## Direct observations

- Exact merged dataset labels: aidav2, immuneindonesia, immunobiologyaging, onek1k, terekhova.
- Eight reference lineage H5ADs contain 8,297,506 disjoint audited cell rows; the five primary lineages contain 7,315,605 cells.
- Every merged object has 18,035 variables, a CSR `.X`, no layers, no `.raw`, and the expected 13 observation columns.
- All `.X/data` arrays are stored as `float64`, not an integer dtype. Across 2,400,000 evenly sampled stored values, 0 were negative and 0 were non-integer-like. Thus the sample supports count semantics, but this is not a full scan of every stored value.
- Raw `donor_id` collisions across datasets: 0. The pipeline nevertheless uses `dataset::donor_id` as the biological identifier.
- Duplicate cell identifiers: 0 after a 64-bit hash screen and exact follow-up when needed.
- Donor biological units with repeated samples: 102.
- Missing age cells across audited objects: 0; missing/unknown sex cells: 0.
- Fine labels observed: 21 exact strings. No labels were merged during the audit.
- SoundLife/Galsky-named artifacts found: 194; neither name occurs in the five reference dataset labels.

## Matrix summary

| lineage | n_cells | n_biological_units | n_donor_observations | n_genes | x_encoding | x_data_dtype | sparsity | sampled_min | sampled_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B cells | 780051 | 2180 | 2331 | 18035 | csr_matrix | float64 | 0.903332 | 1 | 224 |
| NK_ILC | 1088930 | 2179 | 2330 | 18035 | csr_matrix | float64 | 0.900959 | 1 | 118 |
| Monocytes | 1046535 | 2176 | 2327 | 18035 | csr_matrix | float64 | 0.884565 | 1 | 541 |
| CD4_like | 3147275 | 2180 | 2331 | 18035 | csr_matrix | float64 | 0.911149 | 1 | 148 |
| CD8_like | 1252814 | 2180 | 2331 | 18035 | csr_matrix | float64 | 0.902471 | 1 | 134 |
| T_others | 918729 | 2180 | 2331 | 18035 | csr_matrix | float64 | 0.905715 | 1 | 110 |
| DC | 33149 | 1995 | 2134 | 18035 | csr_matrix | float64 | 0.836625 | 1 | 858 |
| pDC | 30023 | 1651 | 1800 | 18035 | csr_matrix | float64 | 0.860993 | 1 | 367 |

## Dataset support

| dataset | n_cells | n_biological_units | n_samples | n_donor_observations | age_min | age_max | female_donors | male_donors | unknown_sex_donors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aidav2 | 1219892 | 625 | 625 | 625 | 19 | 77 | 350 | 275 | 0 |
| immuneindonesia | 369378 | 174 | 174 | 174 | 18 | 60 | 78 | 96 | 0 |
| immunobiologyaging | 3626985 | 234 | 234 | 234 | 40 | 89 | 137 | 97 | 0 |
| onek1k | 1205586 | 981 | 75 | 981 | 19 | 97 | 565 | 416 | 0 |
| terekhova | 1875665 | 166 | 317 | 317 | 25 | 81 | 36 | 130 | 0 |

## Integrity and interpretation notes

- Donor sex inconsistencies: 0; donors with multiple recorded ages: 102; source sample/pool IDs shared by multiple donors: 75.
- **Resolved identifier collision:** the 75 shared source IDs all occur in `onek1k`, where `sample_id` is a pool number shared by 9–14 donors. The user-approved contract therefore retains `source_observation_id = dataset::sample_id` for source provenance and uses the collision-safe `observation_id = dataset::donor_id::sample_id`. Biological independence remains `biological_unit_id = dataset::donor_id`.
- Multiple ages for one donor are reported as repeated longitudinal values, not automatically called errors, because repeated visits can legitimately change age.
- `gene_identifier_summary.tsv` separates source-partition zero expression from dataset-level zero expression. The shared vocabulary is an identifier-presence intersection, so all 18,035 shared IDs are present in every contributing source; zero expression is a separate property.
- The fine-type ontology remains unapproved. Exact `ctype_low` labels and confidence distributions are provided for review before any grouping.
- QC thresholds, GP coverage thresholds, HVGs, dataset roles, and biological fine-label merges are not inferred by this report.

## Artifact inventory

- Lineage H5AD files under `lineages/`: 193
- Merged reference H5AD files: 8
- Split manifests: 20
- Merge manifests: 8
- QC/report artifacts: 299

Complete path inventories and input file metadata are in `audit_manifest.json`.
