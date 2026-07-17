# AGENTS.md — PBMC Immune Health Project North Star

## Project purpose

This repository supports a biological analysis project whose main goal is to derive an **interpretable, transferable immune-health metric from PBMC single-cell RNA-seq data**.

The project should learn healthy immune patterns from several healthy cohorts, validate longitudinal stability in SoundLife, and test transfer to immune-challenged cohorts such as cancer, CMV-related, COVID, or other inflammatory settings.

The central scientific questions are biological:

1. Which cell-type-specific gene programs change reproducibly with healthy ageing?
2. Which program relationships are stable across cohorts and technical settings?
3. How far does an immune-challenged individual deviate from the expected healthy state for their age and sex?
4. Do pathway balances, biological-age residuals, or healthy-manifold deviations associate with clinically meaningful outcomes?
5. Are the resulting scores stable within healthy individuals over time?

The repository is an **analysis repository**, not a general-purpose software package. Favor clarity, reproducibility, modularity, and easy rerunning over abstraction for its own sake.

---

# Core design principles

## 1. Biological interpretation comes first

Every major model output should be traceable to:

- a cell lineage or cell type;
- an interpretable pathway or gene program;
- a donor-level summary;
- a healthy reference expectation;
- an uncertainty estimate;
- a clear biological hypothesis.

Avoid black-box outputs that cannot be connected back to genes, pathways, lineages, or donor-level effects.

## 2. Donors are the biological replicates

Cells increase measurement precision; they do not increase the number of independent human observations.

For age, sex, disease, outcome, stability, and immune-health analyses:

- split train/test by donor;
- aggregate to donor × cell type or donor × lineage;
- prevent donors with many cells from dominating losses;
- report donor counts, not only cell counts;
- use donor-aware bootstrapping and cross-validation.

Never use random cell-level train/test splits for donor-level biological claims.

## 3. Annotation is a replaceable module

The first implementation may use CellTypist because it is fast and adequate for building the end-to-end pipeline.

Later implementations may use:

- CellTypist with majority voting;
- marker-reviewed CellTypist;
- custom CellTypist models;
- scVI/scANVI reference mapping;
- a Jiménez-García-style recursive annotation workflow.

Downstream modules must depend only on a standard annotation contract, not on the internal details of the annotation method.

Changing the annotation version should require changing a configuration value and rerunning downstream stages, not rewriting analysis code.

## 4. Healthy reference learning and disease transfer are separate

The main healthy reference is learned from the five healthy training datasets.

SoundLife is reserved primarily for longitudinal robustness and stability validation.

Disease or immune-challenged cohorts are query datasets. They should not define the healthy reference unless an analysis is explicitly labeled exploratory.

## 5. Agreement across datasets matters more than pooled significance

A signal is especially valuable when it:

- has the same direction across healthy cohorts;
- has similar age-dependent behavior across cohorts;
- survives leave-one-dataset-out training;
- survives age/sex reweighting;
- survives cell-count downsampling;
- remains interpretable at pathway and gene level;
- transfers to SoundLife and external cohorts.

A large pooled model is not automatically trusted.

## 6. Configuration, not hard-coding

Never hard-code:

- absolute data paths;
- HPC project/account names;
- partitions;
- queue names;
- scratch paths;
- output roots;
- conda environments;
- memory, CPU, or GPU requests;
- cluster-specific module commands.

All environment-specific values belong in cluster configuration files.

---

# Non-goals

This repository does not need:

- a public Python package;
- a stable API for external users;
- exhaustive unit-test coverage;
- elaborate class hierarchies;
- a web interface;
- containerization unless it solves a real deployment problem;
- a workflow abstraction more complex than the analysis requires.

Do not spend days engineering a generic framework when a readable function, script, or configuration file is sufficient.

The standard is:

> A future version of David should be able to understand, rerun, and modify the analysis without reverse-engineering hidden assumptions.

---

# Suggested repository structure

```text
immune-health-pbmc/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── environment.yml
├── configs/
│   ├── common.yaml
│   ├── clusters/
│   │   ├── gefion.yaml
│   │   └── genomedk.yaml
│   ├── datasets/
│   │   ├── healthy_dataset_1.yaml
│   │   ├── healthy_dataset_2.yaml
│   │   ├── healthy_dataset_3.yaml
│   │   ├── healthy_dataset_4.yaml
│   │   ├── healthy_dataset_5.yaml
│   │   ├── soundlife.yaml
│   │   └── disease_cohorts.yaml
│   ├── analyses/
│   │   ├── development.yaml
│   │   ├── full_healthy_reference.yaml
│   │   ├── lodo.yaml
│   │   ├── soundlife_validation.yaml
│   │   └── disease_transfer.yaml
│   └── annotation/
│       ├── celltypist_v1.yaml
│       ├── hybrid_v2.yaml
│       └── recursive_v3.yaml
├── profiles/
│   ├── gefion/
│   │   ├── submit.sh
│   │   └── slurm_defaults.yaml
│   └── genomedk/
│       ├── submit.sh
│       └── slurm_defaults.yaml
├── scripts/
│   ├── 00_validate_config.py
│   ├── 01_prepare_dataset.py
│   ├── 02_run_qc.py
│   ├── 03_annotate_cells.py
│   ├── 04_build_lineage_objects.py
│   ├── 05_build_donor_summaries.py
│   ├── 06_train_program_models.py
│   ├── 07_fit_healthy_trajectories.py
│   ├── 08_compute_immune_health.py
│   ├── 09_validate_soundlife.py
│   ├── 10_transfer_disease.py
│   └── 11_build_reports.py
├── src/
│   └── immune_health/
│       ├── config.py
│       ├── io.py
│       ├── qc.py
│       ├── annotation.py
│       ├── aggregation.py
│       ├── balancing.py
│       ├── programs.py
│       ├── trajectories.py
│       ├── metrics.py
│       ├── validation.py
│       └── plotting.py
├── workflows/
│   ├── run_stage.sh
│   ├── submit_stage.sh
│   ├── run_pipeline.sh
│   └── submit_pipeline.sh
├── annotations/
│   ├── lineage_v1_celltypist.tsv
│   ├── lineage_v2_hybrid.tsv
│   └── lineage_v3_recursive.tsv
├── metadata/
│   ├── dataset_manifest.tsv
│   ├── donor_manifest.tsv
│   ├── gene_mapping.tsv
│   └── annotation_ontology.yaml
├── notebooks/
│   ├── exploration/
│   ├── annotation_review/
│   ├── healthy_reference/
│   ├── soundlife/
│   └── disease_transfer/
├── tests/
│   ├── test_config_loading.py
│   ├── test_metadata_contract.py
│   └── test_small_end_to_end.py
└── runs/
    └── <run_id>/
        ├── resolved_config.yaml
        ├── manifests/
        ├── logs/
        ├── models/
        ├── intermediate/
        ├── results/
        ├── figures/
        └── reports/
```

This structure is a guide, not a rigid requirement. Keep it simpler where possible.

---

# Configuration model

Use three layers of configuration.

## 1. Shared scientific configuration

`configs/common.yaml`

```yaml
project:
  name: immune_health_pbmc
  random_seed: 42

annotation:
  version: lineage_v1_celltypist
  column: analysis_lineage
  confidence_column: annotation_confidence
  allow_unknown: true

lineages:
  - T_and_NK
  - B
  - Plasma
  - Mono_and_DC
  - pDC

aggregation:
  biological_unit: donor
  min_cells_per_donor_lineage: 30
  cell_cap_per_donor_lineage: 5000

healthy_reference:
  age_model: spline
  condition_on_sex: true
  use_dataset_balancing: true

validation:
  soundlife_held_out: true
  disease_cohorts_held_out: true
```

## 2. Cluster-specific infrastructure configuration

`configs/clusters/gefion.yaml`

```yaml
cluster:
  name: gefion
  scheduler: slurm
  project_account: cu_0071
  partition_gpu: <gefion_gpu_partition>
  partition_cpu: <gefion_cpu_partition>

paths:
  data_root: /path/on/gefion/data
  project_root: /path/on/gefion/project
  scratch_root: /path/on/gefion/scratch
  output_root: /path/on/gefion/results

environment:
  activation_command: source /path/to/conda.sh
  environment_name: immune-health-pbmc

slurm:
  default_time: "24:00:00"
  default_memory_gb: 64
  default_cpus: 8
  default_gpus: 1
```

`configs/clusters/genomedk.yaml`

```yaml
cluster:
  name: genomedk
  scheduler: slurm
  project_account: <genomedk_project>
  partition_gpu: <genomedk_gpu_partition>
  partition_cpu: <genomedk_cpu_partition>

paths:
  data_root: /path/on/genomedk/data
  project_root: /path/on/genomedk/project
  scratch_root: /path/on/genomedk/scratch
  output_root: /path/on/genomedk/results

environment:
  activation_command: source /path/to/conda.sh
  environment_name: immune-health-pbmc

slurm:
  default_time: "24:00:00"
  default_memory_gb: 64
  default_cpus: 8
  default_gpus: 1
```

## 3. Analysis/run configuration

Example: `configs/analyses/development.yaml`

```yaml
run:
  name: dev_celltypist_tripso
  mode: development

datasets:
  healthy:
    - healthy_dataset_1
    - healthy_dataset_2
  validation: []
  query: []

sampling:
  max_donors_per_dataset: 5
  max_cells_per_donor: 2000

models:
  tripso:
    enabled: true
    epochs: 5
    seeds: [1]

reports:
  generate_full_report: false
```

At runtime, merge:

```text
common.yaml
+ cluster config
+ analysis config
+ optional command-line overrides
```

Always write the fully resolved configuration to:

```text
runs/<run_id>/resolved_config.yaml
```

---

# Cluster selection and execution

The same scientific command should work on either HPC.

```bash
bash workflows/submit_pipeline.sh   --cluster gefion   --analysis full_healthy_reference
```

or:

```bash
bash workflows/submit_pipeline.sh   --cluster genomedk   --analysis full_healthy_reference
```

The wrapper should resolve:

- data paths;
- Slurm account;
- partition;
- memory;
- CPU and GPU requests;
- environment activation;
- output location.

Scientific scripts must not contain `if cluster == "gefion"` logic.

---

# Minimal command-line contract

Every major script should accept:

```bash
python scripts/<stage>.py   --common-config configs/common.yaml   --cluster-config configs/clusters/gefion.yaml   --analysis-config configs/analyses/full_healthy_reference.yaml   --run-id full_healthy_reference_v1
```

Optional:

```bash
--dataset healthy_dataset_1
--lineage T_and_NK
--seed 3
--fold dataset_2
--overwrite
```

Prefer explicit arguments over magical behavior.

---

# Run identity and provenance

Every run must have a stable `run_id`.

Suggested format:

```text
YYYYMMDD_<analysis>_<annotation-version>_<short-tag>
```

Example:

```text
20260710_healthy_reference_lineage-v1_tripso-baseline
```

Every run should record:

```text
resolved configuration
Git commit
Python environment
input manifests
annotation version
random seeds
model configuration
Slurm job IDs
start/end timestamps
software versions
```

A result without its run configuration is not a finished result.

---

# Data contracts

## Cell-level AnnData contract

Required `.obs` columns:

```text
cell_id
dataset_id
study_id
library_id
sample_id
donor_id
age
sex
healthy_status
analysis_lineage
annotation_version
annotation_confidence
```

Recommended QC columns:

```text
n_counts
n_genes
pct_mito
pct_ribo
pct_hb
pct_platelet
doublet_score
```

Required layers:

```text
counts
```

Raw counts must remain available.

## Annotation contract

Every annotation version must export:

```text
cell_id
analysis_lineage
fine_cell_type
annotation_method
annotation_version
confidence
second_best_label
decision_margin
annotation_status
```

Allowed `annotation_status` values:

```text
high_confidence
moderate_confidence
low_confidence
mixed
doublet
low_quality
unknown
```

Downstream stages must use `analysis_lineage`, not tool-specific fields such as CellTypist's raw output columns.

## Donor-level summary contract

One row per:

```text
dataset_id × donor_id × lineage × program
```

Columns should include:

```text
dataset_id
donor_id
age
sex
lineage
program_id
program_score
n_cells
sampling_variance
annotation_version
model_version
```

This table is the main interface between single-cell modeling and donor-level biological analysis.

---

# Pipeline modules

## Module 1 — Dataset preparation

Responsibilities:

- load original dataset;
- harmonize genes;
- harmonize metadata;
- preserve raw counts;
- validate required columns;
- create one canonical AnnData object per dataset.

## Module 2 — Quality control

Responsibilities:

- calculate QC metrics;
- identify low-quality cells;
- calculate or import doublet scores;
- produce donor- and dataset-level QC reports;
- apply explicit, documented filtering.

Filtering rules belong in configuration.

Never silently remove cells.

## Module 3 — Cell annotation

### Version 1

Use CellTypist with:

- raw prediction;
- majority voting;
- confidence values;
- broad ontology collapse;
- marker-based cluster review;
- explicit unknown/low-confidence class.

### Later versions

Support recursive annotation or custom references.

The module must emit the standard annotation table regardless of method.

## Module 4 — Lineage object creation

Create lineage-specific AnnData objects from the chosen annotation version.

Outputs:

```text
lineages/<annotation_version>/<lineage>.h5ad
```

## Module 5 — Donor-aware aggregation

Produce donor × lineage summaries.

Possible representations:

- pseudobulk counts;
- mean pathway activity;
- median pathway activity;
- Tripso embedding centroid;
- embedding variance;
- quantiles;
- cell-state proportions;
- pathway ratios.

Cell counts should determine uncertainty, not biological weight.

## Module 6 — Gene-program modeling

Support multiple approaches without forcing them into one abstraction:

```text
Tripso
simple gene-set scores
rank-based scores
pathway-ratio scores
PCA/factor baselines
optional healthy-manifold models
```

## Module 7 — Healthy trajectory learning

Fit healthy expectations as functions of:

```text
age
sex
dataset
optional CMV status
optional other known covariates
```

Prefer:

- dataset-specific fits;
- hierarchical partial pooling;
- meta-analysis;
- leave-one-dataset-out evaluation.

## Module 8 — Immune-health metric

Candidate components may include:

- pathway biological-age residual;
- healthy-manifold distance;
- robust Mahalanobis distance;
- pathway antagonism ratio;
- lineage-specific deviation;
- composition deviation;
- cross-lineage coherence;
- uncertainty-weighted composite score.

Keep the individual components available. Do not only save a final scalar.

## Module 9 — SoundLife validation

Primary questions:

- Are scores stable within healthy individuals?
- Is between-person variation larger than within-person variation?
- Do transient perturbations return toward baseline?
- Does stability survive cell-count downsampling?
- Are results annotation-version dependent?

## Module 10 — Disease transfer

Map query datasets without refitting the healthy reference.

Report:

- score distributions;
- uncertainty;
- out-of-reference frequency;
- lineage-specific deviations;
- associations with outcomes;
- sensitivity to annotation version and dataset calibration.

---

# Annotation strategy

## Initial development strategy

Use CellTypist to unblock the downstream pipeline.

Recommended progression:

```text
lineage_v1_celltypist
    ↓
lineage_v2_celltypist_plus_marker_review
    ↓
lineage_v3_recursive_reference
```

Run the complete downstream pipeline with `lineage_v1_celltypist`.

Later, rerun the same pipeline with improved annotation versions.

The annotation module is worth upgrading only if it materially changes:

- donor-level lineage proportions;
- program scores;
- age-associated pathways;
- healthy-manifold geometry;
- SoundLife stability;
- disease associations;
- final donor rankings.

## Broad ontology for early development

```text
T_and_NK
B
Plasma
Mono_and_DC
pDC
Platelets
RBC_and_HSPC
Cycling
Doublet_or_ambiguous
Unknown
```

Use finer labels later, but keep the broad ontology stable.

---

# Parallelization strategy

Use Slurm arrays for naturally independent jobs.

Good array dimensions:

```text
dataset
lineage
random seed
LODO fold
model configuration
cell-count cap
HVG strategy
annotation version
```

Use GPUs for:

- scVI/scANVI;
- Tripso;
- large neural healthy-manifold models;
- repeated model sweeps.

Use CPUs for:

- QC;
- pseudobulk aggregation;
- Leiden grids;
- differential expression;
- mixed models;
- meta-analysis;
- reporting.

Prefer many independent one-GPU jobs over complicated distributed training unless memory requires multi-GPU execution.

---

# Development modes

## Smoke mode

Purpose: verify that the pipeline runs.

```text
1–2 datasets
2–3 donors per dataset
small cell cap
1 lineage
1 seed
few epochs
```

Expected runtime: minutes.

## Development mode

Purpose: develop and inspect outputs.

```text
2 datasets
5–10 donors per dataset
moderate cell cap
all broad lineages
1–2 seeds
```

Expected runtime: under a few hours.

## Full mode

Purpose: final healthy-reference analysis.

```text
all five healthy datasets
all eligible donors
all selected lineages
multiple seeds
LODO
cell-count sensitivity
SoundLife validation
```

No script should require full mode to debug basic logic.

---

# Checkpointing and reruns

Every expensive stage should be restartable.

Before computing, a script should check whether:

- expected output exists;
- output metadata matches the requested configuration;
- output is complete;
- `--overwrite` was supplied.

Do not silently reuse outputs from a different configuration.

Use explicit stage completion files:

```text
runs/<run_id>/manifests/<stage>.done.json
```

---

# Notebooks versus scripts

Use scripts for:

- deterministic data processing;
- model training;
- batch jobs;
- aggregation;
- validation metrics;
- figure generation used in the paper.

Use notebooks for:

- exploratory biology;
- annotation review;
- hypothesis generation;
- detailed inspection;
- temporary analyses.

Once a notebook analysis becomes part of the main pipeline, move the stable logic into `src/` or `scripts/`.

---

# Coding style

Prefer:

- small readable functions;
- explicit names;
- type hints for important interfaces;
- docstrings for non-obvious functions;
- logging instead of scattered print statements;
- pathlib instead of manual path concatenation;
- YAML configuration;
- clear failure messages.

Avoid:

- unnecessary object-oriented design;
- giant notebooks as pipelines;
- hidden global variables;
- cluster-specific paths in Python files;
- silent fallback behavior;
- copying the same logic into several scripts.

---

# Minimal testing standard

This is not a software package, but a few tests are essential.

Maintain:

1. configuration-loading test;
2. metadata-contract test;
3. annotation-contract test;
4. tiny end-to-end smoke test;
5. deterministic aggregation test;
6. test that Gefion and GenomeDK configs resolve to the same scientific configuration.

The purpose is to catch expensive HPC mistakes before submitting large jobs.

---

# Decision rules

## Upgrade annotation when

- broad-lineage disagreement is systematic by age, sex, dataset, or chemistry;
- rare-lineage recovery materially changes;
- donor-level program scores change;
- SoundLife stability improves;
- disease associations change;
- CellTypist confidence is poor in important compartments.

## Do not upgrade annotation merely because

- the UMAP looks cleaner;
- the recursive workflow is more sophisticated;
- a tiny number of cells change labels;
- the fine ontology is more detailed but irrelevant to the biological analysis.

## Add a new model only when it answers a distinct question

Examples:

```text
Tripso:
    context-dependent program representation

pathway ratios:
    robustness to baseline shifts

age prediction:
    biological age residuals

healthy manifold:
    multidimensional deviation from expected healthy state

composition module:
    donor-level immune architecture
```

Do not accumulate models only to use GPU hours.

---

# Current planned analysis sequence

## Stage A — Fast end-to-end baseline

1. Prepare five healthy datasets.
2. Apply QC.
3. Annotate with CellTypist.
4. Collapse labels into broad lineages.
5. Build donor × lineage summaries.
6. Train a simple gene-program/pathway baseline.
7. Train initial Tripso models.
8. Fit healthy age/sex trajectories.
9. Compute preliminary immune-health components.
10. Validate longitudinal stability in SoundLife.
11. Transfer to one disease cohort.
12. Identify the parts of the pipeline most sensitive to annotation.

## Stage B — Robustness

1. LODO across the five healthy datasets.
2. Donor-balanced training.
3. Cell-count downsampling.
4. Multiple seeds.
5. Dataset-specific versus pooled fits.
6. Meta-analysis of program-age effects.
7. Compare scalar pathway scores versus Tripso embeddings.
8. Compare annotation v1 and v2.

## Stage C — Annotation upgrade

1. Marker-review uncertain CellTypist clusters.
2. Recursively refine T/NK and Mono/DC.
3. Train custom annotation reference if justified.
4. Export new annotation version.
5. Rerun downstream stages unchanged.
6. Quantify whether biological conclusions improve.

## Stage D — Main biological analysis

1. Freeze healthy-reference model.
2. Freeze annotation version.
3. Freeze pathway/program definitions.
4. Run SoundLife validation.
5. Run external disease cohorts.
6. Relate immune-health components to outcomes.
7. Interpret lineage- and pathway-specific drivers.
8. Build final figures and manuscript analyses.

---

# Instructions for Codex

When modifying this repository:

1. Preserve modular boundaries.
2. Do not hard-code cluster paths or Slurm accounts.
3. Add new environment-specific values to cluster YAML files.
4. Add new scientific values to analysis or common YAML files.
5. Make every expensive stage restartable.
6. Keep annotation method outputs behind the standard annotation contract.
7. Use donor-level splits for biological prediction tasks.
8. Write outputs under a run-specific directory.
9. Save the resolved configuration and provenance.
10. Prefer a clear analysis script over a complex framework.
11. Do not refactor working analysis code without a concrete benefit.
12. Keep the biological question visible in function and output names.
13. Treat unknown or ambiguous cells explicitly.
14. Do not let donors with more cells dominate silently.
15. Ensure a development-mode run exists before submitting full-scale jobs.

When uncertain, choose the implementation that is:

```text
easier to inspect
easier to rerun
easier to compare across annotation/model versions
less likely to hide biological assumptions
```

---

# Final success criteria

The repository succeeds when:

- the same analysis can run on Gefion and GenomeDK by changing only configuration;
- the annotation module can be swapped without rewriting downstream analysis;
- every major result is traceable to a versioned run;
- donor-level biology is not confused with cell-level sample size;
- healthy patterns reproduce across datasets;
- SoundLife shows meaningful longitudinal stability;
- disease cohorts can be scored without retraining the healthy reference;
- final results remain interpretable at lineage, pathway, gene, and donor levels;
- the repository supports the biological paper without becoming a software-engineering project of its own.

The project should remain ambitious scientifically and restrained computationally:

> Build only as much infrastructure as needed to make the biological conclusions trustworthy.
