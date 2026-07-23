# PBMC Immune Health

This repository builds a donor-aware, interpretable PBMC healthy reference and
rehearses zero-shot transfer with five-dataset leave-one-dataset-out (LODO)
evaluation. Donors—not cells—are the independent biological replicates.

The current implementation provides a tested scientific vertical slice:

- read-only H5AD and provenance audit;
- one global donor split shared by all five primary lineages;
- review-only fine-cell-type ontology generation;
- deterministic dataset → donor → fine type → cell sampling;
- composition, pseudobulk, GP-score, PCA, and elastic-net baselines;
- donor-level distribution summaries, distances, decomposition, and uncertainty;
- healthy age/sex trajectories and frozen query scoring;
- conservative TRIPSO wrappers with immutable resources and state guards;
- restartable SLURM arrays generated from JSONL manifests; and
- a central `immune-health` CLI with all pipeline commands.

No large production training job has been launched.

## Audited data contract

The read-only audit observed five reference datasets (`aidav2`,
`immuneindonesia`, `immunobiologyaging`, `onek1k`, and `terekhova`), eight merged
lineages, 2,180 donors, and 18,035 common Ensembl genes. Merged `.X` matrices are
CSR `float64` arrays whose sampled stored values are non-negative and
integer-like; no layer or `.raw` fallback is present.

OneK1K `sample_id` values denote pools shared by donors. The approved identifiers
therefore are:

```text
biological_unit_id    = dataset::donor_id
source_observation_id = dataset::sample_id
observation_id        = dataset::donor_id::sample_id
```

The full evidence boundary and tables are in
[`reports/data_audit/`](reports/data_audit/). The generated candidate at
[`configs/data/fine_type_ontology.generated.yaml`](configs/data/fine_type_ontology.generated.yaml)
remains an immutable audit input. Production uses the hash-bound, David-approved
five-primary-lineage policy in
[`configs/data/fine_type_ontology.approved.yaml`](configs/data/fine_type_ontology.approved.yaml).
It preserves raw labels, applies the 0.70 confidence cutoff cell by cell, and
retains quarantined categories for composition without state inference or
rare-type balancing uplift.

## Install and test

Use Linux and Python 3.10:

```bash
python -m pip install -e ".[dev]"
pytest -q
ruff check src scripts tests slurm/run_manifest_task.py
```

The central command is available as either `immune-health` after installation or
`python -m immune_health.cli` with `PYTHONPATH=src`. Every subcommand supports
`--config`, `--dry-run`, `--seed`, and `--log-level`.

```bash
python -m immune_health.cli --help
```

## Reproducible preparation commands

Audit paths without opening the H5AD matrices:

```bash
python -m immune_health.cli audit-data \
  --config configs/data/reference.yaml \
  --output-dir reports/data_audit \
  --dry-run --seed 42 --log-level INFO
```

Build the global 2,180-donor LODO manifests:

```bash
python -m immune_health.cli make-lodo-folds \
  --config configs/data/reference.yaml \
  --metadata reports/data_audit/donor_summary.tsv.gz \
  --output-dir splits --n-inner-folds 3 --seed 42
```

Generate the conservative ontology candidate and validate real GP resources:

```bash
python -m immune_health.cli build-fine-type-ontology \
  --config configs/data/reference.yaml \
  --input reports/data_audit/fine_type_summary.tsv \
  --output configs/data/fine_type_ontology.generated.yaml \
  --summary-output reports/fine_type_ontology/candidate_summary.tsv

python -m immune_health.cli validate-gene-programs \
  --config configs/gene_programs/default.yaml \
  --output reports/gene_programs/validation.tsv
```

## Smoke tests

The synthetic end-to-end test exercises counts → LODO → baseline → explicit mock
adapter → donor aggregation → healthy trajectory → frozen held-out scoring.

The bounded real-data baseline smoke reads a donor-balanced sparse B-cell subset,
fits PCA on four datasets, and projects Terekhova without refitting:

```bash
python scripts/run_real_baseline_smoke.py \
  --output-dir runs/smoke_real_b_cells \
  --max-donors-per-dataset 2 --max-cells-per-donor 50 --seed 42
```

This is explicitly not a TRIPSO run and does not use a production LODO
vocabulary. Large outputs remain under the ignored `runs/` tree; adjacent JSON
manifests retain provenance.

## TRIPSO and SLURM

TRIPSO source is vendored unchanged under `tripso_code/tripso`; its upstream
revision is documented in [`docs/TRIPSO_PROVENANCE.md`](docs/TRIPSO_PROVENANCE.md).
Create the single pinned environment and validate the real runtime and assets before
training:

```bash
mamba env create --prefix .conda_isolated/immune-health-tripso \
  --file environment.yml

NUMBA_CACHE_DIR=/tmp/immune_health_numba_cache \
MPLCONFIGDIR=/tmp/immune_health_mpl \
.conda_isolated/immune-health-tripso/bin/python \
  scripts/validate_tripso_environment.py \
  --vendor-root tripso_code/tripso \
  --geneformer-root \
    /faststorage/project/immunehealth/Projects/david/external_assets/Geneformer \
  --smoke-mode mock \
  --json-output reports/tripso_environment_pinned.json
```

Generate—but do not submit—the first five-lineage LODO matrix:

```bash
python scripts/generate_job_manifests.py \
  --config configs/experiments/tripso_lodo.yaml \
  --stage stage1 --output-dir slurm/manifests --seed 42
```

This creates 150 rows (five lineages × five held-out datasets × three primary
samplers × two feature sets × one seed). The feature sets are 3,000 or 9,000
fold-local HVGs, each unioned with all retained GP genes and retaining every
eligible cell. The local CPU template retains account `immunehealth`, four hours,
96 GB, and four CPUs. Gefion uses account `cu_0071`; its unknown site resources
remain explicit configuration values. Because Gefion bills exclusive eight-GPU
nodes, CPU phases use `slurm/cpu_nodepack.sbatch` and GPU training/projection use
`slurm/tripso_nodepack.sbatch`. The latter launches eight independent one-GPU
manifest rows per node with GPU binding and single-process Lightning isolation.
The ordinary `tripso_array.sbatch` remains the single-row fallback for clusters
that schedule and bill individual jobs.

Generate the staged CPU preparation plus the exact all-five-healthy Stage-3 inputs:

```bash
python scripts/generate_reference_prep_jobs.py \
  --config configs/experiments/reference_preparation.yaml \
  --output-dir slurm/manifests/reference_prep
```

The final path has no held-out sentinel, trims Terekhova to one deterministic visit
per donor, prepares both HVG+GP vocabularies without cell sampling, and physically
binds all-reference Arrow donors to the fold input consumed by Stage 3. Future
queries are mapped to this frozen vocabulary; query expression cannot redefine it.

## Documentation

See [`docs/pipeline_overview.md`](docs/pipeline_overview.md),
[`docs/gefion_runbook.md`](docs/gefion_runbook.md),
[`docs/production_workflow.md`](docs/production_workflow.md),
[`docs/reference_preparation.md`](docs/reference_preparation.md),
[`docs/post_training_projection.md`](docs/post_training_projection.md),
[`docs/data_contract.md`](docs/data_contract.md),
[`docs/lodo_design.md`](docs/lodo_design.md),
[`docs/gene_programs.md`](docs/gene_programs.md),
[`docs/gefion_compute_plan.md`](docs/gefion_compute_plan.md),
[`docs/donor_distribution_aggregation.md`](docs/donor_distribution_aggregation.md),
[`docs/output_schema.md`](docs/output_schema.md),
[`docs/tripso_environment.md`](docs/tripso_environment.md), and
[`docs/running_on_slurm.md`](docs/running_on_slurm.md).
