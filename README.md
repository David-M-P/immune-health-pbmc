# PBMC Immune Health

## Project goal

This repository supports development of an interpretable, transferable
immune-health metric from PBMC single-cell RNA-seq data. Healthy cohorts will
define expected age- and sex-associated immune patterns, SoundLife will test
longitudinal stability, and immune-challenged cohorts will be scored against the
frozen healthy reference. Donors, rather than individual cells, are the
biological replicates.

The repository currently contains configuration and validation scaffolding only.
Scientific analysis is intentionally not implemented yet.

## Intended module sequence

The planned pipeline is:

1. prepare and harmonize each dataset;
2. calculate and report quality-control metrics;
3. annotate cells and collapse labels to a stable broad-lineage contract;
4. create lineage-specific objects;
5. aggregate cells into donor-aware summaries;
6. model interpretable gene programs;
7. fit healthy age- and sex-associated trajectories;
8. compute immune-health components with uncertainty;
9. validate longitudinal stability in SoundLife;
10. transfer the frozen reference to immune-challenged cohorts; and
11. build traceable reports.

CellTypist is the initial annotation implementation. Downstream stages will use
the standard annotation fields rather than CellTypist-specific output names, so
the annotation method can be upgraded without rewriting the analysis.

## Configuration

Runtime configuration has three layers:

```text
configs/common.yaml
  + configs/clusters/<cluster>.yaml
  + configs/analyses/<analysis>.yaml
```

Gefion and GenomeDK share the scientific and analysis configuration. They differ
only in infrastructure settings such as data roots, output roots, Slurm account,
partitions, and environment activation. Gefion's known project account is
`cu_0071`; all unknown cluster values remain visibly marked with angle-bracket
placeholders. GenomeDK values are independent placeholders and must be filled in
before submission.

Dataset-specific metadata belongs in `configs/datasets/`. The included
`example_dataset.yaml` is a template and does not identify a real cohort.

## Development-mode example

Install the lightweight project dependencies into an active Python 3.10+
environment:

```bash
python -m pip install -e ".[dev]"
```

Then resolve and validate a development configuration:

```bash
python scripts/00_validate_config.py \
  --common-config configs/common.yaml \
  --cluster-config configs/clusters/gefion.yaml \
  --analysis-config configs/analyses/development.yaml \
  --run-id 20260710_development_celltypist
```

This writes `runs/<run-id>/resolved_config.yaml`. Placeholder warnings are
expected in the initial scaffold; replace those values before running data or
Slurm stages.

## Repository layout

```text
configs/             Scientific, analysis, annotation, dataset, and HPC settings
scripts/             Explicit command-line entry points
src/immune_health/   Stable reusable analysis logic
workflows/           Thin execution wrappers (to be added when stages exist)
metadata/            Dataset and donor manifests (not populated yet)
annotations/         Versioned annotation mappings (not populated yet)
tests/               Small safeguards for configuration and contracts
runs/                Run-specific resolved configuration and future outputs
tripso_code/          Vendored third-party TRIPSO source and reproducibility code
```

Raw or patient-level data must not be committed. Generated results, model
outputs, and large single-cell objects should remain outside Git or under ignored
run directories. See [AGENTS.md](AGENTS.md) for the full biological and
engineering guidance.

## Vendored TRIPSO source

TRIPSO and its reproducibility repository are vendored under `tripso_code/`.
Their licenses and imported revisions are documented in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[docs/TRIPSO_PROVENANCE.md](docs/TRIPSO_PROVENANCE.md).
