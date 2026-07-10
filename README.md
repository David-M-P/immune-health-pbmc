# PBMC Immune Health

## Project goal

This project develops an interpretable and transferable immune-health metric from healthy PBMC single-cell RNA-seq cohorts and evaluates it in disease cohorts.

## Repository structure

```text
configs/                 Tracked, machine-neutral configuration examples
docs/                    Provenance and development documentation
notebooks/               Project notebooks
scripts/                 Lightweight command-line utilities
slurm/au/                 Aarhus University HPC job scripts
slurm/gefion/             Gefion job scripts
src/immune_health_pbmc/  Project Python package
tests/                   Project tests
tripso_code/              Vendored TRIPSO source and reproducibility code
```

## Vendored TRIPSO source code

TRIPSO and its reproducibility repository are included directly under `tripso_code/` as ordinary tracked files. No additional GitHub clone or submodule initialization is needed to access them. The vendored source may be modified within this main repository when project work requires it; those changes are then recorded in the main repository history.

The upstream projects remain third-party software governed by their retained licenses. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [docs/TRIPSO_PROVENANCE.md](docs/TRIPSO_PROVENANCE.md).

## Installation

Python 3.10 or newer is required. Dependency installation is intentionally a user-run step:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ./tripso_code/tripso
python -m pip install -e .
```

## Basic verification

After installation, verify that the vendored TRIPSO package is importable:

```bash
python scripts/check_tripso_import.py
```

## Computing environments

GitHub is the canonical source of truth. Local development and the Aarhus University HPC use the GitHub repository directly. Gefion uses an internal GitLab mirror of the GitHub repository and is normally treated as an execution environment.

## Configuration

The YAML files in `configs/` are examples only. Copy the appropriate example to an untracked local configuration and replace its placeholders. Machine-specific paths belong in local configuration files, not committed source code.

## Data policy

Raw data and patient-level information must never be committed. Large processed single-cell objects, generated model outputs, checkpoints, and other derived assets should normally remain outside Git. Any required external assets must be documented without exposing credentials or sensitive locations.

## Development workflow

Begin work by pulling the latest canonical branch, then use a short-lived feature or fix branch for meaningful changes. The laptop and AU HPC push to GitHub; Gefion receives the one-way mirror. See [docs/DEVELOPMENT_WORKFLOW.md](docs/DEVELOPMENT_WORKFLOW.md).

## Third-party attribution

TRIPSO and the TRIPSO reproducibility repository are third-party projects. Their original LICENSE files remain in their vendored directories, and exact imported revisions are recorded in the provenance documentation.

## Current status

The repository currently provides the initial project structure, environment-neutral configuration examples, workflow documentation, and self-contained vendored TRIPSO working trees. Scientific analyses and project-specific implementations will be added as research work proceeds.
