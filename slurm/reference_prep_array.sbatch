#!/usr/bin/env bash
#SBATCH --account=immunehealth
#SBATCH --job-name=immune-health-reference-prep
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4

set -euo pipefail

: "${PROJECT_ROOT:?Set PROJECT_ROOT to the repository checkout}"
: "${REFERENCE_PREP_JOB_MANIFEST:?Set REFERENCE_PREP_JOB_MANIFEST to a JSONL manifest}"
: "${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array task}"

if [[ -n "${ENVIRONMENT_ACTIVATION_SCRIPT:-}" ]]; then
    if [[ ! -f "${ENVIRONMENT_ACTIVATION_SCRIPT}" ]]; then
        echo "Activation script does not exist: ${ENVIRONMENT_ACTIVATION_SCRIPT}" >&2
        exit 2
    fi
    source "${ENVIRONMENT_ACTIVATION_SCRIPT}"
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec python slurm/run_manifest_task.py \
    --manifest "${REFERENCE_PREP_JOB_MANIFEST}" \
    --index "${SLURM_ARRAY_TASK_ID}"
