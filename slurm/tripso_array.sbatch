#!/usr/bin/env bash
#SBATCH --account=immunehealth
#SBATCH --job-name=immune-health-tripso
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err

set -euo pipefail

: "${PROJECT_ROOT:?Set PROJECT_ROOT to the repository checkout}"
: "${TRIPSO_JOB_MANIFEST:?Set TRIPSO_JOB_MANIFEST to a generated JSONL manifest}"
: "${SLURM_ARRAY_TASK_ID:?This script must run as a SLURM array task}"

if [[ -n "${ENVIRONMENT_ACTIVATION_SCRIPT:-}" ]]; then
    if [[ ! -f "${ENVIRONMENT_ACTIVATION_SCRIPT}" ]]; then
        echo "Activation script does not exist: ${ENVIRONMENT_ACTIVATION_SCRIPT}" >&2
        exit 2
    fi
    # The site-owned activation script is explicit cluster configuration.
    source "${ENVIRONMENT_ACTIVATION_SCRIPT}"
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec python slurm/run_manifest_task.py \
    --manifest "${TRIPSO_JOB_MANIFEST}" \
    --index "${SLURM_ARRAY_TASK_ID}"
