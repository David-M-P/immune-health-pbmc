#!/usr/bin/env bash
#SBATCH --account=immunehealth
#SBATCH --job-name=immune-health-cpu-nodepack
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err

set -euo pipefail

: "${PROJECT_ROOT:?Set PROJECT_ROOT to the repository checkout}"
: "${CPU_JOB_MANIFEST:?Set CPU_JOB_MANIFEST to a JSONL job manifest}"
: "${CPU_WORKERS_PER_NODE:?Set CPU_WORKERS_PER_NODE}"
: "${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array task}"
: "${SLURM_CPUS_PER_TASK:?Request --cpus-per-task for each CPU worker}"

if [[ ! "${CPU_WORKERS_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "CPU_WORKERS_PER_NODE must be a positive integer" >&2
    exit 2
fi
if [[ ! "${SLURM_CPUS_PER_TASK}" =~ ^[1-9][0-9]*$ ]]; then
    echo "SLURM_CPUS_PER_TASK must be a positive integer" >&2
    exit 2
fi

if [[ -n "${ENVIRONMENT_ACTIVATION_SCRIPT:-}" ]]; then
    if [[ ! -f "${ENVIRONMENT_ACTIVATION_SCRIPT}" ]]; then
        echo "Activation script does not exist: ${ENVIRONMENT_ACTIVATION_SCRIPT}" >&2
        exit 2
    fi
    source "${ENVIRONMENT_ACTIVATION_SCRIPT}"
fi

if [[ "${SLURM_JOB_NUM_NODES:-1}" != "1" ]]; then
    echo "CPU node packing requires exactly one allocated node" >&2
    exit 2
fi
if [[ -n "${SLURM_NTASKS:-}" ]] \
    && (( SLURM_NTASKS < CPU_WORKERS_PER_NODE )); then
    echo "Request at least ${CPU_WORKERS_PER_NODE} Slurm tasks" >&2
    exit 2
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false

NODEPACK_LOG_ROOT="${CPU_NODEPACK_LOG_ROOT:-${PROJECT_ROOT}/slurm/logs/cpu_nodepack}"
NODEPACK_LOG_DIR="${NODEPACK_LOG_ROOT}/${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${NODEPACK_LOG_DIR}"

# Each task is one independent CPU manifest row. The mapper gives every child a
# normalized one-task Slurm environment, hides CUDA, and requires PyTorch to see
# zero GPUs. This is safe for reference preparation, post-training binding, and
# downstream CPU manifests even when the outer node happens to expose GPUs.
#
# Peer failures do not terminate successful rows. The step still returns nonzero,
# and the ordinary per-row .done.json inventory makes resubmission selective.
exec srun \
    --nodes=1 \
    --ntasks="${CPU_WORKERS_PER_NODE}" \
    --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    --cpu-bind=cores \
    --exact \
    --kill-on-bad-exit=0 \
    --wait=0 \
    --output="${NODEPACK_LOG_DIR}/worker_%t.out" \
    --error="${NODEPACK_LOG_DIR}/worker_%t.err" \
    python -u slurm/run_manifest_nodepack.py \
        --manifest "${CPU_JOB_MANIFEST}" \
        --block-index "${SLURM_ARRAY_TASK_ID}" \
        --workers-per-node "${CPU_WORKERS_PER_NODE}" \
        --expected-visible-gpus 0 \
        "$@"
