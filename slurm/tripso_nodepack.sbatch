#!/usr/bin/env bash
#SBATCH --account=immunehealth
#SBATCH --job-name=immune-health-tripso-nodepack
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err

set -euo pipefail

: "${PROJECT_ROOT:?Set PROJECT_ROOT to the repository checkout}"
: "${TRIPSO_JOB_MANIFEST:?Set TRIPSO_JOB_MANIFEST to a generated JSONL manifest}"
: "${TRIPSO_WORKERS_PER_NODE:?Set TRIPSO_WORKERS_PER_NODE (8 on Gefion)}"
: "${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array task}"
: "${SLURM_CPUS_PER_TASK:?Request --cpus-per-task for each GPU worker}"

if [[ ! "${TRIPSO_WORKERS_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "TRIPSO_WORKERS_PER_NODE must be a positive integer" >&2
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
    echo "Node packing requires exactly one allocated node" >&2
    exit 2
fi
if [[ -n "${SLURM_NTASKS:-}" ]] && (( SLURM_NTASKS < TRIPSO_WORKERS_PER_NODE )); then
    echo "Request at least ${TRIPSO_WORKERS_PER_NODE} Slurm tasks" >&2
    exit 2
fi
if [[ "${SLURM_GPUS_ON_NODE:-}" =~ ^[0-9]+$ ]] \
    && (( SLURM_GPUS_ON_NODE < TRIPSO_WORKERS_PER_NODE )); then
    echo "Request at least ${TRIPSO_WORKERS_PER_NODE} GPUs on the node" >&2
    exit 2
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export IMMUNE_HEALTH_EXPECT_VISIBLE_GPUS=1

NODEPACK_LOG_ROOT="${TRIPSO_NODEPACK_LOG_ROOT:-${PROJECT_ROOT}/slurm/logs/nodepack}"
NODEPACK_LOG_DIR="${NODEPACK_LOG_ROOT}/${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${NODEPACK_LOG_DIR}"

# One Slurm step owns all GPUs.  Each task receives one GPU and a unique
# SLURM_PROCID, which run_manifest_nodepack.py maps to a distinct manifest row.
# --kill-on-bad-exit=0 lets successful rows finish when a peer fails; the srun
# step still returns non-zero, and restart markers make resubmission selective.
exec srun \
    --nodes=1 \
    --ntasks="${TRIPSO_WORKERS_PER_NODE}" \
    --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    --gpus-per-task=1 \
    --gpu-bind=single:1 \
    --cpu-bind=cores \
    --exact \
    --kill-on-bad-exit=0 \
    --wait=0 \
    --output="${NODEPACK_LOG_DIR}/worker_%t.out" \
    --error="${NODEPACK_LOG_DIR}/worker_%t.err" \
    python -u slurm/run_manifest_nodepack.py \
        --manifest "${TRIPSO_JOB_MANIFEST}" \
        --block-index "${SLURM_ARRAY_TASK_ID}" \
        --workers-per-node "${TRIPSO_WORKERS_PER_NODE}" \
        "$@"
