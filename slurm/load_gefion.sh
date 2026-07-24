#!/usr/bin/env bash
# Source this file at the start of every Gefion login:
#   source ./slurm/load_gefion.sh
#
# It loads the concrete deployment configuration, derived run paths, and
# submission functions. It does not unpack data or submit jobs.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf 'This file must be sourced: source %s\n' "$0" >&2
    exit 2
fi

_gefion_loader_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
_gefion_checkout=$(cd -- "${_gefion_loader_dir}/.." && pwd)

if [[ ! -r "${_gefion_checkout}/gefion_env.txt" ]]; then
    printf 'Missing Gefion environment file: %s\n' \
        "${_gefion_checkout}/gefion_env.txt" >&2
    return 2
fi

source "${_gefion_checkout}/gefion_env.txt"

if [[ "${PROJECT_ROOT}" != "${_gefion_checkout}" ]]; then
    printf 'PROJECT_ROOT mismatch:\n  gefion_env.txt: %s\n  checkout:       %s\n' \
        "${PROJECT_ROOT}" "${_gefion_checkout}" >&2
    return 2
fi

export REF_MANIFEST_DIR="${MANIFEST_ROOT}/reference_prep"
export TRAIN_MANIFEST_DIR="${MANIFEST_ROOT}/training"
export PACKED_REF_DIR="${REF_MANIFEST_DIR}/packed"
export STAGE1_MANIFEST="${TRAIN_MANIFEST_DIR}/stage1.jsonl"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

source "${PROJECT_ROOT}/slurm/gefion_submission_helpers.sh"

cd "${PROJECT_ROOT}" || return 2
unset _gefion_loader_dir _gefion_checkout

printf '%s\n' \
    "Loaded Gefion run ${RUN_ID}" \
    "  project: ${PROJECT_ROOT}" \
    "  output:  ${OUTPUT_ROOT}" \
    "  account: ${GEFION_ACCOUNT}" \
    "  queue:   ${GEFION_GPU_PARTITION}" \
    "  packing: ${GPU_WORKERS_PER_NODE} one-GPU workers per node"
