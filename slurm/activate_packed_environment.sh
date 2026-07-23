#!/usr/bin/env bash
# Source this file from a Slurm launcher after exporting IMMUNE_HEALTH_ENV_ROOT.
# The path stays in cluster configuration; this repository file only implements
# the nounset-safe activation contract used by every launcher.

: "${IMMUNE_HEALTH_ENV_ROOT:?Set IMMUNE_HEALTH_ENV_ROOT to the unpacked environment}"

if [[ ! -f "${IMMUNE_HEALTH_ENV_ROOT}/bin/activate" ]]; then
    echo "Packed-environment activation script is missing: ${IMMUNE_HEALTH_ENV_ROOT}/bin/activate" >&2
    return 2 2>/dev/null || exit 2
fi

case $- in
    *u*) IMMUNE_HEALTH_RESTORE_NOUNSET=1 ;;
    *) IMMUNE_HEALTH_RESTORE_NOUNSET=0 ;;
esac

# conda-pack's POSIX activation reads ordinary interactive-shell variables that
# may be unset. The project Slurm launchers deliberately enable nounset first.
set +u
source "${IMMUNE_HEALTH_ENV_ROOT}/bin/activate"
if [[ "${IMMUNE_HEALTH_RESTORE_NOUNSET}" == "1" ]]; then
    set -u
fi
unset IMMUNE_HEALTH_RESTORE_NOUNSET

export PYTHONNOUSERSITE=1
hash -r
