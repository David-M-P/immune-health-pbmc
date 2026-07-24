#!/usr/bin/env bash
# Source this file after gefion_env.txt. It defines restartable node-packed
# submission helpers but does not submit anything by itself.

_gefion_parse_sbatch_job_id() {
    local raw_output="$1"
    local line
    local parsed=""

    # Slurm --parsable normally writes JOB_ID or JOB_ID;CLUSTER. Some Gefion
    # wrappers/plugins can write an informational line to stdout as well. Accept
    # exactly one numeric parsable record and ignore non-record text.
    while IFS= read -r line; do
        line="${line%$'\r'}"
        if [[ "${line}" =~ ^([0-9]+)(\;[^[:space:]]+)?$ ]]; then
            if [[ -n "${parsed}" && "${parsed}" != "${BASH_REMATCH[1]}" ]]; then
                return 1
            fi
            parsed="${BASH_REMATCH[1]}"
        fi
    done <<< "${raw_output}"

    if [[ -z "${parsed}" ]]; then
        return 1
    fi
    printf '%s\n' "${parsed}"
}

plan_array_spec() {
    local manifest="$1"
    local workers="$2"
    local expected_gpus="$3"
    local indices="${4:-}"
    local -a args=(
        --manifest "$manifest"
        --workers-per-node "$workers"
        --expected-visible-gpus "$expected_gpus"
        --plan-only
    )
    if [[ -n "$indices" ]]; then
        args+=(--indices "$indices")
    fi
    "$PY" "$PROJECT_ROOT/slurm/run_manifest_nodepack.py" "${args[@]}" |
        "$PY" -c 'import json,sys; print(json.load(sys.stdin)["slurm_array_spec"])'
}

submit_cpu_pack() {
    local manifest="$1"
    local dependency="${2:-}"
    local indices="${3:-}"
    local array_spec
    local -a runner_args=()
    local -a sbatch_args

    if ! array_spec=$(plan_array_spec \
        "$manifest" "$CPU_WORKERS_PER_NODE" 0 "$indices"); then
        printf 'CPU node-pack planning failed for %s\n' "$manifest" >&2
        return 1
    fi
    if [[ ! "$array_spec" =~ ^0-[0-9]+$ ]]; then
        printf 'CPU planner returned an invalid array for %s: %q\n' \
            "$manifest" "$array_spec" >&2
        return 1
    fi
    if [[ -n "$indices" ]]; then
        runner_args=(--indices "$indices")
    fi

    sbatch_args=(
        --parsable
        --chdir="$PROJECT_ROOT"
        --account="$GEFION_ACCOUNT"
        --partition="$GEFION_CPU_PARTITION"
        --time="$CPU_WALLTIME"
        --nodes=1
        --exclusive
        --ntasks-per-node="$CPU_WORKERS_PER_NODE"
        --cpus-per-task="$CPU_CPUS_PER_WORKER"
        --mem="$CPU_NODE_MEMORY"
        --array="${array_spec}%${MAX_CONCURRENT_CPU_NODES}"
        --output="$OUTPUT_ROOT/slurm_logs/cpu_nodepack/%x-%A_%a.out"
        --error="$OUTPUT_ROOT/slurm_logs/cpu_nodepack/%x-%A_%a.err"
        --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},DATA_ROOT=${DATA_ROOT},REFERENCE_PREP_OUTPUT_ROOT=${REFERENCE_PREP_OUTPUT_ROOT},OUTPUT_ROOT=${OUTPUT_ROOT},CPU_JOB_MANIFEST=${manifest},CPU_WORKERS_PER_NODE=${CPU_WORKERS_PER_NODE},IMMUNE_HEALTH_ENV_ROOT=${ENV_ROOT},ENVIRONMENT_ACTIVATION_SCRIPT=${ACTIVATION_SCRIPT},CPU_NODEPACK_LOG_ROOT=${OUTPUT_ROOT}/slurm_logs/cpu_nodepack"
    )
    if [[ -n "$dependency" ]]; then
        sbatch_args+=(--dependency="afterok:${dependency}")
    fi

    printf 'CPU manifest=%s node_array=%s\n' "$manifest" "$array_spec" >&2
    local submission_output
    if ! submission_output=$(sbatch "${sbatch_args[@]}" \
        "${CPU_OUTER_RESOURCE_ARGS[@]}" \
        "$PROJECT_ROOT/slurm/cpu_nodepack.sbatch" "${runner_args[@]}"); then
        printf 'CPU sbatch failed for %s\n' "$manifest" >&2
        return 1
    fi

    local submitted
    if ! submitted=$(_gefion_parse_sbatch_job_id "$submission_output"); then
        printf 'CPU sbatch returned no unique parsable job ID for %s. Raw stdout follows:\n%s\n' \
            "$manifest" "$submission_output" >&2
        return 1
    fi
    printf '%s\n' "$submitted"
}

submit_gpu_pack() {
    local manifest="$1"
    local dependency="${2:-}"
    local indices="${3:-}"
    local walltime="${4:-$GPU_WALLTIME}"
    local array_spec
    local -a runner_args=()
    local -a sbatch_args

    if ! array_spec=$(plan_array_spec \
        "$manifest" "$GPU_WORKERS_PER_NODE" 1 "$indices"); then
        printf 'GPU node-pack planning failed for %s\n' "$manifest" >&2
        return 1
    fi
    if [[ ! "$array_spec" =~ ^0-[0-9]+$ ]]; then
        printf 'GPU planner returned an invalid array for %s: %q\n' \
            "$manifest" "$array_spec" >&2
        return 1
    fi
    if [[ -n "$indices" ]]; then
        runner_args=(--indices "$indices")
    fi

    sbatch_args=(
        --parsable
        --chdir="$PROJECT_ROOT"
        --account="$GEFION_ACCOUNT"
        --partition="$GEFION_GPU_PARTITION"
        --time="$walltime"
        --nodes=1
        --exclusive
        --ntasks-per-node="$GPU_WORKERS_PER_NODE"
        --cpus-per-task="$CPUS_PER_GPU_WORKER"
        --mem="$GPU_NODE_MEMORY"
        --array="${array_spec}%${MAX_CONCURRENT_GPU_NODES}"
        --output="$OUTPUT_ROOT/slurm_logs/gpu_nodepack/%x-%A_%a.out"
        --error="$OUTPUT_ROOT/slurm_logs/gpu_nodepack/%x-%A_%a.err"
        --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},OUTPUT_ROOT=${OUTPUT_ROOT},TRIPSO_JOB_MANIFEST=${manifest},TRIPSO_WORKERS_PER_NODE=${GPU_WORKERS_PER_NODE},IMMUNE_HEALTH_ENV_ROOT=${ENV_ROOT},ENVIRONMENT_ACTIVATION_SCRIPT=${ACTIVATION_SCRIPT},TRIPSO_NODEPACK_LOG_ROOT=${OUTPUT_ROOT}/slurm_logs/gpu_nodepack"
    )
    if [[ -n "$dependency" ]]; then
        sbatch_args+=(--dependency="afterok:${dependency}")
    fi

    printf 'GPU manifest=%s node_array=%s\n' "$manifest" "$array_spec" >&2
    local submission_output
    if ! submission_output=$(sbatch "${sbatch_args[@]}" \
        "${GPU_OUTER_RESOURCE_ARGS[@]}" \
        "$PROJECT_ROOT/slurm/tripso_nodepack.sbatch" "${runner_args[@]}"); then
        printf 'GPU sbatch failed for %s\n' "$manifest" >&2
        return 1
    fi

    local submitted
    if ! submitted=$(_gefion_parse_sbatch_job_id "$submission_output"); then
        printf 'GPU sbatch returned no unique parsable job ID for %s. Raw stdout follows:\n%s\n' \
            "$manifest" "$submission_output" >&2
        return 1
    fi
    printf '%s\n' "$submitted"
}

_gefion_write_reference_job_ids() {
    local destination="$OUTPUT_ROOT/reference_prep_job_ids.txt"
    local temporary="${destination}.tmp.$$"
    printf '%s\n' \
        "setup=${CPU_SETUP_JOB:-}" \
        "features=${CPU_FEATURES_JOB:-}" \
        "materialize=${CPU_MATERIALIZE_JOB:-}" \
        "tokenize=${CPU_TOKENIZE_JOB:-}" \
        "bind=${CPU_BIND_JOB:-}" \
        > "$temporary"
    mv -- "$temporary" "$destination"
}

gefion_submit_reference_prep() {
    local state_file="$OUTPUT_ROOT/reference_prep_job_ids.txt"
    if [[ -s "$state_file" ]]; then
        printf 'Reference-preparation job state already exists; no jobs submitted:\n' >&2
        sed -n '1,20p' "$state_file" >&2
        return 2
    fi

    CPU_SETUP_JOB=$(submit_cpu_pack "$PACKED_REF_DIR/setup.jsonl") || return
    _gefion_write_reference_job_ids
    CPU_FEATURES_JOB=$(submit_cpu_pack \
        "$REF_MANIFEST_DIR/features.jsonl" "$CPU_SETUP_JOB") || return
    _gefion_write_reference_job_ids
    CPU_MATERIALIZE_JOB=$(submit_cpu_pack \
        "$REF_MANIFEST_DIR/materialize.jsonl" "$CPU_FEATURES_JOB") || return
    _gefion_write_reference_job_ids
    CPU_TOKENIZE_JOB=$(submit_cpu_pack \
        "$PACKED_REF_DIR/tokenize.jsonl" "$CPU_MATERIALIZE_JOB") || return
    _gefion_write_reference_job_ids
    CPU_BIND_JOB=$(submit_cpu_pack \
        "$PACKED_REF_DIR/bind.jsonl" "$CPU_TOKENIZE_JOB") || return
    _gefion_write_reference_job_ids

    export CPU_SETUP_JOB CPU_FEATURES_JOB CPU_MATERIALIZE_JOB
    export CPU_TOKENIZE_JOB CPU_BIND_JOB
    printf 'Submitted reference-preparation dependency chain:\n'
    sed -n '1,20p' "$state_file"
}

_gefion_read_job_id() {
    local key="$1"
    local state_file="$2"
    local value
    value=$(sed -n "s/^${key}=//p" "$state_file")
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
        printf 'Missing or invalid %s job ID in %s\n' "$key" "$state_file" >&2
        return 1
    fi
    printf '%s\n' "$value"
}

gefion_submit_stage1_sentinel() {
    local prep_state="$OUTPUT_ROOT/reference_prep_job_ids.txt"
    local sentinel_state="$OUTPUT_ROOT/stage1_sentinel_job_id.txt"
    if [[ -s "$sentinel_state" ]]; then
        printf 'Stage-1 sentinel state already exists; no jobs submitted:\n' >&2
        sed -n '1,20p' "$sentinel_state" >&2
        return 2
    fi

    CPU_BIND_JOB=$(_gefion_read_job_id bind "$prep_state") || return
    STAGE1_SENTINEL_INDICES='0-5,60-65'
    STAGE1_SENTINEL_JOB=$(submit_gpu_pack \
        "$STAGE1_MANIFEST" "$CPU_BIND_JOB" "$STAGE1_SENTINEL_INDICES" \
        "$SENTINEL_WALLTIME") || return
    export CPU_BIND_JOB STAGE1_SENTINEL_INDICES STAGE1_SENTINEL_JOB
    printf '%s\n' "sentinel=$STAGE1_SENTINEL_JOB" > "$sentinel_state"
    printf 'Submitted Stage-1 sentinel job %s\n' "$STAGE1_SENTINEL_JOB"
}

gefion_submit_stage1_remainder() {
    local sentinel_state="$OUTPUT_ROOT/stage1_sentinel_job_id.txt"
    local stage1_state="$OUTPUT_ROOT/stage1_job_ids.txt"
    if [[ -s "$stage1_state" ]]; then
        printf 'Complete Stage-1 job state already exists; no jobs submitted:\n' >&2
        sed -n '1,20p' "$stage1_state" >&2
        return 2
    fi

    CPU_BIND_JOB=$(
        _gefion_read_job_id bind "$OUTPUT_ROOT/reference_prep_job_ids.txt"
    ) || return
    STAGE1_SENTINEL_JOB=$(
        _gefion_read_job_id sentinel "$sentinel_state"
    ) || return
    STAGE1_REMAINDER_INDICES='6-59,66-149'
    STAGE1_REMAINDER_JOB=$(submit_gpu_pack \
        "$STAGE1_MANIFEST" "$CPU_BIND_JOB" "$STAGE1_REMAINDER_INDICES") || return
    STAGE1_ALL_JOB_IDS="$STAGE1_SENTINEL_JOB:$STAGE1_REMAINDER_JOB"
    export CPU_BIND_JOB STAGE1_SENTINEL_JOB STAGE1_REMAINDER_INDICES
    export STAGE1_REMAINDER_JOB STAGE1_ALL_JOB_IDS
    printf '%s\n' \
        "sentinel=$STAGE1_SENTINEL_JOB" \
        "remainder=$STAGE1_REMAINDER_JOB" \
        > "$stage1_state"
    printf 'Submitted Stage-1 remainder job %s\n' "$STAGE1_REMAINDER_JOB"
}
