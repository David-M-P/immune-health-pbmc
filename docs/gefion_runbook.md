# Gefion end-to-end runbook

This is the least-friction route when raw reference preparation, tokenization,
training, projection, and downstream CPU work all run on Gefion. The Slurm account
is fixed to `cu_0071`. The current deployment values are recorded in
`gefion_env.txt`; a generic template remains below only for future deployments.

Manifest generators never submit jobs. Every submission below is explicit and
restartable. GPU work runs eight independent one-GPU rows per billed node; it is
not an eight-GPU DDP model. CPU work can also be packed into one exclusive node.

## Copy/paste boundary for the current `cu_0071` deployment

Sections 1–3 are development-cluster/SFTP preparation and are retained for
provenance. They do **not** need to be pasted into a Gefion terminal after the
repository, environment archive, source-data archive, and checksum files have
arrived.

For the current deployment, every Gefion-side command needed through the end of
Stage 1 is in Sections 4–10. The concrete scheduler and path values are stored in
the repository file `gefion_env.txt`; do not reconstruct that file by copying a
chat message. Start every new Gefion login with:

```bash
cd /dcai/users/pesdav/cu_0071/immune_health/immune-health-pbmc
source ./slurm/load_gefion.sh
```

The runbook's multiline blocks can therefore be copied from the Gefion checkout
itself (for example with `less docs/gefion_runbook.md`) or viewed in the internal
GitLab web interface. No command from an external machine or chat is required.
The exception is site information that only Gefion can report, such as current
QOS limits; the commands that query it are included below.

`gefion_env.txt` must be committed and pushed with this runbook. If `git status
--short` shows it as untracked on the development cluster, the Gefion clone will
not receive it merely because other repository files were synced.

## Current execution boundary

The implemented path can run, without scientific choices, through:

1. all 412 reference-preparation rows;
2. all 150 Stage-1 Base models;
3. frozen reference and inner-validation projection for those models; and
4. the downstream primitives once a fully hashed candidate plan is supplied.

There are two deliberate manual gates:

- choose two sampler/HVG configurations per lineage after Stage 1, using only
  fixed inner-validation donors;
- choose one winner per lineage after Stage 2, again without consulting the outer
  LODO query.

No implemented command currently writes those choices into the Stage-2/3 YAML,
and no implemented command builds the production downstream candidate plan. Do
not treat the commands after either gate as unattended continuation.

## 1. Commit the execution code before moving it

The Gefion clone must contain the node-pack launchers and the current scientific
configuration. On the development cluster, review and commit the intended changes
before pushing to the GitLab remote used by Gefion:

```bash
cd /faststorage/project/immunehealth/Projects/david/immune-health-pbmc

git status --short
git diff --check

# Stage only files that you have reviewed. Do not add data, environments, runs,
# checkpoints, Arrow directories, or logs.
git add "<REVIEWED_REPOSITORY_FILE_1>" "<REVIEWED_REPOSITORY_FILE_2>"
git commit -m "Add Gefion node-packed production workflow"
git push "<YOUR_GITLAB_REMOTE>" "<YOUR_BRANCH>"
git rev-parse HEAD
```

There is no Gefion GitLab URL configured in this checkout, so it is not safe to
invent one here. Record the exact commit and check out that commit on Gefion.

## 2. Rebuild and validate the transferable environment

Do not reuse an archive made with `conda-pack --ignore-missing-files`. The original
local prefix had pip-overwritten conda records for NumPy, Packaging, and Requests;
such an archive can contain a mixed, non-importable package tree. The environment
definition now pins the conda-side versions to the exact TRIPSO pip versions.

Create a new prefix and a versioned archive. Do not overwrite the currently working
local prefix or an earlier archive while validating the replacement:

```bash
set -euo pipefail

export LOCAL_PROJECT_ROOT=/faststorage/project/immunehealth/Projects/david/immune-health-pbmc
export TRANSFER_ROOT=/faststorage/project/immunehealth/Projects/david/sftp_staging/gefion_v2
export REBUILD_PREFIX="$LOCAL_PROJECT_ROOT/.conda_isolated/immune-health-tripso-transfer-v2"
export ENV_ARCHIVE_NAME=immune-health-tripso-linux-x86_64-v2.tar.gz

mkdir -p "$TRANSFER_ROOT"
test ! -e "$REBUILD_PREFIX"
test ! -e "$TRANSFER_ROOT/$ENV_ARCHIVE_NAME"

cd "$LOCAL_PROJECT_ROOT"
mamba env create \
  --prefix "$REBUILD_PREFIX" \
  --file environment.yml

"$REBUILD_PREFIX/bin/python" -c \
  'import numpy, packaging, requests, torch; assert numpy.__version__ == "1.25.0"; assert packaging.__version__ == "25.0"; assert requests.__version__ == "2.32.3"; assert torch.__version__ == "2.4.1"; print("source imports passed")'

"$REBUILD_PREFIX/bin/conda-pack" \
  --prefix "$REBUILD_PREFIX" \
  --output "$TRANSFER_ROOT/$ENV_ARCHIVE_NAME" \
  --compress-level 1 \
  --n-threads 4
```

Do not add `--dest-prefix`, `--ignore-missing-files`, or
`--ignore-editable-packages`. If packing reports a clobbered conda package, stop
and align that package's conda pin with the required pip version.

Test relocation before upload:

```bash
tar -tzf "$TRANSFER_ROOT/$ENV_ARCHIVE_NAME" \
  bin/activate bin/conda-unpack bin/python conda-meta/history
gzip -t "$TRANSFER_ROOT/$ENV_ARCHIVE_NAME"

VERIFY_PREFIX=$(mktemp -d /tmp/immune-health-packed-env.XXXXXX)
tar -xzf "$TRANSFER_ROOT/$ENV_ARCHIVE_NAME" -C "$VERIFY_PREFIX"
"$VERIFY_PREFIX/bin/python" "$VERIFY_PREFIX/bin/conda-unpack"
"$VERIFY_PREFIX/bin/python" -c \
  'import numpy, packaging, requests, torch; assert numpy.__version__ == "1.25.0"; assert packaging.__version__ == "25.0"; assert requests.__version__ == "2.32.3"; assert torch.__version__ == "2.4.1"; print("relocated imports passed")'

MPLCONFIGDIR=/tmp/immune_health_mpl \
NUMBA_CACHE_DIR=/tmp/immune_health_numba \
"$VERIFY_PREFIX/bin/python" scripts/validate_tripso_environment.py \
  --vendor-root tripso_code/tripso \
  --smoke-mode mock \
  --json-output "$TRANSFER_ROOT/tripso_environment_relocated_mock.json"

REPACK_VALIDATION_DIR="$TRANSFER_ROOT/repack_validation_v2"
REPACK_SMOKE_DIR="$REPACK_VALIDATION_DIR/real_tripso_base_smoke"
test ! -e "$REPACK_VALIDATION_DIR"
mkdir -p "$REPACK_VALIDATION_DIR"

REPACK_SMOKE_JSON=$("$VERIFY_PREFIX/bin/python" -c \
  'import json,sys; print(json.dumps(sys.argv[1:]))' \
  "$VERIFY_PREFIX/bin/python" "$LOCAL_PROJECT_ROOT/scripts/run_real_tripso_smoke.py" \
  --output-dir "$REPACK_SMOKE_DIR" \
  --accelerator cpu)

PYTHONPATH="$LOCAL_PROJECT_ROOT/src" \
MPLCONFIGDIR="$REPACK_VALIDATION_DIR/matplotlib" \
NUMBA_CACHE_DIR="$REPACK_VALIDATION_DIR/numba" \
"$VERIFY_PREFIX/bin/python" "$LOCAL_PROJECT_ROOT/scripts/validate_tripso_environment.py" \
  --vendor-root "$LOCAL_PROJECT_ROOT/tripso_code/tripso" \
  --smoke-mode real \
  --real-smoke-command-json "$REPACK_SMOKE_JSON" \
  --real-smoke-cwd "$LOCAL_PROJECT_ROOT" \
  --json-output "$REPACK_VALIDATION_DIR/tripso_environment.json"

"$VERIFY_PREFIX/bin/python" -c \
  'import json,sys; r=json.load(open(sys.argv[1])); assert r["environment_passed"] is True and r["real_end_to_end_passed"] is True; print("relocated real CPU smoke passed")' \
  "$REPACK_VALIDATION_DIR/tripso_environment.json"

# VERIFY_PREFIX was created by the command above and is the only removal target.
rm -rf -- "$VERIFY_PREFIX"

(cd "$TRANSFER_ROOT" && \
  sha256sum "$ENV_ARCHIVE_NAME" > "$ENV_ARCHIVE_NAME.sha256")
```

The primary hybrid/Base analysis does not need the external full Geneformer
checkout. Transfer it only if the separate 12-layer sensitivity will be run.

## 3. Bundle only the source data needed by the current five lineages

Because all preparation will run on Gefion, transfer the five merged source H5ADs
and curated GMT. Do not materialize the 3k/9k H5ADs or tokenize locally.

```bash
set -euo pipefail

export LOCAL_DATA_ROOT=/faststorage/project/immunehealth/Projects/david/data/intermediate_data

SOURCE_PATHS=(
  reference_lineages/merged/B_cells/merged.h5ad
  reference_lineages/merged/NK_ILC/merged.h5ad
  reference_lineages/merged/Monocytes/merged.h5ad
  reference_lineages/merged/CD4_like/merged.h5ad
  reference_lineages/merged/CD8_like/merged.h5ad
  gene_programs/v1/gene_programs_curated.gmt
)

(cd "$LOCAL_DATA_ROOT" && sha256sum "${SOURCE_PATHS[@]}") \
  > "$TRANSFER_ROOT/reference_source_files.sha256"

tar -C "$LOCAL_DATA_ROOT" \
  -cf "$TRANSFER_ROOT/reference_source_data.tar" \
  "${SOURCE_PATHS[@]}"

(cd "$TRANSFER_ROOT" && \
  sha256sum reference_source_data.tar > reference_source_data.tar.sha256)
```

The source bundle is about 28 GiB. The H5ADs are already internally compressed,
so an uncompressed tar avoids spending substantial CPU for little benefit. Check
Gefion quota before transfer: full preparation writes 160 materialized H5ADs and
160 Arrow datasets, and all-cell projections can be much larger.

Upload the environment archive, both archive checksums, the source tar, and the
per-file checksum manifest by SFTP. Use `reput` for resumable large transfers:

```text
sftp> lcd /faststorage/project/immunehealth/Projects/david/sftp_staging/gefion_v2
sftp> cd /to_gefion
sftp> -mkdir immune_health_v2
sftp> cd immune_health_v2
sftp> reput immune-health-tripso-linux-x86_64-v2.tar.gz
sftp> put   immune-health-tripso-linux-x86_64-v2.tar.gz.sha256
sftp> reput reference_source_data.tar
sftp> put   reference_source_data.tar.sha256
sftp> put   reference_source_files.sha256
```

## 4. One-time Gefion setup

Load the concrete deployment file, all derived manifest paths, and the submission
functions with one command. This opens no editor and submits no jobs:

```bash
cd /dcai/users/pesdav/cu_0071/immune_health/immune-health-pbmc
source ./slurm/load_gefion.sh

test "$GEFION_ACCOUNT" = cu_0071
test "$GEFION_CPU_PARTITION" = defq
test "$GEFION_GPU_PARTITION" = defq
test "$CPU_WALLTIME" = 7-00:00:00
test "$SENTINEL_WALLTIME" = 7-00:00:00
test "$GPU_WALLTIME" = 7-00:00:00
test "$PROJECTION_WALLTIME" = 7-00:00:00
test "$MAX_CONCURRENT_CPU_NODES" = 1
test "$MAX_CONCURRENT_GPU_NODES" = 1
test "${GPU_OUTER_RESOURCE_ARGS[*]}" = "--gpus-per-node=8"
test "${CPU_OUTER_RESOURCE_ARGS[*]}" = "--gpus-per-node=8"

printf '%s\n' \
  "PROJECT_ROOT=$PROJECT_ROOT" \
  "WORK_ROOT=$WORK_ROOT" \
  "INCOMING_ROOT=$INCOMING_ROOT" \
  "OUTPUT_ROOT=$OUTPUT_ROOT" \
  "ENV_ROOT=$ENV_ROOT" \
  "DATA_ROOT=$DATA_ROOT"
```

The seven-day values are kill ceilings: an allocation ends as soon as its work
finishes. Both concurrency limits initially allow one billed node at a time.

Before unpacking, `$INCOMING_ROOT` must be the directory that directly contains
the five transferred payload/checksum files. Confirm that entirely on Gefion:

```bash
ls -lh \
  "$INCOMING_ROOT/immune-health-tripso-linux-x86_64-v2.tar.gz" \
  "$INCOMING_ROOT/immune-health-tripso-linux-x86_64-v2.tar.gz.sha256" \
  "$INCOMING_ROOT/reference_source_data.tar" \
  "$INCOMING_ROOT/reference_source_data.tar.sha256" \
  "$INCOMING_ROOT/reference_source_files.sha256"
```

If these files were placed in a subdirectory such as `immune_health_v2`, change
only the `INCOMING_ROOT` assignment in the repository's `gefion_env.txt`, commit
that path as deployment provenance if appropriate, and source it again. Do not
move or unpack files until all five paths above resolve.

If the checkout does not exist yet, clone the internal GitLab repository and pin
the reviewed commit before unpacking data:

```bash
test ! -e "$PROJECT_ROOT"
mkdir -p "$(dirname "$PROJECT_ROOT")"
git clone "<YOUR_INTERNAL_GITLAB_URL>" "$PROJECT_ROOT"
git -C "$PROJECT_ROOT" checkout "<REVIEWED_COMMIT_SHA>"
```

Discover the unresolved scheduler values on Gefion rather than guessing:

```bash
sacctmgr show associations \
  where user="$USER" account=cu_0071 \
  format=Account,Partition,QOS

sinfo -o '%P %a %l %D %c %m %G'
```

The versioned `configs/clusters/gefion.yaml` is a portable template and
intentionally contains placeholders. Do not open or edit it during this run.
The setup command below writes a separate, placeholder-free runtime-provenance
file directly from the already loaded values. Submission helpers pass those same
values explicitly to Slurm.

Verify and unpack the transferred payloads into new destinations:

```bash
cd "$INCOMING_ROOT"
sha256sum -c immune-health-tripso-linux-x86_64-v2.tar.gz.sha256
sha256sum -c reference_source_data.tar.sha256

test ! -e "$ENV_ROOT"
mkdir -p "$(dirname "$ENV_ROOT")"
mkdir "$ENV_ROOT"
tar -xzf immune-health-tripso-linux-x86_64-v2.tar.gz -C "$ENV_ROOT"
"$ENV_ROOT/bin/python" "$ENV_ROOT/bin/conda-unpack"

test ! -e "$DATA_ROOT/reference_lineages/merged"
mkdir -p "$DATA_ROOT"
tar -xf reference_source_data.tar -C "$DATA_ROOT"
(cd "$DATA_ROOT" && \
  sha256sum -c "$INCOMING_ROOT/reference_source_files.sha256")

mkdir -p \
  "$OUTPUT_ROOT/validation" \
  "$OUTPUT_ROOT/configs" \
  "$MANIFEST_ROOT" \
  "$OUTPUT_ROOT/slurm_logs/cpu_nodepack" \
  "$OUTPUT_ROOT/slurm_logs/gpu_nodepack" \
  "$PROJECT_ROOT/slurm/logs"

"$PY" scripts/write_gefion_runtime_config.py \
  --output "$OUTPUT_ROOT/configs/gefion.runtime.yaml" \
  --project-root "$PROJECT_ROOT" \
  --work-root "$WORK_ROOT" \
  --data-root "$DATA_ROOT" \
  --environment-root "$ENV_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --run-id "$RUN_ID" \
  --account "$GEFION_ACCOUNT" \
  --cpu-partition "$GEFION_CPU_PARTITION" \
  --gpu-partition "$GEFION_GPU_PARTITION" \
  --activation-script "$ACTIVATION_SCRIPT" \
  --cpu-walltime "$CPU_WALLTIME" \
  --gpu-walltime "$GPU_WALLTIME" \
  --projection-walltime "$PROJECTION_WALLTIME" \
  --cpu-memory "$CPU_NODE_MEMORY" \
  --gpu-memory "$GPU_NODE_MEMORY" \
  --cpu-workers "$CPU_WORKERS_PER_NODE" \
  --cpu-cpus-per-worker "$CPU_CPUS_PER_WORKER" \
  --gpu-workers "$GPU_WORKERS_PER_NODE" \
  --gpu-cpus-per-worker "$CPUS_PER_GPU_WORKER" \
  --cpu-node-concurrency "$MAX_CONCURRENT_CPU_NODES" \
  --gpu-node-concurrency "$MAX_CONCURRENT_GPU_NODES" \
  --gpus-per-node 8

if grep -nE '<[A-Z][A-Z0-9_]*>' \
  "$OUTPUT_ROOT/configs/gefion.runtime.yaml"; then
  printf 'Unexpected placeholder in generated runtime configuration\n' >&2
  exit 1
fi

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -c \
  'import numpy, packaging, requests, torch; print(numpy.__version__, packaging.__version__, requests.__version__, torch.__version__)'
git -C "$PROJECT_ROOT" status --porcelain
git -C "$PROJECT_ROOT" rev-parse HEAD
```

The checkout should be clean and its commit should equal the reviewed commit from
Step 1.

## 5. Validate the Gefion GPU boundary

Run this in one interactive eight-GPU allocation. Adjust only the site resource
values already represented by the variables above:

```bash
cd "$PROJECT_ROOT"

salloc \
  --account="$GEFION_ACCOUNT" \
  --partition="$GEFION_GPU_PARTITION" \
  --time=01:00:00 \
  --nodes=1 \
  --exclusive \
  --ntasks-per-node=8 \
  --cpus-per-task="$CPUS_PER_GPU_WORKER" \
  --mem="$GPU_NODE_MEMORY" \
  "${GPU_OUTER_RESOURCE_ARGS[@]}"
```

Inside that allocation, prove that each rank sees exactly one GPU:

```bash
source "$ACTIVATION_SCRIPT"
test "$(command -v python)" = "$ENV_ROOT/bin/python"

srun \
  --nodes=1 \
  --ntasks=8 \
  --cpus-per-task=1 \
  --gpus-per-task=1 \
  --gpu-bind=single:1 \
  --exact \
  --wait=0 \
  python -c \
  'import os, torch; rank=os.environ["SLURM_PROCID"]; n=torch.cuda.device_count(); print("rank={} visible={} count={}".format(rank, os.environ.get("CUDA_VISIBLE_DEVICES"), n), flush=True); assert n == 1'
```

Then run the real one-step Base smoke and write the environment report at the
exact path required by every training row:

```bash
export SMOKE_DIR="$OUTPUT_ROOT/validation/real_tripso_base_smoke_v1"
test ! -e "$SMOKE_DIR"

REAL_SMOKE_JSON=$("$PY" -c \
  'import json,sys; print(json.dumps(sys.argv[1:]))' \
  "$PY" "$PROJECT_ROOT/scripts/run_real_tripso_smoke.py" \
  --output-dir "$SMOKE_DIR" \
  --accelerator cuda)

srun \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="$CPUS_PER_GPU_WORKER" \
  --gpus-per-task=1 \
  --gpu-bind=single:1 \
  --exact \
  "$PY" scripts/validate_tripso_environment.py \
    --vendor-root tripso_code/tripso \
    --smoke-mode real \
    --real-smoke-command-json "$REAL_SMOKE_JSON" \
    --real-smoke-cwd "$PROJECT_ROOT" \
    --json-output "$OUTPUT_ROOT/validation/tripso_environment.json"

"$PY" -c \
  'import json,os; p=os.path.join(os.environ["OUTPUT_ROOT"],"validation","tripso_environment.json"); r=json.load(open(p)); assert r["environment_passed"] is True; assert r["real_end_to_end_passed"] is True; print("Gefion environment and real CUDA smoke passed")'
```

Exit the interactive allocation after both checks pass.

## 6. Generate all preparation and Stage-1 manifests on Gefion

```bash
cd "$PROJECT_ROOT"

export REF_MANIFEST_DIR="$MANIFEST_ROOT/reference_prep"
export TRAIN_MANIFEST_DIR="$MANIFEST_ROOT/training"
export PACKED_REF_DIR="$REF_MANIFEST_DIR/packed"

mkdir -p "$REF_MANIFEST_DIR" "$TRAIN_MANIFEST_DIR" "$PACKED_REF_DIR"

"$PY" scripts/generate_reference_prep_jobs.py \
  --config configs/experiments/reference_preparation.yaml \
  --output-dir "$REF_MANIFEST_DIR"

"$PY" scripts/generate_job_manifests.py \
  --config configs/experiments/tripso_lodo.yaml \
  --stage stage1 \
  --output-dir "$TRAIN_MANIFEST_DIR" \
  --seed 42

"$PY" scripts/combine_job_manifests.py \
  --input "$REF_MANIFEST_DIR/visits.jsonl" \
  --input "$REF_MANIFEST_DIR/final_fold.jsonl" \
  --output "$PACKED_REF_DIR/setup.jsonl"

"$PY" scripts/combine_job_manifests.py \
  --input "$REF_MANIFEST_DIR/lodo_tokenize.jsonl" \
  --input "$REF_MANIFEST_DIR/final_tokenize.jsonl" \
  --output "$PACKED_REF_DIR/tokenize.jsonl"

"$PY" scripts/combine_job_manifests.py \
  --input "$REF_MANIFEST_DIR/lodo_bind.jsonl" \
  --input "$REF_MANIFEST_DIR/final_bind.jsonl" \
  --output "$PACKED_REF_DIR/bind.jsonl"

"$PY" - "$REF_MANIFEST_DIR" "$TRAIN_MANIFEST_DIR/stage1.jsonl" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
expected = {
    "visits": 1,
    "final_fold": 1,
    "features": 30,
    "materialize": 160,
    "lodo_tokenize": 150,
    "lodo_bind": 50,
    "final_tokenize": 10,
    "final_bind": 10,
}
for name, count in expected.items():
    rows = [json.loads(line) for line in (root / f"{name}.jsonl").read_text().splitlines()]
    assert len(rows) == count and all(row["runnable"] is True for row in rows), name
stage1 = [json.loads(line) for line in pathlib.Path(sys.argv[2]).read_text().splitlines()]
assert len(stage1) == 150 and all(row["runnable"] is True for row in stage1)
print("Preparation=412 runnable rows; Stage 1=150 runnable rows")
PY
```

This prepares adaptation, fixed inner-validation, outer-query, and final
all-healthy artifacts. Query materialization/tokenization does not unseal query
evaluation: query cells are excluded from feature fitting and training, and no
query projection job is generated at this point.

## 7. Define reusable node-packed submission helpers

`slurm/load_gefion.sh` has already loaded the repository implementation. Confirm
the functions are present:

```bash
type plan_array_spec submit_cpu_pack submit_gpu_pack
```

This command submits no jobs and normally prints only the three `type`
descriptions. The low-level implementation excerpt below is audit reference;
do not paste it into the shell. The sourced implementation also provides the
one-command stage helpers used below. It calculates the outer node-array range
from the manifest, so row counts are never confused with node-block counts. Its
`sbatch` parser tolerates informational text that a Gefion wrapper or plugin may
add around the one numeric `--parsable` record.

```bash
_gefion_parse_sbatch_job_id() {
  local raw_output="$1"
  local line
  local parsed=""

  while IFS= read -r line; do
    line="${line%$'\r'}"
    if [[ "$line" =~ ^([0-9]+)(\;[^[:space:]]+)?$ ]]; then
      if [[ -n "$parsed" && "$parsed" != "${BASH_REMATCH[1]}" ]]; then
        return 1
      fi
      parsed="${BASH_REMATCH[1]}"
    fi
  done <<< "$raw_output"

  [[ -n "$parsed" ]] || return 1
  printf '%s\n' "$parsed"
}

plan_array_spec() {
  local manifest="$1"
  local workers="$2"
  local expected_gpus="$3"
  local indices="${4:-}"
  local args=(
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
  local runner_args=()
  local sbatch_args

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
  if ! submission_output=$(sbatch \
    "${sbatch_args[@]}" "${CPU_OUTER_RESOURCE_ARGS[@]}" \
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
  local runner_args=()
  local sbatch_args

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
  if ! submission_output=$(sbatch \
    "${sbatch_args[@]}" "${GPU_OUTER_RESOURCE_ARGS[@]}" \
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
```

All generic launcher headers retain the local `immunehealth` default. The explicit
`--account="$GEFION_ACCOUNT"` above overrides it with `cu_0071`.

### If a helper reports an invalid or unparsable job ID

Do not immediately call the helper again. `sbatch` may have accepted the job and
then printed an informational line that confused an older helper. First inspect
the scheduler using the commands below; these checks do not submit or cancel
anything:

```bash
squeue -u "$USER" \
  -o '%.18i %.18F %.8K %.30j %.10T %.10M %.10l %R'

sacct -X -S "$(date +%F)" -u "$USER" \
  --format=JobIDRaw,JobName%30,State,Submit,Start,Elapsed,Timelimit,ExitCode
```

For an array, `%F` in `squeue` is the numeric array job ID that should be used as
a dependency (`%A` remains the corresponding filename substitution). If a newly
submitted `immune-health-cpu-nodepack` or
`immune-health-tripso-nodepack` is present, recover that ID; do not resubmit the
same manifest. The job-specific outer logs also contain `%A` in their filename:

```bash
find "$OUTPUT_ROOT/slurm_logs" -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' \
  | sort -r | head -40
```

The repository helper now accepts the standard `JOB_ID`, `JOB_ID;CLUSTER`, and
those records surrounded by nonnumeric informational text. It deliberately
rejects output containing no numeric parsable record or two different job IDs.
When reporting a remaining problem, preserve the complete text after `Raw stdout
follows:` because it determines whether the submission was accepted.

## 8. Submit all CPU reference preparation

Inspect the plans first:

```bash
for manifest in \
  "$PACKED_REF_DIR/setup.jsonl" \
  "$REF_MANIFEST_DIR/features.jsonl" \
  "$REF_MANIFEST_DIR/materialize.jsonl" \
  "$PACKED_REF_DIR/tokenize.jsonl" \
  "$PACKED_REF_DIR/bind.jsonl"
do
  "$PY" slurm/run_manifest_nodepack.py \
    --manifest "$manifest" \
    --workers-per-node "$CPU_WORKERS_PER_NODE" \
    --expected-visible-gpus 0 \
    --plan-only
done
```

With the current four CPU workers, the outer arrays are `0-0`, `0-7`, `0-39`,
`0-39`, and `0-14`. Submit the entire dependency chain with one command:

```bash
gefion_submit_reference_prep
```

The helper writes `$OUTPUT_ROOT/reference_prep_job_ids.txt` after each successful
`sbatch`, so a terminal disconnect cannot lose an already returned job ID. It
refuses to submit when that state file already exists, preventing accidental
duplication.

The expected final counts are:

```bash
find "$REFERENCE_PREP_OUTPUT_ROOT" -name .done.json | wc -l
# 192: visits + final fold + features + materializations

find "$OUTPUT_ROOT/tripso_inputs" -name .done.json | wc -l
# 220: tokenizations + fold bindings

find "$OUTPUT_ROOT/tripso_inputs" -type d -name tokenized.dataset | wc -l
# 160

find "$OUTPUT_ROOT/tripso_inputs" -name fold_input.json | wc -l
# 60: 50 LODO + 10 final all-healthy
```

A `.failed.json` can remain after a successful retry; the content-verified
`.done.json` is authoritative. Check `sacct` and the per-worker logs under
`$OUTPUT_ROOT/slurm_logs/cpu_nodepack` before proceeding.

## 9. Run Stage 1: sentinel, then the remaining 138 models

The 12-model sentinel covers all six sampler/HVG configurations for the AIDA-v2
fold in B cells and Monocytes:

```bash
export STAGE1_MANIFEST="$TRAIN_MANIFEST_DIR/stage1.jsonl"
export STAGE1_SENTINEL_INDICES='0-5,60-65'

"$PY" slurm/run_manifest_nodepack.py \
  --manifest "$STAGE1_MANIFEST" \
  --workers-per-node 8 \
  --indices "$STAGE1_SENTINEL_INDICES" \
  --plan-only

gefion_submit_stage1_sentinel
```

This uses two node-array elements. Wait for it, inspect all 12 model manifests,
GPU memory, node memory, runtime, sampler audit, sequence QC, and failure markers.
The current seven-day `GPU_WALLTIME` may be retained; if you choose to shorten
it later, base that choice on the slow tail rather than the mean.

Monitor and verify the sentinel with commands already available on Gefion:

```bash
squeue -j "$STAGE1_SENTINEL_JOB" \
  -o '%.18i %.12P %.24j %.8T %.10M %.10l %.6D %R'

sacct -j "$STAGE1_SENTINEL_JOB" \
  --units=G \
  --format=JobID,JobName%30,State,Elapsed,Timelimit,AllocTRES%50,MaxRSS,ExitCode

find "$OUTPUT_ROOT/tripso/stage1" -name model_manifest.json -print | sort
find "$OUTPUT_ROOT/tripso/stage1" -path '*/checkpoints/last.ckpt' -print | sort
find "$OUTPUT_ROOT/tripso/stage1" -name .failed.json -print | sort
find "$OUTPUT_ROOT/slurm_logs/gpu_nodepack" -type f -print | sort
```

Proceed only when `sacct` shows both sentinel array elements completed
successfully, exactly 12 sentinel `model_manifest.json` files and 12 matching
`last.ckpt` files exist, and the worker logs have been reviewed. A seven-day
value is only the kill ceiling. It may be retained for the remainder; reducing
it after the sentinel is optional.

If the login shell was closed while the sentinel ran, restore the variables and
helper functions from Sections 4 and 7, then recover the existing IDs rather
than resubmitting:

```bash
cd /dcai/users/pesdav/cu_0071/immune_health/immune-health-pbmc
source ./slurm/load_gefion.sh
export STAGE1_SENTINEL_INDICES='0-5,60-65'

CPU_BIND_JOB=$(_gefion_read_job_id \
  bind "$OUTPUT_ROOT/reference_prep_job_ids.txt")
STAGE1_SENTINEL_JOB=$(_gefion_read_job_id \
  sentinel "$OUTPUT_ROOT/stage1_sentinel_job_id.txt")
```

Then submit only the rows not already used by the sentinel:

```bash
# Keep the seven-day ceiling from gefion_env.txt unless the sentinel provides a
# well-supported reason to lower it.
export STAGE1_REMAINDER_INDICES='6-59,66-149'

gefion_submit_stage1_remainder
```

The remainder is 138 independent models in 18 node-array elements. The completed
Stage 1 should contain 150 `model_manifest.json` files and 150 `last.ckpt` files.

Monitor and verify the complete Stage 1:

```bash
squeue -j "$STAGE1_SENTINEL_JOB,$STAGE1_REMAINDER_JOB" \
  -o '%.18i %.12P %.24j %.8T %.10M %.10l %.6D %R'

sacct -j "$STAGE1_SENTINEL_JOB,$STAGE1_REMAINDER_JOB" \
  --units=G \
  --format=JobID,JobName%30,State,Elapsed,Timelimit,AllocTRES%50,MaxRSS,ExitCode

test "$(find "$OUTPUT_ROOT/tripso/stage1" -name model_manifest.json | wc -l)" -eq 150
test "$(find "$OUTPUT_ROOT/tripso/stage1" -path '*/checkpoints/last.ckpt' | wc -l)" -eq 150
find "$OUTPUT_ROOT/tripso/stage1" -name .failed.json -print | sort
```

If the final two `test` commands return silently, both expected counts are
correct. Review every failure marker against the corresponding `.done.json` and
worker log; a stale failure marker can remain after a successful retry.

## 10. Bind and project every Stage-1 reference/validation cell

Generate post-training rows only after the real Stage-1 manifest is fixed:

```bash
export POST1_DIR="$MANIFEST_ROOT/post_training/stage1"
export POST1_PACKED_DIR="$POST1_DIR/packed"
mkdir -p "$POST1_DIR" "$POST1_PACKED_DIR"

"$PY" scripts/generate_post_training_jobs.py \
  --training-manifest "$STAGE1_MANIFEST" \
  --output-dir "$POST1_DIR" \
  --batch-size 128 \
  --precision 32 \
  --max-projected-bytes 268435456000

"$PY" scripts/combine_job_manifests.py \
  --input "$POST1_DIR/bind_reference.jsonl" \
  --input "$POST1_DIR/bind_validation.jsonl" \
  --output "$POST1_PACKED_DIR/bind_all.jsonl"

"$PY" scripts/combine_job_manifests.py \
  --input "$POST1_DIR/project_reference.jsonl" \
  --input "$POST1_DIR/project_validation.jsonl" \
  --output "$POST1_PACKED_DIR/project_all.jsonl"

STAGE1_POST_BIND_JOB=$(submit_cpu_pack \
  "$POST1_PACKED_DIR/bind_all.jsonl" "$STAGE1_ALL_JOB_IDS")

STAGE1_POST_PROJECT_JOB=$(submit_gpu_pack \
  "$POST1_PACKED_DIR/project_all.jsonl" \
  "$STAGE1_POST_BIND_JOB" "" "$PROJECTION_WALLTIME")
```

The generator creates 150 reference binds, 150 validation binds, 150 reference
projections, and 150 validation projections. Query files are empty. The combined
GPU projection has 300 rows and therefore 38 eight-worker node elements. Frozen
projection uses every role-specific cell, does not adapt the checkpoint, and
persists only fold-bound training GP candidates.

## 11. Manual gate: choose two configurations per lineage

Stop here until the reference and inner-validation projections have been analysed.
Do not project or inspect the outer query to choose configurations.

The current repository provides conversion, aggregation, endpoint fitting,
validation scoring, and transferable-GP primitives, but it does not yet provide:

- a production candidate-plan builder;
- a command that ranks Stage-1 configurations and edits Stage 2; or
- a dependency-aware Slurm submitter for downstream manifests.

`configs/experiments/downstream_candidate_plan.example.json` must be expanded with
real selected model IDs, candidate endpoints, metadata/resources, and SHA-256
hashes. Stage 1 has only seed 42, so it can rank configurations on inner-validation
behavior, but the multi-seed transferable-GP selector cannot be finalized from
Stage 1 alone.

Once the top two sampler/HVG pairs per lineage have been reviewed, make a
run-specific copy and edit only its Stage-2 entries:

```bash
mkdir -p "$OUTPUT_ROOT/configs"
export RESOLVED_EXPERIMENT="$OUTPUT_ROOT/configs/tripso_lodo.resolved.yaml"
cp configs/experiments/tripso_lodo.yaml "$RESOLVED_EXPERIMENT"

${EDITOR:-vi} "$RESOLVED_EXPERIMENT"
```

Each selected sampler must be one of `native_all_cells`,
`donor_uniform_observed`, or `hybrid`; each HVG size must be `3000` or `9000`.

## 12. Run Stage 2 after the first manual gate

```bash
"$PY" scripts/generate_job_manifests.py \
  --config "$RESOLVED_EXPERIMENT" \
  --stage stage2 \
  --output-dir "$TRAIN_MANIFEST_DIR" \
  --seed 42

"$PY" - "$TRAIN_MANIFEST_DIR/stage2.jsonl" <<'PY'
import json, pathlib, sys
rows = [json.loads(line) for line in pathlib.Path(sys.argv[1]).read_text().splitlines()]
assert len(rows) == 100
assert all(row["runnable"] is True and row["command"] for row in rows)
print("Stage 2 has 100 runnable rows")
PY

export STAGE2_MANIFEST="$TRAIN_MANIFEST_DIR/stage2.jsonl"
STAGE2_TRAIN_JOB=$(submit_gpu_pack \
  "$STAGE2_MANIFEST" "$STAGE1_POST_PROJECT_JOB")
```

Stage 2 trains only seeds 43 and 44: five lineages × five LODO folds × two selected
configurations × two new seeds = 100 models, packed into 13 node elements. The
matching seed-42 Stage-1 model is reused scientifically, but the Stage-2 runner
does not automatically verify that linkage.

Generate and run frozen reference/validation projection for the 100 new models:

```bash
export POST2_DIR="$MANIFEST_ROOT/post_training/stage2"
export POST2_PACKED_DIR="$POST2_DIR/packed"
mkdir -p "$POST2_DIR" "$POST2_PACKED_DIR"

"$PY" scripts/generate_post_training_jobs.py \
  --training-manifest "$STAGE2_MANIFEST" \
  --output-dir "$POST2_DIR" \
  --batch-size 128 \
  --precision 32 \
  --max-projected-bytes 268435456000

"$PY" scripts/combine_job_manifests.py \
  --input "$POST2_DIR/bind_reference.jsonl" \
  --input "$POST2_DIR/bind_validation.jsonl" \
  --output "$POST2_PACKED_DIR/bind_all.jsonl"

"$PY" scripts/combine_job_manifests.py \
  --input "$POST2_DIR/project_reference.jsonl" \
  --input "$POST2_DIR/project_validation.jsonl" \
  --output "$POST2_PACKED_DIR/project_all.jsonl"

STAGE2_POST_BIND_JOB=$(submit_cpu_pack \
  "$POST2_PACKED_DIR/bind_all.jsonl" "$STAGE2_TRAIN_JOB")
STAGE2_POST_PROJECT_JOB=$(submit_gpu_pack \
  "$POST2_PACKED_DIR/project_all.jsonl" \
  "$STAGE2_POST_BIND_JOB" "" "$PROJECTION_WALLTIME")
```

For the Stage-2 comparison, combine the selected Stage-1 seed-42 evidence with
Stage-2 seeds 43/44 in the downstream candidate plan. No automatic seed-family
linker currently does this.

## 13. Manual gate: choose one final configuration per lineage

Using only the three-seed inner-validation evidence, edit the Stage-3 selection in
`$RESOLVED_EXPERIMENT`. Do not use outer LODO query performance. Preserve the
reviewed config as part of the run output.

## 14. Train the final all-healthy models

```bash
"$PY" scripts/generate_job_manifests.py \
  --config "$RESOLVED_EXPERIMENT" \
  --stage stage3 \
  --output-dir "$TRAIN_MANIFEST_DIR" \
  --seed 42

"$PY" - "$TRAIN_MANIFEST_DIR/stage3.jsonl" <<'PY'
import json, pathlib, sys
rows = [json.loads(line) for line in pathlib.Path(sys.argv[1]).read_text().splitlines()]
assert len(rows) == 25
assert all(row["runnable"] is True and row["command"] for row in rows)
print("Stage 3 has 25 runnable rows")
PY

export STAGE3_MANIFEST="$TRAIN_MANIFEST_DIR/stage3.jsonl"
export STAGE3_CORE_INDICES='0-2,5-7,10-12,15-17,20-22'

STAGE3_TRAIN_JOB=$(submit_gpu_pack \
  "$STAGE3_MANIFEST" "$STAGE2_POST_PROJECT_JOB" "$STAGE3_CORE_INDICES")
```

The primary core is 15 models: five lineages × seeds 42–44, packed into two node
elements. Seeds 45–46 are optional replication rows
`3-4,8-9,13-14,18-19,23-24` and can be submitted later.

Project the all-cell healthy reference for the same 15 core models:

```bash
export POST3_DIR="$MANIFEST_ROOT/post_training/stage3"
mkdir -p "$POST3_DIR"

"$PY" scripts/generate_post_training_jobs.py \
  --training-manifest "$STAGE3_MANIFEST" \
  --output-dir "$POST3_DIR" \
  --batch-size 128 \
  --precision 32 \
  --max-projected-bytes 268435456000

STAGE3_POST_BIND_JOB=$(submit_cpu_pack \
  "$POST3_DIR/bind_reference.jsonl" \
  "$STAGE3_TRAIN_JOB" "$STAGE3_CORE_INDICES")

STAGE3_POST_PROJECT_JOB=$(submit_gpu_pack \
  "$POST3_DIR/project_reference.jsonl" \
  "$STAGE3_POST_BIND_JOB" "$STAGE3_CORE_INDICES" "$PROJECTION_WALLTIME")
```

The final model is not one merged checkpoint. It is the 15 authoritative
`model_manifest.json` + `checkpoints/last.ckpt` pairs: one selected configuration
per lineage and three neural seeds, all refit on all five healthy cohorts. The
model manifest, rather than the checkpoint alone, is the transfer artifact because
it binds the vocabulary, GP library, sampler, configuration, training input,
vendor assets, and hashes.

## 15. Downstream fitting and the outer query

After a real, canonical-self-hashed candidate plan exists, Pass 1 can consume both
Stage-1 and Stage-2 reference/validation projection manifests:

```bash
"$PY" scripts/generate_downstream_jobs.py \
  --pass 1 \
  --projection-job-manifest "$POST1_DIR/project_reference.jsonl" \
  --projection-job-manifest "$POST1_DIR/project_validation.jsonl" \
  --projection-job-manifest "$POST2_DIR/project_reference.jsonl" \
  --projection-job-manifest "$POST2_DIR/project_validation.jsonl" \
  --candidate-plan "$OUTPUT_ROOT/configs/downstream_candidate_plan.json" \
  --output-dir "$MANIFEST_ROOT/downstream"
```

The plan must retain `healthy_reference.minimum_exact_sex_donors: 20`. Generated
rows contain logical `depends_on_job_ids`, but no submit driver currently converts
those into Slurm dependencies. Submit only runnable CPU rows in coarse order:

```text
conversion
  -> aggregation and cell bootstrap
  -> endpoints
  -> both reference-weighting fits and reference bootstrap
  -> validation/empirical scoring
  -> transferable-GP selection
```

`slurm/cpu_nodepack.sbatch` can execute each non-empty CPU manifest, but phase
boundaries and runnable-row indices must be reviewed explicitly.

Outer-query projection remains sealed until selected training job IDs and retained
GPs are frozen in the reviewed allowlists. Only then generate query projection
with both `--enable-outer-query-evaluation` and
`--outer-query-selected-job-allowlist`, followed by downstream Pass 2. The outer
query must never feed Stage-2/3 configuration choice or transferable-GP selection.

## 16. Restart and monitoring rules

- Resubmit the same manifest and indices after a failure. A valid `.done.json`
  inventory causes that row to skip; incomplete rows rerun.
- Do not delete or hand-edit `.done.json`, `model_manifest.json`, tokenization
  manifests, projection manifests, or candidate allowlists.
- A stale `.failed.json` is diagnostic, not authoritative, after a successful retry.
- `--array ...%N` limits concurrent nodes, not workers. With eight GPU workers,
  `%4` means up to four billed nodes and 32 simultaneous independent models.
- CPU node packing deliberately sets `CUDA_VISIBLE_DEVICES` empty and validates
  zero visible GPUs. GPU node packing binds one GPU per task and validates exactly
  one visible GPU.
- Never fill a partial node with outer-query work merely to increase occupancy.
  Scientific authorization and leakage control take precedence over utilization.
