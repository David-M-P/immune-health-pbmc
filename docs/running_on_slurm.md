# Running TRIPSO on SLURM

No production jobs are submitted by repository scripts. Experiment expansion and
scheduler submission are separate, inspectable actions.

## Generate manifests

Generate the first five-lineage × five-fold × three-primary-sampler × two-feature-set
× one-seed matrix with:

```bash
python scripts/generate_job_manifests.py \
  --config configs/experiments/tripso_lodo.yaml \
  --stage stage1 \
  --output-dir slurm/manifests \
  --seed 42
```

This writes 150 zero-based JSONL rows and a summary; it does not call `sbatch`.
`--stage all` also writes Stage 2 and Stage 3 manifests. Their rows remain
non-runnable while sampler/HVG values contain explicit selection markers. The
outer held-out cohort is evaluation-only: it must not be used to populate those
markers. Choose two sampler–HVG configurations per lineage from donor-held inner
validation wholly inside the four reference cohorts. Stage 2 contains only
additional seed offsets 1 and 2 (seeds 43 and 44 when the base seed is 42); the
matching Stage-1 seed-42 models are reused rather than retrained. Choose the final
configuration from the same inner-validation evidence across seeds, then reveal
and report the outer query result. If an exploratory run has already inspected
outer results, label it selection-biased rather than a confirmatory LODO estimate.

The two feature sets are `HVG3000 ∪ retained GP genes` and
`HVG9000 ∪ retained GP genes`. Both retain all eligible cells. The three
primary sampling arms are the native vendor loader, donor-uniform sampling with
observed fine-type proportions, and the hybrid sampler. A fully fine-type-balanced
sampler is retained in the YAML as an optional diagnostic but is not expanded by
the primary Stage-1 manifest. To generate that diagnostic deliberately, use a copy
of the experiment YAML with Stage 1 `configuration_selection: all_configured`; this
adds 50 fully-balanced rows and keeps the primary manifest unchanged.

Before Stage 1 (and again for the final all-healthy fit), generate the real
reference inputs; this does not submit jobs:

```bash
python scripts/generate_reference_prep_jobs.py \
  --config configs/experiments/reference_preparation.yaml \
  --output-dir slurm/manifests/reference_prep
```

Run the generated `visits`, `final_fold`, `features`, `materialize`,
`lodo_tokenize`, `lodo_bind`, `final_tokenize`, and `final_bind` manifests with
`afterok` dependencies. The 150 LODO token jobs prepare separate adaptation,
fixed-inner-validation, and sealed query Arrow datasets, while the 50 LODO bind jobs produce every distinct fold input
consumed by the 150 Stage-1 sampler jobs. Query-input binding still happens only
after a model exists. The ten
`final_bind` rows write paths of the form
`${OUTPUT_ROOT}/tripso_inputs/<lineage>/all_healthy/hvg{3000,9000}/fold_input.json`,
which is the exact Stage-3 input template. These descriptors have
`reference_design=all_healthy`, `held_out_dataset=null`, an empty query-donor set,
and a physical Arrow donor-scope proof. The default final fold uses every eligible
healthy donor; an inner validation fold is an optional selection run, not the final
refit.

The reference-preparation cluster example includes both
`REFERENCE_PREP_OUTPUT_ROOT` (feature/materialization products) and `OUTPUT_ROOT`
(shared TRIPSO inputs). The array runner rejects either unresolved placeholder.

## Configure cluster resources

Copy the relevant file in `configs/slurm/` outside version control or supply its
values through your site launcher. Partition, wall time, CPU count per worker,
node memory, output root, project root, and environment activation script are
intentional placeholders. They are not inferred from another cluster.

The generic `tripso_array.sbatch` launcher still runs one manifest row per Slurm
allocation. Do not use it with an exclusive Gefion GPU node: Gefion bills the
complete eight-GPU node, so that combination would leave seven GPUs idle.

CPU phases on an exclusive Gefion node use `cpu_nodepack.sbatch`. It runs a
configurable number of independent manifest rows with CPU core binding, explicitly
hides CUDA, and verifies that each child sees zero GPUs. The same launcher covers
reference preparation, post-training binding, and downstream CPU manifests. Use
`scripts/combine_job_manifests.py` to combine independent same-dependency phases
without losing row provenance. See the
[`Gefion end-to-end runbook`](gefion_runbook.md) for the exact dependency chain.

Gefion production uses `tripso_nodepack.sbatch`. One node-array element requests
one exclusive eight-GPU node; one `srun` step launches eight independent tasks,
each with one GPU and one manifest row. This is not DDP. Before submitting, print
the exact row-to-node mapping:

```bash
python slurm/run_manifest_nodepack.py \
  --manifest slurm/manifests/stage1.jsonl \
  --workers-per-node 8 \
  --plan-only
```

For the 150-row Stage-1 manifest this reports 19 node-array elements and
`slurm_array_spec=0-18`. Submit it as follows, filling every site placeholder:

```bash
sbatch \
  --account=cu_0071 \
  --partition=<GEFION_GPU_PARTITION> \
  --time=<WALLTIME> \
  --nodes=1 \
  --exclusive \
  --ntasks-per-node=8 \
  --gpus-per-node=8 \
  --cpus-per-task=<CPUS_PER_GPU_WORKER> \
  --mem=<MEMORY_PER_NODE> \
  --array=0-18%<MAXIMUM_CONCURRENT_NODES> \
  --export=ALL,PROJECT_ROOT=<PROJECT_ROOT>,OUTPUT_ROOT=<OUTPUT_ROOT>,TRIPSO_JOB_MANIFEST=<PROJECT_ROOT>/slurm/manifests/stage1.jsonl,TRIPSO_WORKERS_PER_NODE=8,IMMUNE_HEALTH_ENV_ROOT=<PACKED_ENV_ROOT>,ENVIRONMENT_ACTIVATION_SCRIPT=<ACTIVATION_SCRIPT> \
  slurm/tripso_nodepack.sbatch
```

The `%` throttle counts exclusive nodes, not GPUs or models: `%4` permits four
nodes and therefore at most 32 concurrent one-GPU workers. If Gefion requires a
typed GRES request, replace `--gpus-per-node=8` with the site-approved
`--gres=gpu:<TYPE>:8`; the inner step still requires one bound GPU per task.

For a non-contiguous selection, pass the same reviewed selection to planning and
submission. The 12 B-cell/Monocyte sentinel rows use two nodes:

```bash
python slurm/run_manifest_nodepack.py \
  --manifest slurm/manifests/stage1.jsonl \
  --workers-per-node 8 \
  --indices '0-5,60-65' \
  --plan-only

sbatch <THE_SAME_NODE_RESOURCE_OPTIONS> \
  --array=0-1%<MAXIMUM_CONCURRENT_NODES> \
  --export=ALL,PROJECT_ROOT=<PROJECT_ROOT>,OUTPUT_ROOT=<OUTPUT_ROOT>,TRIPSO_JOB_MANIFEST=<PROJECT_ROOT>/slurm/manifests/stage1.jsonl,TRIPSO_WORKERS_PER_NODE=8,IMMUNE_HEALTH_ENV_ROOT=<PACKED_ENV_ROOT>,ENVIRONMENT_ACTIVATION_SCRIPT=<ACTIVATION_SCRIPT> \
  slurm/tripso_nodepack.sbatch --indices '0-5,60-65'
```

An `--indices-file` containing comma/range tokens or one index per line is also
supported. Stage-3 core seeds use
`0-2,5-7,10-12,15-17,20-22`, which maps to two nodes.

The launcher uses `--gpus-per-task=1`, `--gpu-bind=single:1`, and an explicit
PyTorch check so every worker sees exactly one GPU. It also rewrites the inherited
eight-task Slurm rank variables before entering Lightning; otherwise Lightning
could mistake the eight independent models for one DDP job. `--wait=0` allows
fast/tail ranks to finish without terminating longer peers. If one worker fails,
the other seven finish, the node element returns non-zero, and a resubmission
skips valid per-row `.done.json` artifacts.

This example documents submission; it is not executed by manifest generation.
Create `slurm/logs` before submission (the repository includes it). Per-worker
logs are written under `slurm/logs/nodepack/<array-job>_<block>/`.

## Post-training all-cell projection

Training itself does not persist the all-cell healthy-reference embeddings. After
the real Stage-1 or resolved Stage-3 manifest is fixed, generate the model-dependent
bind/projection arrays:

```bash
python scripts/generate_post_training_jobs.py \
  --training-manifest slurm/manifests/stage1.jsonl \
  --output-dir slurm/manifests/post_training/stage1 \
  --max-projected-bytes 268435456000
```

Production rows use the fold-bound training-only GP candidate manifest. Stage 1/2
default to every physical reference and validation cell with frozen weights;
Stage 3 defaults to reference only. Outer-query rows require both the explicit
evaluation flag and a hashed selected-job allowlist. Every output publishes a role-bearing
`projection_output_manifest.json`. Bind and project arrays are separate `afterok`
phases, and their runner directories are distinct from Arrow data directories.
See [`post_training_projection.md`](post_training_projection.md) for the exact
contract and all-GP/oversize diagnostic overrides.

## Two-pass downstream CPU manifests

[`generate_downstream_jobs.py`](../scripts/generate_downstream_jobs.py) never calls
`sbatch`. Its candidate plan is canonical-self-hashed, and binds the frozen
fine-type universe, metadata, genes, and vocabulary by SHA-256. Start from
[`downstream_candidate_plan.example.json`](../configs/experiments/downstream_candidate_plan.example.json)
and recompute `manifest_sha256` after replacing every placeholder.

Pass 1 consumes completed reference and inner-validation projections:

```bash
python scripts/generate_downstream_jobs.py \
  --pass 1 \
  --projection-job-manifest slurm/manifests/post_training/stage2/project_reference.jsonl \
  --projection-job-manifest slurm/manifests/post_training/stage2/project_validation.jsonl \
  --candidate-plan configs/experiments/downstream_candidate_plan.json \
  --output-dir slurm/manifests/downstream
```

It emits explicit conversion → per-GP aggregation → role-aware endpoint → both
reference weightings → validation-score dependencies. Aggregation always receives
the frozen fine-type universe. It also emits fine-type-stratified cell embedding-
mean bootstraps, matched-depth empirical endpoint sensitivities, and one central
reference-cross-fit `select-transferable-tripso-gps` job. The cell bootstrap is
not a healthy-reference score standard error. Missing scientific artifacts become
non-runnable rows with an explicit reason and no expected-output claims. Pass 1
never emits outer-query work.

Pass 2 requires the validated `selected_tripso_gps.json` and the same allowlist
that gated query projection. The allowlist must have identical
`selected_training_job_ids` and `allowed_parent_training_job_ids`, bind the
selector file SHA-256, and attest inner-validation-only selection:

```bash
python scripts/generate_downstream_jobs.py \
  --pass 2 \
  --projection-job-manifest slurm/manifests/post_training/stage2/project_query.jsonl \
  --candidate-plan configs/experiments/downstream_candidate_plan.json \
  --selected-gps runs/selection/selected_tripso_gps.json \
  --query-allowlist configs/experiments/outer_query_evaluation_allowlist.json \
  --output-dir slurm/manifests/downstream
```

Only retained query GP/fine-type endpoints are converted and scored. The query is
evaluation-only and cannot feed a selector. `evaluate-lodo` is runnable only when
the allowlist binds an exact preassembled prediction table. Run each non-empty CPU
JSONL with `slurm/tripso_array.sbatch` and
[`downstream_cpu.example.yaml`](../configs/slurm/downstream_cpu.example.yaml):
account `immunehealth`, four hours, 96 GB, four CPUs, and no GPU.

## Validation and restart behavior

For each row the runner:

1. expands configured environment variables and rejects unresolved placeholders;
2. refuses rows whose sampler selection is pending;
3. validates upstream paths, optional hashes, fold schema, and a successful TRIPSO
   environment JSON;
4. writes the resolved `job_spec.json` atomically;
5. sets `PYTHONHASHSEED`, deterministic CUDA workspace configuration, and offline
   W&B unless explicitly configured otherwise;
6. logs interpreter/package versions, SLURM resources, maximum resident memory,
   `nvidia-smi`, and `sstat` where available;
7. executes the worker as an argv list, never through a shell;
8. inventories each expected file and recursively inventories expected directories,
   recording content hashes and a deterministic tree hash before atomically writing
   completion schema v2;
9. writes `.failed.json` with traceback and resource state on failure.

An advisory file lock prevents concurrent tasks from using the same output. A v2
completion marker is trusted only when both its manifest-row fingerprint and a
freshly recomputed output inventory match. Old markers, missing/changed outputs,
directory additions/deletions, and symlinks fail closed. A completion marker for
different job content is never overwritten. To
inspect mechanics without executing a worker, run:

```bash
PROJECT_ROOT="$PWD" OUTPUT_ROOT=<OUTPUT_ROOT> \
python slurm/run_manifest_task.py \
  --manifest slurm/manifests/stage1.jsonl \
  --index 0 \
  --dry-run
```
