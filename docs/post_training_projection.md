# Frozen post-training projection

TRIPSO training does not emit the all-cell embeddings needed to fit a healthy
reference. After each checkpoint finishes, run a separate, inference-only pass over
the exact adaptation Arrow dataset. Stage 1/2 also project the fixed inner donor
fold for model selection. The outer held-out cohort stays sealed and is projected
only for explicitly allowlisted selected jobs. The final all-healthy checkpoint
receives reference projection only; future cohorts are bound later.

## Safety contract

`build-projection-input` binds one tokenization to one immutable model manifest.
The builder and GPU projection both re-read the physical Hugging Face dataset.
They require:

- `reference`: the exact tokenization-manifest path and hash recorded in the
  training fold, with physical donors exactly equal to all lineage-available
  adaptation donors;
- `validation`: physical donors exactly equal the declared lineage-available fixed
  inner fold and are disjoint from adaptation and outer-query donors;
- `query`: for LODO, physical donors exactly equal the lineage-available outer
  held-out fold and are evaluation-only;
- one exact lineage and identical GP-library, gene-vocabulary, token dictionary,
  median dictionary, tokenizer-contract, and model-configuration hashes;
- the exact training-only `projection_gp_candidates.json` already bound through
  feature preparation, tokenization, the fold input, and model provenance;
- every tokenized row, `adapt=false`, no optimizer, evaluation mode, gradients
  disabled, and an unchanged parameter/buffer hash after inference.

The model manifest also records hashes for all five required vendored assets,
including the static Geneformer initializer and tokenizer dictionaries. Full
Geneformer projection retains its validated temporary compatibility context while
the checkpoint is loaded and evaluated.

Outputs are role-labelled Arrow directories:

```text
<projection-data>/embeddings/reference_set
<projection-data>/embeddings/validation_set
<projection-data>/embeddings/query_set
```

The companion `<projection-data>/projection_output_manifest.json` has schema
`immune-health-tripso-projection-output/v1`. It records the role, fold/design,
held-out cohort, lineage, physical cell/donor inventory, ordered retained GP IDs,
embedding dimension, endpoint columns, model/projection-input hashes, exact Arrow
column order, and SHA-256 for every Arrow file. Downstream conversion consumes this
manifest; it never infers role from a directory name.

Training still learns all filtered GPs. The inference adapter intercepts every
vendor test batch before Arrow accumulation and retains only the fold-bound
candidate GP vectors plus required metadata. Unselected vectors and their
`*_prop_genes` columns never enter the accumulated dataset. Bind time records
`n_cells × n_selected_GPs × embedding_dimension × 4` and fails above
`--max-projected-bytes` (250 GiB by default). An oversized run needs the explicit
`--allow-oversized-projection` override; all-GP persistence additionally needs the
explicit `--allow-all-gps` diagnostic option.

The local all-cell datamodule bypasses the vendor 80/10/10 subset only for frozen
projection. It preserves string donor/observation identifiers and does not alter
the training implementation.

## Generate restartable jobs

Generate post-training manifests only after the training manifest has its real,
runnable scientific selections:

```bash
python scripts/generate_post_training_jobs.py \
  --training-manifest slurm/manifests/stage1.jsonl \
  --output-dir slurm/manifests/post_training/stage1 \
  --batch-size 128 \
  --precision 32 \
  --max-projected-bytes 268435456000
```

This only writes JSONL; it never calls `sbatch`. It creates six role/phase files:

```text
bind_reference.jsonl     one row per runnable checkpoint
bind_validation.jsonl    one row per runnable Stage-1/2 checkpoint
bind_query.jsonl         empty by default; allowlisted outer evaluation only
project_reference.jsonl  one row per runnable checkpoint
project_validation.jsonl one row per runnable Stage-1/2 checkpoint
project_query.jsonl      empty by default; allowlisted outer evaluation only
```

For a resolved Stage 3 manifest, only the reference bind/project files contain
rows. No per-GP conversion or aggregation jobs are generated here.

To reveal the outer cohort, create a reviewed JSON manifest with schema
`immune-health-outer-query-evaluation-allowlist/v1`, `selection_basis` set to
`inner_validation_only`, `outer_query_data_consulted_for_selection=false`, unique
`selected_training_job_ids`, and a canonical `manifest_sha256`. Then pass both
`--enable-outer-query-evaluation` and
`--outer-query-selected-job-allowlist <manifest>`. Either option alone fails.

Run bind arrays after their training array succeeds, then run each projection array
after its matching bind array succeeds. CPU binding uses
`slurm/tripso_array.sbatch` on a shared CPU partition or
`slurm/cpu_nodepack.sbatch` on an exclusive Gefion node; billed Gefion GPU
projection uses `slurm/tripso_nodepack.sbatch`. All delegate each row to the same
restart-safe manifest runner. Each row has two deliberately different locations:

- `output_dir`: runner locks, job specification, resource logs, and restart marker;
- `projection_data_dir`: only the atomically published TRIPSO Arrow output.

That separation prevents the runner's `job_spec.json` and resource logs from making
the vendor projection destination nonempty. Projection writes to a sibling partial
directory first, so an interrupted attempt does not poison the final data path.

Bind manifests are CPU work. Use the ordinary single-row array on a shared CPU
partition, or `slurm/cpu_nodepack.sbatch` when Gefion assigns an exclusive node.
Projection manifests are GPU work; on Gefion, pack eight independent projection
rows into each billed exclusive node. Plan the 150-row projection:

```bash
python slurm/run_manifest_nodepack.py \
  --manifest slurm/manifests/post_training/stage1/project_reference.jsonl \
  --workers-per-node 8 \
  --plan-only
```

It requires 19 node-array elements. After the matching CPU bind array succeeds,
submit the GPU projection with explicit Gefion node resources:

```bash
sbatch \
  --account=cu_0071 \
  --partition=<GPU_PARTITION> \
  --time=<WALLTIME> \
  --nodes=1 \
  --exclusive \
  --ntasks-per-node=8 \
  --gpus-per-node=8 \
  --cpus-per-task=<CPUS_PER_PROJECTION> \
  --mem=<MEMORY_PER_NODE> \
  --dependency=afterok:<REFERENCE_BIND_ARRAY_JOB_ID> \
  --array=0-18%<MAXIMUM_CONCURRENT_NODES> \
  --export=ALL,PROJECT_ROOT=<GEFION_CHECKOUT>,OUTPUT_ROOT=<GEFION_OUTPUT>,TRIPSO_JOB_MANIFEST=<GEFION_CHECKOUT>/slurm/manifests/post_training/stage1/project_reference.jsonl,TRIPSO_WORKERS_PER_NODE=8,IMMUNE_HEALTH_ENV_ROOT=<PACKED_ENV_ROOT>,ENVIRONMENT_ACTIVATION_SCRIPT=<ACTIVATION_SCRIPT> \
  slurm/tripso_nodepack.sbatch
```

Repeat for validation after its bind array. Change the node-array upper bound for
smaller selected manifests using `run_manifest_nodepack.py --plan-only`. Full
Geneformer and exceptionally large all-cell projections may require fewer than
eight workers after memory calibration; set `TRIPSO_WORKERS_PER_NODE`,
`--ntasks-per-node`, and the plan's `--workers-per-node` to the same reduced value.

## One checkpoint by hand

Reference binding and projection are explicit:

```bash
python -m immune_health.cli.tokenize_tripso build-projection-input \
  --role reference \
  --tokenization-manifest <fold>/adaptation/tokenization_manifest.json \
  --model-manifest <model>/model_manifest.json \
  --output <model>/post_training/inputs/reference_projection_input.json \
  --use-fold-bound-gp-candidates \
  --max-projected-bytes 268435456000

python -m immune_health.cli project-tripso \
  --model-manifest <model>/model_manifest.json \
  --projection-manifest <model>/post_training/inputs/reference_projection_input.json \
  --output-dir <model>/post_training/projection_data/reference \
  --vendor-root tripso_code/tripso \
  --batch-size 128 \
  --precision 32
```

For model selection, change the role to `validation`, use the fold's validation
tokenization, and choose a separate validation input/output path. Use `query` only
for locked outer evaluation after the allowlist gate. For a future query
against the final all-healthy checkpoint, first materialize and tokenize it against
the frozen final feature set, then use the same query binding command.
