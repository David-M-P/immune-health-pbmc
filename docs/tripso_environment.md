# TRIPSO environment and adapter

## Inspected vendor surface

TRIPSO is vendored under `tripso_code/tripso`; it is not modified by this project.
At inspection time, the tracked vendor tree hash was
`a06a15a04c62bb6b31d713fd022630c17f0b31ee`, introduced by repository commit
`e2bb30c0b3483c29e31a2f4c85378a5cb96dd940`.

The inspected public API is:

- `tripso.pp_and_tokenize(...)` for preprocessing/tokenization;
- `tripso.train(...)`, an alias of `Train.training.run_training`, for Base and
  Global training;
- `tripso.gpEval(...)` and `generate_embeddings(...)` for checkpoint loading and
  embedding extraction.

Production inputs use `python -m immune_health.cli.tokenize_tripso`, which invokes
the inspected `TranscriptomeTokenizer` directly in memory-bounded H5AD row chunks
after fold-local feature selection. This makes `calculate_hvg=false` and
`subsample_by=None` effective by construction, preserves the required metadata,
and writes physical donor-scope and 4,096-position truncation proofs before a fold
can train. It does not modify the vendor source.

Important implementation details are not hidden by the adapter. The vendor
`txDataModule` randomly creates 80/10/10 cell subsets. For donor-aware training,
the local subclass replaces that unrestricted split with a disjoint,
strata-preserving approximately 80/10/10 optimizer split, so singleton and other
small donor/fine-type strata cannot vanish before hierarchical sampling. Real
training is also refused until donor IDs extracted from the physical tokenized
input prove that it contains the adaptation donors only; a fold-table declaration
is not accepted as content proof.
The public embedding helper also selects one subset, so frozen query projection
uses a project-local `QueryOnlyDataModule` subclass and the inspected
`_init_trainer` evaluation surface to send all query cells through Lightning
`test`. This private surface is checked by the environment validator and must be
reviewed when the vendor tree changes.

The vendor training wrapper also hard-wires Weights & Biases login, logging, and
an online API history readback after `Trainer.fit`. `WANDB_MODE=offline` alone is
therefore not sufficient: the readback still requires a server-side run. The
production adapter temporarily replaces only the four vendor tracking globals
(`configure_save_id`, `configure_wandb`, `configure_logger`, and the final
readback rank guard) while the call is active. Lightning's `CSVLogger` writes to
`<model-output>/local_csv/training/`; the canonical table is copied atomically to
`training_metrics.csv`. The vendor datamodule, model construction, Lightning
module, losses, callbacks, checkpoint callback, and `Trainer.fit` are unchanged.
The original globals are restored even if training raises. A partial or changed
vendor tracking surface fails closed in the environment validator and training
adapter rather than silently contacting W&B.

## One transferable environment

[`environment.yml`](../environment.yml) is the single environment for CPU feature
preparation, tokenization, Base/Global training, frozen projection, Arrow conversion,
aggregation, and healthy-reference fitting. It pins Linux Python 3.10, every
uncommented dependency in `tripso_code/tripso/requirements.txt`, PyTorch 2.4.1,
Torchvision 0.19.1, Torchmetrics 1.7.1, and the CUDA 12.1 user-space runtime.
The Gefion compute node still needs an NVIDIA driver compatible with CUDA 12.1.

Create and validate it from the repository root:

```bash
mamba env create --prefix .conda_isolated/immune-health-tripso \
  --file environment.yml

MPLCONFIGDIR=/tmp/immune_health_mpl \
NUMBA_CACHE_DIR=/tmp/immune_health_numba_cache \
.conda_isolated/immune-health-tripso/bin/python \
  scripts/validate_tripso_environment.py \
  --vendor-root tripso_code/tripso \
  --geneformer-root \
    /faststorage/project/immunehealth/Projects/david/external_assets/Geneformer \
  --smoke-mode mock \
  --json-output reports/tripso_environment_pinned.json
```

The validator checks exact dependency versions, imports the real vendor package,
verifies its inspected signatures and bundled resources, and hashes the optional
full Geneformer model. A mock smoke is still labelled as a mock; it is not promoted
to a real training result.

`--smoke-mode mock` proves only that the project frozen-state and no-optimizer
guards work. The JSON deliberately records `real_tripso_training_smoke_passed` as
false. A real tiny training driver can be executed without a shell using
`--smoke-mode real --real-smoke-command-json '["python", "..."]'`; the validator
does not substitute a mock when that command fails.

## Current validation result

The isolated environment at `.conda_isolated/immune-health-tripso` passed on Linux
Python 3.10.14. Every reviewed vendor pin matches, the real `tripso` import and API
checks pass, all five bundled assets hash successfully, and the frozen-projection
guard passes. The full historical Geneformer configuration and weights also pass
their pinned hashes, and its `20,275 x 512` input embedding tensor is bit-for-bit
equal to TRIPSO's bundled static initializer. The validator was then run with the
real Base smoke command and recorded both `environment_passed=true` and
`real_end_to_end_passed=true` in
[`reports/tripso_environment_real_smoke.json`](../reports/tripso_environment_real_smoke.json).

The login node had no visible GPU, so this result does not certify Gefion's driver
or GPU execution. A separate tiny real-training result is recorded by the real
smoke driver; never infer it from `environment_passed` alone.

## Real vendored Base smoke

Run a real one-batch Base optimization before submitting larger jobs:

```bash
.conda_isolated/immune-health-tripso/bin/python \
  scripts/run_real_tripso_smoke.py \
  --output-dir runs/<run_id>/validation/real_tripso_base_smoke
```

The default creates a ten-cell on-disk Hugging Face dataset using real Geneformer
token IDs, writes an eight-gene Ensembl GP, initializes TRIPSO's trainable gene
encoder from the bundled `gf-12L-95M-i4096` static embeddings, and executes one
real forward, backward, optimizer step, and checkpoint save through the vendored
`txDataModule`, `gpTransformerBase`, and `gpBase` classes. It runs on CPU by
default; request `--accelerator cuda` only inside a GPU allocation. The output
directory must be new or empty so a stale checkpoint cannot be mistaken for a
pass.

The validated CPU result is
[`reports/real_tripso_base_smoke/smoke_report.json`](../reports/real_tripso_base_smoke/smoke_report.json).
It recorded finite gene and GP masking losses, exactly one optimizer step, a
non-zero GP-decoder parameter change, and a reloadable checkpoint. This proves
the local vendored training path and static initialization execute. It does not
prove convergence, biological validity, full-data memory sufficiency, the dynamic
sampler, projection, or Gefion GPU/driver compatibility.

The bounded smoke driver instantiates the vendored components directly so it can
stop after exactly one optimizer step. Production does use `tripso.train`, with
the network-free local CSV boundary described above. No model, loss, datamodule,
optimizer, callback, or trainer implementation is replaced in either path.

## Pinned full Geneformer sensitivity

The primary Base analysis uses TRIPSO's bundled May-2025 Geneformer input
embeddings plus a trainable two-layer gene encoder. Full 12-layer Geneformer is a
costlier sensitivity analysis. Download and validate the exact historical model:

```bash
.conda_isolated/immune-health-tripso/bin/python scripts/download_geneformer.py \
  --output-root \
    /faststorage/project/immunehealth/Projects/david/external_assets/Geneformer \
  --manifest-output reports/geneformer_asset_validation.json
```

The download is pinned to revision
`94d98d1774fc21920d2f01dbd23bc22d232a6ec2`. The model SHA-256 is
`4365ba23e393fcfa0e65a94ac64a0983cd788bd23a8d4914f4ab66f85cfe043c`.
The historical snapshot has a model card declaring Apache-2.0 but no standalone
`LICENSE` file; the model card, revision, declared license, and file hashes are
retained in the asset manifest.

On any node that trains or projects the full-Geneformer sensitivity, point the
adapter at the transferred asset root before launching the job:

```bash
export TRIPSO_GENEFORMER_ROOT=/path/on/cluster/external_assets/Geneformer
```

Frozen projection revalidates the pinned configuration and weight hashes, then
installs the same temporary Geneformer-path and forward-signature compatibility
used during training while the checkpoint is reconstructed and evaluated. The
patch is removed after projection and the vendored TRIPSO source remains unchanged.
The primary static-embedding Base does not require this environment variable.

## Moving to Gefion

Keep code and large immutable assets separate:

1. Push the repository (without data, models, or `.conda_isolated`) to GitLab and
   clone it on Gefion.
2. Pack the validated environment on this Linux cluster:

   ```bash
   conda-pack \
     --prefix "$PWD/.conda_isolated/immune-health-tripso" \
     --output immune-health-tripso-linux-x86_64.tar.gz
   ```

3. Transfer that archive, the exact 3k/9k materialized inputs, manifests, and
   `external_assets/Geneformer` over SFTP. Do not put these large files in Git.
   If tokenization was performed locally, transfer the complete Hugging Face
   directory and sidecars too; use the hash-validating
   `relocate-tokenization` procedure in
   [`reference_preparation.md`](reference_preparation.md#relocate-a-tokenization-after-sftp)
   on Gefion instead of editing absolute paths in JSON.
4. On Gefion, unpack into a fixed directory and repair prefixes:

   ```bash
   mkdir -p /path/on/gefion/envs/immune-health-tripso
   tar -xzf immune-health-tripso-linux-x86_64.tar.gz \
     -C /path/on/gefion/envs/immune-health-tripso
   /path/on/gefion/envs/immune-health-tripso/bin/conda-unpack
   ```

5. Re-run the environment validator and a tiny GPU smoke on a compute node before
   launching an array. A packed environment is portable only across compatible
   Linux/glibc systems; Gefion's NVIDIA driver is deliberately revalidated there.

The packed environment contains the project version present when it was built.
The supplied Slurm launchers prepend `${PROJECT_ROOT}/src` to `PYTHONPATH`, so the
GitLab checkout is the authoritative project code after later pulls while binary
dependencies still come from the packed environment. For an interactive shell,
either export the same `PYTHONPATH` or reinstall only the local project (without
resolving dependencies):

```bash
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
# Alternatively:
python -m pip install --no-deps --force-reinstall "${PROJECT_ROOT}"
```

## Asset hashes

| Asset | SHA-256 |
|---|---|
| token dictionary | `67c445f4385127adfc48dcc072320cd65d6822829bf27dd38070e6e787bc597f` |
| gene median file | `a51c53f6a771d64508dfaf61529df70e394c53bd20856926117ae5d641a24bf5` |
| Ensembl mapping | `0819bcbd869cfa14279449b037eb9ed1d09a91310e77bd1a19d927465030e95c` |
| Ensembl dictionary | `8b0fd0521406ed18b2e341ef0acb5f53aa1a62457a07ca5840e1c142f46dd326` |
| 4096-token word embeddings | `637efa884d4007bbec93b7d46f19ecc65c46ddcfe71815a910997b3942e1c4c3` |

Every model manifest additionally records the live repository commit, vendor source
commit/tree hash, fold, held-out dataset, lineage, sampler mode, alpha, lambda, seed,
GP/vocabulary/input/checkpoint hashes, environment, configuration, checkpoint, and
training metrics.

For Slurm array tasks, `run_manifest_task.py` force-sets `WANDB_MODE=disabled`
and `WANDB_SILENT=true` even if the submission environment contains W&B
credentials. Successful training writes, in order, `last.ckpt`,
`tripso_training_result.json`, and `model_manifest.json`; only after those
expected artifacts exist does the runner atomically create `.done.json`.
