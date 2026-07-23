# Fold-local reference preparation

This CPU stage starts from the unbalanced merged raw-count H5ADs under
`data/intermediate_data/reference_lineages/merged`. It never edits those source
files and does not cap or downsample retained cells.

## Biological rules

- Outer and inner roles are joined by `biological_unit_id = dataset::donor_id`.
- Global roles are never reassigned by lineage. Because a donor can have zero cells
  in a merged broad-lineage file, feature preparation writes a self-hashed
  `lineage_donor_scope` containing exact adaptation/validation/query donor subsets
  and every global-fold donor without materializable lineage cells. Materialized
  H5ADs, Arrow tokenization, fold binding, and projection all enforce that scope.
- When Terekhova contributes to reference fitting, one complete observation is
  selected per donor by the lowest SHA-256 of
  `seed::biological_unit_id::observation_id`. Age is not used.
- Production keeps one deterministic Terekhova visit per donor in every role,
  including a LODO query. Retaining all query visits requires the explicit
  `--preserve-all-query-visits` longitudinal sensitivity flag.
- Program expression support and HVGs are learned from adaptation cells only.
- Raw `ctype_low` and `ctype_low_conf` remain unchanged. The approved ontology
  creates canonical `fine_type`, `fine_type_state_eligible`, and
  `fine_type_balance_eligible` columns. Below-threshold cells become
  `low_confidence`; confident unmapped or explicitly quarantined labels become
  `other_confident`. Both categories remain in model inputs and composition.
- Before GPU projection, a storage candidate set is frozen from training-only
  broad-lineage donor pseudobulks. It is the union of transferable within-cohort
  age programs and any prespecified controls; it is not the later fine-type result.
- HVGs are ranked from donor pseudobulks, then combined with equal weight across
  training datasets. The 3,000-gene list is a prefix of the 9,000-gene list.
- The model vocabulary is `retained GP genes union selected HVGs`, in source-H5AD
  gene order.
- `source_cell_id` preserves the H5AD observation name. The unique transport key
  is `cell_key = dataset::source_cell_id`; its name deliberately does not end in
  `_id`, because TRIPSO interprets such metadata fields as integer encodings.

## Focused commands

```bash
python -m immune_health.cli.prepare_reference build-terekhova-visits \
  --metadata splits/global_donor_manifest.tsv \
  --output-dir runs/reference_prep/visit_selection \
  --seed 42

python -m immune_health.cli.prepare_reference select-fold-features \
  --input-h5ad /path/to/reference_lineages/merged/B_cells/merged.h5ad \
  --fold-manifest splits/lodo_aidav2.tsv \
  --visit-manifest runs/reference_prep/visit_selection/terekhova_one_visit.tsv \
  --gene-programs /path/to/gene_programs/v1/gene_programs_curated.gmt \
  --fine-type-ontology configs/data/fine_type_ontology.approved.yaml \
  --output-dir runs/reference_prep/features/B_cells/lodo_aidav2 \
  --lineage "B cells"

python -m immune_health.cli.prepare_reference materialize-fold-h5ad \
  --input-h5ad /path/to/reference_lineages/merged/B_cells/merged.h5ad \
  --preparation-dir runs/reference_prep/features/B_cells/lodo_aidav2 \
  --output-h5ad \
    runs/reference_prep/materialized/B_cells/lodo_aidav2/hvg3000/adaptation/model_input.h5ad \
  --role adaptation \
  --hvg-size 3000
```

For the final model, first declare that all five healthy cohorts are reference
data. There is no sentinel or fabricated held-out cohort:

```bash
python -m immune_health.cli.prepare_reference build-all-healthy-fold \
  --metadata splits/global_donor_manifest.tsv \
  --output-dir runs/reference_prep/folds/all_healthy \
  --healthy-dataset aidav2 \
  --healthy-dataset immuneindonesia \
  --healthy-dataset immunobiologyaging \
  --healthy-dataset onek1k \
  --healthy-dataset terekhova

python -m immune_health.cli.prepare_reference select-fold-features \
  --input-h5ad /path/to/reference_lineages/merged/B_cells/merged.h5ad \
  --fold-manifest runs/reference_prep/folds/all_healthy/all_healthy.tsv \
  --visit-manifest runs/reference_prep/visit_selection/terekhova_one_visit.tsv \
  --gene-programs /path/to/gene_programs/v1/gene_programs_curated.gmt \
  --output-dir runs/reference_prep/features/B_cells/all_healthy \
  --lineage "B cells" \
  --reference-design all_healthy
```

With the default `inner_validation_fold: null`, all 2,180 healthy donors in the
current global manifest are eligible before lineage-specific cell availability is
applied (625 AIDA v2, 174 ImmuneIndonesia, 234 ImmunobiologyAging, 981 OneK1K,
166 Terekhova). Setting an inner fold reserves donors for model selection and is
not the final fit; reset it to null to refit on every eligible donor.

Materialize separate 3,000-HVG-plus-GP and 9,000-HVG-plus-GP H5ADs. Each contains
the exact corresponding union and every eligible cell; the 3,000-HVG run cannot
see genes that occur only in the 9,000-HVG union. Each file also stores
`var["ensembl_id"]` in that exact order and `obs["n_counts"]` calculated from the
complete 18,035-gene source matrix before feature subsetting, as required by the
Geneformer rank-value normalization.

Feature preparation writes `simple_gp_donor_scores.parquet`, cohort-specific
`simple_gp_age_effects.parquet`, `simple_gp_transferability.parquet`, and the
ordered `projection_gp_candidates.{tsv,json}`. The default gate requires at least
20 donors and ten years of age support in at least three training cohorts, 75%
sign agreement, I-squared at most 0.75, and FDR at most 0.05. The JSON binds the
candidate order to the filtered GP database and proves that query data were not
consulted. An empty statistical-plus-control union fails closed. This early,
broad-lineage gate bounds per-cell projection storage; final reporting still
retests within-cohort age effects at fine-type level.

## Exact TRIPSO tokenization

After transferring one materialized H5AD, its vocabulary, and filtered GP CSV to
the compute cluster, tokenize it without another HVG calculation or cell sample:

```bash
python -m immune_health.cli.tokenize_tripso tokenize \
  --input-h5ad runs/reference_prep/materialized/B_cells/lodo_aidav2/hvg3000/adaptation/model_input.h5ad \
  --gene-vocabulary runs/reference_prep/features/B_cells/lodo_aidav2/model_genes_hvg3000.txt \
  --gp-library runs/reference_prep/features/B_cells/lodo_aidav2/gpdb_filtered.csv \
  --projection-gp-candidates runs/reference_prep/features/B_cells/lodo_aidav2/projection_gp_candidates.json \
  --output-dir "${OUTPUT_ROOT}/tripso_inputs/b_cells/lodo_aidav2/hvg3000/adaptation" \
  --role adaptation \
  --row-chunk-size 20000 \
  --nproc 4

python -m immune_health.cli.tokenize_tripso build-fold-input \
  --tokenization-manifest "${OUTPUT_ROOT}/tripso_inputs/b_cells/lodo_aidav2/hvg3000/adaptation/tokenization_manifest.json" \
  --fold-table splits/lodo_aidav2.tsv \
  --output "${OUTPUT_ROOT}/tripso_inputs/b_cells/lodo_aidav2/hvg3000/fold_input.json" \
  --fold-id lodo_aidav2 \
  --held-out-dataset aidav2 \
  --lineage "B cells" \
  --inner-validation-fold 0 \
  --inner-fold-column inner_fold
```

Repeat with `hvg9000`; tokenize adaptation, validation, and query H5ADs separately.
Feature selection saw adaptation donors only. After training, bind validation for
model selection; keep query sealed until the selected-job evaluation gate.

```bash
python -m immune_health.cli.tokenize_tripso build-query-input \
  --tokenization-manifest "${OUTPUT_ROOT}/tripso_inputs/b_cells/lodo_aidav2/hvg3000/query/tokenization_manifest.json" \
  --model-manifest "${OUTPUT_ROOT}/tripso/models/b_cells/lodo_aidav2/hvg3000/model_manifest.json" \
  --output "${OUTPUT_ROOT}/tripso_inputs/b_cells/lodo_aidav2/hvg3000/query_input.json" \
  --use-fold-bound-gp-candidates
```

The tokenization manifest physically inventories adaptation/validation/query donors and keeps
`cell_key`, dataset, donor/observation identifiers, raw and canonical fine type,
both fine-type eligibility flags, and lineage in Arrow.
It reports tokenizer-vocabulary and per-GP coverage plus the fraction of cells whose
expressed-gene rank exceeded 4,094 (4,096 positions minus CLS/EOS). Missing or
empty-token cells are errors; they are never silently dropped. `build-query-input`
refuses any vocabulary, GP, tokenizer-contract, or donor overlap with training.
The manifest also stores a SHA-256 inventory for every physical Arrow file, the
materialized H5AD itself, the approved ontology identity/hash, all three sidecars,
and the tokenizer implementation and dictionaries.

### Relocate a tokenization after SFTP

Absolute paths in the source manifest must not be edited by hand. After copying
the complete tokenization directory, H5AD plus its materialization manifest,
vocabulary, GP database, candidate JSON, and TRIPSO checkout, preserve the source
manifest and create a Gefion-local one:

```bash
cp /gefion/run/adaptation/tokenization_manifest.json \
  /gefion/run/adaptation/tokenization_manifest.source.json

python -m immune_health.cli.tokenize_tripso relocate-tokenization \
  --source-manifest /gefion/run/adaptation/tokenization_manifest.source.json \
  --output-manifest /gefion/run/adaptation/tokenization_manifest.json \
  --tokenized-dataset /gefion/run/adaptation/tokenized.dataset \
  --input-h5ad /gefion/materialized/model_input.h5ad \
  --materialization-manifest /gefion/materialized/model_input.manifest.json \
  --gene-vocabulary /gefion/features/model_genes_hvg3000.txt \
  --gp-library /gefion/features/gpdb_filtered.csv \
  --projection-gp-candidates /gefion/features/projection_gp_candidates.json \
  --vendor-root /gefion/code/tripso_code/tripso \
  --overwrite
```

Relocation first validates the unedited source manifest's self-hash. It then
requires byte-identical H5AD/resources/vendor assets and an exact per-file Arrow
inventory, re-reads physical row, donor, cohort, lineage, and key order, and only
then atomically writes new paths with source-manifest provenance. Use the relocated
manifest for `build-fold-input` and projection binding. A partial SFTP transfer or
one changed Arrow byte is rejected.

The final training descriptor uses the same physical scope proof but no held-out
argument:

```bash
python -m immune_health.cli.tokenize_tripso build-fold-input \
  --tokenization-manifest "${OUTPUT_ROOT}/tripso_inputs/b_cells/all_healthy/hvg3000/adaptation/tokenization_manifest.json" \
  --fold-table runs/reference_prep/folds/all_healthy/all_healthy.tsv \
  --output "${OUTPUT_ROOT}/tripso_inputs/b_cells/all_healthy/hvg3000/fold_input.json" \
  --fold-id all_healthy \
  --reference-design all_healthy \
  --partition-column reference_partition \
  --lineage "B cells"
```

That output path is exactly the `final_input_template` consumed by Stage 3.

## Frozen future query preparation

Do not rerun HVG or GP selection on SoundLife or a disease cohort. Map its raw
lineage H5AD to the final feature manifest instead:

```bash
python -m immune_health.cli.prepare_reference materialize-frozen-query-h5ad \
  --input-h5ad /path/to/unseen/B_cells.h5ad \
  --final-preparation-dir runs/reference_prep/features/B_cells/all_healthy \
  --output-h5ad runs/query/B_cells/hvg3000/model_input.h5ad \
  --lineage "B cells" \
  --hvg-size 3000
```

The mapper preserves all query cells, computes library sizes from the complete
query gene universe, writes missing training genes as zero columns in exact frozen
order, reports coverage, and refuses a reference-cohort name by default. Tokenize
with `--role query`, then use `build-query-input`; its resource hashes must match
the trained final model before frozen projection is allowed.

## CPU Slurm arrays

Generate three restartable manifests:

```bash
python scripts/generate_reference_prep_jobs.py
```

This writes one visit-selection job, one final-fold job, 30 feature-selection jobs
(25 LODO plus five final), 160 materialization jobs, 150 LODO tokenization jobs
(adaptation, fixed validation, and query), 50 LODO adaptation-binding jobs, ten final tokenization
jobs, and ten final fold-binding jobs. Submit the stages in the generated order with
`afterok` dependencies,
using the local CPU template in `configs/slurm/reference_prep_cpu.example.yaml`.
The matching `slurm/reference_prep_array.sbatch` requests the approved local
account, four-hour wall time, 96 GB, and four CPUs. It deliberately makes no
partition or GPU request. Cluster paths and environment activation remain
external settings; export both `REFERENCE_PREP_OUTPUT_ROOT` and `OUTPUT_ROOT` as
shown in the example config. On an exclusive Gefion node, use
`configs/slurm/gefion_cpu.example.yaml` with `slurm/cpu_nodepack.sbatch`; the exact
account is `cu_0071`, while partition, node memory, CPU workers, wall time, and
concurrency remain explicit site values. The full submission chain is in
[`gefion_runbook.md`](gefion_runbook.md).

## Arrow bridge

TRIPSO embedding output is a HuggingFace Arrow directory. Convert only the GP
columns needed by the downstream job:

```bash
python -m immune_health.cli.convert_tripso_arrow \
  --arrow-dataset /path/to/embeddings/query_set \
  --projection-output-manifest \
    /path/to/projection_data/projection_output_manifest.json \
  --cell-metadata \
    runs/reference_prep/features/B_cells/lodo_aidav2/cell_metadata.parquet \
  --embedding-column BLOODGEN3__M10_2 \
  --output-dir runs/reference_prep/converted/B_cells/lodo_aidav2/M10_2
```

The converter first verifies the exact frozen projection output, including its
role, Arrow tree hashes, and frozen GP allowlist. It then performs a one-to-one
`cell_key` join, restores the approved
`observation_id` grouping metadata, preserves Arrow row order, and writes a
float32 NPY, aligned Parquet metadata, and a manifest with ordered-key and array
payload hashes. A row-count-only match is not accepted.
