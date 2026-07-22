# Output schema

Tabular donor outputs use Parquet where possible. Per-cell embeddings use the
vendored chunked Hugging Face Arrow dataset initially, with stable metadata columns;
Zarr is an acceptable replacement when array chunking is needed. JSON is reserved
for manifests, configuration, provenance, validation, and atomic scheduler markers.
Arbitrary Python objects are never stored in table cells.

## Fine-type GP table

One row represents dataset × donor × observation × lineage × fine type × GP × model
seed. Required identity and context columns are:

```text
dataset, donor_id, biological_unit_id, sample_id, source_observation_id,
observation_id, age, sex, lineage, fine_type, gp_id, n_cells,
fine_type_fraction, annotation_confidence_summary
```

State columns are `location_summary`, `covariance_summary`, `covariance_trace`,
`covariance_logdet`, `dispersion`, `age_axis_q10`, `age_axis_q25`, `age_axis_q50`,
`age_axis_q75`, `age_axis_q90`, `healthy_tail_fraction`, `predicted_gp_age`,
`gp_age_acceleration`, `age_matched_distance`, `off_trajectory_distance`,
`sliced_wasserstein_distance`, and `gaussian_wasserstein_distance`.

Uncertainty columns are `cell_sampling_se`, `reference_sampling_se`, and `seed_sd`.
Provenance columns are `model_id`, `fold_id`, `seed`, `annotation_version`,
`gp_library_version`, and `reference_version`. Fixed-length vectors may use Parquet
array columns with declared dimensions; large vectors and covariance factors live in
a separately keyed Zarr/Arrow array.

Cell-level empirical values are not copied into a per-group NPZ. The
`immune-health-empirical-row-index/v1` manifest binds
`empirical_distribution_groups.parquet` and one int64
`empirical_distribution_rows.npy` to the immutable converted float32 NPY and its
Arrow/projection provenance. Consumers memory-map the converted NPY and gather one
group's rows.

## Donor GP endpoint artifact

One `immune-health-donor-gp-endpoint/v1` directory represents exactly one lineage
× fine type × GP and exactly one projection role. It contains
`endpoint_metadata.parquet`, aligned float32 `endpoint_locations.npy`, aligned
float32 `endpoint_covariances.npy`, and `endpoint_manifest.json`. The manifest
records hashes, row order, endpoint identity, model/fold/seed, role, reference
design, held-out dataset, and the complete projection→conversion→aggregation
provenance chain. A LODO reference endpoint contains adaptation rows only; its
query endpoint is physically separate.

Endpoint-backed reference fitting writes two separate frozen models. The
location-only spline is `healthy_reference.json` plus
`healthy_reference_arrays.npz`. The distributional model is the small
`age_kernel_reference.json`; it binds and memory-maps the original endpoint
location/covariance NPYs by hash rather than duplicating them, and uses the same
feature order, donors, and weighting scheme.
Scoring exposes `age_matched_location_distance` and
`age_matched_gaussian_wasserstein_distance` as different quantities; spline
residual covariance is never substituted for donor within-cell covariance.

## Empirical matched-depth score artifact

`immune-health-empirical-matched-depth-score/v1` contains
`empirical_matched_depth_scores.parquet`, replicate-level
`empirical_matched_depth_replicates.parquet`, `empirical_reliability.parquet`, and
`empirical_scoring_manifest.json`. The manifest records the target role, exact
endpoint/index file and content hashes, ordered observation and cell-key hashes,
model/checkpoint/fold/seed binding, aggregation-table binding, deterministic
projection hash, depths, replicate count, and age-kernel settings. The score table
retains `fold_id`, `seed`, and endpoint identity so seed-specific calibrated scalar
scores can be combined without averaging latent coordinates.

`immune-health-seed-score-combination/v1` is a long table of approved scalar metric
means and sample SDs across independent neural seeds. Its manifest records every
input table hash and explicitly states `embedding_coordinates_averaged=false`.

## Final TRIPSO GP selection artifact

One outer fold × lineage selector directory uses schema
`immune-health-tripso-gp-selection/v1` and contains:

- `tripso_gp_cohort_seed_effects.parquet`: one row per lineage × fine type × GP ×
  neural seed × training cohort, including donor count, age span, slope,
  donor-clustered SE, standardized effect, eligibility, model ID, and outer-fold
  identity;
- `tripso_gp_selection.parquet`: one row per candidate endpoint with worst-case
  cohort support/FDR/heterogeneity, seed retention, seed direction/rank/effect
  stability, state-observation coverage, cell-depth/composition diagnostics,
  optional simple-baseline comparison, `retained`, and an explicit reason; and
- `selected_tripso_gps.json`: a canonical self-hashed manifest that binds the two
  tables and every healthy-reference, endpoint, model/checkpoint, fold-input, and
  donor-cross-fit source hash.

The JSON records `query_data_consulted=false`,
`raw_tripso_coordinates_used_for_selection=false`, exact required seeds/cohorts,
thresholds, and `selected_endpoints` as `{lineage, fine_type, gp_id}` objects. It
may validly contain no endpoints with status `complete_no_candidates`. Downstream
query jobs must validate this manifest and must not replace an empty set with an
unfiltered candidate list.

## Lineage GP table

One row represents dataset × donor × observation × lineage × GP × model seed. It
retains:

```text
observed_mixture_score
composition_standardized_state_score
composition_only_score
within_fine_type_heterogeneity
between_fine_type_heterogeneity
total_lineage_heterogeneity
cell_sampling_se
reference_sampling_se
seed_sd
```

It repeats donor/observation identity, age, sex, model, fold, annotation, GP library,
and reference provenance. The three mixture/composition values and three
heterogeneity values remain separate in the first version.

## Embeddings

Per-cell output must retain a stable cell key plus `dataset`, `donor_id`,
`biological_unit_id`, `sample_id`, `source_observation_id`, `observation_id`,
`lineage`, `fine_type`, annotation confidence, each GP embedding, each GP gene
coverage proportion, and the global `cell_token` where produced. Embedding dimension
and GP order are recorded once in the model manifest and array metadata.

## Model manifest

`model_manifest.json` uses schema `immune-health-tripso-model/v1` and records
repository/vendor provenance, fold and held-out dataset, lineage, model type,
sampler mode/alpha/lambda, seed, checkpoint, training metrics, software versions,
model configuration, and SHA-256 hashes for checkpoint, GP library, vocabulary,
fold input, and vendored assets. Frozen query manifests must match the relevant
hashes and configuration fields.

Training observability is network-free. Lightning writes its raw CSV log below
`local_csv/training/`; `training_metrics.csv` is the canonical atomic copy.
`tripso_training_result.json` records its row count, SHA-256, last finite value of
each logged metric, and the fact that no W&B login or API readback was used. The
same summary is embedded in `model_manifest.json`.

## Scheduler markers

`.done.json` contains job ID, manifest-row fingerprint, timestamps, and return code.
`.failed.json` contains the same identity plus error, traceback, and resource log.
Both are written through same-directory temporary files followed by atomic rename.
`resources.before.json` and `resources.after.json` preserve software, memory, GPU,
and scheduler context. A done marker is valid only when all expected outputs exist.

## Missingness

Unavailable rare-fine-type state summaries are null with a reason/eligibility flag;
they are never numeric zero. Undefined metrics due to no age overlap, insufficient
donors, or coverage failure similarly remain typed null values accompanied by a
status column. Cell counts and composition remain available when transcriptional
state is not estimable.
