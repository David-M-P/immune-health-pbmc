# Production workflow: reference training and zero-shot transfer

This is the operational path from the merged lineage count files to a frozen
healthy reference and an unseen query cohort. It deliberately remains a sequence
of restartable CPU and GPU jobs rather than one monolithic command.

| Stage | Typical cluster | Learns from | Main unit |
|---|---|---|---|
| IDs, visits, GP/HVG selection, materialization | local CPU | reference side of the fold | donor |
| Geneformer rank tokenization | local CPU or Gefion CPU | no fitting | cell |
| TRIPSO Base training | Gefion GPU | adaptation cells under a donor-aware sampler | cell draws, donor-controlled |
| Frozen reference/validation projection | Gefion GPU | no fitting | every role-specific cell |
| Locked outer-query projection | Gefion GPU | no fitting; selected jobs only | every held-out cell |
| Distribution aggregation and GP screening | CPU | donor/observation summaries | donor |
| Healthy spline and query scoring | CPU | reference donors only | donor/observation |

## 1. Fix identities and biological roles

The stable identifiers are:

```text
biological_unit_id    = dataset::donor_id
source_observation_id = dataset::sample_id
observation_id        = dataset::donor_id::sample_id
cell_key              = dataset::source_cell_id
```

`biological_unit_id` is the independent-person key. `observation_id` distinguishes
visits or donor-specific aliquots; including the donor is necessary because OneK1K
sample IDs are pools shared by many donors. `cell_key` is the unique row transport
key used to join Arrow output back to metadata.

The same global donor manifest supplies every lineage. A held-out cohort is query;
it cannot enter feature selection, fitting, checkpoint selection, GP selection, or
calibration. A donor's role never changes across lineages, but a lineage file may
contain zero cells for that donor. The preparation manifest therefore freezes the
role-specific physical donor subset and records every globally assigned donor that
cannot be materialized; Arrow data must equal that subset exactly. An internal
donor fold may be withheld for model selection. TRIPSO's
internal cell split is permitted only inside the adaptation-donor Arrow dataset;
it is an optimizer diagnostic, not a biological validation split. For donor-aware
training, the local datamodule replaces the vendor's unrestricted random split
with a deterministic, disjoint approximately 80/10/10 split that guarantees every
observed dataset × donor × fine-type stratum has at least one training cell. Small
strata of one to four cells remain entirely in training. Inner model selection
still uses separate donors.

### Terekhova visits

Exactly one complete Terekhova observation per donor is retained in the production
LODO and final-reference paths. Selection uses the smallest SHA-256 of
`seed::biological_unit_id::observation_id`; age, expression, and outcomes are not
used. This also applies when Terekhova is the held-out LODO query so one donor is
not counted repeatedly. An all-visit query is available only as an explicitly
labelled longitudinal sensitivity and must not be interpreted as extra donors.

## 2. Select GP support and HVGs on CPU

Feature preparation opens each source H5AD in backed mode and streams sparse raw
counts. It never modifies the merged source and never samples cells.

For HVGs, counts are first summed within donor over the selected lineage. Each
donor pseudobulk is normalized to 10,000 counts and log1p transformed. Within each
training cohort, donor variance is standardized in 20 mean-expression bins and
converted to a percentile rank. Cohort percentile ranks are averaged with equal
cohort weight. Thus a cohort with more donors or more cells does not automatically
define the HVG ranking. A gene must be expressed in at least 1% of donors in at
least 75% of the training cohorts under the current defaults.

GP expression/donor support is evaluated on the same training side. The held-out
cohort is not opened to decide which GP survives. Two exact ordered vocabularies
are then created:

```text
top 3,000 training HVGs union every retained GP gene
top 9,000 training HVGs union every retained GP gene
```

The union can contain more than exactly 3,000 or 9,000 genes. Ordering follows the
source H5AD. Separate H5ADs are materialized for each vocabulary and role, and both
variants contain the identical eligible cell rows. The 3k run therefore cannot see
a gene that belongs only to the 9k run.

One unavoidable boundary should remain visible: the requested starting files have
an upstream 18,035-gene universe produced by the earlier multi-cohort merge. HVG
ranking and GP filtering are strictly training-only inside that fixed universe,
but the upstream common-gene universe itself was not reconstructed independently
inside every LODO fold. A completely purist LODO gene-universe test would require
rebuilding fold-specific merges from pre-merge matrices. Future SoundLife/Galsky
projection is still genuinely frozen because those query cohorts do not contribute
to the existing reference merge.

Generate and execute the CPU stages as documented in
[`reference_preparation.md`](reference_preparation.md). The local array requests
the approved account, 4 hours, 96 GB, and 4 CPUs. It creates 1 visit-selection,
30 feature-selection and 160 exact materialization jobs, 150 LODO adaptation/validation/query
tokenization jobs, 50 LODO fold bindings, plus the explicit final fold, final
tokenization, and Stage-3 binding jobs.

## 3. Preserve full library sizes and tokenize every cell

Each materialized H5AD contains only its model vocabulary in `X`, but
`obs["n_counts"]` is calculated from all 18,035 source genes before subsetting.
This is important: recomputing library size from the 3k and 9k matrices would give
the two sensitivity arms different normalization denominators.

The tokenizer, for each cell:

1. divides selected-gene counts by the full-source library size and scales to
   10,000;
2. adjusts genes by the pinned Geneformer training-corpus median;
3. ranks expressed genes within that cell;
4. converts Ensembl IDs to pinned Geneformer token IDs; and
5. stores at most 4,094 ranked genes plus two special tokens.

No cell subsampling and no second HVG calculation occur. A 9k candidate universe
does not mean 9,000 tokens per cell: only expressed genes are ranked, and sequences
above 4,094 genes are truncated. The manifest reports the affected fraction and
per-GP token coverage. It refuses empty/dropped cells, a vocabulary/order mismatch,
or insufficient GP token support.

The physical Arrow dataset retains `cell_key`, dataset, donor, observation, fine
type, and lineage. The fold-input builder applies fixed `inner_fold=0`, declares
those donors as validation, reads the actual adaptation Arrow donors, and proves
that neither validation nor query donors entered training. A declarative table
alone is not accepted as proof.

## 4. Transfer to Gefion

Push code through GitLab. Transfer the packed environment, exact H5ADs or tokenized
Arrow directories, frozen vocabularies, filtered GP CSVs, and their manifests over
SFTP. Include `projection_gp_candidates.json`, every tokenization sidecar, and the
H5AD materialization manifest. Retain relative directory structure and run
`tokenize_tripso relocate-tokenization` on Gefion; it verifies every immutable file
and Arrow shard before atomically rebasing absolute paths. Do not hand-edit the
source JSON. A relocated manifest is the input to fold/projection binding. The
full Geneformer directory is required only for the full-model sensitivity; the
primary hybrid-initialized Base uses the smaller static tensor already vendored in
the repository. Environment packing and unpacking commands are in
[`tripso_environment.md`](tripso_environment.md).

## 5. Understand the three training sampling arms

The primary comparison separates three scientifically different estimands.

### Native all-cell comparator

The vendor loader shuffles its adaptation training cells. A donor with ten times
as many cells contributes approximately ten times as many gradient updates. This
is useful as the conventional cell-weighted comparator, but cell-level training
loss can look excellent while the representation mainly reflects cell-rich donors.

### Donor-uniform, observed composition

The sampler first chooses a cohort with probability proportional to its donor
count (`alpha=1`), then chooses a donor uniformly. Across all cohorts this makes
donors approximately equally exposed. It selects fine types using each donor's
observed proportions in the strata-preserving optimiser training pool
(`lambda=1`), retaining fine-grained biological composition. This is the only
donor-aware arm whose target equals that donor-level mixture exactly; a finite
epoch approaches the target stochastically. It includes `low_confidence` and
`other_confident` through their observed proportions because the uniform component
has zero weight. The training pool is an approximately 80/10/10 cell split that
retains every donor/fine-type stratum, so it closely tracks but need not reproduce
the complete source-cell proportions exactly. Downstream composition is calculated
separately from all projected cells, not from sampler draws.

### Tempered hybrid

The primary donor-aware arm uses `alpha=0.5` and `lambda=0.7`. Cohort size is
square-root tempered, and a donor's fine-type probability is

```text
0.7 * observed donor proportion
+ 0.3 * uniform over that donor's trusted, balance-eligible types.
```

The observed proportions include every physically retained type and sum to one.
The uniform distribution sums to one over only
`fine_type_balance_eligible=true` types and is zero for `low_confidence`,
`other_confident`, and other explicitly ineligible categories. Therefore the
final mixture also sums to one. A trusted rare type receives some of the 30%
uniform uplift; an ineligible type receives only `0.7 * its observed proportion`.
Those cells remain available to training, but annotation uncertainty cannot turn
them into repeatedly oversampled rare states.

This is a controlled compromise, not exact composition preservation. For example,
if a donor has trusted A=60%, trusted B=20%, and low-confidence=20%, the hybrid
targets are A=57%, B=29%, and low-confidence=14%. The observed-composition arm
targets 60%, 20%, and 20%. `lambda=0` is available only as a diagnostic: it draws
uniformly from trusted types and assigns ineligible categories zero probability.
If a donor has no trusted fine type, all sampler modes retain that donor by using
its observed proportions (effective lambda 1); this exceptional fallback and its
reason are explicit in the epoch audit.

Sampling is regenerated deterministically each epoch from seed and epoch. This is
preferable to training forever on one fixed sampled file: common strata rotate
through more cells and no particular rare cell is permanently privileged. Reuse
across epochs is expected; reuse inside a batch is prevented until the entire
positive-probability pool is exhausted. Intended and realized exposure is written
for every cohort, donor, and fine type, including balance eligibility, observed and
uniform components, configured and donor-effective lambda, fallback reason,
zero-probability exclusions, and realized draws. The optimizer-split audit retains
and records source and training cells for every stratum—including
balance-ineligible categories—and aborts if any stratum is lost. The current
primary budget is 256,000 draws per epoch, 20 epochs, batch size 128, on one
GPU/process per job.

Dynamic sampling does not manufacture information for a missing donor/fine-type
combination. `min_cells_per_fine_type=1` physically preserves every observed
label. Older inputs without `fine_type_balance_eligible` safely default to all
types eligible and record that fallback, but approved production inputs must carry
the explicit flag. Regardless of training retention, downstream
empirical-distribution estimates still require enough cells and set underpowered
fine-type state to missing rather than zero.

## 6. Train the hybrid Geneformer-initialized Base

The tokenizer supplies an ordered sequence of gene tokens. In the primary
configuration:

- the `20,275 x 512` gene lookup starts from TRIPSO's static May-2025 Geneformer
  embeddings;
- a trainable two-block, eight-head, 512-dimensional gene encoder contextualizes
  those tokens for this reference dataset;
- masked-gene training hides 25% of gene tokens;
- GP encoders collect the genes belonging to each curated program and learn one
  256-dimensional GP representation per cell and program; and
- GP masked modeling also uses a 25% mask rate.

This is the intended compromise: it transfers Geneformer's learned gene geometry
without paying for the frozen full 12-layer model on every forward pass. The static
initializer has been proven exactly equal to the input-embedding tensor of the
pinned full checkpoint.

`warmup=10` in this vendor implementation means ten complete epochs, not ten steps
or 10%. During those epochs only the trainable gene-encoder loss is used; GP loss
starts afterward. It mirrors the vendor tutorial but is unusually large for a
20-epoch run, so compare it with shorter warmups in the sentinel calibration rather
than treating it as settled.

Full Geneformer instead runs the frozen historical 12-layer contextual encoder.
It requires the downloaded checkpoint and, because that wrapper exposes no
gene-MLM logits, is constrained to `calc_gene_loss=false`, `calc_gp_loss=true`, and
`warmup=0`. It is a sensitivity analysis, not the default.

## 7. Base versus Global

Base directly learns the representations used for GP-age and healthy-deviation
analyses. Sequential Global loads the trained Base, freezes the GP blocks, adds a
cell token/transformer, and learns a whole-cell reconstruction objective. Therefore
Global can add a useful whole-cell embedding and enable Global-only analyses such
as some ablations, but it should not be expected to improve the frozen GP
representation merely because it is a later stage. Keep Base as the main endpoint;
expand Global only if its cell-token endpoint adds prespecified inner-validation
value. The outer held-out cohort is evaluation-only.

## 8. Project the query without adaptation

Query H5ADs are mapped with `materialize-frozen-query-h5ad`; this inserts missing
training genes as zeros in exact frozen order and never recomputes HVGs or filters
GPs. They are then tokenized against the frozen vocabulary, GP definitions,
tokenizer dictionaries, dimensions, and model configuration. The query-input
builder checks hashes and donor disjointness.
Projection loads the selected Base checkpoint, disables optimizer updates and
gradients, sends every query cell through a query-only test loader, and verifies
that model state is unchanged. No query cohort offset is estimated.

TRIPSO writes Hugging Face Arrow. Downstream distribution code consumes aligned
float32 NPY arrays plus Parquet metadata. The focused converter streams one or a
few requested GP columns, joins by unique `cell_key`, verifies key order and array
shape, and hashes both outputs. A mere equality of row counts is rejected. This is
the explicit fix for the former Arrow-versus-NPY inconsistency.

## 9. Aggregate cells without treating them as people

For each dataset × donor × observation × lineage × fine type × GP, retain:

- cell count and fine-type composition;
- robust location of the GP embedding distribution;
- shrinkage dispersion/covariance summaries;
- empirical quantiles when supported; and
- empirical and Gaussian distributional distances when cell depth allows.

Cells contribute to the precision and shape of their donor's distribution. They
do not increase the inferential donor count. Rare fine types can retain composition
while their unmeasurable state is missing. Whole-lineage outputs keep observed
composition, composition-standardized state, composition-only deviation, within-
fine-type heterogeneity, and between-fine-type heterogeneity separate.
`low_confidence` and `other_confident` are retained in the composition denominator,
but the ontology marks them state-ineligible: aggregation writes no state vector
for them and endpoint assembly fails closed if either is requested for GP-age
fitting.

## 10. Select transferable GPs and fit two healthy references

The final GP decision is downstream of reference projection, not a screen on raw
TRIPSO coordinates. First materialize a role=`reference` endpoint for every
training-only candidate GP × fine type × neural seed. At this stage do **not**
materialize or score the outer query. The example below is therefore repeated for
all candidates and seeds, not just for an already selected GP:

```bash
python -m immune_health.cli assemble-donor-gp-endpoint \
  --aggregate-table .../fine_type_distributions.parquet \
  --aggregation-manifest .../donor_distribution_aggregation_manifest.json \
  --projection-output-manifest .../projection_output_manifest.json \
  --lineage "B cells" \
  --fine-type "Naive B" \
  --gp-id BLOODGEN3__M10_2 \
  --output-dir .../endpoints/reference/B_cells/Naive_B/M10_2
```

The endpoint contains aligned metadata, float32 location NPY, float32 covariance
NPY, and a self-hashed manifest. In LODO, the role=`reference` manifest declares
the held-out dataset but rejects it if physically present.

Fit and report two paired references in separate output directories:

```bash
python -m immune_health.cli fit-healthy-reference \
  --metadata .../endpoint_metadata.parquet \
  --features .../endpoint_locations.npy \
  --endpoint-manifest .../endpoint_manifest.json \
  --weighting-scheme donor_pooled \
  --output-dir .../reference/donor_pooled

python -m immune_health.cli fit-healthy-reference \
  --metadata .../endpoint_metadata.parquet \
  --features .../endpoint_locations.npy \
  --endpoint-manifest .../endpoint_manifest.json \
  --weighting-scheme cohort_balanced \
  --output-dir .../reference/cohort_balanced
```

Fit each seed/candidate independently. The fit writes
`training_crossfit_scores.parquet`, where each donor's `predicted_gp_age` was
produced by a healthy trajectory that did not contain that donor. Its
`inner_crossfit_fold`, file hash, score semantics, and training-only scope are
bound into `healthy_reference.json`. The cross-fit table—not the in-sample final
spline score—is the only TRIPSO value accepted by the final selector.

After every candidate exists for every required seed, freeze the final set from
the primary `donor_pooled` references:

```bash
python -m immune_health.cli select-transferable-tripso-gps \
  --reference-manifest-list .../reference_manifests.tsv \
  --lineage "B cells" \
  --fold-id lodo_soundlife \
  --heldout-dataset soundlife \
  --required-training-dataset aidav2 \
  --required-training-dataset immuneindonesia \
  --required-training-dataset immunobiologyaging \
  --required-training-dataset onek1k \
  --required-training-dataset terekhova \
  --required-seed 42 --required-seed 43 --required-seed 44 \
  --weighting-scheme donor_pooled \
  --output-dir .../gp_selection/B_cells
```

The list can be one manifest path per line or a TSV/CSV with a
`reference_manifest` column; relative paths are resolved beside the list. The
selector revalidates the endpoint, model/checkpoint, fold-input manifest, exact
training-cohort universe, held-out exclusion, cross-fit row order, and every file
hash. It requires the same candidate set under every declared seed and fails
closed for a missing seed/cohort, mixed model configuration, duplicate endpoint,
or any query row. Raw location/covariance coordinates are not read for selection.

Within each seed, sex-adjusted age slopes are estimated separately in every
training cohort. Repeated observations from one donor share one unit of total
weight and standard errors are donor-clustered. The screen applies donor/age-span
support, direction concordance, heterogeneity, FDR, effect-size, fine-type state
coverage, median cell depth, residual depth/composition dependence, and
cross-seed retention/direction/rank/effect stability. All thresholds are available
under `tripso_gp_selection` in YAML and as same-named CLI flags. A donor-level
simple score table may be supplied with `--simple-baseline`; it is a labelled
comparison and only becomes a gate when
`minimum_baseline_standardized_improvement` is configured.

The immutable outputs are:

```text
tripso_gp_cohort_seed_effects.parquet
tripso_gp_selection.parquet
selected_tripso_gps.json
```

`selected_tripso_gps.json` is canonical self-hashed, binds both Parquet hashes and
all input artifacts, and exposes exact `lineage`, `fine_type`, `gp_id` records in
`selected_endpoints`. Only after this file validates should the workflow generate
role=`query` endpoint/projection/scoring tasks for those endpoints. An empty result
is recorded as `complete_no_candidates`; it never falls back to every GP.

The held-out dataset is read from a validated LODO endpoint manifest. The old
single-table reference-plus-query input is accepted only with the explicit
`--legacy-combined-lodo-input` compatibility flag.

`donor_pooled` gives every donor one total unit, so a cohort with more donors has
more influence. `cohort_balanced` still equalizes donors within cohort but gives
every cohort the same total weight. Both are scaled to the same total effective
donor count so the ridge penalty remains comparable. Donor-pooled is the primary
population estimate; cohort-balanced is the transfer sensitivity.

For a manifest-bound endpoint, each command writes
`healthy_reference.{json,npz}` (the spline of donor locations) and the small
`age_kernel_reference.json` configuration/binding. The latter reuses the immutable
endpoint location/covariance NPY files by hash instead of copying the potentially
large covariance tensor. The kernel uses the same donor/cohort
weighting scheme and exact-sex matching when support is adequate. Query
distribution shift is the age-matched Gaussian 2-Wasserstein/Bures distance. The
query donor covariance is never compared with the spline's residual covariance of
donor locations; `HealthyTrajectory` explicitly rejects that input. Empirical
sliced Wasserstein remains `not_computed` unless a separately manifest-validated
donor row-index/cell-embedding artifact is supplied.

The approved exact-sex threshold is **20 unique reference donors for the
endpoint**. With 19 or fewer donors of the query sex, the kernel deliberately
falls back to the pooled-sex reference and reports `exact_sex_used=false`; with
20 or more it uses the exact-sex subset (and cohort-balanced scoring additionally
requires support from at least two cohorts). This global endpoint threshold does
not imply good support near every age, so the separate age-local support flags
must still be inspected.

The observed reference cohorts are materially age-unbalanced: for example,
OneK1K has a donor median near 67–68 years, AIDA near 40, and
ImmunobiologyAging begins around age 40. More older donors improve precision in the
older range but can pull knot placement and fitted effects toward that region.
Neither weighting scheme creates young observations in an old-only cohort.
Centered cohort intercepts help when ages overlap; when cohort and age are nearly
separable, a cohort effect and an age effect cannot be identified cleanly.
Consequently, the primary safeguards are age-support flags, age-overlap reporting,
within-cohort slopes, and sign/heterogeneity screening. An inverse-age-bin weighted
fit may be added as a prespecified sensitivity, but it changes the target population
and can make sparse age/sex bins extremely influential, so it is not the default.

The spline includes nonlinear age, sex, and centered training-cohort intercepts.
For an unseen cohort, all cohort-effect columns are set to zero: prediction is at
the weighted average training-cohort intercept. The query cohort never gets a
fitted correction. This makes zero-shot scoring possible but does not make it
immune to a new batch effect.

Every query score receives age-support diagnostics: nearby same-sex donor count,
number of supporting training cohorts, whether age lies in the overall sex-specific
range, and whether it lies in the intersection of every cohort's sex-specific age
range. Extrapolated or weakly supported scores remain visible as such.

## 11. Interpret the primary reference outputs

- `predicted_gp_age`: point on the frozen healthy trajectory closest to the donor.
- `gp_age_acceleration`: predicted GP age minus chronological age.
- `age_matched_location_distance`: Euclidean departure of the donor location from
  the healthy spline at actual age and sex (`age_matched_distance` is retained as
  its compatibility alias).
- `off_trajectory_location_distance`: minimum location departure from any point on
  the healthy path (`off_trajectory_distance` is its compatibility alias).
- `age_matched_gaussian_wasserstein_distance`: distributional departure using the
  donor location and within-cell covariance against the age-kernel healthy
  mixture. Kernel support donors/cohorts, effective support, exact-sex fallback,
  and extrapolation are reported beside it.
- `predicted_distributional_gp_age` and
  `distributional_gp_age_acceleration`: the closest age on the configured
  distributional grid and its difference from chronological age.
- `off_trajectory_gaussian_wasserstein_distance`: the minimum Gaussian/Bures
  departure over that distributional age grid.

Movement along the healthy path is not the same as disease-like abnormality. A
cancer-associated inflammatory pattern may have high age-matched and off-path
distance without meaningfully being an “older” healthy state.

## 12. Quantify uncertainty

Use separate layers:

1. stratified within-observation cell bootstrap for measurement precision;
2. donor-within-cohort bootstrap for healthy-reference uncertainty; and
3. independent Base seeds for model uncertainty.

Do not retrain TRIPSO for thousands of bootstrap replicates. Fit three neural seeds,
calibrate each against its own healthy reference, and run 2,000 donor bootstraps on
CPU for routine intervals (5,000 for final tail estimates if needed). Report cell
sampling SE, reference-bootstrap SE, and seed SD separately.

The runnable stages are:

```bash
# Whole donors are resampled within cohort and sex, then the reference is refitted.
immune-health bootstrap-scores --uncertainty-layer reference \
  --metadata reference/endpoint_metadata.parquet \
  --features reference/endpoint_locations.npy \
  --query-metadata target/one_endpoint_row.parquet \
  --query-features target/one_endpoint_location.npy \
  --output uncertainty/reference_replicates.parquet \
  --manifest uncertainty/reference_replicates.manifest.json \
  --n-bootstrap 2000 --weighting-scheme donor_pooled --seed 42

# Combine only already calibrated scalar endpoint scores across model seeds.
immune-health combine-seed-scores \
  --scores seed_42/empirical_matched_depth_scores.parquet \
  --scores seed_43/empirical_matched_depth_scores.parquet \
  --scores seed_44/empirical_matched_depth_scores.parquet \
  --required-seed 42 --required-seed 43 --required-seed 44 \
  --output uncertainty/seed_summary.parquet
```

The bootstrap stage writes auditable replicates and a self-hashed manifest; it does
not relabel a spline residual covariance as cell dispersion. Summarize its replicate
SD as `reference_sampling_se` downstream. The seed stage requires the same complete
seed set for every biological endpoint and computes `seed_sd` only after each seed
has its own projection and healthy-reference calibration.
