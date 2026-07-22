# Donor distribution aggregation

TRIPSO produces cell-level embeddings, but biological inference starts from donor
observations. For every dataset × donor × observation × lineage × fine type × GP,
the pipeline retains the empirical embedding distribution `z[d,k,g,i]` rather than
immediately replacing it with a broad-lineage centroid.

## Fine-type summaries

For each stratum, record cell count and fraction, annotation confidence, a robust
location, shrinkage covariance summaries, covariance trace and stabilized log
determinant, median distance to centre, and robust distance quantiles. Raw
unrestricted covariance is not valid when cell count is small relative to embedding
dimension.

Age-direction summaries include q10, q25, median, q75, q90, and the fraction outside
the training-only healthy 95% region. Primary distributional comparison uses
deterministic sliced Wasserstein projections; Gaussian/Bures-Wasserstein and
centroid distance are comparators.

## Composition and transcriptional state

The observed fine-type proportion is `p[d,k] = n[d,k] / sum_j n[d,j]`.

Observed-mixture state weights each fine-type state by the donor's actual
composition. It therefore contains both composition change and within-fine-type
transcriptional change. Composition-standardized state replaces those weights with
healthy expected proportions for the same age and sex, isolating state comparisons
from the donor's observed mixture. Composition-only deviation compares the observed
and expected proportions without using transcriptional coordinates. These are
separate outputs, not interchangeable score variants.

## Heterogeneity decomposition

Whole-lineage covariance is decomposed as

```text
total covariance
  = weighted within-fine-type covariance
  + covariance of fine-type centres
```

The first term measures diversity among cells within types; the second measures
separation among type centres. Total lineage heterogeneity combines both. None of
these quantities is an uncertainty estimate.

## Heterogeneity versus uncertainty

Biological heterogeneity describes the width or multimodality of a cell-state
distribution. Uncertainty describes how precisely that heterogeneity or a distance
has been estimated from sampled cells, donors, references, and model seeds. A large
Wasserstein distance does not supply its own standard error.

Cell-sampling uncertainty uses a bootstrap stratified by fine type, optionally with
multinomial or Dirichlet-multinomial composition resampling. Reference uncertainty
resamples training donors and refits trajectories. Seed uncertainty fits and
calibrates each TRIPSO seed separately; arbitrary coordinates from independently
trained models are not averaged. Report cell-sampling SE, reference-bootstrap SE,
and seed SD separately.

## Rare types

Sufficient-cell strata support empirical Wasserstein estimates. Limited-cell strata
use shrinkage location/covariance with visibly larger uncertainty. Insufficient-cell
strata retain their composition contribution but have missing state estimates. A
missing distribution is never encoded as zero. Matched-depth sensitivity at 25, 50,
100, 250, 500, and 1,000 cells will calibrate final thresholds.

## Empirical storage contract

Aggregation does not copy every cell embedding into an
`empirical_distributions.npz`. It writes a compact Parquet group table containing
the exact observation/lineage/fine-type/GP key and contiguous `start`/`stop`
offsets, plus one `int64` `empirical_distribution_rows.npy`. Its manifest binds
those rows to the immutable converted float32 NPY, aligned metadata, ordered
`cell_key` digest, embedding payload hash, Arrow-conversion manifest, and frozen
projection-output manifest.

Distance and bootstrap code must open the source NPY with memory mapping and
gather only the requested group's rows through
`load_empirical_distribution_store`; production code must not recreate an NPZ
copy of the embedding values.

The row-index manifest also binds the exact donor aggregation Parquet hash. This
closes the former Arrow-versus-standalone-NPY ambiguity: Arrow remains the model
output, conversion creates one immutable float32 NPY for numerical access, and the
row index only stores integer row positions into that NPY. Endpoint scoring rejects
an index whose projection, Arrow conversion, aggregation table, GP, or embedding
dimension differs from either endpoint.

## Matched-depth empirical scoring

Run empirical scoring only after assembling one exact reference endpoint and its
matching role-specific target endpoint:

```bash
immune-health score-empirical-endpoint \
  --reference-endpoint-manifest reference/endpoint_manifest.json \
  --query-endpoint-manifest target/endpoint_manifest.json \
  --reference-empirical-index reference/empirical_distribution_manifest.json \
  --query-empirical-index target/empirical_distribution_manifest.json \
  --output-dir empirical_scores \
  --depth 25 --depth 50 --depth 100 \
  --n-replicates 20 --n-projections 32 --age-grid-size 21 \
  --weighting-scheme donor_pooled --seed 42
```

The target role may be `validation` during inner model selection or `query` for the
sealed outer evaluation. It may never be `reference`. For each donor observation,
depth, and replicate, query cells are sampled without replacement. The same number
of reference cells is then generated by first drawing a healthy donor observation
from the deterministic age/sex kernel and then drawing a cell uniformly within that
observation. Donor-rich cell counts therefore do not increase that donor's mixture
weight. `donor_pooled` gives donors equal base exposure; `cohort_balanced` first
gives cohorts equal total exposure and then donors equal exposure within cohort.

The command reports age-matched and minimum-over-age-grid sliced-Wasserstein
distance, predicted empirical GP age, age acceleration, replicate-level values, and
a depth reliability table. `cell_sampling_se` is the sample SD across matched-depth
cell-resampling replicates. It is a measurement-variability estimate and is not
divided by square root of the number of replicates. It is not donor/reference
uncertainty or neural-seed uncertainty. Unsupported depths remain explicit as
`insufficient_query_cells`; they are never imputed as zero.

Work scales approximately as
`query rows × depths × replicates × age-grid points × projections`, with cell depth
and embedding width determining the cost of each projected comparison. For a quick
development check use depths 25/50/100, 20 replicates, 32 projections, and a 21-age
grid. For frozen final estimates use all feasible depths through 1,000, 100--200
replicates, 128 projections, and a 101-age grid. Increase replicates to 500 only for
endpoints whose reliability curve or uncertainty materially affects a conclusion.
Do not expose outer-query artifacts while tuning these settings.

## Age position and abnormality

GP age acceleration is displacement along a learned healthy ageing trajectory after
calibration: it asks whether a donor appears older or younger along healthy change.
Off-trajectory distance is orthogonal or manifold deviation from age-appropriate
healthy states: it asks whether the donor is abnormal in a way that cannot be
explained by healthy ageing. A donor can have little age acceleration but a large
off-trajectory abnormality, or the reverse.
