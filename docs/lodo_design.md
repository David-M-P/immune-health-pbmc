# Leave-one-dataset-out design

Each outer fold holds out exactly one of `aidav2`, `immuneindonesia`,
`immunobiologyaging`, `onek1k`, or `terekhova`. The other four datasets define the
reference side. All five primary lineages use the same global donor manifest so a
person cannot acquire a different role through a lineage, sample, or fine type.
The role assignment is global, but the expected physical donor inventory is the
subset with materializable cells in that lineage after visit selection. Feature
preparation records that subset and every excluded global donor in a self-hashed
`lineage_donor_scope`; a donor is never reassigned merely because a lineage is
absent.

## Why dataset holdout matters

LODO tests transfer across cohort and technical context rather than memorization of
pooled study effects. The fifth healthy dataset is processed as an unseen query,
rehearsing the future SoundLife and Galsky path. Performance is reported for every
fold before aggregation; an overall mean cannot conceal a failed cohort.

## Nested donor roles

Within the four reference datasets, donors—not cells—are assigned to internal
folds. Production fixes `inner_fold=0` as validation before feature preparation;
those donors are excluded from HVG/GP selection and model adaptation. They are
materialized and projected separately for sampler, hyperparameter, GP, and seed
selection. Stratification may use dataset, age bins, and sex, but donor integrity
wins over perfect balance. A donor and all observations remain in one role.

The project adapter distinguishes an internal vendor cell split from biological
validation. If TRIPSO needs cell subsets for numerical optimization, those cells
come exclusively from donors already permitted for adaptation. Internal validation
donors and outer query donors never enter weight updates.

## Leakage boundary

The query dataset cannot influence:

- vocabulary, HVGs, gene mapping decisions, GP inclusion, or fine-type thresholds;
- preprocessing statistics or hierarchical sampling probabilities;
- model, checkpoint, hyperparameter, or seed selection;
- healthy-age GP selection, trajectories, score calibration, or composite weights.

The fold input manifest stores exact lineage-available adaptation, fixed
inner-validation, and outer-query donor IDs, the audited global-fold exclusions,
and all immutable hashes. The model manifest repeats the selection contract.
Validation projection must equal the declared lineage-local inner donor set; outer
query projection must equal the lineage-local held-out donor set and is generated
only after an explicit selected-job allowlist. Every projection checks that model
state is unchanged.

## Donor-hierarchical sampling

Training samples in the order dataset → donor → fine type → cell. Dataset selection
uses

```text
P(dataset=s) proportional to N_s ** alpha
```

and a donor is then selected approximately uniformly. For donor `d` and eligible
fine type `k`, the intended fine-type probability is

```text
q[d,k] = lambda * p[d,k] + (1 - lambda) * u[d,k]
```

where `p[d,k]` is the observed proportion over every retained fine type. `u[d,k]`
is uniform only over that donor's trusted types for which
`fine_type_balance_eligible=true`; it is zero for `low_confidence`,
`other_confident`, and any other explicitly ineligible type. Thus an ineligible
type receives `lambda * p[d,k]` but never the rare-type uniform uplift. The two
components each have their stated total mass, so `q[d,*]` sums to one. A donor
with no trusted fine type cannot receive a meaningful uniform component; rather
than dropping that donor or its cells, the sampler falls back to the donor's
observed proportions (effective lambda 1) and records the donor and reason in the
audit. Inputs made before the reviewed eligibility flag existed default to all
types eligible and record that separate backward-compatible fallback in the audit.

The primary Stage-1 comparison has three different estimands:

1. `native_all_cells` disables the project sampler and uses the native vendor
   training loader, providing the cell-weighted baseline.
2. `donor_uniform_observed` uses alpha 1 and lambda 1. Dataset probability is then
   proportional to donor count, donors are uniform within dataset, and the donor's
   observed fine-type composition is preserved exactly, including the special
   ineligible categories; the uniform term has zero weight.
3. `hybrid` uses alpha 0.5 and lambda 0.7, partially balancing cohorts and fine
   types while retaining most of the observed fine-grained composition. Trusted
   types share the 30% uniform component. Ineligible types remain physically
   present and can be drawn through their 70% observed component only.

`fully_balanced` (alpha 0.5, lambda 0) remains available as an optional diagnostic,
but it is not part of the primary screen because very rare or noisy fine types can
be strongly oversampled. Ineligible categories have zero probability in this
diagnostic and their exclusion is logged explicitly. The optimizer split still
retains at least one training cell from every observed stratum, including these
categories; balance eligibility changes sampling uplift, not physical retention.
The only exception to zero draw probability is an all-ineligible donor: that donor
uses the audited observed-proportion fallback so it is not erased from training.
Intended and realized distributions log eligibility, observed and uniform
components, configured and donor-effective lambda, fallback reason, available
cells, zero-probability strata, and realized draws;
distributed ranks derive distinct deterministic streams from the explicit job
seed. The native arm has no alpha or lambda because it does not invoke the project
sampler.

## Feature-set comparison

Every fold learns features using reference donors only. Two full-cell inputs are
materialized: 3,000 fold-local donor/dataset-aware HVGs plus all retained GP genes,
and 9,000 HVGs plus all retained GP genes. The comparison changes the eligible gene
universe, never the number of cells. Each generated job records `hvg_size`, the
feature-set label, and the fact that retained GP genes are unioned into the
vocabulary. Output and fold-input paths include the feature set, preventing the two
arms from overwriting or silently sharing resources.

Stage 1 trains all 150 lineage × fold × sampler × feature combinations with the
base seed. Stage 2 adds two new seeds for the two selected configurations per
lineage. It reuses the matching Stage-1 base-seed result rather than training that
seed again.

## Evaluation slices

Each fold reports the full query cohort plus age-overlap, sex, cell-depth,
GP-expression-coverage, and fine-type-coverage slices when donor counts permit.
Metrics include age calibration and error, age acceleration, age-matched deviation,
off-trajectory distance, residual dataset predictability, depth sensitivity, seed
stability, and improvement over baselines. Statistical uncertainty is donor-aware.
