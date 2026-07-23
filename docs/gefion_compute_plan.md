# Gefion 10,000-hour compute plan

Treat the allocation as a capped experimental budget, not a reason to run the full
factorial. Gefion bills the complete exclusive eight-GPU node. One allocated
node-hour therefore costs eight GPU-hours whether one or all eight devices are
busy. Production uses the node-pack launcher: eight independent one-GPU models or
projections per node, not one eight-GPU DDP model.

## Calibrate before expanding

Use one GPU per independent model and run eight models concurrently per billed
node. Run full-duration Stage-1 sentinel jobs for two biologically different
lineages (for example B cells and Monocytes), both feature sets, and all three
primary samplers. These 12 jobs are useful results, not throwaway benchmarks, and
occupy two node allocations (eight plus four active GPUs). Record end-to-end time
for token loading, Base training, checkpointing,
and fixed inner-validation projection separately by lineage and HVG size. Keep
outer-query projection out of calibration. Use the 90th-percentile
observed runtime for budgeting.

Also run four short technical checks:

- hybrid Base checkpoint/resume and memory;
- full Geneformer forward/backward and memory;
- sequential Global load/freeze/reconstruction; and
- frozen all-cell validation projection plus Arrow conversion.

Cap calibration near 200–300 GPU-hours. Do not launch the full matrix until the
Gefion CUDA smoke, token coverage, sequence truncation, sampler exposure, and output
hashes have been inspected.

## Main matrix

| Stage | Design | Jobs |
|---|---|---:|
| Base inner-validation screen | 5 lineages × 5 outer folds × 3 samplers × 2 HVG sets × seed 42 | 150 |
| Base inner-validation confirmation | top 2 sampler/HVG pairs per lineage × 5 folds × seeds 43–44 | 100 |
| Final all-healthy reference | winner × 5 lineages × first 3 seeds | 15 |
| Optional production seeds | same final model × seeds 4–5 | 10 |
| Full Geneformer gate | 2 lineages × 2 folds, then at most all 25 | 4 → 25 |
| Sequential Global gate | 2 lineages × 2 folds, then at most all 25 | 4 → 25 |

At eight workers per node, 150 jobs map to 19 node elements, 100 to 13, the
15-job final core to two, and 25 jobs to four. A partial final element is still
billed for all eight GPUs. Never fill a partial element with sealed query work or
an otherwise scientifically unauthorized job merely to improve utilization.

The 150/100 selection jobs are ranked only on donors held inside the four
reference cohorts; outer query cohorts remain sealed until the configuration is
frozen. Stage 2 reuses the matching Stage-1 seed-42 result. The generated Stage-3 manifest
contains five seeds so the final two can be released if budget remains; the core
recommendation is three.

Do not cross full Geneformer × Global × every sampler × both feature sets. Full
Geneformer and Global answer separate sensitivity questions after the Base winner
is known.

## Budget equation

After calibration let:

- `b` be hours for one Base LODO job including reference and inner-validation
  projection (not outer-query evaluation);
- `p` be hours for one final all-healthy Base job and its required projections;
- `g` be hours for one full-Geneformer sensitivity;
- `q` be incremental hours for one sequential-Global job; and
- `C_query` be the later SoundLife/Galsky projection allowance.

The three-seed core is approximately:

```text
250*b + 15*p + up_to_25*g + up_to_25*q + calibration + C_query
```

These are GPU-hours. To estimate exclusive-node wall time for a homogeneous
eight-way packed phase, divide the fully occupied portion by eight and add complete
node billing for each partial bundle. The `%` limit in a node-packed Slurm array
counts concurrent nodes, so `%4` means at most 32 simultaneous model workers.

Reserve at least 15% (about 1,500 hours) for failed/requeued jobs, locked outer-query projection,
and justified follow-up. As a planning example only, if `b=20`, `p=25`, `g=60`,
and `q=15`, the Base screen costs 3,000 hours, confirmation 2,000, final reference
375, full Geneformer 1,500, Global 375, and calibration about 200: 7,450 hours,
leaving 2,550 hours of reserve. These are not runtime predictions.

If Base calibration is slower, use successive halving: complete two or three outer
folds for all six sampler/HVG pairs, retain three based on prespecified donor-level
metrics and rare-fine-type exposure, finish all five folds for those, then retain
two for extra seeds. Never eliminate a configuration based on one lineage alone.

## Hyperparameters worth searching

Prespecify the two HVG sets and three sampling estimands; alpha and lambda are not
continuous tuning parameters in the primary analysis. They define whose cells the
model learns from.

On two sentinel lineages, use inner donor validation to test:

- learning rate: `5e-5`, `1e-4`, `2e-4`;
- warmup epochs: at least the vendor-like `10` versus a shorter `2` (and `0` if
  learning curves show stable joint training);
- total exposure, decided from learning curves rather than an arbitrary epoch grid;
  and
- effective batch size only for memory/throughput, using gradient accumulation to
  keep the effective batch comparable.

Initially fix hidden size 512, GP latent size 256, two blocks, eight heads, weight
decay `1e-4`, both masking rates at `0.25`, and the 4,096-token cap. A broad search
over architecture depth, latent size, weight decay, masking, alpha, and lambda would
consume the allocation while making model selection unstable.

For an unbiased LODO estimate, make hyperparameter decisions on donor-held inner
validation data and use the outer cohort once. The preparation CLI supports an
`--inner-validation-fold` that removes those donors before HVG/GP selection; train
on the adaptation H5AD and project the validation H5AD with role `validation`.
Only after a selected-job allowlist is frozen should the outer query be projected.
If configurations are instead chosen using the five outer LODO results,
report every candidate and label the chosen configuration's LODO performance as
selection-biased rather than a clean final test.

## Model-selection endpoint

Do not select by cell-level validation loss alone. Use a prespecified donor-level,
cohort-balanced transfer score, with guardrails for:

- donor-level age calibration and off-trajectory behavior;
- performance in every fold, not only the pooled mean;
- rare fine-type exposure and output coverage;
- residual cohort predictability;
- gene/GP token coverage and 4,094-gene truncation;
- stability across the first and subsequent seeds; and
- improvement over composition, pseudobulk GP, and PCA baselines.

## Bootstrap and replication

Neural replication and statistical bootstrap answer different questions:

- use three Base seeds for primary model variability;
- keep all five LODO cohort holdouts for transfer variability;
- use approximately 100 stratified cell bootstraps for development and 500–1,000
  for final measurement-uncertainty summaries where necessary;
- use 2,000 donor-within-cohort bootstrap replicates for routine reference and
  performance confidence intervals, increasing to 5,000 for final tail estimates;
  and
- never retrain TRIPSO thousands of times for a donor bootstrap.

Calibrate each neural seed against its own healthy reference before combining
donor-level quantities. Report cell-sampling SE, donor/reference-bootstrap SE, and
between-seed SD separately.
