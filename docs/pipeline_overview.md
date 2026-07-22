# Pipeline overview

The pipeline learns a healthy PBMC reference from independent donors, freezes it,
and scores an unseen cohort through the same interface that will later be used for
SoundLife and Galsky. Cells are repeated measurements within a person: they improve
the precision of a donor's state distribution, but they do not create additional
biological replicates.

## Stages

1. Audit read-only merged lineage H5ADs and their provenance.
2. Add stable donor and observation identifiers and construct one global donor
   split shared by all lineages.
3. Build each LODO fold and its training-only gene vocabulary, GP library, fine-type
   decisions, sampler, and internal donor folds.
4. Fit donor-level baselines and the hybrid Geneformer-initialized TRIPSO Base using
   only donors permitted to adapt in that fold. Treat sequential Global and full
   Geneformer as gated sensitivity analyses.
5. Freeze vocabulary, tokenizer, preprocessing, GP definitions, model configuration,
   embedding dimensions, weights, checkpoint choice, and random seed.
6. Project every held-out query cell without an optimizer or gradient update.
7. Retain fine-type GP distributions and aggregate them into distinct composition,
   cell-state, heterogeneity, and uncertainty components.
8. Discover healthy-age directions and fit calibration using reference donors only.
9. Score the fifth dataset and report its fold separately before any cross-fold
   summary.

The primary lineages are `B cells`, `NK_ILC`, `Monocytes`, `CD4_like`, and
`CD8_like`. `T_others`, `DC`, and `pDC` remain audit targets rather than first-pass
model targets.

## What is fold-specific

Every item that could learn from the data is rebuilt inside the four-dataset
reference side of a LODO fold: feature and HVG selection, identifier mapping
coverage decisions, GP filtering, fine-type eligibility, sampler distributions,
preprocessing statistics, TRIPSO fitting, early stopping, checkpoint selection,
healthy-age GP selection, trajectories, score calibration, and composite weights.
The held-out dataset contributes none of these choices.

The vendor TRIPSO datamodule creates an internal random cell 80/10/10 split. The
native cell-weighted comparator retains that optimizer behavior inside an input
already restricted to adaptation donors. Donor-aware arms replace it locally with
a disjoint, approximately 80/10/10 split that guarantees every observed dataset ×
donor × fine-type stratum contributes training cells. Neither is reported as
biological validation; internal model selection remains grouped by donor.

## Frozen transfer

Held-out healthy data are deliberately treated like a future SoundLife or Galsky
query. The projection adapter loads a validated checkpoint, disables gradients,
uses Lightning's test loop over the complete query dataset, and verifies that the
full model state hash is unchanged. Its output is an out-of-core Arrow dataset from
the vendored embedding writer. A resource or coverage mismatch is an error, not an
invitation to retrain on the query.

## Comparison path

Pseudobulk expression, pathway scores, PCA and composition-only models use the same
donor folds and held-out scoring contract. Their donor-level outputs share model,
fold, seed, annotation, vocabulary, and reference provenance with TRIPSO outputs.
This makes it possible to determine whether learned distributions improve on
simpler and more interpretable representations.
