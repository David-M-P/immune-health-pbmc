# Gene-program provenance and curation

## Current production library

The inspected production resource is
`data/intermediate_data/gene_programs/v1/gene_programs_curated.gmt`; TRIPSO uses
the equivalent column-oriented `gpdb_curated.csv`. Its immutable provenance is in
`source_manifest.json`. The library contains 500 programs and 24,364 retained
program-gene memberships:

| Source | Source programs | Retained programs |
|---|---:|---:|
| BloodGen3 | 382 | 369 |
| MSigDB Hallmark 2026.1.Hs | 50 | 50 |
| PROGENy through decoupler 2.1.6 | 14 | 14 |
| MSigDB Reactome 2026.1.Hs | 1,839 | 67 |
| Total | 2,285 | 500 |

CollecTRI and a project-specific immune-age library are supported as explicit
configuration slots, but they are not silently fabricated: both remain absent from
the current production bundle. The downstream transferable-age screen discovers
age-associated programs from the source library using training donors; it is not a
substitute for a separately curated immune-age resource.

## Identifier mapping

Source symbols and Ensembl identifiers were mapped using the project gene maps and
the pinned HGNC complete set dated 2026-01-06. Exact symbols, previous/alias
symbols, source Ensembl IDs, unmapped rows, one-to-many mappings, and many-to-one
mappings are recorded. The production TRIPSO resources use Ensembl IDs. Fold
preparation never silently converts an Ensembl vocabulary as though it contained
symbols; the training adapter rejects that configuration.

## Global curation and overlap control

The source-universe curation was deterministic and recorded before the current LODO
analysis. Programs with fewer than 12 or more than 300 mapped genes were removed.
Programs between 20 and 150 genes and near the 60-gene target were preferred.
Source priority was Hallmark, PROGENy, BloodGen3, then Reactome, with immune/PBMC
Reactome terms prioritized and unrelated Reactome topics deprioritized.

Redundancy was handled directly: a candidate was removed when its Jaccard overlap
with an already retained higher-priority program exceeded 0.70 or its overlap
coefficient exceeded 0.85. The library was then capped at 500 programs. Of 2,285
candidates, 461 failed the minimum size, 61 exceeded the maximum size, 61 were
removed as redundant, and 1,202 fell beyond the 500-program cap. The retained
program size has median 29, mean 48.7, and range 12–201 genes.

This makes high overlap less severe, not impossible. Two pathways may share a
biological core while remaining below both thresholds, and BloodGen3 modules can
partially overlap broader Hallmark or Reactome biology. Do not interpret 500 GP
tests as 500 independent hypotheses. Report the overlap matrix or clusters when
interpreting selected programs, and perform sensitivity analyses at the pathway
family level.

Flags such as `tbd`, cell identity, ribosomal/translation, cell cycle, and
mitochondrial were retained as annotations rather than silently deleted. Among the
500 programs, 172 carry `tbd`, 61 cell-identity, 24 ribosomal/translation, 17
cell-cycle, and 7 mitochondrial flags. These programs can be excluded in a
prespecified sensitivity analysis, but their removal should not be decided after
looking at query outcomes.

## Fold-local filtering

Global curation does not guarantee that every program is measurable in every
lineage or LODO fold. Within each four-cohort training side, preparation checks:

- mapped and tokenizable gene count;
- program size;
- expression and donor coverage across training cohorts; and
- remaining pairwise Jaccard redundancy.

The held-out cohort is not used. Every retained GP gene is unioned with the 3,000
or 9,000 training-only HVGs so a pathway gene cannot disappear merely because it
is not highly variable. The materialized H5AD, frozen vocabulary, GP CSV, token
coverage table, and hashes bind the exact program definition used by training and
query projection.

## Selecting transferable ageing programs

For each lineage and fine type, every training-only candidate is first projected
under each independent TRIPSO seed and aggregated to a role=`reference` donor
endpoint. A donor-grouped inner cross-fit then predicts GP age: the trajectory
that scores a donor was fitted without that donor. The final selector accepts only
the manifest-bound `predicted_gp_age` column from those cross-fit tables; it does
not choose programs from in-sample fitted values or raw TRIPSO coordinates.

A sex-adjusted predicted-GP-age slope is fitted separately within every training
cohort, with repeated observations sharing one donor's total weight and standard
errors clustered by donor. The audit reports each cohort's donor count, age span,
slope, uncertainty, standardized effect, neural seed, and model identity. It also
reports measurable-state coverage, median/total cells, and age/sex/cohort-adjusted
associations of GP-age acceleration with log cell depth and logit fine-type
composition.

A fixed-effect summary is used only as a screen. The default candidate rule requires
at least three eligible training cohorts, at least 75% sign concordance, I-squared
at most 0.75, and Benjamini–Hochberg FDR at most 0.05. Cohort-specific results remain
visible; a significant pooled result cannot hide an opposite or unsupported
cohort. Selection is repeated inside each LODO training fold and is frozen before
the fifth cohort is scored. Every declared neural seed must contain the identical
candidate endpoint set. Final retention additionally requires the prespecified
fraction of seeds to pass, stable slope direction across seeds, and any configured
rank/effect-variance thresholds. A donor-level simple GP score can be included as
an auditable baseline comparison. The selected set is frozen in the self-hashed
`selected_tripso_gps.json`; missing seeds/cohorts and query leakage are errors, not
reasons to fall back to all candidates.
