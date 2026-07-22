# Data contract

## Biological and observation identifiers

These three identifiers are approved and have different meanings:

| Field | Definition | Meaning |
|---|---|---|
| `biological_unit_id` | `dataset::donor_id` | Independent human biological replicate |
| `source_observation_id` | `dataset::sample_id` | Source-provided sample, visit, or sequencing-pool label |
| `observation_id` | `dataset::donor_id::sample_id` | Donor-specific sample or visit used for scoring |

Components must be non-empty and cannot themselves contain `::`. Donor IDs are
never assumed to be globally unique. A source sample can be shared by several
donors, as observed for OneK1K sequencing pools, so `source_observation_id` is
retained for provenance but is not used as a unique donor observation. Repeated
Terekhova samples obtain separate `observation_id` values while retaining one
`biological_unit_id` for longitudinal grouping.

## Audited reference inputs

The five exact healthy dataset values are `aidav2`, `immuneindonesia`,
`immunobiologyaging`, `onek1k`, and `terekhova`. The merged lineage objects contain
CSR matrices with 18,035 identically ordered Ensembl IDs. The audited `.X` storage
is `float64`, and sampled stored values were non-negative and integer-like; no
counts layer or `.raw` was present. Model preparation must preserve sparse storage
and verify integer-like raw counts rather than assuming the floating dtype is
normalized expression.

Required cell metadata are:

- `dataset`, `donor_id`, `sample_id`, `age`, and `sex`;
- broad `lineage`;
- fine label `ctype_low` and confidence `ctype_low_conf`;
- higher label `ctype_high` and confidence `ctype_high_conf`;
- technical fields such as `chemistry`, `batch`, and `pct_mt` where available.

Derived annotations must also record their ontology and annotation versions.
Missing and low-confidence fine types remain explicit; they are not silently
relabelled. An insufficiently sampled state is missing, not a zero vector.

## Fold input contract

A TRIPSO fold descriptor records disjoint lists of adaptation, internal validation,
and query `biological_unit_id` values. All samples, fine types, cells, and lineages
from one biological unit inherit the same role. The descriptor also binds the
tokenized input, GP library, vocabulary, sampler manifest, and their hashes.

TRIPSO training refuses a biological split unit other than donor, refuses query
adaptation flags, and refuses the vendor's cell-weighted sampler as a substitute
for the project donor-hierarchical sampler. The prepared tokenized directory passed
to the vendor may contain adaptation donors only. Real training requires a donor
inventory extracted from that physical tokenized dataset; declaring donor roles in
the fold table alone is not accepted as proof.

## Query contract

Query inputs must provide the same identifier and metadata fields, raw sparse
counts, and donor-specific observations. Genes are mapped to the frozen vocabulary;
missing-gene and per-GP coverage are reported. Vocabulary, GP library, tokenizer,
preprocessing, embedding dimension, model configuration, weights, and reference
calibration must match the model manifest. Below-threshold coverage fails unless a
separately recorded override is explicitly authorized.

Input H5ADs are read-only. Derived H5ADs, tokenized datasets, manifests, models, and
reports belong under the configured output root, never in the source data tree.
