"""Donor-safe contracts for fold-specific TRIPSO inputs and resources."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class TripsoContractError(ValueError):
    """Raised when an input would violate a biological or resource contract."""


IDENTIFIER_CONTRACT = {
    "biological_unit_id": "dataset::donor_id",
    "source_observation_id": "dataset::sample_id",
    "observation_id": "dataset::donor_id::sample_id",
}

TRAIN_PARTITIONS = frozenset(
    {"train", "training", "inner_train", "adaptation", "reference"}
)
VALIDATION_PARTITIONS = frozenset(
    {"validation", "val", "inner_validation", "tuning", "early_stopping"}
)
QUERY_PARTITIONS = frozenset({"query", "test", "held_out", "heldout"})
KNOWN_PARTITIONS = TRAIN_PARTITIONS | VALIDATION_PARTITIONS | QUERY_PARTITIONS
REFERENCE_DESIGNS = frozenset({"lodo", "all_healthy"})
PROJECTION_GP_CANDIDATE_SCHEMA = "immune-health-projection-gp-candidates/v1"

REQUIRED_VENDOR_ASSETS = (
    "tripso/Utils/geneformer_token_dictionary_may2025.pkl",
    "tripso/Utils/geneformer_gene_median_file_may2025.pkl",
    "tripso/Utils/geneformer_ensembl_mapping_file_may2025.pkl",
    "tripso/Utils/geneformer_ensembl_dictionary_may2025.pkl",
    "tripso/Utils/gf-12L-95M-i4096_word_embeddings_may2025.pt",
)


def _component(value: Any, name: str) -> str:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        raise TripsoContractError(f"{name} must be non-empty")
    if "::" in text:
        raise TripsoContractError(
            f"{name} contains reserved identifier separator '::': {text!r}"
        )
    return text


def make_identifiers(dataset: Any, donor_id: Any, sample_id: Any) -> dict[str, str]:
    """Build the approved biological, source-observation, and observation IDs."""
    dataset_text = _component(dataset, "dataset")
    donor_text = _component(donor_id, "donor_id")
    sample_text = _component(sample_id, "sample_id")
    return {
        "biological_unit_id": f"{dataset_text}::{donor_text}",
        "source_observation_id": f"{dataset_text}::{sample_text}",
        "observation_id": f"{dataset_text}::{donor_text}::{sample_text}",
    }


def validate_identifiers(row: Mapping[str, Any]) -> dict[str, str]:
    """Validate any present derived IDs and return their canonical values."""
    canonical = make_identifiers(row["dataset"], row["donor_id"], row["sample_id"])
    for name, expected in canonical.items():
        actual = row.get(name)
        if actual is not None and str(actual).strip() != expected:
            raise TripsoContractError(
                f"Invalid {name} for dataset={row['dataset']!r}, "
                f"donor={row['donor_id']!r}, sample={row['sample_id']!r}: "
                f"expected {expected!r}, observed {actual!r}"
            )
    return canonical


def sha256_path(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash one regular file without loading it into memory."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Cannot hash missing regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    """Return a stable hash for a JSON-compatible value."""
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_projection_gp_candidates(
    path: Path,
    *,
    gp_library_path: Path,
) -> dict[str, Any]:
    """Validate the immutable training-only GP projection allowlist."""

    path = Path(path).resolve()
    gp_library_path = Path(gp_library_path).resolve()
    if not path.is_file() or not gp_library_path.is_file():
        raise FileNotFoundError(
            "Projection GP candidate manifest and filtered GP library are required"
        )
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != PROJECTION_GP_CANDIDATE_SCHEMA:
        raise TripsoContractError(
            f"Unsupported projection GP candidate manifest: {path}"
        )
    content = dict(manifest)
    claimed = content.pop("manifest_content_sha256", None)
    if claimed != canonical_json_hash(content):
        raise TripsoContractError(
            f"Projection GP candidate content hash does not match: {path}"
        )
    program_ids = manifest.get("program_ids")
    if (
        not isinstance(program_ids, list)
        or not program_ids
        or len(program_ids) != len(set(map(str, program_ids)))
    ):
        raise TripsoContractError(
            "Projection GP candidate program_ids must be a nonempty unique list"
        )
    program_ids = list(map(str, program_ids))
    if manifest.get("program_ids_ordered_sha256") != canonical_json_hash(program_ids):
        raise TripsoContractError(
            "Projection GP candidate ordered-program digest does not match"
        )
    binding = manifest.get("binding")
    if not isinstance(binding, Mapping) or binding.get("gpdb_sha256") != sha256_path(
        gp_library_path
    ):
        raise TripsoContractError(
            "Projection GP candidates are bound to another filtered GP library"
        )
    with gp_library_path.open(encoding="utf-8", newline="") as handle:
        available = next(csv.reader(handle), [])
    expected_order = [name for name in available if name in set(program_ids)]
    if expected_order != program_ids:
        raise TripsoContractError(
            "Projection GP candidates are absent from or out of GP-library order"
        )
    if manifest.get("query_data_consulted") is not False:
        raise TripsoContractError(
            "Projection GP candidate manifest does not prove query exclusion"
        )
    if manifest.get("selection_level") != "donor_lineage_pseudobulk":
        raise TripsoContractError(
            "Projection GP candidate selection level is not the approved donor gate"
        )
    return manifest


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write a JSON mapping in the destination directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def read_table(path: Path) -> list[dict[str, str]]:
    """Read a CSV/TSV metadata or split table."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Table does not exist: {path}")
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise TripsoContractError(f"Table has no header: {path}")
        return [dict(row) for row in reader]


def _as_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n", ""}:
        return False
    raise TripsoContractError(f"{name} must be boolean, observed {value!r}")


def _adaptation_allowed(row: Mapping[str, Any], partition_column: str) -> bool:
    if "eligible_for_reference_fitting" in row and row[
        "eligible_for_reference_fitting"
    ] not in {None, ""}:
        return _as_bool(
            row["eligible_for_reference_fitting"],
            name="eligible_for_reference_fitting",
        )
    if "adaptation_allowed" in row and row["adaptation_allowed"] not in {None, ""}:
        return _as_bool(row["adaptation_allowed"], name="adaptation_allowed")
    partition = str(row.get(partition_column, "")).strip().lower()
    if partition not in KNOWN_PARTITIONS:
        raise TripsoContractError(
            f"Unknown {partition_column} value {partition!r}; provide an explicit "
            "adaptation_allowed column if the split uses different labels"
        )
    return partition in TRAIN_PARTITIONS


@dataclass(frozen=True)
class ValidatedFoldRows:
    """Validated donor sets for one outer fold."""

    adaptation_donors: tuple[str, ...]
    validation_donors: tuple[str, ...]
    query_donors: tuple[str, ...]
    datasets: tuple[str, ...]
    observations: tuple[str, ...]


def validate_fold_rows(
    rows: Sequence[Mapping[str, Any]],
    held_out_dataset: str | None,
    *,
    partition_column: str = "partition",
    reference_design: str = "lodo",
) -> ValidatedFoldRows:
    """Prove donor separation and the approved identifier mapping for one fold."""
    if not rows:
        raise TripsoContractError("Fold metadata contains no rows")
    if reference_design not in REFERENCE_DESIGNS:
        raise TripsoContractError(
            f"reference_design must be one of {sorted(REFERENCE_DESIGNS)}"
        )
    if reference_design == "lodo":
        held_out_dataset = _component(held_out_dataset, "held_out_dataset")
    elif held_out_dataset is not None:
        raise TripsoContractError(
            "all_healthy reference design cannot declare a held-out dataset"
        )

    roles_by_donor: dict[str, set[str]] = {}
    adaptation: set[str] = set()
    validation: set[str] = set()
    query: set[str] = set()
    datasets: set[str] = set()
    observations: set[str] = set()
    observation_owner: dict[str, str] = {}

    for index, row in enumerate(rows, start=1):
        missing = [name for name in ("dataset", "donor_id") if name not in row]
        if missing:
            raise TripsoContractError(
                f"Fold row {index} lacks required columns: {', '.join(missing)}"
            )
        dataset = _component(row["dataset"], "dataset")
        donor_id = _component(row["donor_id"], "donor_id")
        donor = f"{dataset}::{donor_id}"
        if row.get("biological_unit_id") not in {None, "", donor}:
            raise TripsoContractError(
                f"Invalid biological_unit_id in fold row {index}: expected {donor!r}, "
                f"observed {row.get('biological_unit_id')!r}"
            )
        observation = None
        if row.get("sample_id") not in {None, ""}:
            canonical = validate_identifiers(row)
            observation = canonical["observation_id"]
        can_adapt = _adaptation_allowed(row, partition_column)

        if reference_design == "lodo" and dataset == held_out_dataset and can_adapt:
            raise TripsoContractError(
                f"Held-out dataset {held_out_dataset!r} contains adaptation donor "
                f"{donor!r}"
            )
        if reference_design == "lodo" and dataset == held_out_dataset:
            role = "query"
            query.add(donor)
        elif can_adapt:
            role = "adaptation"
            adaptation.add(donor)
        else:
            role = "validation"
            validation.add(donor)

        roles_by_donor.setdefault(donor, set()).add(role)
        if observation is not None:
            prior_owner = observation_owner.setdefault(observation, donor)
            if prior_owner != donor:
                raise TripsoContractError(
                    f"Observation {observation!r} maps to two biological units"
                )
            observations.add(observation)
        datasets.add(dataset)

    crossing = {
        donor: roles for donor, roles in roles_by_donor.items() if len(roles) > 1
    }
    if crossing:
        example = next(iter(sorted(crossing.items())))
        raise TripsoContractError(
            "A donor crosses adaptation/validation/query roles: "
            f"{example[0]} -> {sorted(example[1])}"
        )
    if not adaptation:
        raise TripsoContractError("Fold has no donors permitted for model adaptation")
    if reference_design == "lodo" and not query:
        raise TripsoContractError(
            f"Fold has no query donors from held-out dataset {held_out_dataset!r}"
        )
    if reference_design == "all_healthy" and query:
        raise TripsoContractError(
            "all_healthy reference design cannot contain query donors"
        )

    return ValidatedFoldRows(
        adaptation_donors=tuple(sorted(adaptation)),
        validation_donors=tuple(sorted(validation)),
        query_donors=tuple(sorted(query)),
        datasets=tuple(sorted(datasets)),
        observations=tuple(sorted(observations)),
    )


def prepare_fold_input_manifest(
    *,
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    fold_id: str,
    held_out_dataset: str | None,
    lineage: str,
    tokenized_dataset_path: Path,
    gp_library_path: Path,
    gene_vocabulary_path: Path,
    projection_gp_candidates_path: Path,
    source_h5ad_path: Path | None = None,
    sampler_manifest_path: Path | None = None,
    tokenization_manifest_path: Path | None = None,
    tokenized_biological_unit_ids: Iterable[str] | None = None,
    partition_column: str = "partition",
    reference_design: str = "lodo",
    inner_validation_fold: int | None = None,
    inner_fold_column: str | None = None,
    lineage_donor_scope_validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a fold descriptor that permits adaptation only on listed donors."""
    validated = validate_fold_rows(
        rows,
        held_out_dataset,
        partition_column=partition_column,
        reference_design=reference_design,
    )
    expected_tokenized_donors = set(validated.adaptation_donors)
    if tokenized_biological_unit_ids is None:
        tokenized_scope_validation: dict[str, Any] = {
            "status": "not_performed",
            "required_before_real_training": True,
            "note": (
                "Provide donor IDs extracted from the physical tokenized dataset; "
                "a fold table alone does not prove dataset contents."
            ),
        }
    else:
        observed_tokenized_donors = {
            str(value).strip() for value in tokenized_biological_unit_ids
        }
        if any(
            not value or value.lower() in {"nan", "none", "null"}
            for value in observed_tokenized_donors
        ):
            raise TripsoContractError(
                "tokenized biological_unit_id values must be non-empty"
            )
        missing_donors = sorted(expected_tokenized_donors - observed_tokenized_donors)
        unexpected_donors = sorted(
            observed_tokenized_donors - expected_tokenized_donors
        )
        if missing_donors or unexpected_donors:
            raise TripsoContractError(
                "Tokenized dataset donor scope differs from adaptation donors; "
                f"missing={missing_donors[:5]}, unexpected={unexpected_donors[:5]}"
            )
        tokenized_scope_validation = {
            "status": "passed",
            "required_before_real_training": True,
            "n_tokenized_biological_units": len(observed_tokenized_donors),
            "biological_unit_ids_sha256": canonical_json_hash(
                sorted(observed_tokenized_donors)
            ),
        }
    required_paths = {
        "tokenized_dataset_path": Path(tokenized_dataset_path),
        "gp_library_path": Path(gp_library_path),
        "gene_vocabulary_path": Path(gene_vocabulary_path),
        "projection_gp_candidates_path": Path(projection_gp_candidates_path),
    }
    if source_h5ad_path is not None:
        required_paths["source_h5ad_path"] = Path(source_h5ad_path)
    if sampler_manifest_path is not None:
        required_paths["sampler_manifest_path"] = Path(sampler_manifest_path)
    if tokenization_manifest_path is not None:
        required_paths["tokenization_manifest_path"] = Path(tokenization_manifest_path)
    missing = [
        f"{name}={path}" for name, path in required_paths.items() if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Fold resources are missing: " + ", ".join(missing))
    projection_candidates = validate_projection_gp_candidates(
        Path(projection_gp_candidates_path),
        gp_library_path=Path(gp_library_path),
    )

    row_digest_payload = []
    for row in rows:
        dataset = _component(row["dataset"], "dataset")
        donor_id = _component(row["donor_id"], "donor_id")
        digest_row = {
            "dataset": dataset,
            "donor_id": donor_id,
            "biological_unit_id": f"{dataset}::{donor_id}",
            "partition": str(row.get(partition_column, "")),
            "adaptation_allowed": _adaptation_allowed(row, partition_column),
        }
        if row.get("sample_id") not in {None, ""}:
            ids = validate_identifiers(row)
            digest_row.update({"sample_id": str(row["sample_id"]), **ids})
        row_digest_payload.append(digest_row)

    manifest: dict[str, Any] = {
        "schema_version": "immune-health-tripso-fold-input/v1",
        "fold_id": _component(fold_id, "fold_id"),
        "reference_design": reference_design,
        "held_out_dataset": (
            _component(held_out_dataset, "held_out_dataset")
            if held_out_dataset is not None
            else None
        ),
        "lineage": _component(lineage, "lineage"),
        "biological_split_unit": "donor",
        "random_cell_split_for_biological_evaluation": False,
        "vendor_internal_cell_split_scope": "adaptation_donors_only",
        "tokenized_dataset_scope_validation": tokenized_scope_validation,
        "identifier_contract": dict(IDENTIFIER_CONTRACT),
        "adaptation_biological_unit_ids": list(validated.adaptation_donors),
        "validation_biological_unit_ids": list(validated.validation_donors),
        "query_biological_unit_ids": list(validated.query_donors),
        "lineage_donor_scope_validation": (
            json.loads(json.dumps(dict(lineage_donor_scope_validation)))
            if lineage_donor_scope_validation is not None
            else {
                "status": "not_available",
                "global_fold_donor_inventory_used": True,
            }
        ),
        "inner_model_selection": {
            "enabled": inner_validation_fold is not None,
            "validation_fold": inner_validation_fold,
            "fold_column": inner_fold_column,
            "selection_role": (
                "validation" if inner_validation_fold is not None else None
            ),
            "outer_query_used_for_model_selection": False,
        },
        "n_observations": len(validated.observations),
        "datasets": list(validated.datasets),
        "inputs": {name: str(path.resolve()) for name, path in required_paths.items()},
        "hashes": {
            "fold_rows_sha256": canonical_json_hash(row_digest_payload),
            "gp_library_sha256": sha256_path(Path(gp_library_path)),
            "gene_vocabulary_sha256": sha256_path(Path(gene_vocabulary_path)),
            "projection_gp_candidates_sha256": sha256_path(
                Path(projection_gp_candidates_path)
            ),
            "projection_gp_program_ids_ordered_sha256": projection_candidates[
                "program_ids_ordered_sha256"
            ],
        },
    }
    if sampler_manifest_path is not None and Path(sampler_manifest_path).is_file():
        manifest["hashes"]["sampler_manifest_sha256"] = sha256_path(
            Path(sampler_manifest_path)
        )
    if (
        tokenization_manifest_path is not None
        and Path(tokenization_manifest_path).is_file()
    ):
        manifest["hashes"]["tokenization_manifest_sha256"] = sha256_path(
            Path(tokenization_manifest_path)
        )
    manifest["manifest_sha256"] = canonical_json_hash(manifest)
    atomic_write_json(output_path, manifest)
    return manifest


def load_fold_input_manifest(path: Path) -> dict[str, Any]:
    """Load and revalidate a fold input descriptor."""
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != "immune-health-tripso-fold-input/v1":
        raise TripsoContractError(f"Unsupported fold manifest schema: {path}")
    reference_design = manifest.get("reference_design", "lodo")
    if reference_design not in REFERENCE_DESIGNS:
        raise TripsoContractError("Fold input has an invalid reference_design")
    held_out_dataset = manifest.get("held_out_dataset")
    if reference_design == "lodo" and not held_out_dataset:
        raise TripsoContractError("LODO fold input lacks held_out_dataset")
    if reference_design == "all_healthy" and held_out_dataset is not None:
        raise TripsoContractError(
            "all_healthy fold input cannot declare held_out_dataset"
        )
    if reference_design == "all_healthy" and manifest.get("query_biological_unit_ids"):
        raise TripsoContractError("all_healthy fold input cannot contain query donors")
    if manifest.get("biological_split_unit") != "donor":
        raise TripsoContractError("TRIPSO adaptation requires donor biological splits")
    if manifest.get("random_cell_split_for_biological_evaluation") is not False:
        raise TripsoContractError("Cell-level biological evaluation is forbidden")
    if manifest.get("vendor_internal_cell_split_scope") != "adaptation_donors_only":
        raise TripsoContractError(
            "Vendor cell splits must be restricted to adaptation donors"
        )
    if manifest.get("identifier_contract") != IDENTIFIER_CONTRACT:
        raise TripsoContractError(
            "Fold manifest uses a non-approved identifier contract"
        )

    donor_sets = [
        set(manifest.get("adaptation_biological_unit_ids", [])),
        set(manifest.get("validation_biological_unit_ids", [])),
        set(manifest.get("query_biological_unit_ids", [])),
    ]
    if not donor_sets[0]:
        raise TripsoContractError("Fold manifest needs adaptation donors")
    if reference_design == "lodo" and not donor_sets[2]:
        raise TripsoContractError("LODO fold manifest needs query donors")
    if reference_design == "all_healthy" and donor_sets[2]:
        raise TripsoContractError(
            "all_healthy fold manifest must have an empty query donor set"
        )
    if any(
        left & right
        for i, left in enumerate(donor_sets)
        for right in donor_sets[i + 1 :]
    ):
        raise TripsoContractError("A biological unit appears in multiple fold roles")

    scope_validation = manifest.get("lineage_donor_scope_validation")
    if not isinstance(scope_validation, Mapping):
        raise TripsoContractError(
            "Fold input lacks its lineage donor-scope validation record"
        )
    scope_status = scope_validation.get("status")
    if scope_status == "passed":
        role_names = ("adaptation", "validation", "query")
        expected_counts = scope_validation.get("n_expected_biological_units_by_role")
        expected_hashes = scope_validation.get(
            "expected_biological_unit_ids_by_role_sha256"
        )
        excluded = scope_validation.get(
            "global_fold_biological_unit_ids_excluded_by_original_role"
        )
        if not all(
            isinstance(item, Mapping)
            for item in (expected_counts, expected_hashes, excluded)
        ):
            raise TripsoContractError(
                "Lineage donor-scope validation lacks role-specific evidence"
            )
        reconstructed: set[str] = set()
        for role, donors in zip(role_names, donor_sets, strict=True):
            ordered = sorted(map(str, donors))
            if expected_counts.get(role) != len(ordered) or expected_hashes.get(
                role
            ) != canonical_json_hash(ordered):
                raise TripsoContractError(
                    f"Lineage donor-scope validation differs for role {role!r}"
                )
            raw_excluded = excluded.get(role)
            if not isinstance(raw_excluded, list):
                raise TripsoContractError(
                    f"Lineage donor-scope exclusions lack role {role!r}"
                )
            excluded_donors = [str(value) for value in raw_excluded]
            if excluded_donors != sorted(set(excluded_donors)):
                raise TripsoContractError(
                    f"Lineage donor-scope exclusions are invalid for {role!r}"
                )
            if donors & set(excluded_donors):
                raise TripsoContractError(
                    f"A donor is both expected and excluded for role {role!r}"
                )
            reconstructed.update(donors)
            reconstructed.update(excluded_donors)
        reconstructed_ordered = sorted(reconstructed)
        if scope_validation.get("n_global_fold_biological_units") != len(
            reconstructed_ordered
        ) or scope_validation.get(
            "global_fold_biological_unit_ids_sha256"
        ) != canonical_json_hash(reconstructed_ordered):
            raise TripsoContractError(
                "Lineage donor-scope validation does not reconstruct the global fold"
            )
        if scope_validation.get("global_fold_donor_inventory_used") is not False:
            raise TripsoContractError(
                "Lineage-aware fold input incorrectly claims global donor expectations"
            )
        scope_hash = scope_validation.get("lineage_donor_scope_sha256")
        if not isinstance(scope_hash, str) or len(scope_hash) != 64:
            raise TripsoContractError(
                "Lineage donor-scope validation lacks its source hash"
            )
    elif scope_status == "not_available":
        if scope_validation.get("global_fold_donor_inventory_used") is not True:
            raise TripsoContractError("Legacy donor-scope fallback is malformed")
    else:
        raise TripsoContractError("Fold input has an invalid donor-scope status")

    inner_selection = manifest.get("inner_model_selection")
    if inner_selection is not None:
        if not isinstance(inner_selection, Mapping):
            raise TripsoContractError("Fold input has invalid inner_model_selection")
        if inner_selection.get("outer_query_used_for_model_selection") is not False:
            raise TripsoContractError(
                "Outer-query data cannot be used for inner model selection"
            )
        if inner_selection.get("enabled"):
            validation_fold = inner_selection.get("validation_fold")
            fold_column = inner_selection.get("fold_column")
            if (
                isinstance(validation_fold, bool)
                or not isinstance(validation_fold, int)
                or validation_fold < 0
                or not isinstance(fold_column, str)
                or not fold_column.strip()
                or inner_selection.get("selection_role") != "validation"
            ):
                raise TripsoContractError(
                    "Fold input has an invalid fixed inner-validation declaration"
                )
            if not donor_sets[1]:
                raise TripsoContractError(
                    "Enabled inner selection requires validation donors"
                )

    expected_manifest_hash = manifest.get("manifest_sha256")
    hash_payload = dict(manifest)
    hash_payload.pop("manifest_sha256", None)
    if expected_manifest_hash != canonical_json_hash(hash_payload):
        raise TripsoContractError(f"Fold manifest hash does not match content: {path}")
    for name, raw_path in manifest.get("inputs", {}).items():
        if not Path(raw_path).exists():
            raise FileNotFoundError(f"Fold input {name} does not exist: {raw_path}")
    inputs = manifest.get("inputs", {})
    candidate_path = inputs.get("projection_gp_candidates_path")
    gp_library_path = inputs.get("gp_library_path")
    if not candidate_path or not gp_library_path:
        raise TripsoContractError(
            "Fold input lacks its training-only projection GP candidate binding"
        )
    candidates = validate_projection_gp_candidates(
        Path(candidate_path), gp_library_path=Path(gp_library_path)
    )
    if sha256_path(Path(candidate_path)) != manifest.get("hashes", {}).get(
        "projection_gp_candidates_sha256"
    ):
        raise TripsoContractError(
            "Fold-bound projection GP candidate file changed after binding"
        )
    if candidates["program_ids_ordered_sha256"] != manifest.get("hashes", {}).get(
        "projection_gp_program_ids_ordered_sha256"
    ):
        raise TripsoContractError(
            "Fold-bound projection GP candidate order changed after binding"
        )
    return manifest


@dataclass(frozen=True)
class ResourceValidation:
    """Resolved immutable resources and their content hashes."""

    paths: Mapping[str, str]
    hashes: Mapping[str, str]


def validate_tripso_resources(
    *,
    vendor_root: Path,
    gp_library_path: Path,
    gene_vocabulary_path: Path,
    checkpoint_path: Path | None = None,
    expected_hashes: Mapping[str, str] | None = None,
    vendor_asset_names: Iterable[str] = REQUIRED_VENDOR_ASSETS,
) -> ResourceValidation:
    """Require real TRIPSO code/assets and verify optional pinned hashes."""
    vendor_root = Path(vendor_root).resolve()
    resources: dict[str, Path] = {
        "vendor_setup": vendor_root / "setup.py",
        "vendor_requirements": vendor_root / "requirements.txt",
        "gp_library": Path(gp_library_path).resolve(),
        "gene_vocabulary": Path(gene_vocabulary_path).resolve(),
    }
    for relative_name in vendor_asset_names:
        key = "vendor_asset:" + relative_name
        resources[key] = vendor_root / relative_name
    if checkpoint_path is not None:
        resources["checkpoint"] = Path(checkpoint_path).resolve()

    missing = [
        f"{name}={path}" for name, path in resources.items() if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Required TRIPSO resources are absent; assets are never synthesized: "
            + ", ".join(missing)
        )
    empty = [
        f"{name}={path}" for name, path in resources.items() if path.stat().st_size == 0
    ]
    if empty:
        raise TripsoContractError("TRIPSO resources are empty: " + ", ".join(empty))

    hashes = {name: sha256_path(path) for name, path in resources.items()}
    for name, expected in (expected_hashes or {}).items():
        if name not in hashes:
            raise TripsoContractError(f"Expected hash names unknown resource {name!r}")
        if hashes[name] != expected:
            raise TripsoContractError(
                f"Hash mismatch for {name}: expected {expected}, observed "
                f"{hashes[name]}"
            )
    return ResourceValidation(
        paths={name: str(path) for name, path in resources.items()}, hashes=hashes
    )
