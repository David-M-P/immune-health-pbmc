"""Hash-bound donor inventories for broad-lineage reference artifacts."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

LINEAGE_DONOR_SCOPE_SCHEMA = "immune-health-lineage-donor-scope/v1"
MATERIALIZED_PREPARATION_ROLES = ("adaptation", "validation", "query")


def canonical_json_digest(payload: object) -> str:
    """Hash a JSON-compatible value using the repository's canonical encoding."""

    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_lineage_donor_scope(
    value: Mapping[str, Any], *, lineage: str | None = None
) -> dict[str, Any]:
    """Validate the fold-local donor inventory available in one lineage.

    Global donor folds intentionally contain people who can have zero cells in a
    particular broad-lineage file. This contract records the donors that can be
    physically materialized for each role, plus every global-fold donor excluded
    for lack of role-eligible lineage cells. Their union must reconstruct the
    complete global donor fold, so accidental Arrow donor loss cannot be waived.
    """

    scope = json.loads(json.dumps(dict(value)))
    if scope.get("schema_version") != LINEAGE_DONOR_SCOPE_SCHEMA:
        raise ValueError("Unsupported lineage donor-scope schema")
    observed_lineage = str(scope.get("lineage", "")).strip()
    if not observed_lineage or (
        lineage is not None and observed_lineage != str(lineage)
    ):
        raise ValueError("Lineage donor scope has an invalid lineage")
    if scope.get("scope_unit") != "biological_unit_id":
        raise ValueError("Lineage donor scope must use biological_unit_id")

    role_values = scope.get("biological_unit_ids_by_preparation_role")
    role_counts = scope.get("n_biological_units_by_preparation_role")
    role_hashes = scope.get("biological_unit_ids_by_preparation_role_sha256")
    if not all(
        isinstance(item, Mapping) for item in (role_values, role_counts, role_hashes)
    ):
        raise ValueError("Lineage donor scope lacks role-specific inventories")

    occupied: set[str] = set()
    for role in MATERIALIZED_PREPARATION_ROLES:
        raw = role_values.get(role)
        if not isinstance(raw, list):
            raise ValueError(f"Lineage donor scope lacks role {role!r}")
        donors = [str(item).strip() for item in raw]
        if (
            donors != sorted(donors)
            or len(donors) != len(set(donors))
            or any(not donor or donor.count("::") != 1 for donor in donors)
        ):
            raise ValueError(
                f"Lineage donor scope role {role!r} is not a sorted unique "
                "biological-unit inventory"
            )
        overlap = occupied & set(donors)
        if overlap:
            raise ValueError(
                "A biological unit crosses lineage preparation roles: "
                f"{sorted(overlap)[:5]}"
            )
        occupied.update(donors)
        if role_counts.get(role) != len(donors):
            raise ValueError(f"Lineage donor count is wrong for role {role!r}")
        if role_hashes.get(role) != canonical_json_digest(donors):
            raise ValueError(f"Lineage donor hash is wrong for role {role!r}")

    missing_key = "global_fold_biological_unit_ids_without_materialized_role_cells"
    raw_missing = scope.get(missing_key)
    if not isinstance(raw_missing, list):
        raise ValueError("Lineage donor scope lacks its global-fold exclusions")
    missing = [str(item).strip() for item in raw_missing]
    if (
        missing != sorted(missing)
        or len(missing) != len(set(missing))
        or any(not donor or donor.count("::") != 1 for donor in missing)
        or occupied & set(missing)
    ):
        raise ValueError("Lineage donor-scope exclusions are invalid")
    missing_count = scope.get(
        "n_global_fold_biological_units_without_materialized_role_cells"
    )
    if missing_count != len(missing):
        raise ValueError("Lineage donor-scope exclusion count is wrong")

    reconstructed_global = sorted(occupied | set(missing))
    if scope.get("n_global_fold_biological_units") != len(reconstructed_global):
        raise ValueError("Lineage donor scope does not reconstruct the global fold")
    if scope.get("global_fold_biological_unit_ids_sha256") != canonical_json_digest(
        reconstructed_global
    ):
        raise ValueError("Lineage donor scope has the wrong global-fold hash")
    expected_hash = scope.get("scope_sha256")
    content = dict(scope)
    content.pop("scope_sha256", None)
    if expected_hash != canonical_json_digest(content):
        raise ValueError("Lineage donor-scope self-hash is invalid")
    return scope
