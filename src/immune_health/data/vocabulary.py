"""Fold-specific gene vocabulary construction that never opens held-out data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py

from immune_health.provenance import stable_hash


def _decode(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


@dataclass(frozen=True)
class TrainingVocabulary:
    genes: tuple[str, ...]
    heldout_dataset: str
    training_datasets: tuple[str, ...]
    opened_sources: tuple[str, ...]
    excluded_sources: tuple[str, ...]
    vocabulary_hash: str

    def manifest(self) -> dict[str, object]:
        return {
            "heldout_dataset": self.heldout_dataset,
            "training_datasets": list(self.training_datasets),
            "n_genes": len(self.genes),
            "vocabulary_hash": self.vocabulary_hash,
            "opened_sources": list(self.opened_sources),
            "excluded_sources_not_opened": list(self.excluded_sources),
            "selection_scope": "training_sources_only",
        }


def _source_path(data_root: Path, source: dict[str, object]) -> Path:
    return (
        data_root
        / "lineages"
        / str(source["source_dataset_id"])
        / str(source["lineage_filename"])
    )


def read_source_gene_ids(path: Path, field: str = "unified_ensembl") -> tuple[str, ...]:
    """Read only a source H5AD's small var array in read-only mode."""
    with h5py.File(path, "r") as handle:
        if "var" not in handle or field not in handle["var"]:
            raise ValueError(f"Source H5AD lacks var/{field}: {path}")
        values = tuple(
            _decode(value).split(".", 1)[0] for value in handle["var"][field][:]
        )
    if len(values) != len(set(values)):
        raise ValueError(
            f"Source gene IDs are not unique after version stripping: {path}"
        )
    return values


def build_training_vocabulary(
    merge_manifest_path: Path,
    data_root: Path,
    heldout_dataset: str,
    *,
    allowed_training_datasets: Iterable[str] | None = None,
) -> TrainingVocabulary:
    """Intersect source vocabularies using the four training datasets only.

    The held-out source path is recorded as excluded but is never opened. This
    avoids the subtle leakage that would arise from reusing the precomputed
    five-dataset intersection in a production LODO fit.
    """
    manifest = json.loads(merge_manifest_path.read_text())
    sources = list(manifest.get("sources", []))
    if not sources:
        raise ValueError(f"Merge manifest contains no sources: {merge_manifest_path}")
    observed_datasets = {str(source["dataset"]) for source in sources}
    if heldout_dataset not in observed_datasets:
        raise ValueError(
            f"Held-out dataset {heldout_dataset!r} is absent from merge sources"
        )
    allowed = (
        observed_datasets - {heldout_dataset}
        if allowed_training_datasets is None
        else set(map(str, allowed_training_datasets))
    )
    if heldout_dataset in allowed:
        raise ValueError("Held-out dataset cannot enter training vocabulary selection")
    unknown = allowed - observed_datasets
    if unknown:
        raise ValueError(f"Unknown training datasets: {sorted(unknown)}")

    selected_sources = [
        source for source in sources if str(source["dataset"]) in allowed
    ]
    excluded_sources = [
        source for source in sources if str(source["dataset"]) == heldout_dataset
    ]
    if not selected_sources or not excluded_sources:
        raise ValueError(
            "Vocabulary selection needs training and held-out source files"
        )

    first_order: tuple[str, ...] | None = None
    common: set[str] | None = None
    opened: list[str] = []
    for source in selected_sources:
        path = _source_path(data_root, source)
        if not path.is_file():
            raise FileNotFoundError(
                f"Configured training source does not exist: {path}"
            )
        genes = read_source_gene_ids(path)
        opened.append(str(path.resolve()))
        if first_order is None:
            first_order = genes
            common = set(genes)
        else:
            assert common is not None
            common.intersection_update(genes)
    assert first_order is not None and common is not None
    ordered = tuple(gene for gene in first_order if gene in common)
    if not ordered:
        raise ValueError("Training-only gene intersection is empty")
    excluded = tuple(
        str(_source_path(data_root, source).resolve()) for source in excluded_sources
    )
    if set(opened).intersection(excluded):
        raise AssertionError("Held-out source was opened during vocabulary selection")
    return TrainingVocabulary(
        genes=ordered,
        heldout_dataset=str(heldout_dataset),
        training_datasets=tuple(sorted(allowed)),
        opened_sources=tuple(opened),
        excluded_sources=excluded,
        vocabulary_hash=stable_hash(ordered),
    )
