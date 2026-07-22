"""Small synthetic AnnData fixture with known donor-level effects."""

from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from immune_health.data.ids import add_stable_identifiers

SYNTHETIC_DATASETS = ("cohort_a", "cohort_b", "cohort_c", "cohort_d", "heldout_e")
SYNTHETIC_LINEAGES = ("B cells", "NK_ILC", "Monocytes", "CD4_like", "CD8_like")


@dataclass(frozen=True)
class SyntheticTruth:
    held_out_dataset: str
    age_up_genes: tuple[str, ...]
    dispersion_gene: str
    composition_age_lineage: str


def make_synthetic_pbmc(
    *,
    donors_per_dataset: int = 4,
    cells_per_donor_lineage: int = 24,
    n_genes: int = 24,
    seed: int = 42,
) -> tuple[ad.AnnData, SyntheticTruth]:
    """Create sparse counts with age, dispersion and composition ground truth.

    The last dataset is intentionally designated as a zero-shot query cohort.
    Donor zero in every dataset has two samples, exercising repeated-observation
    handling without ever changing the biological split unit.
    """
    if donors_per_dataset < 2 or cells_per_donor_lineage < 8 or n_genes < 8:
        raise ValueError("Synthetic fixture dimensions are too small for its contracts")
    rng = np.random.default_rng(seed)
    gene_ids = np.asarray([f"ENSG{index:011d}" for index in range(1, n_genes + 1)])
    gene_symbols = np.asarray([f"GENE_{index}" for index in range(n_genes)])
    fine_types = {
        "B cells": ("Naive B cells", "Memory B cells"),
        "NK_ILC": ("CD16+ NK cells", "CD16- NK cells"),
        "Monocytes": ("Classical monocytes", "Non-classical monocytes"),
        "CD4_like": ("Tcm/Naive helper T cells", "Tem/Effector helper T cells"),
        "CD8_like": ("Tcm/Naive cytotoxic T cells", "Tem/Temra cytotoxic T cells"),
    }
    rows: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []
    cell_number = 0
    for dataset_index, dataset in enumerate(SYNTHETIC_DATASETS):
        for donor_index in range(donors_per_dataset):
            donor_id = f"D{donor_index:02d}"
            base_age = 24 + 14 * donor_index + dataset_index
            sex = "female" if donor_index % 2 == 0 else "male"
            visits = 2 if donor_index == 0 else 1
            for visit in range(visits):
                sample_id = f"{donor_id}_V{visit + 1}"
                age = float(base_age + visit)
                for lineage_index, lineage in enumerate(SYNTHETIC_LINEAGES):
                    first_type, second_type = fine_types[lineage]
                    older_fraction = np.clip(0.2 + (age - 20) / 100, 0.2, 0.75)
                    if lineage != "B cells":
                        older_fraction = 0.5
                    n_second = int(round(cells_per_donor_lineage * older_fraction))
                    labels = np.asarray(
                        [first_type] * (cells_per_donor_lineage - n_second)
                        + [second_type] * n_second,
                        dtype=object,
                    )
                    rng.shuffle(labels)
                    for fine_type in labels:
                        base = np.full(n_genes, 1.5 + 0.15 * lineage_index)
                        base[0] += max(age - 20, 0) / 12
                        base[1] += max(age - 20, 0) / 20
                        if fine_type == second_type:
                            base[3] += 2.0
                        dataset_scale = 1.0 + 0.03 * dataset_index
                        if age >= 55:
                            # Gamma-Poisson mixing creates a known older-donor
                            # dispersion increase without changing independence.
                            base[2] *= rng.gamma(shape=1.2, scale=1 / 1.2)
                        counts = rng.poisson(np.maximum(base * dataset_scale, 0.01))
                        rows.append(counts.astype(np.float64))
                        metadata.append(
                            {
                                "cell_id": f"cell_{cell_number:07d}",
                                "dataset": dataset,
                                "donor_id": donor_id,
                                "sample_id": sample_id,
                                "age": age,
                                "sex": sex,
                                "lineage": lineage,
                                "ctype_high": "T cells"
                                if lineage in {"CD4_like", "CD8_like"}
                                else lineage,
                                "ctype_high_conf": 0.99,
                                "ctype_low": fine_type,
                                "ctype_low_conf": 0.95,
                                "chemistry": f"synthetic_v{dataset_index % 2 + 1}",
                                "batch": dataset,
                                "pct_mt": 2.0,
                            }
                        )
                        cell_number += 1
    obs = add_stable_identifiers(pd.DataFrame(metadata).set_index("cell_id"))
    matrix = sparse.csr_matrix(np.vstack(rows), dtype=np.float64)
    var = pd.DataFrame(
        {"unified_ensembl": gene_ids, "gene_symbol": gene_symbols}, index=gene_ids
    )
    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    truth = SyntheticTruth(
        held_out_dataset=SYNTHETIC_DATASETS[-1],
        age_up_genes=(gene_ids[0], gene_ids[1]),
        dispersion_gene=gene_ids[2],
        composition_age_lineage="B cells",
    )
    return adata, truth
