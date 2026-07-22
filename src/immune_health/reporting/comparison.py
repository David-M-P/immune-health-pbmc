"""LODO comparison reports that expose every held-out dataset separately."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

REQUIRED_METRIC_COLUMNS = {
    "heldout_dataset",
    "lineage",
    "method",
    "metric",
    "value",
}


def validate_metric_table(frame: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_METRIC_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Comparison metrics are missing columns: {missing}")
    duplicates = frame.duplicated(
        ["heldout_dataset", "lineage", "method", "metric"], keep=False
    )
    if duplicates.any() and "seed" not in frame:
        raise ValueError(
            "Repeated fold/lineage/method/metric rows require a seed column"
        )


def _markdown_table(frame: pd.DataFrame) -> str:
    def render(value: object) -> str:
        if pd.isna(value):
            return "NA"
        if isinstance(value, float):
            return f"{value:.5g}"
        return str(value).replace("|", "\\|")

    columns = list(map(str, frame.columns))
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def build_comparison_report(
    metrics: pd.DataFrame,
    output_path: Path,
    *,
    expected_heldout_datasets: Iterable[str] | None = None,
) -> str:
    """Write fold-first results, followed by explicitly labelled aggregates."""
    validate_metric_table(metrics)
    expected = set(expected_heldout_datasets or ())
    observed = set(metrics["heldout_dataset"].astype(str))
    if expected and observed != expected:
        raise ValueError(
            "Cannot build complete LODO report; held-out datasets differ: "
            f"expected={sorted(expected)}, observed={sorted(observed)}"
        )
    if "seed" in metrics:
        fold_table = (
            metrics.groupby(
                ["heldout_dataset", "lineage", "method", "metric"], observed=True
            )["value"]
            .agg(value="mean", seed_sd="std", n_seeds="count")
            .reset_index()
        )
    else:
        fold_table = metrics.copy()
    aggregate = (
        fold_table.groupby(["lineage", "method", "metric"], observed=True)["value"]
        .agg(mean_across_folds="mean", sd_across_folds="std", n_folds="count")
        .reset_index()
    )
    lines = [
        "# Donor-aware LODO comparison",
        "",
        "Every held-out dataset is reported before aggregation. A failed or missing "
        "fold "
        "is never replaced by an overall average.",
        "",
    ]
    for heldout in sorted(observed):
        lines.extend(
            [
                f"## Held out: {heldout}",
                "",
                _markdown_table(
                    fold_table.loc[
                        fold_table["heldout_dataset"].astype(str) == heldout
                    ].sort_values(["lineage", "metric", "method"])
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Aggregate across completed folds",
            "",
            _markdown_table(aggregate.sort_values(["lineage", "metric", "method"])),
            "",
        ]
    )
    report = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    return report
