#!/usr/bin/env python3
"""Run one real, bounded vendored TRIPSO Base optimization step.

This smoke test deliberately creates a tiny on-disk Hugging Face dataset and
an Ensembl-ID gene-program CSV.  It then trains the real vendored
``gpTransformerBase`` through the real vendored ``gpBase`` Lightning module.
It does not call ``tripso.train`` because that convenience wrapper requires a
Weights & Biases login and API readback even for an offline one-batch run.

The default exercises the production-relevant hybrid initialization: the
trainable from-scratch gene encoder is initialized with the static Geneformer
12-layer embedding table bundled with TRIPSO.  The run stays on CPU unless an
operator explicitly requests CUDA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VENDOR_ROOT = REPOSITORY_ROOT / "tripso_code" / "tripso"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New or empty output directory for synthetic inputs and checkpoint",
    )
    parser.add_argument(
        "--vendor-root",
        type=Path,
        default=DEFAULT_VENDOR_ROOT,
        help="Vendored TRIPSO package root",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--n-cells", type=int, default=10)
    parser.add_argument(
        "--embedding-init",
        choices=("geneformer-static", "random"),
        default="geneformer-static",
        help=(
            "Use the bundled Geneformer word embeddings (production-relevant) "
            "or a small random embedding table (faster diagnostic)"
        ),
    )
    parser.add_argument(
        "--accelerator",
        choices=("cpu", "cuda"),
        default="cpu",
        help="CUDA is used only when explicitly requested and available",
    )
    return parser.parse_args(argv)


def _prepare_empty_output(output_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.iterdir())
    if existing:
        raise FileExistsError(
            f"Smoke output directory must be empty: {output_dir}; "
            f"found {len(existing)} existing entries"
        )
    return output_dir


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _select_real_geneformer_tokens(
    token_dictionary_path: Path,
    n_genes: int = 12,
) -> tuple[list[str], list[int]]:
    with token_dictionary_path.open("rb") as handle:
        token_dictionary = pickle.load(handle)
    candidates = sorted(
        (
            (str(gene), int(token))
            for gene, token in token_dictionary.items()
            if str(gene).startswith("ENSG") and int(token) >= 4
        ),
        key=lambda item: item[1],
    )
    if len(candidates) < n_genes:
        raise ValueError(
            f"Token dictionary has only {len(candidates)} usable Ensembl genes"
        )
    chosen = candidates[:n_genes]
    return [item[0] for item in chosen], [item[1] for item in chosen]


def _write_synthetic_inputs(
    *,
    output_dir: Path,
    token_dictionary_path: Path,
    n_cells: int,
    seed: int,
) -> dict[str, Any]:
    # Delay this comparatively expensive import until the output contract has
    # been checked, and write a real Dataset.save_to_disk artifact.
    import pandas as pd
    from datasets import Dataset

    genes, tokens = _select_real_geneformer_tokens(token_dictionary_path)
    rng = random.Random(seed)
    sequences: list[list[int]] = []
    for cell_index in range(n_cells):
        ranked_tokens = tokens.copy()
        rng.shuffle(ranked_tokens)
        # Geneformer-tokenized cells begin with the vendored <cls> token (2).
        sequences.append([2, *ranked_tokens])

    tokenized_path = output_dir / "tokenized.dataset"
    dataset = Dataset.from_dict(
        {
            "input_ids": sequences,
            "length": [len(sequence) for sequence in sequences],
            "idx": list(range(n_cells)),
        }
    )
    dataset.save_to_disk(str(tokenized_path))

    # Eight present genes are enough to guarantee non-padding GP inputs while
    # keeping the model and loss tiny. Ensembl IDs bind gene_format=ensembl.
    gp_library_path = output_dir / "tiny_gp_library.csv"
    pd.DataFrame({"SMOKE_GP": genes[:8]}).to_csv(gp_library_path, index=False)

    return {
        "tokenized_dataset_path": str(tokenized_path),
        "gp_library_path": str(gp_library_path),
        "genes": genes,
        "tokens": tokens,
        "n_cells": n_cells,
        "sequence_length_including_cls": len(sequences[0]),
        "gp_size": 8,
    }


def _scalar_metric(metrics: dict[str, Any], name: str) -> float:
    value = metrics.get(name)
    if value is None:
        raise RuntimeError(f"Lightning did not report required metric {name!r}")
    if hasattr(value, "detach"):
        value = value.detach().cpu().item()
    scalar = float(value)
    if not (float("-inf") < scalar < float("inf")):
        raise RuntimeError(f"Non-finite {name}: {scalar}")
    return scalar


def run_real_smoke(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = _prepare_empty_output(args.output_dir)
    vendor_root = args.vendor_root.resolve()
    if not (vendor_root / "tripso" / "__init__.py").is_file():
        raise FileNotFoundError(f"Not a vendored TRIPSO root: {vendor_root}")
    if args.n_cells < max(10, args.batch_size * 2):
        raise ValueError("n_cells must be at least 10 and at least two batches")
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")

    # Keep matplotlib and model telemetry fully local on HPC compute nodes.
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / "matplotlib_cache"))
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    if str(vendor_root) not in sys.path:
        sys.path.insert(0, str(vendor_root))

    import numpy as np
    import pandas as pd
    import pytorch_lightning as pl
    import torch
    import tripso
    from pytorch_lightning.callbacks import ModelCheckpoint
    from tripso.Datamodules.datamodule import txDataModule
    from tripso.Models.gp_model import gpTransformerBase
    from tripso.Trainers.trainer import gpBase

    if args.accelerator == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--accelerator=cuda requested, but CUDA is unavailable")

    token_dictionary_path = Path(tripso.TOKEN_DICTIONARY_FILE).resolve()
    static_embedding_path = (
        Path(tripso.__file__).resolve().parent
        / "Utils"
        / "gf-12L-95M-i4096_word_embeddings_may2025.pt"
    )
    synthetic = _write_synthetic_inputs(
        output_dir=output_dir,
        token_dictionary_path=token_dictionary_path,
        n_cells=args.n_cells,
        seed=args.seed,
    )

    pl.seed_everything(args.seed, workers=True)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    if args.embedding_init == "geneformer-static":
        hidden_size = 512
        attention_heads = 8
        use_gene_embeddings: str | bool = "gf-12L-95M-i4096"
    else:
        hidden_size = 32
        attention_heads = 4
        use_gene_embeddings = False

    # tokenization_input_size=4096 selects the inspected May-2025 dictionaries.
    # max_seq_len is independently bounded to the twelve-gene smoke universe.
    bert_config = {
        "hidden_size": hidden_size,
        "num_hidden_layers": 1,
        "num_attention_heads": attention_heads,
        "tokenization_input_size": 4096,
        "tokenization_vocab_size": 20275,
        "max_seq_len": len(synthetic["genes"]),
        "mlm_masking_prob": 1.0,
        "use_pos_emb": "sin_cos",
        "use_l2_norm": False,
        "use_flash": False,
        "torch_dtype": "float32",
    }

    gpdb = pd.read_csv(synthetic["gp_library_path"])
    model = gpTransformerBase(
        database=gpdb,
        gp_inputs=["SMOKE_GP"],
        do_ensembl_conversion=False,
        num_heads=attention_heads,
        n_blocks=1,
        mgm_mask_ratio=1.0,
        use_flash=False,
        fm_encoder_pkg="from_scratch",
        fm_encoder_name="gf-12L-95M-i4096",
        bert_config=bert_config,
        gp_latent_size=hidden_size,
        use_gene_embeddings=use_gene_embeddings,
        use_l2_norm=False,
        all_genes=synthetic["genes"],
        warmup=0,
    )
    lightning_model = gpBase(
        model=model,
        output_dir=str(output_dir),
        lr=1e-4,
        weight_decay=0.0,
        lr_scheduler="ReduceLROnPlateau",
        total_epochs=1,
        calc_gp_loss=True,
        calc_gene_loss=True,
        warmup=0,
        hparam_save="ignore_model",
    )
    datamodule = txDataModule(
        folder=synthetic["tokenized_dataset_path"],
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=True,
        sampler=None,
        seed=args.seed,
        model_input_size=4096,
        output_dir=str(output_dir),
    )

    monitored_parameter = lightning_model.model.multi_gp_encoder.encoder[
        0
    ].decoder.weight
    monitored_before = monitored_parameter.detach().cpu().clone()
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="real-tripso-base-smoke-{epoch:02d}-{step}",
        save_top_k=0,
        save_last=True,
        save_weights_only=True,
    )
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=1,
        strategy="auto",
        precision=32,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        accumulate_grad_batches=1,
        logger=False,
        callbacks=[checkpoint_callback],
        enable_checkpointing=True,
        enable_model_summary=False,
        enable_progress_bar=False,
        deterministic=True,
        log_every_n_steps=1,
    )

    started = time.perf_counter()
    trainer.fit(lightning_model, datamodule=datamodule)
    elapsed_seconds = time.perf_counter() - started

    monitored_after = monitored_parameter.detach().cpu()
    maximum_parameter_change = float(
        torch.max(torch.abs(monitored_after - monitored_before)).item()
    )
    if trainer.global_step != 1:
        raise RuntimeError(
            f"Expected one optimizer step, observed {trainer.global_step}"
        )
    if maximum_parameter_change <= 0:
        raise RuntimeError("Optimizer step did not change the monitored GP decoder")

    train_loss = _scalar_metric(trainer.callback_metrics, "train/loss_epoch")
    gene_loss = _scalar_metric(
        trainer.callback_metrics, "train/gene_masking_loss_epoch"
    )
    gp_loss = _scalar_metric(trainer.callback_metrics, "train/SMOKE_GP_MGM_loss_epoch")

    checkpoint_path = Path(checkpoint_callback.last_model_path).resolve()
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
        raise RuntimeError("Lightning did not create a non-empty last checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not checkpoint.get("state_dict"):
        raise RuntimeError("Saved checkpoint has no model state_dict")

    report = {
        "passed": True,
        "run_type": "real_vendored_tripso_base_training_smoke",
        "synthetic_data": True,
        "scope": (
            "One real CPU/CUDA forward, backward, optimizer step, and checkpoint; "
            "not a convergence, biological-validity, dynamic-sampler, or full-data test"
        ),
        "vendor": {
            "module_path": str(Path(tripso.__file__).resolve()),
            "vendor_root_requested": str(vendor_root),
            "token_dictionary_path": str(token_dictionary_path),
            "token_dictionary_sha256": _sha256(token_dictionary_path),
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "pytorch_lightning": pl.__version__,
            "accelerator": args.accelerator,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
        },
        "model": {
            "model_type": "Base",
            "fm_encoder_pkg": "from_scratch",
            "embedding_init": args.embedding_init,
            "static_embedding_path": (
                str(static_embedding_path)
                if args.embedding_init == "geneformer-static"
                else None
            ),
            "static_embedding_sha256": (
                _sha256(static_embedding_path)
                if args.embedding_init == "geneformer-static"
                else None
            ),
            "hidden_size": hidden_size,
            "gene_identifier_format": "ensembl",
            "calc_gene_loss": True,
            "calc_gp_loss": True,
            "bert_config": bert_config,
        },
        "data": synthetic,
        "training": {
            "epochs": 1,
            "limit_train_batches": 1,
            "optimizer_steps": trainer.global_step,
            "batch_size": args.batch_size,
            "train_loss": train_loss,
            "gene_masking_loss": gene_loss,
            "gp_masking_loss": gp_loss,
            "monitored_gp_decoder_maximum_absolute_change": (maximum_parameter_change),
            "elapsed_seconds": elapsed_seconds,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
            "sha256": _sha256(checkpoint_path),
            "state_dict_entries": len(checkpoint["state_dict"]),
        },
    }
    report_path = output_dir / "smoke_report.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_real_smoke(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
