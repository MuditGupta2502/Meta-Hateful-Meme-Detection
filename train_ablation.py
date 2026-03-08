"""
train_ablation.py — Ablation study: compare 4 fusion strategies.

Runs all four variants sequentially and prints a comparison table
(matching the style of the 2025 Elsevier paper's Table 3).

Variants
--------
  1. image_only      — CLIP image embedding → MLP
  2. text_only       — CLIP text  embedding → MLP
  3. late_fusion     — concat(img, txt) → MLP   (no cross-attention)
  4. cross_attention — our full bidirectional cross-attention model

Usage
-----
    python train_ablation.py
    python train_ablation.py --config configs/config.yaml --epochs 10
    python train_ablation.py --config configs/config_smote.yaml --epochs 20
    python train_ablation.py --variants image_only text_only  # run subset
"""

import argparse
import json
import os
import sys
import time

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import CLIPProcessor

sys.path.insert(0, os.path.dirname(__file__))

from src.datasets.meme_dataset import HatefulMemeDataset
from src.models.ablation_models import (
    ImageOnlyClassifier,
    TextOnlyClassifier,
    LateFusionClassifier,
)
from src.models.hateful_meme_model import HatefulMemeClassifier
from src.training.sampler import make_balanced_sampler
from src.training.trainer import Trainer


# ──────────────────────────────────────────────────────────────────────────────
ALL_VARIANTS = ["image_only", "text_only", "late_fusion", "cross_attention"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablation study for fusion strategy")
    parser.add_argument("--config",   default="configs/config.yaml")
    parser.add_argument("--epochs",   type=int,   default=None)
    parser.add_argument("--variants", nargs="+",  default=ALL_VARIANTS,
                        choices=ALL_VARIANTS,
                        help="Which variants to run (default: all four)")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────────────────────
def build_datasets(cfg: dict, processor: CLIPProcessor):
    train_ds = HatefulMemeDataset(
        jsonl_path=cfg["data"]["train_path"],
        image_root=cfg["data"]["image_root"],
        processor=processor,
        split="train",
    )
    dev_ds = HatefulMemeDataset(
        jsonl_path=cfg["data"]["dev_path"],
        image_root=cfg["data"]["image_root"],
        processor=processor,
        split="dev",
    )
    return train_ds, dev_ds


def build_loaders(
    train_ds: HatefulMemeDataset,
    dev_ds: HatefulMemeDataset,
    cfg: dict,
    use_balanced: bool,
) -> tuple[DataLoader, DataLoader]:
    bs = cfg["training"]["batch_size"]
    use_pin = torch.cuda.is_available()

    if use_balanced:
        sampler = make_balanced_sampler(train_ds)
        train_loader = DataLoader(
            train_ds, batch_size=bs,
            sampler=sampler,        # shuffle must be False with sampler
            num_workers=0, pin_memory=use_pin,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=bs, shuffle=True,
            num_workers=0, pin_memory=use_pin,
        )

    val_loader = DataLoader(
        dev_ds, batch_size=bs, shuffle=False,
        num_workers=0, pin_memory=use_pin,
    )
    return train_loader, val_loader


# ──────────────────────────────────────────────────────────────────────────────
def build_model(variant: str, cfg: dict) -> torch.nn.Module:
    mc = cfg["model"]
    kwargs = dict(
        clip_model_name=mc["clip_model"],
        freeze_encoders=mc["freeze_encoders"],
        embed_dim=mc["embed_dim"],
        hidden_dim=mc["fusion_hidden_dim"],
        dropout=mc["dropout"],
    )
    if variant == "image_only":
        return ImageOnlyClassifier(**kwargs)
    elif variant == "text_only":
        return TextOnlyClassifier(**kwargs)
    elif variant == "late_fusion":
        return LateFusionClassifier(**kwargs)
    else:  # cross_attention
        return HatefulMemeClassifier(
            clip_model_name=mc["clip_model"],
            freeze_encoders=mc["freeze_encoders"],
            embed_dim=mc["embed_dim"],
            num_attention_heads=mc["num_attention_heads"],
            fusion_hidden_dim=mc["fusion_hidden_dim"],
            dropout=mc["dropout"],
        )


# ──────────────────────────────────────────────────────────────────────────────
def print_table(results: dict) -> None:
    """Print a comparison table like the 2025 Elsevier paper's Table 3."""
    header = f"{'Variant':<22} {'Val AUROC':>10} {'Val Acc':>9} {'Val F1':>8} {'Macro-F1':>10}"
    print("\n" + "=" * 65)
    print("  ABLATION STUDY — Fusion Strategy Comparison")
    print("=" * 65)
    print(header)
    print("-" * 65)
    for variant, r in results.items():
        if "error" in r:
            print(f"  {variant:<20}   ERROR: {r['error']}")
        else:
            print(
                f"  {variant:<20} "
                f"{r['best_val_auroc']:>10.4f} "
                f"{r['best_val_accuracy']:>8.1%} "
                f"{r['best_val_f1']:>8.4f} "
                f"{r.get('best_val_f1_macro', 0.0):>10.4f}"
            )
    print("=" * 65)


# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.epochs is not None:
        cfg["training"]["num_epochs"] = args.epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_balanced = cfg["training"].get("use_balanced_sampler", False)

    print(f"\nDevice          : {device}")
    print(f"Epochs          : {cfg['training']['num_epochs']}")
    print(f"Balanced sampler: {use_balanced}")
    print(f"Variants        : {args.variants}\n")

    # Shared processor (loaded once, reused across variants)
    processor = CLIPProcessor.from_pretrained(cfg["model"]["clip_model"])
    train_ds, dev_ds = build_datasets(cfg, processor)

    results: dict = {}
    ablation_dir = os.path.join(cfg["paths"].get("output_dir", "outputs"), "ablation")
    os.makedirs(ablation_dir, exist_ok=True)

    for variant in args.variants:
        print(f"\n{'#'*60}")
        print(f"  Variant: {variant.upper()}")
        print(f"{'#'*60}")

        t_start = time.time()
        try:
            train_loader, val_loader = build_loaders(train_ds, dev_ds, cfg, use_balanced)
            model = build_model(variant, cfg)

            params = model.count_parameters()
            print(f"  Trainable params: {params['trainable']:,} / {params['total']:,}")

            trainer_cfg = {
                **cfg["training"],
                "model_dir": os.path.join(ablation_dir, variant),
            }
            trainer = Trainer(model, train_loader, val_loader, trainer_cfg, device)
            history = trainer.train()

            best_auroc = max(history["val_auroc"])
            best_epoch = history["val_auroc"].index(best_auroc) + 1
            best_acc   = history["val_accuracy"][best_epoch - 1]
            best_f1    = history["val_f1"][best_epoch - 1]
            best_f1m   = history.get("val_f1_macro", [0.0] * len(history["val_auroc"]))[best_epoch - 1]

            results[variant] = {
                "best_val_auroc":    best_auroc,
                "best_val_accuracy": best_acc,
                "best_val_f1":       best_f1,
                "best_val_f1_macro": best_f1m,
                "best_epoch":        best_epoch,
                "runtime_s":         round(time.time() - t_start, 1),
            }
            print(f"  Done in {results[variant]['runtime_s']}s")

        except Exception as exc:
            results[variant] = {"error": str(exc)}
            print(f"  FAILED: {exc}")

    # Print and save table
    print_table(results)

    table_path = os.path.join(ablation_dir, "ablation_results.json")
    with open(table_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {table_path}")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
