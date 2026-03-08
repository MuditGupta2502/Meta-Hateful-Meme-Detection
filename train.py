"""
train.py — Entry-point for training the Hateful Meme Classifier.

Usage
-----
    # from the project root, with venv active:
    python train.py
    python train.py --config configs/config.yaml
    python train.py --epochs 5 --batch_size 16 --lr 5e-5
"""

import argparse
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import CLIPProcessor

# Ensure project root is on the path when called as a script
sys.path.insert(0, os.path.dirname(__file__))

from src.datasets.meme_dataset import HatefulMemeDataset
from src.models.hateful_meme_model import HatefulMemeClassifier
from src.training.trainer import Trainer
from src.training.sampler import make_balanced_sampler


# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Hateful Meme Classifier")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--epochs",     type=int,   default=None, help="Override num_epochs")
    parser.add_argument("--batch_size", type=int,   default=None, help="Override batch_size")
    parser.add_argument("--lr",         type=float, default=None, help="Override learning_rate")
    parser.add_argument("--no_freeze",  action="store_true",      help="Unfreeze CLIP encoders")
    parser.add_argument("--smote",      action="store_true",      help="Use balanced oversampler (SMOTE-style)")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────────────────────
def build_dataloaders(
    cfg: dict,
    processor: CLIPProcessor,
    use_balanced_sampler: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders."""

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

    use_pin = torch.cuda.is_available()

    if use_balanced_sampler:
        print("  [Sampler] Using balanced WeightedRandomSampler (SMOTE-style)")
        sampler = make_balanced_sampler(train_ds)
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg["training"]["batch_size"],
            sampler=sampler,        # shuffle must be False when sampler is set
            num_workers=0,
            pin_memory=use_pin,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg["training"]["batch_size"],
            shuffle=True,
            num_workers=0,
            pin_memory=use_pin,
        )

    val_loader = DataLoader(
        dev_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=use_pin,
    )
    return train_loader, val_loader


# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # CLI overrides
    if args.epochs is not None:
        cfg["training"]["num_epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["training"]["learning_rate"] = args.lr
    if args.no_freeze:
        cfg["model"]["freeze_encoders"] = False
    # --smote flag OR use_balanced_sampler in config both enable balanced sampling
    use_balanced = args.smote or cfg["training"].get("use_balanced_sampler", False)
    if use_balanced:
        # Route output to a separate directory so Phase 1 weights are untouched.
        # Only reroute if --smote was explicitly passed AND the config didn't
        # already specify a smote-specific path (avoids "models_smote_smote" bug).
        current_dir = cfg["paths"].get("model_dir", "outputs/models")
        if "models_smote" not in current_dir:
            cfg["paths"]["model_dir"] = current_dir.replace("models", "models_smote")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")

    # ── CLIP processor ──────────────────────────────────────────────────
    clip_model_name = cfg["model"]["clip_model"]
    print(f"Loading CLIP processor: {clip_model_name}")
    processor = CLIPProcessor.from_pretrained(clip_model_name)

    # ── Data ────────────────────────────────────────────────────────────
    print("Building datasets …")
    train_loader, val_loader = build_dataloaders(cfg, processor, use_balanced_sampler=use_balanced)
    print(f"  Train : {len(train_loader.dataset):,} samples")
    print(f"  Val   : {len(val_loader.dataset):,} samples")

    # ── Model ───────────────────────────────────────────────────────────
    print("\nBuilding model …")
    model = HatefulMemeClassifier(
        clip_model_name=clip_model_name,
        freeze_encoders=cfg["model"]["freeze_encoders"],
        embed_dim=cfg["model"]["embed_dim"],
        num_attention_heads=cfg["model"]["num_attention_heads"],
        fusion_hidden_dim=cfg["model"]["fusion_hidden_dim"],
        dropout=cfg["model"]["dropout"],
    )

    param_info = model.count_parameters()
    print(
        f"  Parameters — total: {param_info['total']:,}  "
        f"trainable: {param_info['trainable']:,}  "
        f"frozen: {param_info['frozen']:,}"
    )

    # ── Trainer config (flat dict) ───────────────────────────────────────
    trainer_cfg = {
        **cfg["training"],
        "model_dir": cfg["paths"]["model_dir"],
    }

    # ── Train ───────────────────────────────────────────────────────────
    trainer = Trainer(model, train_loader, val_loader, trainer_cfg, device)
    history = trainer.train()

    return history


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
