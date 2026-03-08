"""
Trainer for the multimodal hateful meme classifier.

Handles:
  - Training loop (epoch iteration, forward pass, backward, optimiser step)
  - Validation loop (no-grad inference + metric computation)
  - Learning-rate schedule (cosine with warmup via HuggingFace)
  - Checkpoint saving (best-AUROC model + latest model)
  - Training history serialisation to JSON
"""

import os
import json
import time
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from src.training.losses import WeightedFocalLoss
from src.evaluation.metrics import compute_metrics, pretty_print_metrics


class Trainer:
    """
    Args:
        model        : HatefulMemeClassifier instance
        train_loader : DataLoader for the training set
        val_loader   : DataLoader for the validation set
        config       : flat dict drawn from configs/config.yaml
        device       : torch.device
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # ── Loss ────────────────────────────────────────────────────────
        raw_weights = config.get("class_weights")
        class_weights = (
            torch.tensor(raw_weights, dtype=torch.float32).to(device)
            if raw_weights else None
        )
        self.criterion = WeightedFocalLoss(
            class_weights=class_weights,
            gamma=config.get("focal_gamma", 2.0),
        )

        # ── Optimiser (only trainable params) ───────────────────────────
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable_params,
            lr=config["learning_rate"],
            weight_decay=config.get("weight_decay", 1e-4),
        )

        # ── LR schedule: cosine with linear warmup ───────────────────────
        total_steps = len(train_loader) * config["num_epochs"]
        warmup_steps = config.get("warmup_steps", 100)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # ── State ────────────────────────────────────────────────────────
        self.best_auroc: float = 0.0
        self.history: dict = {
            "train_loss": [],
            "val_loss": [],
            "val_auroc": [],
            "val_f1": [],
            "val_f1_macro": [],
            "val_accuracy": [],
        }

        self.model_dir: str = config.get("model_dir", "outputs/models")
        os.makedirs(self.model_dir, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────
    def _train_epoch(self) -> float:
        """One full pass over the training set. Returns mean loss."""
        self.model.train()
        total_loss = 0.0
        max_grad_norm = self.config.get("max_grad_norm", 1.0)

        for batch in tqdm(self.train_loader, desc="  train", leave=False, ncols=90):
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            pixel_values   = batch["pixel_values"].to(self.device)
            labels         = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask, pixel_values)
            loss = self.criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    # ──────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> tuple[dict, np.ndarray, np.ndarray]:
        """Inference over `loader`. Returns (metrics_dict, logits, labels)."""
        self.model.eval()
        total_loss = 0.0
        all_logits, all_labels = [], []

        for batch in tqdm(loader, desc="  eval ", leave=False, ncols=90):
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            pixel_values   = batch["pixel_values"].to(self.device)
            labels         = batch["label"].to(self.device)

            logits = self.model(input_ids, attention_mask, pixel_values)
            loss = self.criterion(logits, labels)

            total_loss += loss.item()
            all_logits.append(logits.cpu().float().numpy())
            all_labels.append(labels.cpu().numpy())

        logits_np = np.concatenate(all_logits, axis=0)
        labels_np = np.concatenate(all_labels, axis=0)

        metrics = compute_metrics(logits_np, labels_np)
        metrics["loss"] = total_loss / len(loader)
        return metrics, logits_np, labels_np

    # ──────────────────────────────────────────────────────────────────
    def _save_checkpoint(self, epoch: int, metrics: dict, filename: str) -> None:
        path = os.path.join(self.model_dir, filename)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": {k: v for k, v in metrics.items() if k != "report"},
            },
            path,
        )

    # ──────────────────────────────────────────────────────────────────
    def train(self) -> dict:
        """Full training run. Returns the history dict."""
        num_epochs = self.config["num_epochs"]
        print(f"\n{'='*60}")
        print(f"  Starting training for {num_epochs} epoch(s)")
        print(f"  Device : {self.device}")
        print(f"  Train batches : {len(self.train_loader)}")
        print(f"  Val   batches : {len(self.val_loader)}")
        print(f"{'='*60}\n")

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            print(f"Epoch {epoch}/{num_epochs}")

            train_loss = self._train_epoch()
            val_metrics, _, _ = self._evaluate(self.val_loader)

            elapsed = time.time() - t0

            # ── record ────────────────────────────────────────────────
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics.get("loss", 0.0))
            self.history["val_auroc"].append(val_metrics.get("auroc", 0.0))
            self.history["val_f1"].append(val_metrics.get("f1", 0.0))
            self.history["val_f1_macro"].append(val_metrics.get("f1_macro", 0.0))
            self.history["val_accuracy"].append(val_metrics.get("accuracy", 0.0))

            # ── print ─────────────────────────────────────────────────
            print(
                f"  Train loss : {train_loss:.4f}   "
                f"Val loss : {val_metrics.get('loss', 0):.4f}   "
                f"({elapsed:.1f}s)"
            )
            pretty_print_metrics(val_metrics, prefix="Val")

            # ── save best ─────────────────────────────────────────────
            cur_auroc = val_metrics.get("auroc", 0.0)
            if cur_auroc > self.best_auroc:
                self.best_auroc = cur_auroc
                self._save_checkpoint(epoch, val_metrics, "best_model.pt")
                print(f"  *** Best model saved  (AUROC = {self.best_auroc:.4f}) ***")

            # ── save latest ───────────────────────────────────────────
            self._save_checkpoint(epoch, val_metrics, "latest_model.pt")

        # Save history to disk
        hist_path = os.path.join(self.model_dir, "training_history.json")
        with open(hist_path, "w") as f:
            json.dump(self.history, f, indent=2)

        print(f"\nTraining complete.  Best Val AUROC = {self.best_auroc:.4f}")
        print(f"History saved to {hist_path}")
        return self.history

    # ──────────────────────────────────────────────────────────────────
    def evaluate_test(self, test_loader: DataLoader) -> dict:
        """Run inference on the test set and return metrics (if labels exist)."""
        metrics, logits, labels = self._evaluate(test_loader)
        return metrics, logits, labels
