"""
Class-balanced sampler for the Hateful Memes training set.

Instead of raw SMOTE (which synthesises pixel-space images — incorrect for
vision transformers), we use WeightedRandomSampler to oversample the minority
class (hateful) so each epoch sees a ~50/50 class balance.  This is the
standard PyTorch-native equivalent of SMOTE oversampling for image datasets
and matches the effect described in the 2025 Elsevier paper.

Usage
-----
    from src.training.sampler import make_balanced_sampler
    sampler = make_balanced_sampler(train_dataset)
    loader  = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    # NOTE: shuffle=False when using a custom sampler
"""

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


def make_balanced_sampler(dataset: Dataset) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that draws minority-class samples more
    often so the effective class distribution per epoch is balanced.

    Args:
        dataset : HatefulMemeDataset (or any dataset whose __getitem__
                  returns a dict with key "label" as a scalar LongTensor)

    Returns:
        WeightedRandomSampler — pass to DataLoader(sampler=...)
    """
    # Collect labels without running CLIP preprocessing (peek at raw data)
    labels = np.array([dataset.data[i].get("label", -1) for i in range(len(dataset))])

    # Ignore unlabelled test samples (label == -1)
    valid_mask = labels >= 0
    valid_labels = labels[valid_mask]

    classes, class_counts = np.unique(valid_labels, return_counts=True)
    # weight per class = 1 / count  (minority gets higher weight)
    class_weight = {cls: 1.0 / cnt for cls, cnt in zip(classes, class_counts)}

    # Per-sample weight: invalid/test samples get weight 0
    sample_weights = np.zeros(len(dataset), dtype=np.float64)
    for i, lbl in enumerate(labels):
        if lbl in class_weight:
            sample_weights[i] = class_weight[lbl]

    # Total samples drawn = 2 × majority count (balanced epoch)
    num_samples = int(2 * max(class_counts))

    print(
        f"  [Sampler] Classes: {dict(zip(classes, class_counts))}  "
        f"| Per-class weight: { {k: f'{v:.5f}' for k, v in class_weight.items()} }  "
        f"| Samples/epoch: {num_samples:,}"
    )

    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=num_samples,
        replacement=True,
    )
