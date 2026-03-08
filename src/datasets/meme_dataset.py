import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import CLIPProcessor

from src.utils.data_loader import load_jsonl


class HatefulMemeDataset(Dataset):
    """
    PyTorch Dataset for the Meta Hateful Memes Challenge.

    Each sample is pre-processed through CLIPProcessor so batches
    contain ready-to-use tensors (no raw PIL images in collate).

    Args:
        jsonl_path  : path to the .jsonl split file
        image_root  : directory that contains the meme PNG files
        processor   : CLIPProcessor instance (shared across splits)
        split       : one of "train" | "dev" | "test"
    """

    def __init__(
        self,
        jsonl_path: str,
        image_root: str,
        processor: CLIPProcessor,
        split: str = "train",
    ) -> None:
        self.data = load_jsonl(jsonl_path)
        self.image_root = image_root
        self.processor = processor
        self.split = split

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.data)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        sample = self.data[idx]

        text: str = sample["text"]
        # test set has no label field → use -1 as sentinel
        label: int = sample.get("label", -1)
        sample_id: int = sample["id"]

        # ---- load image ------------------------------------------------
        image_filename = os.path.basename(sample["img"])
        image_path = os.path.join(self.image_root, image_filename)
        image = Image.open(image_path).convert("RGB")

        # ---- CLIP preprocessing ----------------------------------------
        encoded = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77,          # CLIP hard limit
        )

        return {
            # shapes after .squeeze(0): (77,), (77,), (3,224,224)
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
            "id": sample_id,
            "text": text,           # kept for error-analysis / counterfactuals
        }
