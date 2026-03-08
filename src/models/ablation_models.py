"""
Ablation model variants for the Hateful Meme Detection project.

Four variants are compared in the ablation study (matching the 2025
Elsevier paper's table format):

  1.  ImageOnlyClassifier      — CLIP image embedding → MLP (no text)
  2.  TextOnlyClassifier       — CLIP text  embedding → MLP (no image)
  3.  LateFusionClassifier     — concat(img_embed, txt_embed) → MLP
                                  (simple late fusion, no cross-attention)
  4.  HatefulMemeClassifier    — our full cross-attention model (from
                                  src/models/hateful_meme_model.py)

All share the same frozen CLIP backbone and the same ClassificationHead
architecture so performance differences reflect only the fusion strategy.
"""

import torch
import torch.nn as nn

from .clip_encoder import CLIPEncoder
from .fusion import ClassificationHead


# ──────────────────────────────────────────────────────────────────────────────

class ImageOnlyClassifier(nn.Module):
    """
    Unimodal baseline: CLIP image embedding only.
    Input dimension to MLP = embed_dim (512).
    """

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        freeze_encoders: bool = True,
        embed_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = CLIPEncoder(clip_model_name, freeze=freeze_encoders)
        self.classifier = ClassificationHead(
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_classes=2,
            dropout=dropout,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        image_embeds, _ = self.encoder(input_ids, attention_mask, pixel_values)
        return self.classifier(image_embeds)

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


# ──────────────────────────────────────────────────────────────────────────────

class TextOnlyClassifier(nn.Module):
    """
    Unimodal baseline: CLIP text embedding only.
    Input dimension to MLP = embed_dim (512).
    """

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        freeze_encoders: bool = True,
        embed_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = CLIPEncoder(clip_model_name, freeze=freeze_encoders)
        self.classifier = ClassificationHead(
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_classes=2,
            dropout=dropout,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        _, text_embeds = self.encoder(input_ids, attention_mask, pixel_values)
        return self.classifier(text_embeds)

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


# ──────────────────────────────────────────────────────────────────────────────

class LateFusionClassifier(nn.Module):
    """
    Multimodal late-fusion baseline: concat(img_embed, txt_embed) → MLP.
    No cross-attention — this is the strategy used by the 2025 Elsevier paper
    (RoBERTa + ResNet50 concatenation).

    Input dimension to MLP = 2 * embed_dim (1024).
    """

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        freeze_encoders: bool = True,
        embed_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = CLIPEncoder(clip_model_name, freeze=freeze_encoders)
        self.classifier = ClassificationHead(
            input_dim=2 * embed_dim,   # concat of image + text
            hidden_dim=hidden_dim,
            num_classes=2,
            dropout=dropout,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        image_embeds, text_embeds = self.encoder(input_ids, attention_mask, pixel_values)
        fused = torch.cat([image_embeds, text_embeds], dim=-1)  # (B, 1024)
        return self.classifier(fused)

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}
