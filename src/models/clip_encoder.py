"""
Frozen CLIP encoder for extracting aligned image and text embeddings.

Uses openai/clip-vit-base-patch32 by default, which produces 512-dim
L2-normalised embeddings for both modalities in a shared space.
"""

import torch
import torch.nn as nn
from transformers import CLIPModel


class CLIPEncoder(nn.Module):
    """
    Wraps a pretrained CLIP model and exposes image/text embeddings.

    When `freeze=True` (the default) all CLIP parameters are frozen so
    only the downstream fusion + classification layers are trained.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        freeze: bool = True,
    ) -> None:
        super().__init__()

        self.clip = CLIPModel.from_pretrained(model_name)

        if freeze:
            for param in self.clip.parameters():
                param.requires_grad = False

        # Projected (post-linear) embedding dimension
        self.embed_dim: int = self.clip.config.projection_dim  # 512 for ViT-B/32

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image_embeds : (B, embed_dim) — L2-normalised image projections
            text_embeds  : (B, embed_dim) — L2-normalised text projections
        """
        outputs = self.clip(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            return_dict=True,
        )
        # Both tensors are already L2-normalised by CLIP's projection head
        return outputs.image_embeds, outputs.text_embeds
