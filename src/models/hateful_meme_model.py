"""
Full multimodal classifier for hateful meme detection.

Pipeline
--------
    CLIPEncoder (frozen)
        └─ image_embeds (B, 512)
        └─ text_embeds  (B, 512)
            │
    CrossAttentionFusion
        └─ fused (B, 1024)          # [img_ctx ‖ txt_ctx]
            │
    ClassificationHead (MLP)
        └─ logits (B, 2)
"""

import torch
import torch.nn as nn

from .clip_encoder import CLIPEncoder
from .fusion import CrossAttentionFusion, ClassificationHead


class HatefulMemeClassifier(nn.Module):
    """
    End-to-end multimodal model for the Hateful Memes Challenge.

    Args:
        clip_model_name   : HuggingFace model identifier for CLIP
        freeze_encoders   : if True, CLIP parameters are not updated
        embed_dim         : CLIP projection dimension (512 for ViT-B/32)
        num_attention_heads: heads in the cross-attention fusion layer
        fusion_hidden_dim : first hidden dim of the MLP classifier
        dropout           : dropout rate in both fusion and classifier
    """

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        freeze_encoders: bool = True,
        embed_dim: int = 512,
        num_attention_heads: int = 4,
        fusion_hidden_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.encoder = CLIPEncoder(clip_model_name, freeze=freeze_encoders)

        # Verify the actual embed_dim matches expectations
        assert self.encoder.embed_dim == embed_dim, (
            f"CLIP embed_dim mismatch: expected {embed_dim}, "
            f"got {self.encoder.embed_dim}"
        )

        self.fusion = CrossAttentionFusion(
            embed_dim=embed_dim,
            num_heads=num_attention_heads,
            dropout=0.1,
        )

        # Fused representation = 2 * embed_dim (img_ctx concatenated with txt_ctx)
        self.classifier = ClassificationHead(
            input_dim=2 * embed_dim,
            hidden_dim=fusion_hidden_dim,
            num_classes=2,
            dropout=dropout,
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,        # (B, 77)
        attention_mask: torch.Tensor,   # (B, 77)
        pixel_values: torch.Tensor,     # (B, 3, 224, 224)
    ) -> torch.Tensor:                  # (B, 2)
        image_embeds, text_embeds = self.encoder(input_ids, attention_mask, pixel_values)
        fused = self.fusion(image_embeds, text_embeds)
        logits = self.classifier(fused)
        return logits

    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return raw CLIP embeddings — useful for analysis & visualisation."""
        return self.encoder(input_ids, attention_mask, pixel_values)

    # ------------------------------------------------------------------
    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}
