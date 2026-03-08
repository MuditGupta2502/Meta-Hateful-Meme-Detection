"""
Fusion and classification modules for multimodal hateful meme detection.

Architecture
------------
CrossAttentionFusion
    - Image embeddings attend to text  (image-guided by text context)
    - Text  embeddings attend to image (text-guided  by image context)
    - Both attended vectors are residual-added and layer-normalised
    - Outputs: concatenated [img_out ‖ txt_out] of shape (B, 2*embed_dim)

ClassificationHead
    - 3-layer MLP: input → hidden → hidden/2 → num_classes
    - GELU activation + dropout between layers
"""

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """
    Bidirectional cross-attention fusion between image and text embeddings.

    Args:
        embed_dim  : dimension of incoming CLIP embeddings (512)
        num_heads  : number of attention heads
        dropout    : dropout on attention weights
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Image tokens attending to text context
        self.img_to_text = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        # Text tokens attending to image context
        self.text_to_img = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.norm_img = nn.LayerNorm(embed_dim)
        self.norm_txt = nn.LayerNorm(embed_dim)

    # ------------------------------------------------------------------
    def forward(
        self,
        image_embeds: torch.Tensor,  # (B, embed_dim)
        text_embeds: torch.Tensor,   # (B, embed_dim)
    ) -> torch.Tensor:               # (B, 2 * embed_dim)
        # Treat each embedding as a single-token sequence for MHA
        img = image_embeds.unsqueeze(1)  # (B, 1, D)
        txt = text_embeds.unsqueeze(1)   # (B, 1, D)

        # image query  ←  text key/value
        img_ctx, _ = self.img_to_text(query=img, key=txt, value=txt)
        img_out = self.norm_img(img + img_ctx).squeeze(1)  # (B, D)

        # text  query  ←  image key/value
        txt_ctx, _ = self.text_to_img(query=txt, key=img, value=img)
        txt_out = self.norm_txt(txt + txt_ctx).squeeze(1)  # (B, D)

        # Concatenate both context-enriched representations
        return torch.cat([img_out, txt_out], dim=-1)       # (B, 2D)


# ──────────────────────────────────────────────────────────────────────────────

class ClassificationHead(nn.Module):
    """
    3-layer MLP classification head.

    Args:
        input_dim   : dimension of the fused representation (2 * embed_dim)
        hidden_dim  : width of the first hidden layer
        num_classes : 2 for binary hateful/non-hateful classification
        dropout     : dropout probability applied after each hidden layer
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, num_classes)
        return self.mlp(x)
