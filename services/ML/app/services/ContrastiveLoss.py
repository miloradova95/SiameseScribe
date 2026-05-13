import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    """Pulls a confirmed positive pair together: L = mean(||emb_a - emb_p||^2)."""

    def forward(self, emb_anchor: torch.Tensor, emb_positive: torch.Tensor) -> torch.Tensor:
        diff = emb_anchor - emb_positive
        return torch.mean(torch.sum(diff ** 2, dim=1))
