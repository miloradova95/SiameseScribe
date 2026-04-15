import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, anchor_emb, positive_emb, negative_emb):
        dist_ap = F.pairwise_distance(anchor_emb, positive_emb)
        dist_an = F.pairwise_distance(anchor_emb, negative_emb)
        loss = torch.mean(torch.clamp(dist_ap - dist_an + self.margin, min=0.0))
        return loss
