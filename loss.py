import torch
import torch.nn as nn


class CrossentropyLoss(nn.Module):

    def __init__(self):
        self.eps = 1e-8
        super.__init__()

    def forward(self, pred, true):
        return -torch.mean(torch.sum(true * torch.log(pred + self.eps), dim=-1))


class SemiHardTripletLoss(nn.Module):

    def __init__(self, margin):
        self.margin = margin
        super.__init__()

    def forward(self, anc, pos, neg):
        """
        anc: embeddings from anchor inputs
        pos: embeddings from positive inputs
        neg: embeddings from negative inputs

        shape: (batch_size, num_features)
        """
        diff_pos = torch.sum((anc - pos) ** 2, dim=-1)
        diff_neg = torch.sum((anc - neg) ** 2, dim=-1)
        loss = torch.clamp(diff_pos - diff_neg + self.margin, 0.0, None)
        loss = torch.mean(loss)
        return loss
