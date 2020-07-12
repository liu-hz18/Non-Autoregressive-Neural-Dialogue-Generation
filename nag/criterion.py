import torch
from torch import nn
from torch.nn import functional as F


class LabelSmoothedCrossEntropyLoss(nn.Module):
    """
    Cross Entropy loss with label smoothing.
    For training, the loss is smoothed with parameter eps, while for evaluation, the smoothing is disabled.
    """
    def __init__(self, eps, ignore_index=-100, weight=None):
        super(LabelSmoothedCrossEntropyLoss, self).__init__()
        self.eps = eps
        self.ignore_index = ignore_index
        self.weight = weight

    def forward(self, input, target):
        # [batch, c, d1, ..., dk]
        log_soft = F.log_softmax(input, dim=1)
        loss = log_soft * -1.
        # [batch, d1, ..., dk]
        nll_loss = F.nll_loss(
            log_soft, target, self.weight,
            None, self.ignore_index, None, 'mean')
        if self.training:
            # [batch, c, d1, ..., dk]
            inf_mask = loss.eq(float('inf'))
            # [batch, d1, ..., dk]
            smooth_loss = loss.masked_fill(inf_mask, 0.).sum(dim=1)
            eps_i = self.eps / (1.0 - inf_mask.float()).sum(dim=1)
            return nll_loss * (1. - self.eps) + (smooth_loss * eps_i).mean()
        else:
            return nll_loss


def neighbor_cosine_similarity(hidden):
    hidden_shift_left = hidden[:][1:][:]
    hidden_shift_right = hidden[:][:-1][:]
    return torch.mean(F.cosine_similarity(hidden_shift_left, hidden_shift_right, dim=2))


def similarity_regularization(hidden, out, alpha=0.1):
    L_hidden = neighbor_cosine_similarity(hidden)
    L_prob = neighbor_cosine_similarity(out)
    loss = 1 + L_hidden * (1 - L_prob)
    return alpha * loss
