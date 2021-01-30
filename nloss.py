import torch
import torch.nn as nn
import torch.nn.functional as F

#All Avaliable.
class normLSFLoss(nn.Module):
    def __init__(self, gamma=0.0, smoothing=0.0, reduction='mean', isCos=False, isNorm=False):
        super(normLSFLoss, self).__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.reduction = reduction
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.isCos = isCos
        self.isNorm = isNorm
    def forward(self, logits, target):
        with torch.no_grad():
            num_classes = logits.size(1)
            target = target.clone().detach()
            lb_pos, lb_neg = 1. - self.smoothing + (self.smoothing / num_classes) , self.smoothing / num_classes
            lb_one_hot = torch.empty(size=(target.size(0), num_classes),  device=target.device).fill_(lb_neg).scatter_(1, target.data.unsqueeze(1), lb_pos)
        if self.isCos:
            logs = self.log_softmax(F.normalize(logits))
            lb_one_hot =  F.normalize(lb_one_hot)
        else:
            logs = self.log_softmax(logits)
        pt = torch.exp(logs)
        if self.isNorm:
            loss = -torch.sum((1-pt).pow(self.gamma)*logs * lb_one_hot, dim=1) / (- logs.sum(dim=1))
        else:
            loss = -torch.sum((1-pt).pow(self.gamma)*logs * lb_one_hot, dim=1)          
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss
    