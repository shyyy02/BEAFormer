import torch
from torch import nn
import torch.nn.functional as F


class BCEWithLogitsLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, targets):
        logits = logits.float().flatten(1)
        targets = targets.float().flatten(1)
        loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction=self.reduction
        )
        return loss


class BCELoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, targets, weights = None):
        logits = logits.float().flatten(1)
        targets = targets.float().flatten(1)
        if weights is not None:
            weights = weights.float().flatten(1)
        loss = F.binary_cross_entropy(logits, targets, weights, reduction=self.reduction)
        return loss


class CrossEntropyLoss(nn.Module):
    def __init__(self, reduction="mean", ignore_index=-100):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        logits = logits.float()
        targets = targets.long()
        return F.cross_entropy(
            logits, targets, reduction=self.reduction, ignore_index=self.ignore_index
        )


class DiceLoss(nn.Module):
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index
    
    def _one_hot(self, targets, num_classes):
        classes = torch.arange(num_classes, device=targets.device)
        one_hot = (targets[:, None, :, :] == classes[None, :, None, None]).float()
        return one_hot

    def _binary_dice(self, logits, targets, weights):
        if len(logits.shape) == 4:
            logits = logits.squeeze(1)

        mask = (targets != self.ignore_index).float()
        if weights is not None:
            mask = mask * weights.float()
        probs = (torch.sigmoid(logits.float()) * mask).flatten(1)
        targets = (targets.float() * mask).flatten(1)
        
        intersection = (probs * targets).sum(1)
        union = (probs**2 + targets**2).sum(1)
        
        dice = (2.0 * intersection + 1e-5) / (union + 1e-5)
        dice_loss = 1 - dice
        
        return dice_loss.mean()

    def _multiclass_dice(self, logits, targets, weights):
        mask = (targets != self.ignore_index).float().unsqueeze(1)
        if weights is not None:
            mask = mask * weights.float().unsqueeze(1)
        probs = (F.softmax(logits.float(), dim=1) * mask).flatten(2)
        targets = self._one_hot(targets.long(), logits.shape[1]).flatten(2)

        intersection = (probs * targets).sum(2)
        union = (probs**2 + targets**2).sum(2)

        dice = (2.0 * intersection + 1e-5) / (union + 1e-5)
        dice_loss = 1 - dice

        return dice_loss.mean()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor = None):
        if len(logits.shape) == 3 or logits.shape[1] == 1:
            return self._binary_dice(logits, targets, weights)
        else:
            return self._multiclass_dice(logits, targets, weights)
