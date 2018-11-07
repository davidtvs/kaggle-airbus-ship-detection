import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, weight=1, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        if reduction.lower() == "none":
            self.reduction_op = None
        elif reduction.lower() == "mean":
            self.reduction_op = torch.mean
        elif reduction.lower() == "sum":
            self.reduction_op = torch.sum
        else:
            raise ValueError(
                "expected one of ('none', 'mean', 'sum'), got {}".format(reduction)
            )

    def forward(self, input, target):
        if input.dim() == 4:
            input = input.permute(0, 2, 3, 1)
            input = input.contiguous().view(-1, input.size(-1))
        elif input.dim() != 2:
            raise ValueError(
                "expected input of size 4 or 2, got {}".format(input.dim())
            )

        if target.dim() == 3:
            target = target.contiguous().view(-1)
        elif target.dim() != 1:
            raise ValueError(
                "expected target of size 3 or 1, got {}".format(target.dim())
            )

        if target.dim() != input.dim() - 1:
            raise ValueError(
                "expected target dimension {} for input dimension {}, got {}".format(
                    input.dim() - 1, input.dim(), target.dim()
                )
            )

        m = input.size(0)
        probabilities = nn.functional.softmax(input[range(m), target], dim=0)
        focal = self.weight * (1 - probabilities).pow(self.gamma)
        ce = nn.functional.cross_entropy(input, target, reduction="none")
        loss = focal * ce

        if self.reduction_op is not None:
            return self.reduction_op(loss)
        else:
            return loss
