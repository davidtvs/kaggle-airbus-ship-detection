import torch
import torch.nn as nn
import utils


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


class DiceLoss(nn.Module):
    """https://github.com/pytorch/pytorch/issues/1249"""

    def __init__(self, reduction="mean", eps=1e-6):
        super().__init__()
        self.eps = eps
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
        if input.dim() != 2 and input.dim() != 4:
            raise ValueError(
                "expected input of size 4 or 2, got {}".format(input.dim())
            )

        if target.dim() != 1 and target.dim() != 3:
            raise ValueError(
                "expected target of size 3 or 1, got {}".format(target.dim())
            )

        if input.dim() == 4 and target.dim() == 3:
            reduce_dims = (3, 2)
        elif input.dim() == 2 and target.dim() == 1:
            reduce_dims = 1
        else:
            raise ValueError(
                "expected target dimension {} for input dimension {}, got {}".format(
                    input.dim() - 1, input.dim(), target.dim()
                )
            )

        target_onehot = utils.to_onehot(target, input.size(1))
        probabilities = nn.functional.softmax(input, 1)

        # dice = 2 * (t * p) / (t^2 + p^2)
        num = torch.sum(target_onehot * probabilities, dim=reduce_dims)
        den_t = torch.sum(target_onehot * target_onehot, dim=reduce_dims)
        den_p = torch.sum(probabilities * probabilities, dim=reduce_dims)
        loss = -2 * (num / (den_t + den_p + self.eps))

        if self.reduction_op is not None:
            return self.reduction_op(loss)
        else:
            return loss
