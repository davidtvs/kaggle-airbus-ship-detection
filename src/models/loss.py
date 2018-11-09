import torch
import torch.nn as nn
import utils


class BCE2dWithLogitsLoss(nn.Module):
    def __init__(self, weight=None, reduction="elementwise_mean", pos_weight=None):
        super().__init__()
        self.bce_logits = nn.BCEWithLogitsLoss(
            weight=weight, reduction=reduction, pos_weight=pos_weight
        )

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.float()
        return self.bce_logits(input, target)


class BinaryFocalWithLogitsLoss(nn.Module):
    """Computes the focal loss with logits for binary data.

    The Focal Loss is designed to address the one-stage object detection scenario in
    which there is an extreme imbalance between foreground and background classes during
    training (e.g., 1:1000). Focal loss is defined as:

    FL = alpha(1 - p)^gamma * CE(p, y)
    where p are the probabilities, after applying the sigmoid to the logits, alpha is a
    balancing parameter, gamma is the focusing parameter, and CE(p, y) is the
    cross entropy loss. When gamma=0 and alpha=1 the focal loss equals cross entropy.

    See: https://arxiv.org/abs/1708.02002

    Arguments:
        gamma (float, optional): focusing parameter. Default: 2.
        alpha (float, optional): balancing parameter. Default: 0.25.
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'mean'
        eps (float, optional): small value to avoid division by zero. Default: 1e-6.

    """

    def __init__(self, gamma=2, alpha=0.25, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
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
        if input.size() != target.size():
            raise ValueError(
                "size mismatch, {} != {}".format(input.size(), target.size())
            )
        elif target.unique(sorted=True).tolist() not in [[0, 1], [0], [1]]:
            raise ValueError("target values are not binary")

        input = input.view(-1)
        target = target.view(-1)

        # Following the paper: probabilities = probabilities if y=1; otherwise,
        # probabilities = 1-probabilities
        probabilities = torch.sigmoid(input)
        probabilities[target != 1] = 1 - probabilities[target != 1]

        # Compute the loss
        focal = self.alpha * (1 - probabilities).pow(self.gamma)
        bce = nn.functional.binary_cross_entropy_with_logits(
            input, target, reduction="none"
        )
        loss = focal * bce

        if self.reduction_op is not None:
            return self.reduction_op(loss)
        else:
            return loss


class FocalWithLogitsLoss(nn.Module):
    """Computes the focal loss with logits.

    The Focal Loss is designed to address the one-stage object detection scenario in
    which there is an extreme imbalance between foreground and background classes during
    training (e.g., 1:1000). Focal loss is defined as:

    FL = alpha(1 - p)^gamma * CE(p, y)
    where p are the probabilities, after applying the softmax layer to the logits,
    alpha is a balancing parameter, gamma is the focusing parameter, and CE(p, y) is the
    cross entropy loss. When gamma=0 and alpha=1 the focal loss equals cross entropy.

    See: https://arxiv.org/abs/1708.02002

    Arguments:
        gamma (float, optional): focusing parameter. Default: 2.
        alpha (float, optional): balancing parameter. Default: 0.25.
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'mean'
        eps (float, optional): small value to avoid division by zero. Default: 1e-6.

    """

    def __init__(self, gamma=2, alpha=0.25, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
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
        focal = self.alpha * (1 - probabilities).pow(self.gamma)
        ce = nn.functional.cross_entropy(input, target, reduction="none")
        loss = focal * ce

        if self.reduction_op is not None:
            return self.reduction_op(loss)
        else:
            return loss


class BinaryDiceWithLogitsLoss(nn.Module):
    """Computes the Sørensen–Dice loss with logits for binary data.

    Dice_coefficient = 2 * intersection(X, Y) / (|X| + |Y|)
    where, X and Y are sets of binary data, in this case, probabilities and targets.
    |X| and |Y| are the cardinalities of the corresponding sets. Probabilities are
    computed using the sigmoid.

    The optimizer minimizes the loss function therefore:
    Dice_loss = -Dice_coefficient
    (min(-x) = max(x))

    See: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    Arguments:
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'mean'
        eps (float, optional): small value to avoid division by zero. Default: 1e-6.

    """

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
        if input.size() != target.size():
            raise ValueError(
                "size mismatch, {} != {}".format(input.size(), target.size())
            )
        elif target.unique(sorted=True).tolist() not in [[0, 1], [0], [1]]:
            raise ValueError("target values are not binary")

        input = input.view(-1)
        target = target.view(-1)

        # Dice = 2 * intersection(X, Y) / (|X| + |Y|)
        # X and Y are sets of binary data, in this case, probabilities and targets
        # |X| and |Y| are the cardinalities of the corresponding sets
        probabilities = torch.sigmoid(input)
        num = torch.sum(target * probabilities)
        den_t = torch.sum(target)
        den_p = torch.sum(probabilities)
        loss = -2 * (num / (den_t + den_p + self.eps))

        if self.reduction_op is not None:
            return self.reduction_op(loss)
        else:
            return loss


class DiceWithLogitsLoss(nn.Module):
    """Computes the Sørensen–Dice loss with logits.

    Dice_coefficient = 2 * intersection(X, Y) / (|X| + |Y|)
    where, X and Y are sets of binary data, in this case, predictions and targets.
    |X| and |Y| are the cardinalities of the corresponding sets. Probabilities are
    computed using softmax.

    The optimizer minimizes the loss function therefore:
    Dice_loss = -Dice_coefficient
    (min(-x) = max(x))

    See: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    Arguments:
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'mean'
        eps (float, optional): small value to avoid division by zero. Default: 1e-6.

    """

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
            reduce_dims = (0, 3, 2)
        elif input.dim() == 2 and target.dim() == 1:
            reduce_dims = 0
        else:
            raise ValueError(
                "expected target dimension {} for input dimension {}, got {}".format(
                    input.dim() - 1, input.dim(), target.dim()
                )
            )

        target_onehot = utils.to_onehot(target, input.size(1))
        probabilities = nn.functional.softmax(input, 1)

        # Dice = 2 * intersection(X, Y) / (|X| + |Y|)
        # X and Y are sets of binary data, in this case, probabilities and targets
        # |X| and |Y| are the cardinalities of the corresponding sets
        num = torch.sum(target_onehot * probabilities, dim=reduce_dims)
        den_t = torch.sum(target_onehot, dim=reduce_dims)
        den_p = torch.sum(probabilities, dim=reduce_dims)
        loss = -2 * (num / (den_t + den_p + self.eps))

        if self.reduction_op is not None:
            return self.reduction_op(loss)
        else:
            return loss
