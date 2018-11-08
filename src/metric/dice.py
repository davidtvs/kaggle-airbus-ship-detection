import numpy as np
from metric import metric
from utils import to_onehot


class BinaryDice(metric.Metric):
    """Computes the Sørensen–Dice coefficient for binary data.

    Dice = 2 * intersection(X, Y) / (|X| + |Y|)
    where, X and Y are sets of binary data, in this case, predictions and targets.
    |X| and |Y| are the cardinalities of the corresponding sets.

    See: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    Arguments:
        name (str, optional): a name for the metric. Default: bin_dice.
        eps (float, optional): small value to avoid division by zero. Default: 1e-6.

    """

    def __init__(self, name="bin_dice", eps=1e-6):
        super().__init__(name)
        self.eps = eps
        self.intersection = 0
        self.cardinality_p = 0
        self.cardinality_t = 0

    def reset(self):
        """Clears previously added predicted and target pairs."""
        self.intersection = 0
        self.cardinality_p = 0
        self.cardinality_t = 0

    def add(self, predicted, target):
        """Adds the predicted and target pair to the Dice coefficient computation.

        Arguments:
            predicted (torch.Tensor): a (N, *) tensor of predictions, where * means
                any number of additional dimensions
            target (torch.Tensor): a (N, *) tensor of targets, where * means any number
                of additional dimensions

        """
        # Parameter check
        if predicted.size() != target.size():
            raise ValueError(
                "size mismatch, {} != {}".format(predicted.size(), target.size())
            )

        # Flatten the tensor and convert to numpy
        predicted = predicted.cpu().view(-1).numpy()
        target = target.cpu().view(-1).numpy()

        if tuple(np.unique(predicted)) not in [(0, 1), (0,), (1,)]:
            raise ValueError("predicted values are not binary")
        if tuple(np.unique(target)) not in [(0, 1), (0,), (1,)]:
            raise ValueError("target values are not binary")

        self.intersection += np.sum(target * predicted)
        self.cardinality_t += np.sum(target)
        self.cardinality_p += np.sum(predicted)

    def value(self):
        """Computes the Dice coefficient.

        Returns:
            float: the Dice coefficient.
        """
        return 2 * (
            self.intersection / (self.cardinality_t + self.cardinality_p + self.eps)
        )


class Dice(metric.Metric):
    """Computes the Sørensen–Dice coefficient per class and corresponding mean.

    Dice = 2 * intersection(X, Y) / (|X| + |Y|)
    where, X and Y are sets of binary data, in this case, predictions and targets.
    |X| and |Y| are the cardinalities of the corresponding sets.

    See: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    Arguments:
        num_classes (int): number of classes in the classification problem
        name (str, optional): a name for the metric. Default: dice.
        eps (float, optional): small value to avoid division by zero. Default: 1e-6.

    """

    def __init__(self, num_classes, name="dice", eps=1e-6):
        super().__init__(name)
        self.num_classes = num_classes
        self.eps = eps
        self.intersection = np.zeros((self.num_classes,))
        self.cardinality_p = np.zeros((self.num_classes,))
        self.cardinality_t = np.zeros((self.num_classes,))

    def reset(self):
        """Clears previously added predicted and target pairs."""
        self.intersection = np.zeros((self.num_classes,))
        self.cardinality_p = np.zeros((self.num_classes,))
        self.cardinality_t = np.zeros((self.num_classes,))

    def add(self, predicted, target):
        """Adds the predicted and target pair to the Dice coefficient.

        Arguments:
            predicted (torch.Tensor): A (N, H, W) or a (H, W) tensor of integer encoded
                predictions in the range [0, num_classes-1].
            target (torch.Tensor): A (N, H, W) or a (H, W) tensor of integer encoded
                target values in the range [0, num_classes-1].

        """
        # Parameter check
        if predicted.size() != target.size():
            raise ValueError(
                "size mismatch, {} != {}".format(predicted.size(), target.size())
            )
        elif predicted.min() < 0 or predicted.max() > self.num_classes - 1:
            raise ValueError("predicted values outside range [0, num_classes-1]")
        elif target.min() < 0 or target.max() > self.num_classes - 1:
            raise ValueError("target values outside range [0, num_classes-1]")

        predicted = to_onehot(predicted, self.num_classes).numpy()
        target = to_onehot(target, self.num_classes).numpy()

        self.intersection += np.sum(target * predicted, axis=(3, 2, 0))
        self.cardinality_t += np.sum(target, axis=(3, 2, 0))
        self.cardinality_p += np.sum(predicted, axis=(3, 2, 0))

    def value(self):
        """Computes the mean Dice coefficient.

        Returns:
            float: the mean Dice coefficient.
        """
        return np.mean(self.value_class())

    def value_class(self):
        """Computes the Dice coefficient per class.

        Returns:
            numpy.ndarray: An array where each element corresponds to the Dice of the
            corresponding class.
        """
        return 2 * (
            self.intersection / (self.cardinality_t + self.cardinality_p + self.eps)
        )
