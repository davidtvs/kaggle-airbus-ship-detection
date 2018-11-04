import numpy as np
import torch
from metric import metric


class Accuracy(metric.Metric):
    """Computes the accuracy.

    accuracy = correct_predictions / total_predictions

    Arguments:
        name (str): a name for the metric. Default: acc.
    """

    def __init__(self, name="acc"):
        super().__init__(name)
        self.reset()

    def reset(self):
        self.correct_pred = 0
        self.total = 0

    def add(self, predicted, target):
        """Computes the accuracy of the predictions.

        Arguments:
            predicted (torch.Tensor or numpy.ndarray): tensor/array of predicted scores
                obtained from the model.
            target (torch.Tensor or numpy.ndarray): tensor/array of ground-truth labels.

        """
        # If target and/or predicted are tensors, convert them to numpy arrays
        if torch.is_tensor(predicted):
            predicted = predicted.cpu().numpy()
        if torch.is_tensor(target):
            target = target.cpu().numpy()

        predicted = predicted.flatten()
        target = target.flatten()

        self.correct_pred += np.sum(predicted == target)
        self.total += len(target)

    def value(self):
        """
        Returns:
            float: The accuracy.
        """
        # Just in case we get a division by 0, ignore/hide the error
        with np.errstate(divide="ignore", invalid="ignore"):
            res = np.divide(self.correct_pred, self.total)

        return res
