import numpy as np
from metric import metric, utils


def binary_iou(predicted, target, eps=1e-6):
    tp = np.sum(target * predicted)
    fn = np.sum(target) - tp
    fp = np.sum(predicted) - tp
    return tp / (tp + fn + fp + 1e-6)


class BinaryIoU(metric.Metric):
    """Computes the intersection over union (IoU) for binary data.

    The intersection over union (IoU) is defined as:

        IoU = true_positive / (true_positive + false_positive + false_negative).

    See: https://en.wikipedia.org/wiki/Jaccard_index

    Arguments:
        name (str, optional): a name for the metric. Default: bin_miou.
        eps (float, optional): small value to avoid division by zero. Default: 1e-6.

    """

    def __init__(self, name="bin_iou", eps=1e-6):
        super().__init__(name)
        self.eps = eps
        self.iou_history = []

    def reset(self):
        """Clears previously added predicted and target pairs."""
        self.iou_history = []

    def add(self, predicted, target):
        """Adds the predicted and target pair to the IoU metric.

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
        elif tuple(np.unique(target)) not in [(0, 1), (0,), (1,)]:
            raise ValueError("target values are not binary")

        self.iou_history.append(binary_iou(predicted, target, eps=self.eps))

    def value(self):
        """Computes the IoU.

        Returns:
            float: the IoU.
        """
        return np.mean(self.iou_history)


class IoU(metric.Metric):
    """Computes the intersection over union (IoU) per class and corresponding
    mean (mIoU).

    The predictions are first accumulated in a confusion matrix
    and the IoU is computed from it as follows:

        IoU = true_positive / (true_positive + false_positive + false_negative).

    See: https://en.wikipedia.org/wiki/Jaccard_index

    Arguments:
        num_classes (int): number of classes in the classification problem
        ignore_index (int or iterable, optional): Index of the classes to ignore when
            computing the IoU. Can be an int, or any iterable of ints. Default: None.
        name (str, optional): a name for the metric. Default: miou.
        eps (float, optional): small value to avoid division by zero. Default: 1e-6.

    """

    def __init__(self, num_classes, ignore_index=None, name="miou", eps=1e-6):
        super().__init__(name)
        self.num_classes = num_classes
        self.eps = eps
        self.conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)

        if ignore_index is None:
            self.ignore_index = None
        elif isinstance(ignore_index, int):
            self.ignore_index = (ignore_index,)
        else:
            try:
                self.ignore_index = tuple(ignore_index)
            except TypeError:
                raise ValueError("'ignore_index' must be an int or iterable")

    def reset(self):
        """Clears previously added predicted and target pairs."""
        self.conf_matrix.fill(0)

    def add(self, predicted, target):
        """Adds the predicted and target pair to the IoU metric.

        Arguments:
            predicted (torch.Tensor): A (N, H, W) or a (H, W) tensor of integer encoded
                target values in the range [0, num_classes-1].
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

        # Add to the confusion matrix
        self.conf_matrix += utils.confusion_matrix(
            self.num_classes, predicted.view(-1), target.view(-1)
        )

    def value(self):
        """Computes the mean IoU.

        Returns:
            float: the mean IoU.
        """
        return np.mean(self.value_class())

    def value_class(self):
        """Computes the IoU per class.

        Returns:
            numpy.ndarray: An array where each element corresponds to the IoU of that
            class.
        """
        if self.ignore_index is not None:
            for index in self.ignore_index:
                self.conf_matrix[:, self.ignore_index] = 0
                self.conf_matrix[self.ignore_index, :] = 0
        true_positive = np.diag(self.conf_matrix)
        false_positive = np.sum(self.conf_matrix, 0) - true_positive
        false_negative = np.sum(self.conf_matrix, 1) - true_positive

        iou = true_positive / (
            true_positive + false_positive + false_negative + self.eps
        )

        return iou
