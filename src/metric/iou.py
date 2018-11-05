import numpy as np
from metric import metric, utils


class IoU(metric.Metric):
    """Computes the intersection over union (IoU) per class and corresponding
    mean (mIoU).

    Intersection over union (IoU) is a common evaluation metric for semantic
    segmentation. The predictions are first accumulated in a confusion matrix
    and the IoU is computed from it as follows:

        IoU = true_positive / (true_positive + false_positive + false_negative).

    Arguments:
        num_classes (int): number of classes in the classification problem
        normalized (boolean, optional): Determines whether or not the confusion matrix
            is normalized or not. Default: False.
        ignore_index (int or iterable, optional): Index of the classes to ignore when
            computing the IoU. Can be an int, or any iterable of ints. Default: None.
        name (str): a name for the metric. Default: miou.
    """

    def __init__(self, num_classes, normalized=False, ignore_index=None, name="miou"):
        super().__init__(name)
        self.num_classes = num_classes
        self.normalized = normalized
        self.conf_matrix = np.ndarray((num_classes, num_classes), dtype=np.int32)

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

        The mean computation ignores NaN elements in the IoU array.

        Returns:
            float: the mean IoU.
        """
        return np.nanmean(self.class_iou())

    def class_iou(self):
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

        # Just in case we get a division by 0, ignore/hide the error
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = true_positive / (true_positive + false_positive + false_negative)

        return iou
