import numpy as np
from metric import metric, iou
from utils import split_ships


def f_score(prediction_masks, target_masks, beta=2, thresholds=np.arange(0.5, 1, 0.05)):
    """Computes the F(beta, thresholds) score at different intersection over union (IoU)
    thresholds.

    At each threshold value, the F Score value is calculated based on the number of
    true positives (TP), false negatives (FN), and false positives (FP) resulting from
    comparing the predicted object to all ground truth objects:

        F(beta, t) = (1 + beta^2) * TP(t) / ((1 + beta^2) * TP(t) + beta^2 * FN(t) + FP(t))

    A true positive is counted when a single predicted object matches a ground truth
    object with an IoU above the threshold. A false positive indicates a predicted
    object had no associated ground truth object. A false negative indicates a ground
    truth object had no associated predicted object. The average F Score of a single
    image is then calculated as the mean of the above F Score values at each IoU
    threshold.

    Implementation modified from: https://www.kaggle.com/markup/f2-metric-optimized

    Arguments:
        prediction_masks (numpy.ndarray): prediction masks with shape (n, H, W) where n
            is the number of predicted objects.
        target_masks (numpy.ndarray): target masks with shape (n, H, W) where n is the
            number of ground-truth objects.
        beta (int, optional): recall-precision weight. Default: 2.
        thresholds (numpy.ndarray, optional): IoU thresholds.
            Default: np.arange(0.5, 1, 0.05).

        Returns:
            float: the F-score.

    """
    # If the target is empty return 1 if the prediction is also empty; otherwise
    # return 0
    if np.sum(target_masks) == 0:
        return float(np.sum(prediction_masks) == 0)

    iou_arr = []
    pred_idx_found = []
    for target in target_masks:
        for pred_idx, pred in enumerate(prediction_masks):
            # Check if this prediction mask has already been matched to a target mask
            if pred_idx not in pred_idx_found:
                curr_iou = iou.binary_iou(pred, target)
                if curr_iou > np.min(thresholds):
                    iou_arr.append(curr_iou)
                    # Matched a prediction with a target, remember the index so we don't
                    # match it to another target mask
                    pred_idx_found.append(pred_idx)
                    break

    # F score computation
    fscore_total, tp, fn, fp = 0, 0, 0, 0
    beta_sq = beta * beta
    iou_np = np.array(iou_arr)
    for th in thresholds:
        tp = np.sum(iou_np > th)
        fp = len(prediction_masks) - tp
        fn = len(target_masks) - tp
        fscore_total += (1 + beta_sq) * tp / ((1 + beta_sq) * tp + beta_sq * fn + fp)

    return fscore_total / len(thresholds)


class AirbusFScoreApprox(metric.Metric):
    """Computes the approximate F(beta, thresholds) score at different intersection over
    union (IoU) thresholds.

    The result is approximate because ships from ground-truth masks are split using
    their connectivity. Ships specified in the RLE as seprate but connected are detected
    as a single ship in this metric.
    For details about the F score computation see f_score().

    Arguments:
        beta (int, optional): recall-precision weight. Default: 2.
        thresholds (numpy.ndarray, optional): IoU thresholds.
            Default: np.arange(0.5, 1, 0.05).
        min_size(int, optional): only blobs above this size in pixels are labeled as
            ships, essentially noise removal. Default: 18.
        max_ships_error (int, optional): maximum number of ships allowed in a single
            image. If surpassed, a ValueError is raised. Default: 100.
        name (str, optional): a name for the metric. Default: fscore_approx.

    """

    def __init__(
        self,
        beta=2,
        thresholds=np.arange(0.5, 1, 0.05),
        min_size=18,
        max_ships_error=30,
        name="fscore_approx",
    ):
        super().__init__(name)
        self.thresholds = thresholds
        self.beta = beta
        self.min_size = min_size
        self.max_ships_error = max_ships_error
        self.fscore_history = []

    def reset(self):
        """Clears previously added predicted and target pairs."""
        self.fscore_history = []

    def add(self, predicted, target):
        """Adds the predicted and target pair to the IoU metric.

        Arguments:
            predicted (torch.Tensor): a (N, 1, H, W) binary tensor of predictions.
            target (torch.Tensor): a (N, 1, H, W) binary tensor of targets.

        """
        # Parameter check
        if predicted.size() != target.size():
            raise ValueError(
                "size mismatch, {} != {}".format(predicted.size(), target.size())
            )
        elif tuple(predicted.unique(sorted=True)) not in [(0, 1), (0,), (1,)]:
            raise ValueError("predicted values are not binary")
        elif tuple(target.unique(sorted=True)) not in [(0, 1), (0,), (1,)]:
            raise ValueError("target values are not binary")

        # Flatten the tensor and convert to numpy
        predicted = predicted.squeeze().cpu().numpy()
        target = target.squeeze().cpu().numpy()

        for p, t in zip(predicted, target):
            # Try to split the segmentation mask in into one mask per ship
            # This process might raise an error if too many ships are found, especially
            # during the early stages of training.
            try:
                predicted_ships = split_ships(p, self.min_size, self.max_ships_error)
            except ValueError:
                # Catch the error and give this image a 0 score.
                self.fscore_history.append(0)
                break

            # Note that here we want to fail if too many ships are found, it should
            # never happen
            target_ships = split_ships(t, min_size=0)
            score = f_score(
                predicted_ships,
                target_ships,
                beta=self.beta,
                thresholds=self.thresholds,
            )
            self.fscore_history.append(score)

    def value(self):
        """Computes the F Score.

        Returns:
            float: the F Score.
        """
        if len(self.fscore_history) == 0:
            return 0
        else:
            return np.mean(self.fscore_history)
