import numpy as np
import torch


def confusion_matrix(num_classes, predicted, target, normalized=False):
    """Constructs a confusion matrix.

    Supports multi-class problems but does not support multi-label, multi-class
    problems.
    Modified from: https://github.com/pytorch/tnt/blob/master/torchnet/meter/confusionmeter.py

    Arguments:
        num_classes (int): number of classes in the classification problem.
        predicted (torch.Tensor or numpy.ndarray): Can be a tensor/array of one-hot
            predictions with shape (N x num_classes), or an integer encoded tensor/array
            with shape (N,) and values in the range [0, num_classes-1].
        target (torch.Tensor or numpy.ndarray): Can be a tensor/array of one-hot targets
            with shape (N x num_classes), or an integer encoded tensor/array with shape
            (N,) and values in the range [0, num_classes-1].
        normalized (boolean, optional): Determines whether or not the confusion matrix
            is normalized or not. Default: False.

    """
    # If target and/or predicted are tensors, convert them to numpy arrays
    if torch.is_tensor(predicted):
        predicted = predicted.cpu().numpy()
    if torch.is_tensor(target):
        target = target.cpu().numpy()

    if predicted.shape[0] != target.shape[0]:
        raise ValueError("target and predicted do not match")

    # predicted validation and pre-processing
    if np.ndim(predicted) != 1:
        # Asssume one-hot encoding (N, num_classes)
        if predicted.shape[1] != num_classes:
            raise ValueError(
                "one-hot predictions has {} classes but {} were requested".format(
                    predicted.shape[1], num_classes
                )
            )

        # Convert from one-hot encoding to int encoding
        predicted = np.argmax(predicted, 1)
    else:
        # Assume int encoding (N,)
        if predicted.max() > num_classes - 1 and predicted.min() < 0:
            raise ValueError("predicted values out of range [0, num_classes-1]")

    # target validation and pre-processing
    if np.ndim(target) != 1:
        # Assume one-hot encoding (N, num_classes)
        if target.shape[1] != num_classes:
            raise ValueError(
                "one-hot targets has {} classes but {} were requested".format(
                    target.shape[1], num_classes
                )
            )
        elif (target < 0).any() and (target > 1).any():
            raise ValueError("found values different than 0 and 1 in one-hot targets")
        elif (target.sum(1) != 1).any():
            raise ValueError("multi-label setting is not supported")

        # Convert from one-hot encoding to int encoding
        target = np.argmax(target, 1)
    else:
        # Assume int encoding (N,)
        if target.max() > num_classes - 1 and target.min() < 0:
            raise ValueError("target values out of range [0, num_classes-1]")

    # Tricnum_classes for bincounting 2 arrays together
    x = predicted + num_classes * target
    bincount_2d = np.bincount(x.astype(np.int32), minlength=num_classes ** 2)
    assert bincount_2d.size == num_classes ** 2
    conf = bincount_2d.reshape((num_classes, num_classes))

    if normalized:
        conf = conf.astype(np.float32)
        conf = conf / conf.sum(1).clip(min=1e-12)[:, None]

    return conf
