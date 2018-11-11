import os
import errno
import json
import torch
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data.utils import rle_encode


def to_onehot_np(y, num_classes=None, axis=0, dtype="float32"):
    """Converts a class numpy.ndarray (integers) to a one hot numpy.ndarray.

    Modified from: https://github.com/keras-team/keras/blob/master/keras/utils/np_utils.py#L9

    Arguments:
        y (numpy.ndarray): array of integer values in the range
            [0, num_classes - 1] to be one hot encoded.
        num_classes (int, optional): total number of classes. If set to None,
            num_classes = max(y) + 1. Default: None.
        axis (int, optional): the axis where the one hot classes are encoded.
            E.g. when set to 1 and the size of y is (5, 5) the output is
            (5, num_classes, 5). Default: 0.
        dtype (torch.dtype, optional): the output data type, as a string (float32,
            float64, int32...). Default: float32.

    Returns:
        A one hot representation of the input numpy.ndarray.
    """
    y = np.array(y, dtype="int")
    if not num_classes:
        num_classes = np.max(y) + 1
    elif np.amax(y) > num_classes - 1 or np.amin(y) < 0:
        raise ValueError("y values outside range [0, {}]".format(num_classes - 1))

    input_shape = y.shape
    y = y.ravel()
    n = y.shape[0]
    output_shape = list(input_shape)
    output_shape.append(num_classes)
    axis_order = list(range(len(input_shape)))
    axis_order.insert(axis, -1)

    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    categorical = np.reshape(categorical, output_shape)

    return np.transpose(categorical, axis_order)


def to_onehot_tensor(tensor, num_classes=None, axis=0, dtype=torch.float):
    """Converts a class tensor (integers) to a one hot tensor.

    Arguments:
        tensor (torch.Tensor): tensor of integer values in the range
            [0, num_classes - 1] to be converted into a one hot tensor.
        num_classes (int, optional): total number of classes. If set to None,
            num_classes = max(tensor) + 1. Default: None.
        axis (int, optional): the axis where the one hot classes are encoded.
            E.g. when set to 1 and the size of tensor is (5, 5) the output is
            (5, num_classes, 5). Default: 0.
        dtype (torch.dtype, optional): the output data type. Default: torch.float.

    Returns:
        A one hot representation of the input tensor.
    """
    tensor = torch.tensor(tensor, dtype=torch.long)
    if not num_classes:
        num_classes = torch.max(tensor).item() + 1
    elif tensor.max() > num_classes - 1 or tensor.min() < 0:
        raise ValueError("tensor values outside range [0, {}]".format(num_classes - 1))

    out_shape = list(tensor.size())
    out_shape.insert(axis, num_classes)

    tensor = tensor.unsqueeze(axis)
    onehot = torch.zeros(out_shape, dtype=dtype)
    onehot.scatter_(axis, tensor, 1)

    return onehot


def logits_to_pred_sigmoid(logits):
    """Function to transform logits into predictions.

    Applies the sigmoid function to the logits and rounds the result
    (threshold at 0.5).

    Arguments:
        logits (torch.Tensor): logits output by the model.

    Returns:
        torch.Tensor: The predictions.
    """
    return torch.sigmoid(logits).round()


def save_config(filepath, config):
    with open(filepath, "w") as outfile:
        json.dump(config, outfile, indent=4, sort_keys=True)


def load_config(filepath):
    with open(filepath, "r") as infile:
        config = json.load(infile)

    return config


def save_summary(filepath, args, config, losses, metrics):
    """Saves the model in a specified directory with a specified name.save

    Arguments:
        filepath (str): path to the location where the model will be saved.
        args (dict): the command-line arguments
        config (dict): the configuration used for training.
        losses (dict): the losses.
        metrics (dict): the metrics.

    """
    # Create the directory for the checkpoint in case it doesn't exist
    checkpoint_dir = os.path.dirname(filepath)
    try:
        os.makedirs(checkpoint_dir)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    metrics_dict = {m.name: m.value() for m in metrics}
    out = {"args": args, "config": config, "losses": losses, "metrics": metrics_dict}
    with open(filepath, "w") as summary_file:
        json.dump(out, summary_file, indent=4, sort_keys=True)


def imshow_batch(
    images, targets=None, nrow=8, padding=2, scale_each=False, pad_value=0
):
    """Displays a batch of images and, optionally, of targets.

    Arguments:
        images (torch.Tensor): a 4D mini-batch tensor of shape (B, C, H, W).
        targets (torch.Tensor): a 4D mini-batch tensor of shape (B, C, H, W). When set
            to None only the batch of images is displayed.

    """
    if not isinstance(images, torch.Tensor) or images.dim() != 4:
        raise ValueError(
            "expected '{0}', got '{1}'".format(type(torch.Tensor), type(images))
        )

    # Make a grid with the images
    images = torchvision.utils.make_grid(
        images, nrow=nrow, padding=padding, scale_each=scale_each, pad_value=pad_value
    ).numpy()

    # Check if targets is a Tensor. If it is, display it; otherwise, show the images
    if isinstance(targets, torch.Tensor) and targets.dim() == 4:
        targets = torchvision.utils.make_grid(
            targets,
            nrow=nrow,
            padding=padding,
            scale_each=scale_each,
            pad_value=pad_value,
        ).numpy()

        fig, axarr = plt.subplots(3, 1)
        axarr[0].set_title("Batch of images")
        axarr[0].axis("off")
        axarr[0].imshow(np.transpose(images, (1, 2, 0)))

        axarr[1].set_title("Batch of targets")
        axarr[1].axis("off")
        axarr[1].imshow(np.transpose(targets, (1, 2, 0)))

        axarr[2].set_title("Targets overlayed with images")
        axarr[2].axis("off")
        axarr[2].imshow(np.transpose(images, (1, 2, 0)))
        axarr[2].imshow(np.transpose(targets, (1, 2, 0)), alpha=0.5)
    else:
        plt.imshow(np.transpose(images, (1, 2, 0)))
        plt.axis("off")
        plt.gca().set_title("Batch of samples")

    plt.show()


def dataloader_info(dataloader):
    """Displays information about a given dataloader.

    Prints the size of the dataset, the dimensions of the images and target images, and
    displays a batch of images and targets with `imshow_batch()`.

    Arguments:
        dataloader (torch.utils.data.Dataloader): the dataloader.

    """
    images, targets = iter(dataloader).next()
    print("Number of images:", len(dataloader.dataset))
    print("Image size:", images.size())
    print("Targets size:", targets.size())
    imshow_batch(images, targets)


def make_submission(save_dir, image_ids, predictions):
    rle = []
    for pred in predictions:
        rle.append(rle_encode(pred))

    submission_df = pd.DataFrame({"ImageId": image_ids, "EncodedPixels": rle})
    submission_path = os.path.join(save_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
