import os
import errno
import json
import torch
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data.utils import rle_encode


def save_config(filepath, config):
    with open(filepath, "w") as outfile:
        json.dump(config, outfile, indent=4, sort_keys=True)


def load_config(filepath):
    with open(filepath, "r") as infile:
        config = json.load(infile)

    return config


def save_summary(filepath, config, losses, metrics):
    """Saves the model in a specified directory with a specified name.save

    Arguments:
        filepath (str): path to the location where the model will be saved
        config (dict): a dictionary with the configuration used for training.
        losses (dict): a dictionary of losses.
        metrics (dict): a dictionary of metrics.

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
    out = {"config": config, "losses": losses, "metrics": metrics_dict}
    with open(filepath, "w") as summary_file:
        json.dump(out, summary_file, indent=4, sort_keys=True)


def imshow_batch(images, targets=None):
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
    images = torchvision.utils.make_grid(images).numpy()

    # Check if targets is a Tensor. If it is, display it; otherwise, show the images
    if isinstance(targets, torch.Tensor) and targets.dim() == 4:
        targets = torchvision.utils.make_grid(targets).numpy()

        fig, axarr = plt.subplots(3, 1)
        axarr[0].set_title("Batch of samples")
        axarr[0].axis("off")
        axarr[0].imshow(np.transpose(images, (1, 2, 0)))

        axarr[1].set_title("Batch of targets")
        axarr[1].axis("off")
        axarr[1].imshow(np.transpose(targets, (1, 2, 0)))

        axarr[2].set_title("Targets overlayed with samples")
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
