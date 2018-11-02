import os
import errno
import torch
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data.utils import rle_encode


def save_summary(filepath, losses, metrics, args):
    """Saves the model in a specified directory with a specified name.save

    Arguments:
        filepath (str): path to the location where the model will be saved
        losses (dict): a dictionary of losses.
        metrics (dict): a dictionary of metrics.
        args (ArgumentParser): the command-line arguments.

    """
    # Create the directory for the checkpoint in case it doesn't exist
    checkpoint_dir = os.path.dirname(filepath)
    try:
        os.makedirs(checkpoint_dir)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    # Save arguments
    with open(filepath, "w") as summary_file:
        # Write arguments
        summary_file.write("Arguments\n")
        str_list = ["{0}: {1}".format(arg, getattr(args, arg)) for arg in args]
        str_list = sorted(str_list)
        summary_file.write("\n".join(str_list))
        summary_file.write("\n")

        # Write losses
        summary_file.write("\nLosses\n")
        str_list = ["{0}: {1}".format(key, losses[key]) for key in losses]
        str_list = sorted(str_list)
        summary_file.write("\n".join(str_list))
        summary_file.write("\n")

        # Write metrics
        summary_file.write("\nMetrics\n")
        str_list = ["{0}: {1}".format(key, metrics[key].value()) for key in metrics]
        str_list = sorted(str_list)
        summary_file.write("\n".join(str_list))
        summary_file.write("\n")


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
