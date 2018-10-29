import os
import errno
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt


def imshow_batch(images, targets=None):
    """Displays a batch of images and, optionally, of targets.

    Arguments:
        images (torch.Tensor): a 4D mini-batch tensor of shape (B, C, H, W).
        targets (torch.Tensor): a 4D mini-batch tensor of shape (B, C, H, W). When set
            to None only the batch of images is displayed.

    """
    if not isinstance(images, torch.Tensor) or images.dim() != 4:
        raise RuntimeError(
            "Expected '{0}', got '{1}'".format(type(torch.Tensor), type(images))
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


def save_checkpoint(model, optimizer, metric, epoch, args):
    """Saves the model in a specified directory with a specified name.save

    Arguments:
        model (torch.nn.Module): the model to save.
        optimizer (torch.optim): the model optimizer.
        epoch (int): the current epoch.
        metric (object): the performance metric.
        args (ArgumentParser): the command-line arguments.

    """
    name = args.model_name

    if not os.path.isdir(args.checkpoint_dir):
        raise IOError("The directory '{0}' doesn't exist.".format(args.checkpoint_dir))

    # Create the subdirectory for the checkpoint in case it doesn't exist
    save_dir = os.path.join(args.checkpoint_dir, name)
    try:
        os.mkdir(save_dir)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    # Save model
    model_path = os.path.join(save_dir, name)
    checkpoint = {
        "epoch": epoch,
        "metric": metric,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, model_path + ".pth")

    # Save arguments
    summary_filename = os.path.join(save_dir, name + "_args.txt")
    with open(summary_filename, "w") as summary_file:
        sorted_args = sorted(vars(args))
        for arg in sorted_args:
            arg_str = "{0}: {1}\n".format(arg, getattr(args, arg))
            summary_file.write(arg_str)
