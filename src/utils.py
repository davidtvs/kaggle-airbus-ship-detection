import os
import errno
import torch
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data.utils import rle_encode


class EarlyStopping(object):
    """Stop training when a metric has stopped improving.

    Arguments:
        trainer (Trainer): Instance of the Trainer class.
        mode (str): One of `min`, `max`. In `min` mode, the trainer is stopped when the
            quantity monitored has stopped decreasing; in `max` mode it will be stopped
            when the quantity monitored has stopped increasing. Default: 'min'.
        patience (int): Number of epochs with no improvement after which the training
            is stopped. For example, if `patience = 2`, the first 2 epochs with no
            improvement are ignored; on the 3rd epoch without improvement the trainer
            is stopped. Default: 20.
        threshold (float): Improvements are only considered as improvements when it
            exceeds the `threshold`. Default: 1e-4.

        """

    def __init__(self, trainer, mode="min", patience=20, threshold=1e-4):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")

        self.trainer = trainer
        self.mode = mode
        self.patience = patience
        self.num_bad_epochs = 0
        if mode == "min":
            self.best = np.inf
            self.threshold = -threshold
            self.cmp_op = np.less
        else:
            self.best = -np.inf
            self.threshold = threshold
            self.cmp_op = np.greater

    def step(self, metric):
        """Stops training if the metric has not improved and exceeded `patience`.

        Arguments:
            metric (metric.Metric): quantity to monitor.

        """
        if self.cmp_op(metric - self.threshold, self.best):
            self.best = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs > self.patience:
                self.trainer.stop = True


class ModelCheckpoint(object):
    """Save the model after epoch.

    Arguments:
        trainer (Trainer): Instance of the Trainer class.
        filepath (str): path to the location where the model will be saved
        mode (str): One of `min`, `max`. In `min` mode, the checkpoint is saved when the
            quantity monitored reaches a new minimum; in `max` mode it will be saved
            when the quantity monitored reaches a new maximum. Default: 'min'.
        threshold (float): Improvements are only considered as improvements when it
            exceeds the `threshold`. Default: 1e-4.

    """

    def __init__(self, trainer, filepath, mode="min", threshold=1e-4):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")

        self.filepath = filepath
        self.trainer = trainer
        self.mode = mode
        if mode == "min":
            self.best = np.inf
            self.threshold = -threshold
            self.cmp_op = np.less
        else:
            self.best = -np.inf
            self.threshold = threshold
            self.cmp_op = np.greater

    def step(self, metric):
        """Saves the model if the specified metric improved.

        Arguments:
            metric (metric.Metric): quantity to monitor.

        """
        if self.cmp_op(metric - self.threshold, self.best):
            self.best = metric
            save_checkpoint(
                self.filepath,
                self.trainer.epoch,
                self.trainer.model,
                self.trainer.optimizer,
                self.trainer.losses,
                self.trainer.metrics,
                self.trainer.args,
            )


def save_checkpoint(filepath, epoch, model, optimizer, losses, metrics, args):
    """Saves the model in a specified directory with a specified name.save

    Arguments:
        epoch (int): the current epoch.
        model (torch.nn.Module): the model to save.
        optimizer (torch.optim): the model optimizer.
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

    # Save model
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "losses": losses,
        "metrics": metrics,
        "args": args,
    }
    torch.save(checkpoint, filepath)

    # Save arguments
    summary_filename = os.path.join(checkpoint_dir, "summary.txt")
    with open(summary_filename, "w") as summary_file:
        # Write arguments
        summary_file.write("Arguments\n")
        sorted_args = sorted(vars(args))
        for arg in sorted_args:
            arg_str = "{0}: {1}\n".format(arg, getattr(args, arg))
            summary_file.write(arg_str)
        summary_file.write("\n")

        # Write losses
        summary_file.write("Losses\n")
        str_list = ["{0}: {1}".format(key, losses[key]) for key in losses]
        str_list = sorted(str_list)
        summary_file.write("\n".join(str_list))
        summary_file.write("\n")

        # Write metrics
        summary_file.write("Metrics\n")
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


def make_submission(save_dir, image_ids, predictions):
    rle = []
    for pred in predictions:
        rle.append(rle_encode(pred))

    submission_df = pd.DataFrame({"ImageId": image_ids, "EncodedPixels": rle})
    submission_path = os.path.join(save_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
