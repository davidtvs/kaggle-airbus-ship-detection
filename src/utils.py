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

    # Make a grid with the images
    images = torchvision.utils.make_grid(images).numpy()

    if targets is not None:
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
    images, targets = iter(dataloader).next()
    print("Number of images:", len(dataloader.dataset))
    print("Image size:", images.size())
    print("Targets size:", targets.size())
    imshow_batch(images, targets)
