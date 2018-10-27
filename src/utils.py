import torchvision
import numpy as np
import matplotlib.pyplot as plt


def imshow_batch(images, labels):
    """Displays two grids of images. The top grid displays ``images``
    and the bottom grid ``labels``

    Keyword arguments:
    - images (``Tensor``): a 4D mini-batch tensor of shape
    (B, C, H, W)
    - labels (``Tensor``): a 4D mini-batch tensor of shape
    (B, C, H, W)

    """

    # Make a grid with the images and labels and convert it to numpy
    images = torchvision.utils.make_grid(images).numpy()
    labels = torchvision.utils.make_grid(labels).numpy()

    fig, axarr = plt.subplots(3, 1)
    plt.tight_layout(h_pad=0.1, w_pad=0.1)

    axarr[0].set_title("Batch of samples")
    axarr[0].axis("off")
    axarr[0].imshow(np.transpose(images, (1, 2, 0)))

    axarr[1].set_title("Batch of targets")
    axarr[1].axis("off")
    axarr[1].imshow(np.transpose(labels, (1, 2, 0)))

    axarr[2].set_title("Targets overlayed with samples")
    axarr[2].axis("off")
    axarr[2].imshow(np.transpose(images, (1, 2, 0)))
    axarr[2].imshow(np.transpose(labels, (1, 2, 0)), alpha=0.5)

    plt.show()
