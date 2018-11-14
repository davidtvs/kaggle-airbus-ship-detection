import torch
import numpy as np
import cv2
from scipy.ndimage import label
from utils import to_onehot_np


def sigmoid_threshold(tensor, threshold=0.5, high=1, low=0):
    """Applies the sigmoid function to the tensor and thresholds the values

    out_tensor(x) = low if tensor(x) <= threshold
                  = high if tensor(x) > threshold

    Arguments:
        tensor (torch.Tensor): the tensor to threshold.

    Returns:
        torch.Tensor: same shape as the input with values {low, high}.
    """
    high = torch.Tensor([high]).to(tensor.device)
    low = torch.Tensor([low]).to(tensor.device)
    out = torch.sigmoid(tensor)
    return torch.where(out > threshold, high, low)


def resize(img, output_size):
    return cv2.resize(img, output_size, interpolation=cv2.INTER_NEAREST)


def split_ships(img, max_ships=30, on_max_error=False, dtype="uint8"):
    """Takes a mask of ships and splits them into different individual masks.

    Uses a structuring element to define connected blobs (ships in this case),
    scipy.ndimage.label does all the work.
    See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html

    Arguments:
        img (numpy.ndarray): the mask of ships to split with size (H, W).
        max_ships(int, optional): maximum number of ships allowed in a single
            image. If surpassed and on_max_error is True a ValueError is raised; if
            on_max_error is False, the smaller blobs are set to background until the
            number of ships is below this threshold. Default: 30.
        on_max_error (int, optional): if True, raises an error if more than max_ships
            are found in a single image. Default: False.

    Returns:
        numpy.ndarray: the masks of individual ships with size (n, H, W), where n is the
        number of ships. If there are no ships, returns a array of size (1, H, W) filled
        with zeros.

    """
    # The background is also labeled
    max_blobs = max_ships + 1

    # No blobs/ships, return empty mask
    if np.sum(img) == 0:
        return np.expand_dims(img, 0)

    # Labels blobs/ships in the image
    labeled_ships, num_ships = label(img)
    if num_ships > max_blobs:
        if on_max_error:
            raise ValueError(
                "too many ships found {}, expect a maximum of {}".format(
                    num_ships, max_ships
                )
            )
        else:
            # Compute the size of each labeled blob and get the corresponding size so
            # that only max_blobs remain
            blob_sizes = np.bincount(labeled_ships.ravel())
            sorted_blob_idx = np.argsort(blob_sizes)
            too_small = np.zeros_like(blob_sizes, dtype=bool)
            too_small[sorted_blob_idx[:-max_blobs]] = True

            # Labels that are below min_size are set to background, the remaining
            # objects are relabeled
            mask = too_small[labeled_ships]
            labeled_ships[mask] = 0
            labeled_ships, num_ships = label(labeled_ships)

    # For convenience, each ship is isolated in an image. Achieving this is equivalent
    # to converting labeled_ships into its one hot form and then removing the first
    # channel which is the background
    out = to_onehot_np(labeled_ships, num_ships + 1, dtype=dtype)[1:]

    return out


def imfill(img, color=1):
    _, contours, _ = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv2.drawContours(img, [cnt], 0, color, -1)

    return img


def fill_oriented_bbox(img, fill_threshold=None, color=1):
    # For some reason it needs a copy else it raises an error
    _, contours, _ = cv2.findContours(
        img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    out = np.zeros_like(img, dtype=np.uint8)
    for cnt in contours:
        # Compute the oriented bounding box
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        obbox = np.zeros_like(img, dtype=np.uint8)
        cv2.fillPoly(obbox, [box], color)

        if fill_threshold is not None:
            # Fill the contour so we can compare it to the oriented bounding box later
            cnt_fill = np.zeros_like(img, dtype=np.uint8)
            cv2.fillPoly(cnt_fill, [cnt], color)

            # Compare the areas and return the filled bounding box only if the ratio is
            # lower than fill_threshold
            if np.sum(obbox) / np.sum(cnt_fill) < fill_threshold:
                out = np.where(out > 0, out, obbox)
            else:
                out = np.where(out > 0, out, cnt_fill)
        else:
            out = np.where(out > 0, out, obbox)

    return out
