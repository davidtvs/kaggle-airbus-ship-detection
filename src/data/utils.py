import numpy as np


def rle_encode(mask):
    """Encodes a mask in run-length format.

    Source: https://www.kaggle.com/paulorzp/run-length-encode-and-decode

    Arguments:
        mask (numpy.ndarray): the mask to encode

    Returns:
        str: The run-length encoded pixels formatted as
            (start length) string

    """
    # Transpose is needed because RLE is numbered from top to bottom, then left to right
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape):
    """Decodes a run-length encoded mask.

    Source: https://www.kaggle.com/paulorzp/run-length-encode-and-decode

    Arguments:
        mask_rle (str): the run-length encoded pixels formatted as (start length)
        shape (tuple): the dimensions of the array to return (height, width)

    Returns:
        numpy.ndarray: the mask with `shape` dimensions

    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1

    # Transpose is needed because RLE is numbered from top to bottom, then left to right
    return mask.reshape(shape).T
