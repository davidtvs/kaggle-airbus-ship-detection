import os
import pandas as pd
import numpy as np
from .utils import rle_decode
from torch.utils.data import Dataset
from PIL import Image


class AirbusShipDataset(Dataset):

    # Dataset directories
    train_dir = "train_v2"
    test_dir = "test_v2"

    # Run-length encoded target mask
    rle_filename = "train_ship_segmentations_v2.csv"

    def __init__(self, root_dir, mode="train", transform=None, target_transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform

        if self.mode.lower() in ("train", "val"):
            # Get the list of images from the training set
            data_dir = os.path.join(root_dir, self.train_dir)
            self.data = [
                os.path.join(root_dir, self.train_dir, f) for f in os.listdir(data_dir)
            ]

            # Read CSV with run-length encoding
            target_path = os.path.join(root_dir, self.rle_filename)
            self.target_df = pd.read_csv(target_path).set_index("ImageId")
        elif self.mode.lower() == "test":
            # Get the list of images from the test set
            data_dir = os.path.join(root_dir, self.test_dir)
            self.data = os.listdir(data_dir)

            # The test set ground-truth is not public
            self.target_df = None
        else:
            raise RuntimeError(
                "Unexpected dataset mode. Supported modes are: train, val and test"
            )

    def __getitem__(self, index):
        """Gets a sample and target pair from the dataset.

        Arguments:
            index (int): index of the item in the dataset

        Returns:
            tuple: (image, target) where `image` is a `PIL.Image` and `target` is a
            `numpy.ndarray` mask.

        """

        # Load image from disk
        img = Image.open(self.data[index])

        # Create the target from the run-length encoding
        if self.target_df is not None:
            # Get the RLE code by selecting all rows with the image filename
            rle_code = self.target_df.loc[
                os.path.basename(self.data[index])
            ].values.flatten()
            if len(rle_code) > 1:
                rle = " ".join(rle_code)
            else:
                rle = rle_code[0]

            # Iterate over each row and decode
            target = np.zeros(img.size)
            if pd.notna(rle):
                target += rle_decode(rle, img.size)

            target = np.expand_dims(target, -1)
        else:
            target = None

        # Apply transforms if there are any
        if self.transform:
            img = self.transform(img)

        if self.target_transform and target is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.data)
