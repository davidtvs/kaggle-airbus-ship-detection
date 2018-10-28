import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from .utils import rle_decode


class AirbusShipDataset(Dataset):

    # Dataset directories
    train_dir = "train_v2"
    test_dir = "test_v2"

    # Run-length encoded target mask
    rle_filename = "train_ship_segmentations_v2.csv"

    def __init__(
        self,
        root_dir,
        mode="train",
        transform=None,
        target_transform=None,
        random_state=None,
        val_split_size=0.2,
    ):
        self.root_dir = root_dir
        self.mode = mode.lower()
        self.transform = transform
        self.target_transform = target_transform

        if self.mode in ("train", "val"):
            # Get the list of images from the training set
            data_dir = os.path.join(root_dir, self.train_dir)
            train_names = os.listdir(data_dir)

            # Read CSV with run-length encoding
            rle_path = os.path.join(root_dir, self.rle_filename)
            rle_df = pd.read_csv(rle_path).set_index("ImageId")

            # Split the dataset into training and validation
            # (shuffle must be false otherwise the order will not match the order of the
            # data frame)
            train_names, val_names = train_test_split(
                train_names,
                test_size=val_split_size,
                shuffle=False,
                random_state=random_state,
            )
            train_target_df = rle_df.loc[train_names]
            val_target_df = rle_df.loc[val_names]

            if self.mode == "train":
                self.data = [os.path.join(data_dir, f) for f in train_names]
                self.target_df = train_target_df
            else:
                self.data = [
                    os.path.join(root_dir, self.train_dir, f) for f in val_names
                ]
                self.target_df = val_target_df

        elif self.mode == "test":
            # Get the list of images from the test set
            data_dir = os.path.join(root_dir, self.test_dir)
            self.data = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]

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
        target = np.zeros(img.size)
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
            if pd.notna(rle):
                target += rle_decode(rle, img.size)

        target = np.expand_dims(target, -1)

        # Apply transforms if there are any
        if self.transform:
            img = self.transform(img)

        if self.target_transform and target is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.data)
