import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from .utils import rle_decode


class AirbusShipDataset(Dataset):
    # Corrupted train images to discard
    train_corrupted = ["6384c3e78.jpg"]

    # data_pathset directories
    train_dir = "train_v2"
    test_dir = "test_v2"

    # Run-length encoded CSV files
    clf_filename = "train_ship_segmentations_v2.csv"
    seg_filename = "train_ship_segmentations_seg.csv"

    # Overlapping images in training set
    overlap_filename = "duplicates.csv"

    weights_enet = [1.42340848, 47.8291847]
    weights_mfb = [0.5013302, 188.44108246]

    def __init__(
        self,
        root_dir,
        for_segmentation,
        mode="train",
        transform=None,
        target_transform=None,
        train_val_split=0.2,
        data_slice=1.0,
        return_path=False,
        remove_overlap="none",
        random_state=None,
    ):
        self.root_dir = root_dir
        self.mode = mode.lower()
        self.transform = transform
        self.target_transform = target_transform
        self.return_path = return_path
        self.remove_overlap = remove_overlap.lower()
        if self.remove_overlap not in ("train", "val", "none"):
            raise ValueError(
                "expected one of (train, val, none) fore remove_overlap, got {}".format(
                    remove_overlap
                )
            )
        if for_segmentation:
            rle_filename = self.seg_filename
        else:
            rle_filename = self.clf_filename

        if self.mode in ("train", "val"):
            # Read CSV with run-length encoding
            rle_path = os.path.join(root_dir, rle_filename)
            rle_df = pd.read_csv(rle_path).set_index("ImageId")

            # Remove the corrupted images
            rle_df.drop(index=self.train_corrupted, inplace=True, errors="ignore")

            # Get list of images in the training set. Unique gets rid of the duplicated
            # entries of images with more than one ship
            data_names = rle_df.index.unique().tolist()
            if data_slice:
                data_names = self._slice(data_names, data_slice)

            # Split the data_pathset into training and validation
            data_names, val_names = train_test_split(
                data_names,
                test_size=train_val_split,
                shuffle=False,
                random_state=random_state,
            )

            # Remove overlapping images
            (data_names, val_names) = self._remove_overlap(data_names, val_names)

            train_target_df = rle_df.loc[data_names]
            val_target_df = rle_df.loc[val_names]

            data_dir = os.path.join(root_dir, self.train_dir)

            if self.mode == "train":
                # Training images and targets
                self.data_path = [os.path.join(data_dir, f) for f in data_names]
                self.target_df = train_target_df
            else:
                # Validation images and targets
                self.data_path = [os.path.join(data_dir, f) for f in val_names]
                self.target_df = val_target_df

        elif self.mode == "test":
            # Get the list of images from the test set
            data_dir = os.path.join(root_dir, self.test_dir)
            data_names = sorted(os.listdir(data_dir))
            data_names = self._slice(data_names, data_slice)
            self.data_path = [os.path.join(data_dir, f) for f in data_names]

            # The test set ground-truth is not public
            self.target_df = None
        else:
            raise RuntimeError(
                "Unexpected data_pathset mode. Supported modes are: train, val and test"
            )

    def __getitem__(self, index):
        """Gets a sample and target pair from the data_pathset.

        Arguments:
            index (int): index of the item in the data_pathset

        Returns:
            tuple: (image, target) if `return_path` is False or
            (image, target, image_path) if `return_path` is True. `image` and `target`
            are `PIL.Image` and `image_path` is a string.

        """
        # Load image from disk
        img_path = self.data_path[index]
        img = Image.open(img_path)

        # Create the target from the run-length encoding
        target = np.zeros(img.size)
        if self.target_df is not None:
            # Get the RLE code by selecting all rows with the image filename
            rle_code = self.target_df.loc[os.path.basename(img_path)].values.flatten()
            if len(rle_code) > 1:
                rle = " ".join(rle_code)
            else:
                rle = rle_code[0]

            # Iterate over each row and decode
            if pd.notna(rle):
                target += rle_decode(rle, img.size)

        target = Image.fromarray(target)

        if self.transform:
            img = self.transform(img)

        if self.target_transform and target is not None:
            target = self.target_transform(target)

        out = (img, target)
        if self.return_path:
            out += (img_path,)

        return out

    def __len__(self):
        """Returns the length of the data_pathset."""
        return len(self.data_path)

    def _slice(self, x, slice_size):
        """Returns a slice of the specified list."""
        if isinstance(slice_size, int):
            pass
        elif isinstance(slice_size, float):
            slice_size = int(len(x) * slice_size)
        else:
            raise TypeError("slice_size must be an int or a float")

        return x[:slice_size]

    def _remove_overlap(self, train, val):
        """Remove overlapping images from a list of image filenames.

        Source: https://www.kaggle.com/iafoss/list-of-overlapping-images-for-validation-set/notebook

        Arguments:
            train (string list): list of image names in the training set.
            val (string list): list of image names in the validation set.

        Returns:
            (string list, string list): tuple of (train, val) datasets without
            overlapped images.
        """
        if self.remove_overlap == "train":
            search_names = val
            remove_names = train
        elif self.remove_overlap == "val":
            search_names = train
            remove_names = val
        else:
            return (train, val)

        overlap_path = os.path.join(self.root_dir, self.overlap_filename)
        overlap_df = pd.read_csv(overlap_path).set_index("ImageId")

        overlap_imgs = []
        for name in search_names:
            s = overlap_df.loc[name, "duplicates"]
            if not isinstance(s, float):
                for idx in s.split():
                    overlap_imgs.append(idx)
        overlap_imgs = set(overlap_imgs)
        remove_names = [name for name in remove_names if name not in overlap_imgs]

        if self.remove_overlap == "train":
            out = (remove_names, search_names)
        else:
            out = (search_names, remove_names)

        return out
