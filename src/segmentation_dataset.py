import os
import pandas as pd
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import utils
import transforms as ctf
import models.classifier as classifier
from engine import predict
from args import segmentation_dataset_args
from data.airbus import AirbusShipDataset


if __name__ == "__main__":
    # Get arguments from the command-line and json configuration
    args = segmentation_dataset_args()
    config = utils.load_config(args.config)

    num_classes = 1
    input_dim = (config["img_h"], config["img_w"])
    checkpoint_path = config["model_checkpoint"]

    if checkpoint_path is None:
        raise ValueError("model checkpoint hasn't been specified")
    if not os.path.isfile(checkpoint_path):
        raise ValueError("the model checkpoint doesn't exist")

    # Compose the image transforms to be applied to the data
    image_transform = transforms.Compose(
        [transforms.Resize(input_dim), transforms.ToTensor()]
    )
    target_transform = transforms.Compose(
        [transforms.Resize(input_dim), ctf.TargetHasShipTensor()]
    )

    # Initialize the dataset in training mode without data split
    print("Loading training dataset...")
    dataset = AirbusShipDataset(
        config["dataset_dir"],
        False,
        mode="train",
        transform=image_transform,
        target_transform=target_transform,
        train_val_split=0.0,
    )
    dataloader = data.DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["workers"],
    )
    if config["dataset_info"]:
        utils.dataloader_info(dataloader)

    # Initialize ship or no-ship detection network and then laod the weigths
    print("Loading ship detection model...")
    net = classifier.resnet(config["resnet_size"], num_classes)

    print("Loading model weights from {}...".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    net.load_state_dict(checkpoint["model"])

    print()
    print("Generating predictions...")
    predictions, targets = predict(net, dataloader, config["device"])

    print()
    print("Generating segmentation dataset...")
    true_targets = targets == 1
    false_positives = (predictions == 1) & (targets == 0)

    # Select from the full training set the images that are either true positives or
    # false positives
    csv_path = os.path.join(config["dataset_dir"], dataset.clf_filename)
    df = pd.read_csv(csv_path).set_index("ImageId")
    image_id = df.index.unique()
    df = df.loc[(image_id[true_targets]) | (image_id[false_positives])]

    csv_path = os.path.join(config["dataset_dir"], dataset.seg_filename)
    df.to_csv(csv_path, index=False)
    print("Done! Saved dataset for segmentation in {}".format(csv_path))
