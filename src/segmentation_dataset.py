import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as tf
import utils
import transforms as ctf
import models.classifier as classifier
from post_processing import sigmoid_threshold
from engine import predict_batch
from args import config_args
from data.airbus import AirbusShipDataset


if __name__ == "__main__":
    # For some reason this script hits the limit of file descriptors and an error like
    # "RuntimeError: received 0 items of ancdata" is raised. Uncomment the following
    # line to solve the problem
    # torch.multiprocessing.set_sharing_strategy("file_system")

    # Get arguments from the command-line and json configuration
    args = config_args()
    config = utils.load_config(args.config)

    input_dim = (config["img_h"], config["img_w"])
    checkpoint_path = config["model_checkpoint"]

    if checkpoint_path is None:
        raise ValueError("model checkpoint hasn't been specified")
    if not os.path.isfile(checkpoint_path):
        raise ValueError("the model checkpoint doesn't exist")

    # Compose the image transforms to be applied to the data
    image_transform = tf.Compose([tf.Resize(input_dim), tf.ToTensor()])
    target_transform = tf.Compose([tf.Resize(input_dim), ctf.TargetHasShipTensor()])

    # Initialize the dataset in training mode without data split
    print("Loading training dataset...")
    dataset = AirbusShipDataset(
        config["dataset_dir"],
        False,
        mode="train",
        transform=image_transform,
        target_transform=target_transform,
        train_val_split=0.0,
        data_slice=1.0,
        return_path=True,
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
    num_classes = 1
    net = classifier.resnet(config["resnet_size"], num_classes)
    print(net)

    print("Loading model weights from {}...".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    net.load_state_dict(checkpoint["model"])

    print()
    print("Generating predictions...")
    # Get the requested device and move the model to that device in evaluation mode then
    # make predictions batch by batch
    device = torch.device(config["device"])
    net = net.to(device).eval()
    target_list = []
    pred_list = []
    path_list = []
    for step, (img_batch, target_batch, path_batch) in enumerate(tqdm(dataloader)):
        img_batch = img_batch.to(device)
        pred_batch = predict_batch(net, img_batch, sigmoid_threshold)
        pred_list.extend(pred_batch.squeeze(1))
        target_list.extend(target_batch)
        path_list.extend([os.path.basename(fp) for fp in path_batch])

    # Convert to numpy arrays to enable indexing
    predictions = np.array(pred_list)
    targets = np.array(target_list)
    paths = np.array(path_list)

    print()
    print("Generating training dataset for segmentation...")
    true_targets = targets == 1
    true_positives = (predictions == 1) & (targets == 1)
    false_positives = (predictions == 1) & (targets == 0)
    false_negatives = (predictions == 0) & (targets == 1)
    print("True positives: {}".format(np.sum(true_positives)))
    print("False positives: {}".format(np.sum(false_positives)))
    print("False negatives: {}".format(np.sum(false_negatives)))

    # Select from the full training set the images that have ships or are false
    # positives
    csv_path = os.path.join(config["dataset_dir"], dataset.clf_filename)
    df = pd.read_csv(csv_path).set_index("ImageId")
    df = df.loc[paths[true_targets | false_positives]]

    csv_path = os.path.join(os.path.dirname(args.config), dataset.seg_filename)
    df.to_csv(csv_path)
    print("Done! Saved training dataset for segmentation in: {}".format(csv_path))
