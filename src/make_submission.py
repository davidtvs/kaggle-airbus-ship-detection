import os
from tqdm import tqdm
import pandas as pd
import torch
import torch.utils.data as data
import torchvision.transforms as tf
import models
import utils
import transforms as ctf
import post_processing as pp
from engine import predict_batch
from args import config_args
from data.airbus import AirbusShipDataset
from data.utils import rle_encode


if __name__ == "__main__":
    # Get arguments from the command-line and json configuration
    args = config_args()
    config = utils.load_config(args.config)

    input_dim = (config["img_h"], config["img_w"])
    output_dim = (config["out_h"], config["out_w"])
    clf_checkpoint = config["clf_checkpoint"]
    seg_checkpoint = config["seg_checkpoint"]

    if clf_checkpoint is None:
        raise ValueError("classifier model checkpoint hasn't been specified")
    elif seg_checkpoint is None:
        raise ValueError("segmentation model checkpoint hasn't been specified")
    elif not os.path.isfile(clf_checkpoint):
        raise ValueError("the classifier model checkpoint doesn't exist")
    elif not os.path.isfile(seg_checkpoint):
        raise ValueError("the segmentation model checkpoint doesn't exist")

    # Compose the image transforms to be applied to the data
    image_transform = tf.Compose([tf.Resize(input_dim), tf.ToTensor()])
    # This is not relevant, test labels don't exist
    target_transform = tf.Compose(
        [tf.Resize(input_dim), tf.ToTensor(), ctf.Threshold()]
    )

    # Initialize the dataset in training mode without data split
    print("Loading training dataset...")
    dataset = AirbusShipDataset(
        config["dataset_dir"],
        False,
        mode="test",
        transform=image_transform,
        target_transform=target_transform,
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
    clf_net = models.resnet(config["clf_resnet"], num_classes)

    print("Loading model weights from {}...".format(clf_checkpoint))
    checkpoint = torch.load(clf_checkpoint, map_location=torch.device("cpu"))
    clf_net.load_state_dict(checkpoint["model"])

    print("Loading segmentation model...")
    model_str = config["seg_model"].lower()
    if model_str == "enet":
        seg_net = models.ENet(num_classes)
    elif model_str == "linknet":
        seg_net = models.LinkNet(num_classes)
    elif model_str == "linknet34":
        seg_net = models.LinkNet(num_classes, 34)
    elif model_str == "dilatedunet":
        seg_net = models.DilatedUNet(classes=num_classes)
    else:
        raise ValueError(
            "requested unknown model {}, expect one of "
            "(ENet, LinkNet, LinkNet34, DilatedUNet)".format(config["model"])
        )

    print("Loading model weights from {}...".format(seg_checkpoint))
    checkpoint = torch.load(seg_checkpoint, map_location=torch.device("cpu"))
    seg_net.load_state_dict(checkpoint["model"])

    print()
    print("Constructing combined network (classifier + segmentation network)...")
    net = models.ComboNet(
        clf_net,
        seg_net,
        clf_output_fn=pp.sigmoid_threshold,
        seg_output_fn=pp.sigmoid_threshold,
    )
    print(net)

    print()
    print("Generating predictions...")
    # Initialize the lists that will store submission data
    image_id_list = []
    encoded_pixels_list = []

    # Get the requested device and move the model to that device in evaluation mode then
    # make predictions batch by batch
    device = torch.device(config["device"])
    net = net.to(device).eval()
    for step, (img_batch, target_batch, path_batch) in enumerate(tqdm(dataloader)):
        img_batch = img_batch.to(device)
        pred_batch = predict_batch(net, img_batch)

        # Show the images and predictions if requested
        if config["show_predictions"]:
            utils.imshow_batch(
                img_batch, torch.from_numpy(pred_batch), pad_value=1, padding=4
            )

        # Iterate over each prediction in the batch and split the segmented ships into
        # individual masks
        for (pred, path) in zip(pred_batch, path_batch):
            image_id = os.path.basename(path)
            pred = pred.squeeze(0).astype("uint8")

            # Post processing
            if pred.shape != output_dim:
                pred = pp.resize(pred, output_dim)
            if config["imfill"]:
                pred = pp.imfill(pred)
            if config["oriented_bbox"]:
                pred = pp.fill_oriented_bbox(pred, config["oriented_bbox_th"])

            # Split ships into masks of isolated ships
            split_pred_masks = pp.split_ships(pred, dtype="uint8")

            # Iterate over the individual masks and encode them in run-length form,
            # finally, append to the submission data frame a new row with the image file
            # name and the encoded pixels
            for ship_mask in split_pred_masks:
                encoded_pixels = rle_encode(ship_mask)
                image_id_list.append(image_id)
                encoded_pixels_list.append(encoded_pixels)

    # Construct the path where the submission will be saved (same directory as config
    # file with name specified by the submission_name setting in the config file).
    csv_filename = config["submission_name"] + ".csv"
    csv_path = os.path.join(os.path.dirname(args.config), csv_filename)

    # Build the submission data frame and save it as a csv file
    df = pd.DataFrame({"ImageId": image_id_list, "EncodedPixels": encoded_pixels_list})
    df.to_csv(csv_path, index=False)
    print()
    print("Done! Saved submission in: {}".format(csv_path))
