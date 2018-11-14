import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as tf
import utils
import metric
import models
import transforms as ctf
from engine import evaluate
from args import config_args
from data.airbus import AirbusShipDataset
import post_processing as pp


if __name__ == "__main__":
    # Get arguments from the command-line and json configuration
    args = config_args()
    config = utils.load_config(args.config)

    # Compose the image transforms to be applied to the data
    input_dim = (config["img_h"], config["img_w"])
    output_dim = (config["out_h"], config["out_w"])
    model_checkpoint = config["model_checkpoint"]

    if model_checkpoint is None:
        raise ValueError("segmentation model checkpoint hasn't been specified")
    elif not os.path.isfile(model_checkpoint):
        raise ValueError("the segmentation model checkpoint doesn't exist")

    # Compose the image transforms to be applied to the data
    image_transform = tf.Compose([tf.Resize(input_dim), tf.ToTensor()])
    target_transform = tf.Compose(
        [tf.Resize(input_dim), tf.ToTensor(), ctf.Threshold()]
    )

    # Initialize the datasets and dataloaders
    print("Loading validation dataset...")
    dataset = AirbusShipDataset(
        config["dataset_dir"],
        True,
        mode="val",
        transform=image_transform,
        target_transform=target_transform,
        train_val_split=config["val_split"],
        data_slice=config["slice_factor"],
        random_state=23,
        remove_overlap=config["remove_overlap"],
    )
    dataloader = data.DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["workers"],
    )
    if config["dataset_info"]:
        utils.dataloader_info(dataloader)

    # Initialize ship segmentation network
    num_classes = 1
    model_str = config["model"].lower()
    print("Loading ship segmentation model ({})...".format(model_str))
    if model_str == "enet":
        net = models.ENet(num_classes)
    elif model_str == "linknet":
        net = models.LinkNet(num_classes)
    elif model_str == "linknet34":
        net = models.LinkNet(num_classes, 34)
    elif model_str == "dilatedunet":
        net = models.DilatedUNet(classes=num_classes)
    else:
        raise ValueError(
            "requested unknown model {}, expect one of "
            "(ENet, LinkNet, LinkNet34, DilatedUNet)".format(config["model"])
        )
    print(net)

    print("Loading model weights from {}...".format(model_checkpoint))
    checkpoint = torch.load(model_checkpoint, map_location=torch.device("cpu"))
    net.load_state_dict(checkpoint["model"])

    # Metrics to evaluate
    metrics = metric.MetricList(
        [metric.AirbusFScoreApprox(), metric.BinaryIoU(), metric.BinaryDice()]
    )

    def logits_to_pred(logits):
        # Convert from logits to prediction using sigmoid and thresholding at 0.5
        pred = pp.sigmoid_threshold(logits)

        # Convert to numpy and apply the opencv post processing functions
        if config["oriented_bbox"] or config["imfill"]:
            if pred.dim() == 4:
                pred = pred.squeeze(1)
            pred_np = pred.cpu().numpy().astype("uint8")

            # Iterate over each sample and apply the post processing operations
            pp_list = []
            for p in pred_np:
                if p.shape != output_dim:
                    p = pp.resize(p, output_dim)
                if config["imfill"]:
                    p = pp.imfill(p)
                if config["oriented_bbox"]:
                    p = pp.fill_oriented_bbox(p, config["oriented_bbox_th"])
                pp_list.append(p)

            # Convert the post-processed list of predictions in tensor predictions with
            # the same shape as the logits
            pp_pred = np.array(pp_list, dtype="float32")
            out = torch.from_numpy(pp_pred)
            if logits.dim() == 4:
                out = out.unsqueeze(1)
        else:
            out = pred

        return out

    # Get the requested device and move the model to that device in evaluation mode then
    # make predictions batch by batch
    metrics = evaluate(
        net, dataloader, metrics, output_fn=logits_to_pred, device=config["device"]
    )
    print(metrics)
