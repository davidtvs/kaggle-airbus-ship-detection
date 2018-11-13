import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as tf
from torch.optim.lr_scheduler import ReduceLROnPlateau
import utils
import metric
import engine
import models
import models.loss as loss
import transforms as ctf
from args import train_args
from data.airbus import AirbusShipDataset
from post_processing import sigmoid_threshold

# Run only if this module is being run directly
if __name__ == "__main__":

    # Get arguments from the command-line and json configuration
    args = train_args()
    config = utils.load_config(args.config)

    # Compose the image transforms to be applied to the data
    input_dim = (config["img_h"], config["img_w"])
    image_transform = tf.Compose([tf.Resize(input_dim), tf.ToTensor()])
    target_transform = tf.Compose(
        [tf.Resize(input_dim), tf.ToTensor(), ctf.Threshold()]
    )

    # Initialize the datasets and dataloaders
    print("Loading training dataset...")
    trainset = AirbusShipDataset(
        config["dataset_dir"],
        True,
        mode="train",
        transform=image_transform,
        target_transform=target_transform,
        train_val_split=config["val_split"],
        data_slice=config["slice_factor"],
        random_state=23,
    )
    train_loader = data.DataLoader(
        trainset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["workers"],
    )
    if config["dataset_info"]:
        utils.dataloader_info(train_loader)

    print("Loading validation dataset...")
    valset = AirbusShipDataset(
        config["dataset_dir"],
        True,
        mode="val",
        transform=image_transform,
        target_transform=target_transform,
        train_val_split=config["val_split"],
        data_slice=config["slice_factor"],
        random_state=23,
    )
    val_loader = data.DataLoader(
        valset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["workers"],
    )
    if config["dataset_info"]:
        utils.dataloader_info(val_loader)

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

    # Loss function
    loss_name = config["loss"].lower()
    if loss_name == "bce":
        criterion = nn.BCEWithLogitsLoss()
    elif loss_name == "bfl":
        criterion = loss.BinaryFocalWithLogitsLoss()
    elif loss_name == "bdl":
        criterion = loss.BinaryDiceWithLogitsLoss()
    elif loss_name == "bce_bdl":
        criterion = loss.BCE_BDWithLogitsLoss()
    elif loss_name == "bce_logbdl":
        criterion = loss.BCE_LogBDWithLogitsLoss()
    else:
        raise ValueError(
            "requested unknown loss {}, expect one of "
            "(bce, bfl, bdl, bce_bdl, bce_logbdl)".format(config["loss"])
        )

    # Optimizer: adam
    optimizer_name = config["optimizer"].lower()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=config["lr_rate"])
    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(net.parameters(), lr=config["lr_rate"])
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            net.parameters(), lr=config["lr_rate"], momentum=0.9
        )
    else:
        raise ValueError(
            "unknown optimizer {}, expect one of (Adam, RMSprop, SGD)".format(
                config["optimizer"]
            )
        )

    # If a model checkpoint has been specified try to load its weights
    start_epoch = 1
    metrics = metric.MetricList(
        [metric.AirbusFScoreApprox(), metric.BinaryIoU(), metric.BinaryDice()]
    )
    if args.model_checkpoint:
        print("Loading weights from {}...".format(args.model_checkpoint))
        checkpoint = torch.load(args.model_checkpoint, map_location=torch.device("cpu"))
        net.load_state_dict(checkpoint["model"])

        # If the --resume flag is specified, training will continue from the checkpoint
        # as if it was never aborted. Otherwise, training will take only the already
        # loaded weights start from scratch
        if args.resume:
            start_epoch = checkpoint["epoch"] + 1
            optimizer.load_state_dict(checkpoint["optimizer"])
            metrics = checkpoint["metrics"]
            print(
                "Resuming training from epoch {}: Metrics - {}".format(
                    start_epoch, metrics
                )
            )

    checkpoint_dir = os.path.join(config["checkpoint_dir"], config["model_name"])
    model_path = os.path.join(checkpoint_dir, config["model_name"] + ".pth")

    # Set up learning rate secheduling, model checkpoints, and early stopping. The mode
    # argument is set to max because the quantity that will be monitored is the first
    # metric in the 'metrics' MetricList
    lr_scheduler = ReduceLROnPlateau(
        optimizer, mode="max", patience=config["lr_patience"], verbose=True
    )
    model_checkpoint = engine.ModelCheckpoint(model_path, mode="max")
    early_stopping = engine.EarlyStopping(mode="max", patience=config["stop_patience"])

    # Train the model
    print()
    train = engine.Trainer(
        net,
        optimizer,
        criterion,
        metrics,
        config["epochs"],
        start_epoch=start_epoch,
        lr_scheduler=lr_scheduler,
        early_stop=early_stopping,
        model_checkpoint=model_checkpoint,
        device=config["device"],
    )
    net, checkpoint = train.fit(train_loader, val_loader, output_fn=sigmoid_threshold)

    # Save a summary file containing the args, losses, and metrics
    config_path = os.path.join(checkpoint_dir, os.path.basename(args.config))
    utils.save_config(config_path, config)
    summary_path = os.path.join(checkpoint_dir, "summary.json")
    utils.save_summary(
        summary_path, vars(args), config, checkpoint["losses"], checkpoint["metrics"]
    )
