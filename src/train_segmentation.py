import os
import torch
import torch.utils.data as data
import torchvision.transforms as tf
from torch.optim.lr_scheduler import ReduceLROnPlateau

import utils
import metric
import engine
import models
import transforms as ctf
from args import train_args
from data.airbus import AirbusShipDataset

# Run only if this module is being run directly
if __name__ == "__main__":

    # Get arguments from the command-line and json configuration
    args = train_args()
    config = utils.load_config(args.config)

    # Compose the image transforms to be applied to the data
    input_dim = (config["img_h"], config["img_w"])
    image_transform = tf.Compose([tf.Resize(input_dim), tf.ToTensor()])
    target_transform = tf.Compose([tf.Resize(input_dim), ctf.ToLongTensor()])

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
    num_classes = 2
    model_str = config["model"].lower()
    print("Loading ship segmentation model ({})...".format(model_str))
    if model_str == "enet":
        net = models.ENet(num_classes)
    elif model_str == "linknet":
        net = models.LinkNet(num_classes)
    else:
        raise ValueError(
            "requested unknown model {}, expect one of (ENet, LinkNet)".format(
                model_str
            )
        )

    # Loss function: Combines the activation function nn.LogSoftmax() and the loss
    # function nn.NLLLoss() into a single class. Therefore, it expects logits instead of
    # probabilities.
    criterion = torch.nn.CrossEntropyLoss()

    def logits_to_pred(logits):
        """Function to transform logits into predictions.

        Arguments:
            logits (torch.Tensor): logits output by the model.

        Returns:
            torch.Tensor: The predictions.
        """
        _, predictions = logits.max(1)
        return predictions

    # Optimizer: adam
    optimizer = torch.optim.Adam(net.parameters(), lr=config["lr_rate"])

    # If a model checkpoint has been specified try to load its weights
    start_epoch = 1
    metrics = metric.MetricList([metric.IoU(num_classes), metric.Accuracy()])
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
    net, checkpoint = train.fit(train_loader, val_loader, output_fn=logits_to_pred)

    # Save a summary file containing the args, losses, and metrics
    config_path = os.path.join(checkpoint_dir, os.path.basename(args.config))
    utils.save_config(config_path, config)
    summary_path = os.path.join(checkpoint_dir, "summary.json")
    utils.save_summary(
        summary_path, vars(args), config, checkpoint["losses"], checkpoint["metrics"]
    )
