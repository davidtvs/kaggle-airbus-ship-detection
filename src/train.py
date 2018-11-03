import os
import torch
import torch.utils.data as data
import torchvision.transforms as tf
from torch.optim.lr_scheduler import ReduceLROnPlateau

import utils
import metric
import transforms as ctf
import models.ship_noship as sns
from models.trainer import Trainer, ModelCheckpoint, EarlyStopping
from args import get_train_args
from data.airbus import AirbusShipDataset

# Run only if this module is being run directly
if __name__ == "__main__":
    # Get arguments from the command-line
    args = get_train_args()

    num_classes = 1
    input_dim = 224

    # Compose the image transforms to be applied to the data
    image_transform = tf.Compose([tf.Resize(input_dim), tf.ToTensor()])

    target_transform = tf.Compose([tf.Resize(input_dim), ctf.TargetHasShipTensor()])

    # Initialize the datasets and dataloaders
    print("Loading training dataset...")
    trainset = AirbusShipDataset(
        args.dataset_dir,
        mode="train",
        transform=image_transform,
        target_transform=target_transform,
        train_val_split=args.val_split,
        data_slice=args.slice_factor,
        random_state=23,
    )
    train_loader = data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )
    if args.dataset_info:
        utils.dataloader_info(train_loader)

    print()
    print("Loading validation dataset...")
    valset = AirbusShipDataset(
        args.dataset_dir,
        mode="val",
        transform=image_transform,
        target_transform=target_transform,
        train_val_split=args.val_split,
        data_slice=args.slice_factor,
        random_state=23,
    )
    val_loader = data.DataLoader(
        valset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )
    if args.dataset_info:
        utils.dataloader_info(val_loader)

    # Initialize ship or no-ship detection network
    print()
    print("Loading ship detection model...")
    snsnet = sns.resnet(34, num_classes)

    # Loss function: binary cross entropy with logits. Expects logits therefore the
    # output layer must return a logits instead of probabilities
    criterion = torch.nn.BCEWithLogitsLoss()

    # Optimizer with learning rate scheduling
    optimizer = torch.optim.Adam(snsnet.parameters(), lr=args.learning_rate)
    lr_scheduler = ReduceLROnPlateau(
        optimizer, mode="max", patience=args.lr_patience, verbose=True
    )

    # Save a checkpoint after an epoch with better score
    checkpoint_path = os.path.join(
        args.checkpoint_dir, args.model_name, args.model_name + ".pth"
    )
    model_checkpoint = ModelCheckpoint(checkpoint_path, mode="max")

    # Metrics: accuracy. The validation accuracy is the quantity monitored in
    # lr_scheduler, early_stopping, and model_checkpoint
    metrics = metric.MetricList([metric.Accuracy()])
    early_stopping = EarlyStopping(mode="max", patience=args.stop_patience)

    # Train the model
    print()
    train = Trainer(
        snsnet,
        optimizer,
        criterion,
        metrics,
        args.epochs,
        lr_scheduler=lr_scheduler,
        early_stop=early_stopping,
        model_checkpoint=model_checkpoint,
        device=args.device,
    )
    train.fit(train_loader, val_loader)
