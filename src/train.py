import time
import os
from tqdm import tqdm

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

import utils
import metric
import models.net as models
from args import get_train_args
from data.airbus import AirbusShipDataset
from transforms import TargetHasShipTensor


class Trainer:
    def __init__(self, model, args):
        self.args = args
        self.checkpoint_path = os.path.join(
            args.checkpoint_dir, args.model_name, args.model_name + ".pth"
        )
        self.stop = False
        self.device = torch.device(self.args.device)
        self.model = model.to(self.device)
        self.epoch = 0
        self.num_epochs = args.epochs
        self.metrics = {"acc": metric.Accuracy()}
        self.main_metric_key = "acc"
        self.losses = {"train": 0, "val": 0}

        # Loss function: binary cross entropy with logits. Expects logits therefore the
        # output layer must return a logits instead of probabilities
        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate
        )
        self.lr_scheduler = ReduceLROnPlateau(
            self.optimizer, patience=args.lr_patience, mode="max", verbose=True
        )
        self.early_stopping = utils.EarlyStopping(
            self, patience=args.stop_patience, mode="max"
        )
        self.checkpoint = utils.ModelCheckpoint(self, self.checkpoint_path, mode="max")

    def run_epoch(self, dataloader, is_training):
        # Set model to training mode if training; otherwise, set it to evaluation mode
        if is_training:
            self.model.train()
        else:
            self.model.eval()

        # Initialize running metrics
        running_loss = 0.0
        self._metrics_reset()

        # Iterate over data.
        for step, (inputs, targets) in enumerate(tqdm(dataloader)):
            # Move data to the proper device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Run a single iteration
            step_loss = self.run_step(inputs, targets, is_training)
            running_loss += step_loss

        epoch_loss = running_loss / len(dataloader.dataset)

        if not is_training:
            metric_val = self.metrics[self.main_metric_key].value()
            self.lr_scheduler.step(metric_val)
            self.early_stopping.step(metric_val)
            self.checkpoint.step(metric_val)

        return epoch_loss

    def run_step(self, inputs, targets, is_training):
        # Zero the parameter gradients
        self.optimizer.zero_grad()

        # Forward
        # Track history only if training
        with torch.set_grad_enabled(is_training):
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Apply the sigmoid function to get the prediction from the logits
            preds = torch.sigmoid(outputs).detach().round_()

            # Backward only if training
            if is_training:
                loss.backward()
                self.optimizer.step()

        # Statistics
        loss = loss.item() * inputs.size(0)
        self._metrics_update(preds.squeeze_(), targets.squeeze_())

        return loss

    def fit(self, train_dataloader, val_dataloader):
        # Get the current time to know how much time it took to train the model
        since = time.time()

        # Start training the model
        for self.epoch in range(self.num_epochs):
            print("Epoch {}/{}".format(self.epoch, self.args.epochs - 1))
            print("-" * 80)

            print("Training")
            self.losses["train"] = self.run_epoch(train_dataloader, is_training=True)
            print("Loss - {:.4f}".format(self.losses["train"]))
            print("Metrics - {}".format(self._metrics_str()))
            print()

            print("Validation")
            self.losses["val"] = self.run_epoch(val_dataloader, is_training=False)
            print("Loss - {:.4f}".format(self.losses["val"]))
            print("Metrics - {}".format(self._metrics_str()))
            print()

            # Check if we have to stop early
            if self.stop:
                print("Epoch {}: early stopping".format(self.epoch))
                break

        # Load the best model weights
        checkpoint = torch.load(self.checkpoint_path)
        self.model.load_state_dict(checkpoint["model"])
        self.metrics = checkpoint["metrics"]
        self.losses = checkpoint["losses"]
        print("-" * 80)
        print()
        print("Best validation score in epoch {}".format(self.epoch))
        print("Metrics - {}".format(self._metrics_str()))

        time_elapsed = time.time() - since
        print(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        print()

        return self.model

    def _metrics_str(self):
        if not isinstance(self.metrics, dict):
            raise TypeError("expect type 'dict' for 'metrics'")

        str_list = [
            "{0}: {1:.4f}".format(key, self.metrics[key].value())
            for key in self.metrics
        ]
        return " - ".join(str_list)

    def _metrics_reset(self):
        if not isinstance(self.metrics, dict):
            raise TypeError("expect type 'dict' for 'metrics'")

        for key in self.metrics:
            self.metrics[key].reset()

    def _metrics_update(self, prediction, target):
        if not isinstance(self.metrics, dict):
            raise TypeError("expect type 'dict' for 'metrics'")

        for key in self.metrics:
            self.metrics[key].add(prediction, target)


# Run only if this module is being run directly
if __name__ == "__main__":
    # Get arguments from the command-line
    args = get_train_args()

    num_classes = 1
    input_dim = 224

    # Compose the image transforms to be applied to the data
    image_transform = transforms.Compose(
        [transforms.Resize(input_dim), transforms.ToTensor()]
    )

    target_transform = transforms.Compose(
        [transforms.Resize(input_dim), TargetHasShipTensor()]
    )

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
    snsnet = models.resnet_snsnet(34, num_classes)
    print(snsnet)

    # Train the model
    print()
    train = Trainer(snsnet, args)
    train.fit(train_loader, val_loader)
