import time
import copy
from tqdm import tqdm

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import utils
import models.net as models
from args import get_arguments
from data.airbus import AirbusShipDataset
from transforms import TargetHasShipTensor


class Trainer:
    def __init__(self, model, args):
        self.args = args
        self.device = torch.device(self.args.device)
        self.model = model.to(self.device)

        # Loss function: binary cross entropy with logits. Expects logits therefore the
        # output layer must return a logits instead of probabilities
        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate
        )

    def make_epoch(self, dataloader, epoch, is_training):
        # Set model to training mode if training; otherwise, set it to evaluation mode
        if is_training:
            self.model.train()
        else:
            self.model.eval()

        running_loss, running_corrects = self.make_step(dataloader, is_training)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)

        return epoch_loss, epoch_acc

    def make_step(self, dataloader, is_training):
        # Initialize running metrics
        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for step, (inputs, labels) in enumerate(tqdm(dataloader)):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward
            # Track history only if training
            with torch.set_grad_enabled(is_training):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Apply the sigmoid function to get the prediction from the logits
                preds = torch.sigmoid(outputs).round_()

                # Backward only if training
                if is_training:
                    loss.backward()
                    self.optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        return running_loss, running_corrects

    def fit(self, train_dataloader, val_dataloader):
        # Get the current time to know how much time it took to train the model
        since = time.time()

        val_acc_history = []
        best_acc = 0.0
        best_model = copy.deepcopy(self.model.state_dict())

        # Save the model (mostly to find out if the saving process succeeds)
        utils.save_checkpoint(self.model, self.optimizer, best_acc, 0, self.args)

        # Start training the model
        for epoch in range(self.args.epochs):
            print("Epoch {}/{}".format(epoch, self.args.epochs - 1))
            print("-" * 80)

            epoch_loss, epoch_acc = self.make_epoch(
                train_dataloader, epoch, is_training=True
            )
            print("Training - Loss: {:.4f} Acc: {:.4f}".format(epoch_loss, epoch_acc))

            epoch_loss, epoch_acc = self.make_epoch(
                val_dataloader, epoch, is_training=False
            )
            print("Validation - Loss: {:.4f} Acc: {:.4f}".format(epoch_loss, epoch_acc))

            # Deep copy the model
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(self.model.state_dict())
                utils.save_checkpoint(
                    self.model, self.optimizer, epoch, best_acc, self.args
                )

            val_acc_history.append(epoch_acc)
            print()

        time_elapsed = time.time() - since
        print(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        print("Best val Acc: {:4f}".format(best_acc))

        # Load the best model weights
        self.model.load_state_dict(best_model)
        return self.model, val_acc_history


# Run only if this module is being run directly
if __name__ == "__main__":
    # Get arguments from the command-line
    args = get_arguments()

    # Initialize ship or no-ship detection network
    print("Loading ship detection model")
    num_classes = 1
    snsnet, input_dim = models.r34_sns_net(num_classes)
    print(snsnet)

    # Compose the image transforms to be applied to the data
    image_transform = transforms.Compose(
        [transforms.Resize(input_dim), transforms.ToTensor()]
    )

    target_transform = transforms.Compose(
        [transforms.Resize(input_dim), TargetHasShipTensor()]
    )

    # Initialize the datasets and dataloaders
    print()
    print("Loading training dataset")
    trainset = AirbusShipDataset(
        args.dataset_dir,
        mode="train",
        transform=image_transform,
        target_transform=target_transform,
    )
    train_loader = data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )
    if args.dataset_info:
        utils.dataloader_info(train_loader)

    print()
    print("Loading validation dataset")
    valset = AirbusShipDataset(
        args.dataset_dir,
        mode="val",
        transform=image_transform,
        target_transform=target_transform,
    )
    val_loader = data.DataLoader(
        valset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )
    if args.dataset_info:
        utils.dataloader_info(val_loader)

    # Train the model
    print()
    train = Trainer(snsnet, args)
    train.fit(train_loader, val_loader)
