import time
import copy

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import utils
from data.airbus import AirbusShipDataset

root_dir = "/media/davidtvs/Storage/Datasets/airbus-ship-detection"


def train_model(model, dataloaders, criterion, optimizer, num_epochs, device):
    """Trains a model on the training set and evaluates it on the validation set.

    The model is trained on the training dataset (`dataloaders['train']`), minimizing
    `criterion` using `optimizer` as the optimization algorithm. In each epoch, after
    training, the model is evaluated on the validation set (`dataloaders['val']`).

    Arguments:
        model (torch.nn.Module): the model instance to train.
        dataloaders (torch.utils.data.Dataloader): Provides single or multi-process
            iterators over the dataset.
        criterion (torch.nn): The loss criterion.
        optimizer (torch.optim): The optimization algorithm.
        num_epochs (int): The number of training epochs.
        device (torch.device): Object representing the device on which to perform the
            training.

    """
    since = time.time()

    val_acc_history = []

    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model.state_dict())
            if phase == "val":
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model)
    return model, val_acc_history


# Run only if this module is being run directly
if __name__ == "__main__":
    trainset = AirbusShipDataset(
        root_dir,
        mode="train",
        transform=transforms.ToTensor(),
        target_transform=transforms.ToTensor(),
    )
    train_loader = data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=4)
    print(len(trainset))
    images, targets = iter(train_loader).next()
    utils.imshow_batch(images, targets)

    valset = AirbusShipDataset(
        root_dir,
        mode="val",
        transform=transforms.ToTensor(),
        target_transform=transforms.ToTensor(),
    )
    val_loader = data.DataLoader(valset, batch_size=4, shuffle=True, num_workers=4)
    print(len(valset))
    images, targets = iter(val_loader).next()
    utils.imshow_batch(images, targets)

    testset = AirbusShipDataset(
        root_dir,
        mode="test",
        transform=transforms.ToTensor(),
        target_transform=transforms.ToTensor(),
    )
    test_loader = data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=4)
    print(len(testset))
    images, targets = iter(test_loader).next()
    utils.imshow_batch(images)
