import os
import time
import errno
from tqdm import tqdm
import torch
import numpy as np


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        metrics,
        num_epochs,
        start_epoch=1,
        lr_scheduler=None,
        early_stop=None,
        model_checkpoint=None,
        device=None,
    ):
        self.criterion = criterion
        self.metrics = metrics
        self.start_epoch = start_epoch
        self.epoch = self.start_epoch
        self.num_epochs = num_epochs
        self.lr_scheduler = lr_scheduler
        self.early_stop = early_stop
        if self.early_stop is not None:
            self.early_stop.set_trainer(self)
        self.model_checkpoint = model_checkpoint
        if self.model_checkpoint is not None:
            self.model_checkpoint.set_trainer(self)

        # If device is None select GPU if available; otherwise, select CPU
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device)

        # If the optimizer is loaded from a checkpoint the states are loaded to the CPU,
        # during training, if the device is the GPU the optimizer will raise an error
        # because it'll expect a CPU tensor. To solve this problem the optimizer state
        # is manually moved to the correct device.
        # See https://github.com/pytorch/pytorch/issues/2830 for more details.
        self.optimizer = optimizer
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)

        self.stop = False
        self.losses = {"train": 0, "val": 0}

    def fit(self, train_dataloader, val_dataloader, output_fn=None):
        # Get the current time to know how much time it took to train the model
        since = time.time()

        # Start training the model
        for self.epoch in range(self.start_epoch, self.num_epochs + 1):
            print("Epoch {}/{}".format(self.epoch, self.num_epochs))
            print("-" * 80)

            print("Training")
            self.losses["train"] = self.run_epoch(
                train_dataloader, is_training=True, output_fn=output_fn
            )
            print("loss: {:.4f} {}".format(self.losses["train"], self.metrics))
            print()

            print("Validation")
            self.losses["val"] = self.run_epoch(
                val_dataloader, is_training=False, output_fn=output_fn
            )
            print("loss: {:.4f} {}".format(self.losses["val"], self.metrics))
            print()

            # Check if we have to stop early
            if self.epoch < self.num_epochs and self.stop:
                print("Epoch {}: early stopping".format(self.epoch))
                break

        # Load the best model weights
        best_checkpoint = self.model_checkpoint.load()
        self.model.load_state_dict(best_checkpoint["model"])
        self.epoch = best_checkpoint["epoch"]
        self.metrics = best_checkpoint["metrics"]
        self.losses = best_checkpoint["losses"]
        print("-" * 80)
        print()
        print("Best validation score in epoch {}".format(self.epoch))
        print("Metrics - {}".format(self.metrics))

        time_elapsed = time.time() - since
        print(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        print()

        return self.model, best_checkpoint

    def run_epoch(self, dataloader, is_training, output_fn=None):
        # Set model to training mode if training; otherwise, set it to evaluation mode
        if is_training:
            self.model.train()
        else:
            self.model.eval()

        # Initialize running metrics
        running_loss = 0.0
        self.metrics.reset()

        # Iterate over data.
        for step, (inputs, targets) in enumerate(tqdm(dataloader)):
            # Move data to the proper device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Run a single iteration
            step_loss = self.run_step(inputs, targets, is_training, output_fn=output_fn)
            running_loss += step_loss

        epoch_loss = running_loss / len(dataloader.dataset)

        if not is_training:
            metric_val = self.metrics.first()[1].value()
            self.lr_scheduler.step(metric_val)
            self.early_stop.step(metric_val)
            self.model_checkpoint.step(metric_val)

        return epoch_loss

    def run_step(self, inputs, targets, is_training, output_fn=None):
        # Zero the parameter gradients
        self.optimizer.zero_grad()

        # Forward
        # Track history only if training
        with torch.set_grad_enabled(is_training):
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward only if training
            if is_training:
                loss.backward()
                self.optimizer.step()

            # Apply the output function to the model output (e.g. to convert from logits
            # to predictions)
            outputs = outputs.detach()
            if output_fn is not None:
                outputs = output_fn(outputs)

        # Statistics
        loss = loss.item() * inputs.size(0)
        self.metrics.add(outputs, targets)

        return loss


def predict(model, dataloader, output_fn=None, device=None):
    pred_list = list(
        predict_yield_batch(model, dataloader, output_fn=output_fn, device=device)
    )
    predictions = np.concatenate(pred_list, axis=0)

    return predictions


def predict_yield_batch(model, dataloader, output_fn=None, device=None):
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model = model.to(device).eval()

    # Get the current time to know how much time it took to make the predictions
    since = time.time()
    for step, (images, _) in enumerate(tqdm(dataloader)):
        images = images.to(device)
        yield predict_batch(model, images, output_fn=output_fn)

    time_elapsed = time.time() - since
    print(
        "Predictions complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )


def predict_batch(model, input, output_fn=None):
    # We don't want to compute gradients, deactivate the autograd engine, this also
    # saves a lot of memory
    with torch.no_grad():
        # Do a froward pass with the images and apply the sigmoid function to get
        # the prediction
        outputs = model(input)
        # Note: Because gradients are not computed there is no need to detach from
        # the graph
        if output_fn is not None:
            outputs = output_fn(outputs)

    return outputs.cpu().numpy()


class EarlyStopping(object):
    """Stop training when a metric has stopped improving.

    Arguments:
        mode (str): One of `min`, `max`. In `min` mode, the trainer is stopped when the
            quantity monitored has stopped decreasing; in `max` mode it will be stopped
            when the quantity monitored has stopped increasing. Default: 'min'.
        patience (int): Number of epochs with no improvement after which the training
            is stopped. For example, if `patience = 2`, the first 2 epochs with no
            improvement are ignored; on the 3rd epoch without improvement the trainer
            is stopped. Default: 20.
        threshold (float): Improvements are only considered as improvements when it
            exceeds the `threshold`. Default: 1e-4.

        """

    def __init__(self, mode="min", patience=20, threshold=1e-4):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown")

        self.trainer = None
        self.mode = mode
        self.patience = patience
        self.num_bad_epochs = 0
        if mode == "min":
            self.best = np.inf
            self.threshold = -threshold
            self.cmp_op = np.less
        else:
            self.best = -np.inf
            self.threshold = threshold
            self.cmp_op = np.greater

    def set_trainer(self, trainer):
        """Sets the trainer class instance.

        Arguments:
            trainer (Trainer): Instance of the Trainer class.

        """
        if not isinstance(trainer, Trainer):
            raise TypeError("inappropriate type for 'trainer'")
        self.trainer = trainer

    def step(self, metric):
        """Stops training if the metric has not improved and exceeded `patience`.

        Arguments:
            metric (metric.Metric): quantity to monitor.

        """
        if self.cmp_op(metric - self.threshold, self.best):
            self.best = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs > self.patience:
                self.trainer.stop = True


class ModelCheckpoint(object):
    """Save the model after epoch.

    Arguments:
        filepath (str): path to the location where the model will be saved
        mode (str): One of `min`, `max`. In `min` mode, the checkpoint is saved when the
            quantity monitored reaches a new minimum; in `max` mode it will be saved
            when the quantity monitored reaches a new maximum. Default: 'min'.
        threshold (float): Improvements are only considered as improvements when it
            exceeds the `threshold`. Default: 1e-4.

    """

    def __init__(self, filepath, mode="min", threshold=1e-4):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown")

        self.filepath = filepath
        self.trainer = None
        self.mode = mode
        if mode == "min":
            self.best = np.inf
            self.threshold = -threshold
            self.cmp_op = np.less
        else:
            self.best = -np.inf
            self.threshold = threshold
            self.cmp_op = np.greater

    def set_trainer(self, trainer):
        """Sets the trainer class instance.

        Arguments:
            trainer (Trainer): Instance of the Trainer class.

        """
        if not isinstance(trainer, Trainer):
            raise TypeError("inappropriate type for 'trainer'")
        self.trainer = trainer

    def step(self, metric):
        """Saves the model if the specified metric improved.

        Arguments:
            metric (metric.Metric): quantity to monitor.

        """
        if self.cmp_op(metric - self.threshold, self.best):
            self.best = metric
            self.save()

    def save(self):
        # Create the directory for the checkpoint in case it doesn't exist
        checkpoint_dir = os.path.dirname(self.filepath)
        try:
            os.makedirs(checkpoint_dir)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass

        # Make sure the model is in training mode to save the state of layers like
        # batch normalization and dropout.
        model = self.trainer.model.train()

        # Save model
        checkpoint = {
            "epoch": self.trainer.epoch,
            "model": model.state_dict(),
            "optimizer": self.trainer.optimizer.state_dict(),
            "losses": self.trainer.losses,
            "metrics": self.trainer.metrics,
        }
        torch.save(checkpoint, self.filepath)

    def load(self):
        return torch.load(self.filepath)
