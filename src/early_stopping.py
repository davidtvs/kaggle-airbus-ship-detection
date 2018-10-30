import numpy as np
from functools import partial


class EarlyStopping(object):
    """Stop training when a metric has stopped improving.

    Arguments:
        trainer (Trainer): Instance of the Trainer class.
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

    def __init__(self, trainer, mode="min", patience=20, threshold=1e-4):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")

        self.trainer = trainer
        self.mode = mode
        self.patience = patience
        self.threshold = threshold
        self.is_better = partial(self._cmp, mode, threshold)
        if mode == "min":
            self.best = np.inf
        else:
            self.best = -np.inf
        self.num_bad_epochs = 0

    def step(self, metrics):
        current = metrics
        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs > self.patience:
                self.trainer.stop = True

    def _cmp(self, mode, threshold, value, target):
        if mode == "min":
            return value < target - threshold
        else:
            return value > target + threshold
