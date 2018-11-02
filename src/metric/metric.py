from collections import OrderedDict


class Metric(object):
    """Base class for all metrics.

    From: https://github.com/pytorch/tnt/blob/master/torchnet/meter/meter.py
    """

    def reset(self):
        pass

    def add(self, predicted, target):
        pass

    def value(self):
        pass


class MetricContainer(object):
    def __init__(self, metrics):
        if isinstance(metrics, OrderedDict):
            self.metrics = metrics
        else:
            raise TypeError(
                "expect 'OrderedDict' for 'metrics' parameter; got '{}'".format(
                    type(metrics)
                )
            )

    def reset(self):
        for key in self.metrics:
            self.metrics[key].reset()

    def add(self, predicted, target):
        for key in self.metrics:
            self.metrics[key].add(predicted, target)

    def value(self):
        return [self.metrics[key].value() for key in self.metrics]

    def first(self):
        key = next(iter(self.metrics))
        return (key, self.metrics[key])

    def __iter__(self):
        return iter(self.metrics)

    def __str__(self):
        str_list = [
            "{}: {:.4f}".format(key, self.metrics[key].value()) for key in self.metrics
        ]
        return " - ".join(str_list)
