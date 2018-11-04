class Metric(object):
    """Base class for all metrics.

    From: https://github.com/pytorch/tnt/blob/master/torchnet/meter/meter.py

    Arguments:
        name (str): a name for the metric. Default: miou.
    """

    def __init__(self, name):
        self.name = name

    def reset(self):
        pass

    def add(self, predicted, target):
        pass

    def value(self):
        pass


class MetricList(object):
    def __init__(self, metrics):
        # Make sure we get a list even if metrics is some other type of iterable
        self.metrics = [m for m in metrics]

    def reset(self):
        for m in self.metrics:
            m.reset()

    def add(self, predicted, target):
        for m in self.metrics:
            m.add(predicted, target)

    def value(self):
        return [m.value() for m in self.metrics]

    def first(self):
        m = next(iter(self.metrics))
        return (m.name, m)

    def __iter__(self):
        return iter(self.metrics)

    def __str__(self):
        str_list = ["{}: {:.4f}".format(m.name, m.value()) for m in self.metrics]
        return " - ".join(str_list)
