import torch
import torch.nn as nn
import torchvision.models as models


def resnet(
    num_layers,
    num_classes,
    feature_extraction=False,
    use_pretrained=True,
    dropout_p=0.5,
):
    """Builds a ResNet based network with an adaptive head.

    Arguments:
        num_layers (int): number of layers. One of (18, 34, 50, 101, 152).
        num_classes (int): the number of classes that the model will output.
        feature_extraction (boolean): when True, the pretrained layers are frozen and
            gradients can only be computed for new layers; when False, gradients are
            computed for all layers. Default: False.
        use_pretrained (boolean): if True, returns a model pre-trained on ImageNet.
            Default: True.
        dropout_p (float): probability of an element to be zeroed. Default: 0.5.

    Returns:
        The model.
    """
    if num_layers == 18:
        model = models.resnet18(pretrained=use_pretrained)
    elif num_layers == 34:
        model = models.resnet34(pretrained=use_pretrained)
    elif num_layers == 50:
        model = models.resnet50(pretrained=use_pretrained)
    elif num_layers == 101:
        model = models.resnet101(pretrained=use_pretrained)
    elif num_layers == 152:
        model = models.resnet152(pretrained=use_pretrained)
    else:
        raise ValueError(
            "expected one of {0} for 'num_layers'; got {1}".format(
                (18, 34, 50, 101, 152), num_layers
            )
        )

    # If in feature extraction mode, do not compute gradients for existing parameters
    if feature_extraction:
        set_parameter_requires_grad(model, False)

    # Replace the average pooling and fully connected layer (the last two layers) with
    # adaptive pooling and a custom head from adaptive_head()
    in_features = model.fc.in_features
    head = adaptive_head(in_features, num_classes, dropout_p)
    model = nn.Sequential(*list(model.children())[:-2], head)

    return model


def set_parameter_requires_grad(model, requires_grad):
    """Sets the .requires_grad attribute of the parameters in the model.

    Modified from:
    https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

    Arguments:
        model (torch.nn.Module): the neural network container.
        requires_grad (boolean): when False, all parameters are frozen and gradients are
        not computed; when True, gradients are computed for all layers.
    """
    for param in model.parameters():
        param.requires_grad = requires_grad


def adaptive_head(in_features, num_classes, dropout_p):
    """Creates the head of a model that can accept images of any size.

    Modified from fastai:
    https://github.com/fastai/fastai/blob/1ad3caafc123cb35fea8b63fee3b82301310207b/fastai/vision/learner.py

    Arguments:
        in_features (int): number of input features (the same as the number of output
            features of the layer before this module).
        num_classes (int): the number of classes that the model will output.
        dropout_p (float): probability of an element to be zeroed. Default: 0.5.

    Returns:
        torch.nn.Sequential: A sequential model container.
    """
    # AdaptiveConcatPool2d concatenates two volumes each with in_features
    return nn.Sequential(
        AdaptiveConcatPool2d(1),
        Flatten(),
        nn.BatchNorm1d(in_features * 2),
        nn.Dropout(dropout_p),
        nn.Linear(in_features * 2, in_features),
        nn.ReLU(),
        nn.BatchNorm1d(in_features),
        nn.Dropout(dropout_p),
        nn.Linear(in_features, num_classes),
    )


class AdaptiveConcatPool2d(nn.Module):
    """Concatenates `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`.

    Modified from fastai:
    https://github.com/fastai/fastai/blob/14c02c2009af212e5030ff0f777246826ed4f9dc/fastai/layers.py

    Arguments:
        out_size (int or tuple): the target output size of the image of the form H x W.
            Can be a tuple (H, W) or a single H for a square image H x H. H and W can be
            either a int, or None which means the size will be the same as that of the
            input.

    """

    def __init__(self, out_size):
        super().__init__()
        if out_size is not None:
            if not (isinstance(out_size, int) or isinstance(out_size, tuple)):
                raise ValueError(
                    "expected int or tuple for 'out_size'; got {0}".format(
                        type(out_size)
                    )
                )
            if isinstance(out_size, tuple) and len(out_size) != 2:
                raise ValueError(
                    "expected a tuple of length 2; got {}".format(len(out_size))
                )
        self.avg_adapt = nn.AdaptiveAvgPool2d(out_size)
        self.max_adapt = nn.AdaptiveMaxPool2d(out_size)

    def forward(self, x):
        return torch.cat([self.max_adapt(x), self.avg_adapt(x)], 1)


class Flatten(nn.Module):
    """Flattens a layer preserving the batch dimension."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Keep the batch dimension and flatten the remaining
        return x.view(x.size(0), -1)
