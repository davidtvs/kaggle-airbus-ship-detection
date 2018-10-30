import torch
import torch.nn as nn
import torchvision.models as models


def r18_sns_net(
    num_classes, feature_extraction=False, use_pretrained=True, dropout_p=0.5
):
    # Initialize ResNet model for ship detection; this model will simply output whether
    # or not a ship is in the image
    model = models.resnet18(pretrained=use_pretrained)

    # If in feature extraction mode, do not compute gradients for existing parameters
    if feature_extraction:
        set_parameter_requires_grad(model, False)

    # Replace the average pooling and fully connected layer (the last two layers) with
    # adaptive pooling and a custom head from adaptive_head()
    in_features = model.fc.in_features
    head = adaptive_head(in_features, num_classes, dropout_p)
    model = nn.Sequential(*list(model.children())[:-2], head)

    return model


def r34_sns_net(
    num_classes, feature_extraction=False, use_pretrained=True, dropout_p=0.5
):
    # Initialize ResNet model for ship detection; this model will simply output whether
    # or not a ship is in the image
    model = models.resnet34(pretrained=use_pretrained)

    # If in feature extraction mode, do not compute gradients for existing parameters
    if feature_extraction:
        set_parameter_requires_grad(model, False)

    # Replace the average pooling and fully connected layer (the last two layers) with
    # adaptive pooling and a custom head from adaptive_head()
    in_features = model.fc.in_features
    head = adaptive_head(in_features, num_classes, dropout_p)
    model = nn.Sequential(*list(model.children())[:-2], head)

    return model


def adaptive_head(in_features, num_classes, dropout_p):
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
    """Concatenates `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."""

    def __init__(self, out_size):
        super().__init__()
        if not (isinstance(out_size, int) or isinstance(out_size, tuple)):
            raise ValueError(
                "expected int or tuple for 'out_size'; got {0}".format(type(out_size))
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
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Keep the batch dimension and flatten the remaining
        return x.view(x.size(0), -1)


def set_parameter_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad
