import torch.nn as nn
import torchvision.models as models


def r18_sns_net(num_classes, feature_extraction=False, use_pretrained=True):
    # Initialize ResNet model for ship detection; this model will simply output whether
    # or not a ship is in the image
    model = models.resnet18(pretrained=use_pretrained)

    # If in feature extraction mode, do not compute gradients for existing parameters
    if feature_extraction:
        set_parameter_requires_grad(model, False)

    # Replace the fully connected layer with one that has a single output
    fc_infeatures = model.fc.in_features
    model.fc = nn.Linear(fc_infeatures, 1)
    input_dim = (224, 224)

    return model, input_dim


def r34_sns_net(num_classes, feature_extraction=False, use_pretrained=True):
    # Initialize ResNet model for ship detection; this model will simply output whether
    # or not a ship is in the image
    model = models.resnet34(pretrained=use_pretrained)

    # If in feature extraction mode, do not compute gradients for existing parameters
    if feature_extraction:
        set_parameter_requires_grad(model, False)

    # Replace the fully connected layer with one that has a single output
    fc_infeatures = model.fc.in_features
    model.fc = nn.Linear(fc_infeatures, 1)
    input_dim = (224, 224)

    return model, input_dim


def set_parameter_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad
