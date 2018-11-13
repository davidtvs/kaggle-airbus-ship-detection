from collections import OrderedDict
import torch.nn as nn
import torchvision.models as models


class LinkNet(nn.Module):
    def __init__(self, num_classes, resnet_size=18, pretrained_encoder=True):
        super().__init__()
        self.num_classes = num_classes

        # The LinkNet encoder is a ResNet18 without the last average pooling layer and
        # the fully connected layer
        if resnet_size == 18:
            resnet = models.resnet18(pretrained=pretrained_encoder)
        elif resnet_size == 34:
            resnet = models.resnet34(pretrained=pretrained_encoder)
        else:
            raise ValueError(
                "expected 18 or 34 for resnet_size, got {}".format(resnet_size)
            )
        encoder_list = list(resnet.named_children())[:-2]
        self.encoder = nn.Sequential(OrderedDict([*encoder_list]))

        # Construct the decoder
        self.layer4_d = DecoderBlock(512, 256, stride=2, padding=1)
        self.layer3_d = DecoderBlock(256, 128, stride=2, padding=1)
        self.layer2_d = DecoderBlock(128, 64, stride=2, padding=1)
        self.layer1_d = DecoderBlock(64, 64, stride=1, padding=1)
        self.tconv1_d = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1)
        self.bn1_d = nn.BatchNorm2d(32)
        self.relu1_d = nn.ReLU()
        self.conv1_d = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2_d = nn.BatchNorm2d(32)
        self.relu2_d = nn.ReLU()
        self.tconv2_d = nn.ConvTranspose2d(32, self.num_classes, 3, stride=2, padding=1)

    def forward(self, x):
        input_x = x

        # Have to access the output of a few layers in the encoder to make the skip
        # connections. For that, iterate over all modules in the encoder, do the
        # forward pass and save the output for the layers that are needed
        skip = {}
        for name, module in self.encoder.named_children():
            x = module(x)
            if name in ("conv1", "maxpool", "layer1", "layer2", "layer3"):
                skip[name] = x

        x = skip["layer3"] + self.layer4_d(x, skip["layer3"].size())
        x = skip["layer2"] + self.layer3_d(x, skip["layer2"].size())
        x = skip["layer1"] + self.layer2_d(x, skip["layer1"].size())
        x = self.layer1_d(x, skip["maxpool"].size())
        x = self.tconv1_d(x, skip["conv1"].size())
        x = self.bn1_d(x)
        x = self.relu1_d(x)
        x = self.conv1_d(x)
        x = self.bn2_d(x)
        x = self.relu2_d(x)

        return self.tconv2_d(x, input_x.size())


class DecoderBlock(nn.Module):
    """Creates a decoder block.

    Decoder block architecture:
    1. Conv2D
    2. BatchNormalization
    3. ReLU
    4. Conv2DTranspose
    5. BatchNormalization
    6. ReLU
    7. Conv2D
    8. BatchNormalization
    9. ReLU

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        padding=0,
        output_padding=0,
        projection_ratio=4,
        bias=False,
    ):
        super().__init__()

        proj_channels = in_channels // projection_ratio
        self.conv1 = nn.Conv2d(in_channels, proj_channels, 1)
        self.bn1 = nn.BatchNorm2d(proj_channels)
        self.relu1 = nn.ReLU()
        self.tconv = nn.ConvTranspose2d(
            proj_channels,
            proj_channels,
            3,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.bn2 = nn.BatchNorm2d(proj_channels)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(proj_channels, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU()

    def forward(self, x, output_size=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.tconv(x, output_size=output_size)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = self.bn3(x)
        return self.relu3(x)
