import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_planes, out_planes, stride):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False
    )


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None, stride=1) -> None:
        super().__init__()
        self.expansion = 4
        self.downsample = downsample
        self.conv1 = conv1x1(in_channels, out_channels, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = conv1x1(out_channels, out_channels * self.expansion, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        if self.downsample is not None:
            identity = self.downsample(identity)

        return self.relu(x + identity)


class ResNet(nn.Module):
    def __init__(
        self, block: ConvBlock, layers: list[int], img_channels: int, n_classes: int
    ) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(img_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layers(block, layers[0], 64, stride=1)
        self.layer2 = self._make_layers(block, layers[1], 128, stride=2)
        self.layer3 = self._make_layers(block, layers[2], 256, stride=2)
        self.layer4 = self._make_layers(block, layers[3], 512, stride=2)

        # avg-pool + fc
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, n_classes)

    def _make_layers(self, block, n_blocks, out_channels, stride):
        downsample = None
        layers = []

        if stride != 1 or self.inplanes != out_channels * 4:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, out_channels * 4, stride),
                nn.BatchNorm2d(out_channels * 4),
            )
        layers.append(block(self.inplanes, out_channels, downsample, stride))
        self.inplanes = out_channels * 4

        for _ in range(1, n_blocks):
            layers.append(block(self.inplanes, out_channels))  #

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def ResNet50(img_channels: int = 3, n_classes: int = 10):
    return ResNet(ConvBlock, [3, 4, 6, 3], img_channels, n_classes)


def ResNet101(img_channels: int = 3, n_classes: int = 10):
    return ResNet(ConvBlock, [3, 4, 23, 3], img_channels, n_classes)


def ResNet152(img_channels: int = 3, n_classes: int = 10):
    return ResNet(ConvBlock, [3, 8, 36, 3], img_channels, n_classes)


if __name__ == "__main__":
    x = torch.randn(1, 3, 224, 224)
    model = ResNet50(img_channels=3, n_classes=10)
    print(model(x).shape)
