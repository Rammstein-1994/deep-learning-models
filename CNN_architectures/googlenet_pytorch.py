import torch
import torch.nn as nn


class GoogLeNet(nn.Module):
    def __init__(self, in_channels: int = 3, n_classes: int = 1000) -> None:
        super(GoogLeNet, self).__init__()

        self.net = nn.Sequential(
            conv_block(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            conv_block(64, 192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # in_channels, out_1x1, r_3x3, out_3x3, r_5x5, out_5x5, out_1x1_pool
            Inception_block(192, 64, 96, 128, 16, 32, 32),
            Inception_block(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Inception_block(480, 192, 96, 208, 16, 48, 64),
            Inception_block(512, 160, 112, 224, 24, 64, 64),
            Inception_block(512, 128, 128, 256, 24, 64, 64),
            Inception_block(512, 112, 144, 288, 32, 64, 64),
            Inception_block(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Inception_block(832, 256, 160, 320, 32, 128, 128),
            Inception_block(832, 384, 192, 384, 48, 128, 128),
            nn.AvgPool2d(kernel_size=7, stride=1),
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(1024, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class Inception_block(nn.Module):
    def __init__(
        self, in_channels, out_1x1, r_3x3, out_3x3, r_5x5, out_5x5, out_1x1_pool
    ) -> None:
        super(Inception_block, self).__init__()

        self.filter_1x1 = conv_block(in_channels, out_1x1, kernel_size=1)
        self.filter_3x3 = nn.Sequential(
            conv_block(in_channels, r_3x3, kernel_size=1),
            conv_block(r_3x3, out_3x3, kernel_size=3, padding=1),
        )
        self.filter_5x5 = nn.Sequential(
            conv_block(in_channels, r_5x5, kernel_size=1),
            conv_block(r_5x5, out_5x5, kernel_size=5, padding=2),
        )
        self.filter_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_1x1_pool, kernel_size=1),
        )

    def forward(self, x):
        return torch.concat(
            [
                self.filter_1x1(x),
                self.filter_3x3(x),
                self.filter_5x5(x),
                self.filter_pool(x),
            ],
            dim=1,
        )


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channles, **kwargs) -> None:
        super(conv_block, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channles, **kwargs),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    x = torch.randn(100, 3, 224, 224)
    model = GoogLeNet(in_channels=3, n_classes=1000)
    print(model(x).shape)
