import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, in_channels: int = 3, n_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=11, stride=4),  # b x 96 x 54 x 54
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # b x 96 x 26 x 26
            nn.Conv2d(96, 256, kernel_size=5, padding=2),  # b x 256 x 26 x 26
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # b x 256 x 12 x 12
            nn.Conv2d(256, 384, kernel_size=3, padding=1),  # b x 384 x 12 x 12
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),  # b x 384 x 12 x 12
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # b x 256 x 12 x 12
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # b x 256 x 5 x 5
            nn.Flatten(),
        )

        self.fn_block = nn.Sequential(
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, n_classes),
        )

    def forward(self, x):
        return self.fn_block(self.conv_block(x))


if __name__ == "__main__":
    x = torch.randn(100, 3, 224, 224)
    model = AlexNet(in_channels=3, n_classes=1000)
    print(model(x).shape)
