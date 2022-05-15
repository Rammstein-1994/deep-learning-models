import torch
import torch.nn as nn

######################
# LeNet Architecture #
######################################################################################
# C1: 1x32x32 -> conv(in=1, out=6, k=5x5, s=1x1, p=0) -> (32+2*0-5/1)+1=28 -> 6x28x28
# S2: 6x28x28 -> pool(k=2, s=2) -> 6x14x14
# C3: 6x14x14 -> conv(in=6, out=16, k=5x5, s=1x1, p=0) -> 16x10x10
# S4: 16x10x10 -> pool(k=2, s=2) -> 16x5x5
# C5: 16x5x5 -> conv(in=16, out=120, k=5x5, s=1x1, p=0) -> 120x1x1
# flatten: 120x1x1 -> (batch_size, 120)
# F6: (batch_size, 120) -> Linear(in=120, out=84) -> (batch_size, 84)
# OUTPUT: (batch_size, 84) -> Linear(in=84, out=10)  -> (batch_size, 10)
######################################################################################


class LeNet(nn.Module):
    def __init__(self, in_channels: int = 1, n_classes: int = 10) -> None:
        super(LeNet, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=0),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0),
            nn.Sigmoid(),
            nn.Flatten(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, n_classes),
        )

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    x = torch.randn(64, 1, 32, 32)
    model = LeNet()
    print(model(x).shape)
