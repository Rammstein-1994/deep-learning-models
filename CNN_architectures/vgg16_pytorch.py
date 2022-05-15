import torch
import torch.nn as nn

VGG16 = [
    64,
    64,
    "Max_pool",
    128,
    128,
    "Max_pool",
    256,
    256,
    256,
    "Max_pool",
    512,
    512,
    512,
    "Max_pool",
    512,
    512,
    512,
    "Max_pool",
]


class VGG16_Net(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, vgg_architecture) -> None:
        super(VGG16_Net, self).__init__()
        self.in_channels = in_channels

        # Conv. layers
        self.conv_layers = self.create_conv_block(vgg_architecture)

        # fully-connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, n_classes),
        )

    def create_conv_block(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x

                layers.extend(
                    [
                        nn.Conv2d(
                            in_channels,
                            out_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                        nn.ReLU(),
                    ]
                )
                in_channels = x
            else:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(self.conv_layers(x))


if __name__ == "__main__":
    x = torch.randn(1, 3, 224, 224)
    model = VGG16_Net(in_channels=3, n_classes=1000, vgg_architecture=VGG16)
    print(model(x).shape)
