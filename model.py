import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True
    ):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Model(nn.Module):

    def __init__(self, height: int, width: int, out_channels: int, num_class: int):
        super(Model, self).__init__()

        layers = [ConvBlock(3, out_channels)]
        for i in range(3):
            layers += [ConvBlock(out_channels, out_channels)]
        layers += [nn.AvgPool2d(kernel_size=(height, width), stride=1)]
        layers += [nn.Softmax(num_class)]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
