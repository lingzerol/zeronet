import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base.block import *
from ..base.container import *
from ..base.function import *

class UNet(nn.Module):

    def __init__(self, in_channels, out_channels, inner_channels, kernel_size, stride=1,
                 padding=0, factor=2, num_inner_layers=3, dropout=0.5, norm="BatchNorm2d", activation="Tanh",
                 inner_activation="PReLu", padtype="replicate"):
        super(UNet, self).__init__()
        self.networks = []

        inner_channels = int(inner_channels*(factor**num_inner_layers))
        self.networks.append(UNetBlock(inner_channels, int(inner_channels/factor),
                                       kernel_size=kernel_size, stride=stride, padding=padding, dropout=dropout,
                                       innermost=True, norm=norm, activation=inner_activation, inner_activation=inner_activation, padtype=padtype))
        inner_channels = int(inner_channels/factor)
        for _ in range(num_inner_layers-1):
            self.networks.append(UNetBlock(inner_channels, int(inner_channels/factor),
                                           kernel_size=kernel_size, stride=stride, padding=padding, dropout=dropout,
                                           subblock=self.networks[-1], norm=norm, activation=inner_activation, inner_activation=inner_activation, padtype=padtype))
            inner_channels = int(inner_channels/factor)

        self.networks.append(UNetBlock(inner_channels, out_channels,
                                       kernel_size=kernel_size, stride=stride, padding=padding, dropout=dropout,
                                       input_nc=in_channels, outermost=True, subblock=self.networks[-1], norm=None,
                                       activation=activation, inner_activation=inner_activation, padtype=padtype))
        self.add_module("Unet", self.networks[-1])

    def forward(self, x, layer=None):
        if layer is not None and layer < len(self.networks):
            return self.networks[layer](x)
        else:
            return self.networks[-1](x)

    def __len__(self):
        return len(self.networks)


class ConvNet(nn.Module):

    def __init__(self, in_channels, out_channels, inner_channels, kernel_size, stride=1,
                 padding=0, factor=2, num_inner_layers=3, dropout=0.5, norm="BatchNorm2d",
                 activation="Tanh", inner_activation="PReLu"):
        super(ConvNet, self).__init__()

        self.networks = Interative_Sequential()

        self.networks.add_module("input_Conv", Conv2dBlock(
            in_channels, inner_channels, kernel_size, stride, padding, dropout, norm, inner_activation))

        for i in range(num_inner_layers):
            self.networks.add_module("inner_Conv_%d" % (i), Conv2dBlock(
                inner_channels, int(inner_channels*factor),  kernel_size, stride, padding, dropout, norm, inner_activation))
            inner_channels = int(inner_channels*factor)
        self.networks.add_module("output_Conv", Conv2dBlock(
            inner_channels, out_channels, kernel_size, stride, padding, 0, None, activation))

    def forward(self, x, layer=-1, every=False):
        return self.networks(x, layer, every)

    def __len__(self):
        return len(self.networks)


class ConvTransposeNet(nn.Module):

    def __init__(self, in_channels, out_channels, inner_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, factor=2, num_inner_layers=3, dropout=0.5, norm="BatchNorm2d",
                 activation="Tanh", inner_activation="PReLu"):
        super(ConvTransposeNet, self).__init__()

        self.networks = Interative_Sequential()

        self.networks.add_module("input_ConvTranspose", ConvTranspose2dBlock(
            in_channels, inner_channels, kernel_size, stride, padding, output_padding, 0, norm, inner_activation))

        for i in range(num_inner_layers):
            self.networks.add_module("inner_ConvTranspose_%d" % (i), ConvTranspose2dBlock(
                inner_channels, int(inner_channels*factor),  kernel_size, stride, padding, output_padding, dropout, norm, inner_activation))
            inner_channels = int(inner_channels*factor)
        self.networks.add_module("output_ConvTranspose", ConvTranspose2dBlock(
            inner_channels, out_channels, kernel_size, stride, padding, output_padding, dropout, None, activation))

    def forward(self, x, layer=-1, every=False):
        return self.networks(x, layer, every)

    def __len__(self):
        return len(self.networks)


class ResNet(nn.Module):

    def __init__(self, in_channels, out_channels, inner_channels, kernel_size, stride=1,
                 padding=0, output_padding=0,  factor=2, num_layers=2, num_res_blocks=3, dropout=0.5, norm="BatchNorm2d",
                 activation="Tanh", inner_activation="PReLu", padtype="replicate", mode="BottleNeck"):
        super(ResNet, self).__init__()

        self.networks = []
        if num_layers > 0:
            self.networks.add_module("in_Conv2dBlock", Conv2dBlock(in_channels, inner_channels, kernel_size=kernel_size,
                                                                   stride=stride, padding=padding, dropout=dropout, norm=norm, activation=inner_activation))
        elif in_channels != inner_channels:
            self.networks.add_module("in_Conv2dBlock", Conv2dBlock(in_channels, inner_channels, kernel_size=1,
                                                                   stride=1, padding=0, dropout=dropout, norm=norm, activation=inner_activation))

        ngf = inner_channels
        for i in range(num_layers-1):
            self.networks.add_module("Conv2dBlock_%d" % (i), Conv2dBlock(ngf, int(ngf*factor), kernel_size=kernel_size,
                                                                         stride=stride, padding=padding, dropout=dropout, norm=norm, activation=inner_activation))
            ngf = int(ngf*factor)

        for i in range(num_res_blocks):
            self.networks.add_module("ResBlock_%d" % (i), ResnetBlock(ngf, dropout=dropout, norm=norm, activation=inner_activation,
                                                                      inner_activation=inner_activation, padtype=padtype, mode=mode))

        for i in range(num_layers-1):
            self.networks.add_module("ConvTranspose2dBlock_%d" % (i), ConvTranspose2dBlock(ngf, int(ngf/factor), kernel_size=kernel_size,
                                                                                           stride=stride, padding=padding, output_padding=output_padding, dropout=dropout, norm=norm, activation=inner_activation))
            ngf = int(ngf/factor)
        if num_layers > 0:
            self.networks.add_module("out_ConvTranspose2dBlock", ConvTranspose2dBlock(inner_channels, out_channels, kernel_size=kernel_size,
                                                                                      stride=stride, padding=padding, output_padding=output_padding, dropout=0, norm=None, activation=activation))
        elif inner_channels != out_channels:
            self.networks.add_module("out_ConvTranspose2dBlock", ConvTranspose2dBlock(inner_channels, out_channels, kernel_size=1,
                                                                                      stride=1, padding=0, output_padding=0, dropout=0, norm=None, activation=activation))

    def forward(self, x, layer=-1, every=False):
        return self.networks(x, layer, every)

    def __len__(self):
        return len(self.networks)
