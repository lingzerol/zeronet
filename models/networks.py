import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import *
from .factory import *
from .base_factory import *


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

        self.networks = []

        self.networks.append(Conv2dBlock(
            in_channels, inner_channels, kernel_size, stride, padding, dropout, norm, inner_activation))
        self.add_module("input_Conv", self.networks[-1])

        for i in range(num_inner_layers):
            self.networks.append(Conv2dBlock(
                inner_channels, int(inner_channels*factor),  kernel_size, stride, padding, dropout, norm, inner_activation))
            self.add_module("inner_Conv_%d" % (i), self.networks[-1])
            inner_channels = int(inner_channels*factor)
        self.networks.append(Conv2dBlock(
            inner_channels, out_channels, kernel_size, stride, padding, 0, None, activation))
        self.add_module("output_Conv", self.networks[-1])

    def forward(self, x, layer=-1, every=False):
        if layer < 0:
            layer = len(self.networks)+layer+1
        if every:
            result = []
        for i in range(max(1, min(layer, len(self.networks)))):
            x = self.networks[i](x)
            if every:
                result.append(x)
        if every:
            return result
        else:
            return x

    def __len__(self):
        return len(self.networks)


class ConvTransposeNet(nn.Module):

    def __init__(self, in_channels, out_channels, inner_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, factor=2, num_inner_layers=3, dropout=0.5, norm="BatchNorm2d",
                 activation="Tanh", inner_activation="PReLu"):
        super(ConvTransposeNet, self).__init__()

        self.networks = []

        self.networks.append(ConvTranspose2dBlock(
            in_channels, inner_channels, kernel_size, stride, padding, output_padding, 0, norm, inner_activation))
        self.add_module("input_ConvTranspose", self.networks[-1])

        for i in range(num_inner_layers):
            self.networks.append(ConvTranspose2dBlock(
                inner_channels, int(inner_channels*factor),  kernel_size, stride, padding, output_padding, dropout, norm, inner_activation))
            self.add_module("inner_ConvTranspose_%d" % (i), self.networks[-1])
            inner_channels = int(inner_channels*factor)
        self.networks.append(ConvTranspose2dBlock(
            inner_channels, out_channels, kernel_size, stride, padding, output_padding, dropout, None, activation))
        self.add_module("output_ConvTranspose", self.networks[-1])

    def forward(self, x, layer=-1, every=False):
        if layer < 0:
            layer = len(self.networks)+layer+1
        if every:
            result = []
        for i in range(max(1, min(layer, len(self.networks)))):
            x = self.networks[i](x)
            if every:
                result.append(x)
        if every:
            return result
        else:
            return x

    def __len__(self):
        return len(self.networks)


class ResNet(nn.Module):

    def __init__(self, in_channels, out_channels, inner_channels, kernel_size, stride=1,
                 padding=0, output_padding=0,  factor=2, num_layers=2, num_res_blocks=3, dropout=0.5, norm="BatchNorm2d",
                 activation="Tanh", inner_activation="PReLu", padtype="replicate", mode="BottleNeck"):
        super(ResNet, self).__init__()

        self.networks = []
        self.networks.append(Conv2dBlock(in_channels, inner_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dropout=dropout, norm=norm, activation=inner_activation))
        self.add_module("in_Conv2dBlock", self.networks[-1])
        ngf = inner_channels
        for i in range(num_layers-1):
            self.networks.append(Conv2dBlock(ngf, int(ngf*factor), kernel_size=kernel_size,
                                             stride=stride, padding=padding, dropout=dropout, norm=norm, activation=inner_activation))
            self.add_module("Conv2dBlock_%d" % (i), self.networks[-1])
            ngf = int(ngf*factor)

        for i in range(num_res_blocks):
            self.networks.append(ResnetBlock(ngf, norm=norm, activation=inner_activation,
                                             inner_activation=inner_activation, padtype=padtype, mode=mode))
            self.add_module("ResBlock_%d" % (i), self.networks[-1])

        for i in range(num_layers-1):
            self.networks.append(ConvTranspose2dBlock(ngf, int(ngf/factor), kernel_size=kernel_size,
                                                      stride=stride, padding=padding, output_padding=output_padding, dropout=dropout, norm=norm, activation=inner_activation))
            self.add_module("ConvTranspose2dBlock_%d" % (i), self.networks[-1])
            ngf = int(ngf/factor)
        self.networks.append(ConvTranspose2dBlock(inner_channels, out_channels, kernel_size=kernel_size,
                                                  stride=stride, padding=padding, output_padding=output_padding, dropout=0, norm=None, activation=activation))
        self.add_module("out_ConvTranspose2dBlock", self.networks[-1])

    def forward(self, x, layer=-1, every=False):
        if layer < 0:
            layer = len(self.networks)+layer+1
        if every:
            result = []
        for i in range(max(1, min(layer, len(self.networks)))):
            x = self.networks[i](x)
            if every:
                result.append(x)
        if every:
            return result
        else:
            return x

    def __len__(self):
        return len(self.networks)


class ConvNetworkFactory(NetworkFactory):

    def __init__(self):
        super(ConvNetworkFactory, self).__init__()

        self.exists_model_name = ["ConvNet", "ConvTransposeNet", "UNet", "ResNet",
                                  "Conv2dBlock", "ConvTranspose2dBlock", "ResnetBlock", "UNetBlock"]

    def define(self, param, in_channels, out_channels, subblock=None):
        module_type = param["type"]

        if module_type not in self.exists_model_name:
            raise RuntimeError("module not exists!")

        factor = param["factor"] if "factor" in param else 2
        sub_in_channels = param["in_channels"] if "in_channels" in param else in_channels
        sub_out_channels = param["out_channels"] if "out_channels" in param else out_channels
        inner_channels = param["inner_channels"] if "inner_channels" in param else 128
        num_inner_layers = param["num_inner_layers"] if "num_inner_layers" in param else 3
        num_layers = param["num_layers"] if "num_layers" in param else 2
        num_res_blocks = param["num_res_blocks"] if "num_res_blocks" in param else 3
        kernel_size = param["kernel_size"] if "kernel_size" in param else 3
        stride = param["stride"] if "stride" in param else 1
        padding = param["padding"] if "padding" in param else 0
        output_padding = param["output_padding"] if "output_padding" in param else 0
        norm = param["norm"] if "norm" in param else None
        inner_activation = param["inner_activation"] if "inner_activation" in param else None
        activation = param["activation"] if "activation" in param else None
        dropout = param["dropout"] if "dropout" in param else 0
        padtype = param["padtype"] if "padtype" in param else "replicate"
        outermost = param["outermost"] if "outermost" in param else False
        innermost = param["innermost"] if "innermost" in param else False
        mode = param["mode"] if "mode" in param else "BottleNeck"

        if module_type == "ConvNet":
            return ConvNet(sub_in_channels, sub_out_channels, inner_channels, kernel_size, stride,
                           padding, factor, num_inner_layers, dropout, norm, activation, inner_activation)
        elif module_type == "ConvTransposeNet":
            return ConvTransposeNet(sub_in_channels, sub_out_channels, inner_channels, kernel_size, stride,
                                    padding, output_padding, factor, num_inner_layers, dropout, norm, activation, inner_activation)
        elif module_type == "UNet":
            return UNet(sub_in_channels, sub_out_channels, inner_channels, kernel_size, stride,
                        padding, factor, num_inner_layers, dropout, norm, activation, inner_activation, padtype)
        elif module_type == "ResNet":
            return ResNet(sub_in_channels, sub_out_channels, inner_channels, kernel_size, stride,
                          padding, output_padding, factor, num_layers, num_res_blocks, dropout, norm, activation, inner_activation, padtype, mode=mode)
        elif module_type == "Conv2dBlock":
            return Conv2dBlock(sub_in_channels, sub_out_channels, kernel_size, stride, padding, dropout,  norm, activation)
        elif module_type == "ConvTranspose2dBlock":
            return ConvTranspose2dBlock(sub_in_channels, sub_out_channels, kernel_size, stride, padding, output_padding, dropout, norm, activation)
        elif module_type == "ResNetBlock":
            return ResNetBlock(sub_in_channels, sub_out_channels, norm, activation,
                               inner_activation, padtype, mode=mode)
        elif module_type == "UNetBlock":
            return UNetBlock(inner_channels, out_channels, kernel_size, stride=stride,
                             padding=padding, dropout=dropout, input_nc=in_channels, outermost=outermost,
                             innermost=innermost, subblock=subblock, norm=norm, activation=activation,
                             inner_activation=inner_activation, padtype=padtype)


class ConvArchitectFactory(BaseArchitectFactory):

    def __init__(self):
        super(ConvArchitectFactory, self).__init__()
        self.conv_network_factory = ConvNetworkFactory()

    def define_single_network(self, param, in_channels=None, out_channels=None):
        module_type = param["type"]
        if module_type in self.network_factory.exists_model_name:
            return self.network_factory.define(param)
        elif module_type in self.conv_network_factory.exists_model_name:
            return self.conv_network_factory.define(
                param, in_channels, out_channels)
        else:
            raise RuntimeError("module not exists!")

    def define_network(self, param, in_channels=None, out_channels=None):
        result = []
        for key in param.keys():
            id = param[key]["id"]
            module_type = param[key]["type"]
            if module_type in self.network_factory.exists_model_name:
                result.append([id, self.network_factory.define(param[key])])
            elif module_type in self.conv_network_factory.exists_model_name:
                result.append([id, self.conv_network_factory.define(
                    param[key], in_channels, out_channels)])
            else:
                raise RuntimeError("module not exists!")
        result.sort(key=lambda x: x[0])
        result = [r[1] for r in result]
        return nn.Sequential(*result)


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
