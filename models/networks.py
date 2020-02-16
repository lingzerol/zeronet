import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import *
from .factory import NetworkFactory


class UNet(nn.Module):

    def __init__(self, in_channels, out_channels, inner_channels, kernel_size, stride=1,
                 padding=0, factor=2, num_inner_layers=3, dropout=0.5, norm="BatchNorm2d", activation="Tanh",
                 inner_activation="PReLu", padtype="replicate"):
        super(UNet, self).__init__()
        self.network = None

        inner_channels = inner_channels*(factor**num_inner_layers)
        self.network = UNetBlock(inner_channels, inner_channels//factor,
                                 kernel_size=kernel_size, stride=stride, padding=padding, dropout=dropout,
                                 innermost=True, norm=norm, activation=inner_activation, inner_activation=inner_activation, padtype=padtype)
        inner_channels //= factor
        for _ in range(num_inner_layers-1):
            self.network = UNetBlock(inner_channels, inner_channels//factor,
                                     kernel_size=kernel_size, stride=stride, padding=padding, dropout=dropout,
                                     subblock=self.network, norm=norm, activation=inner_activation, inner_activation=inner_activation, padtype=padtype)
            inner_channels //= factor

        self.network = UNetBlock(inner_channels, out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding, dropout=dropout,
                                 input_nc=in_channels, outermost=True, subblock=self.network, norm=None,
                                 activation=activation, inner_activation=inner_activation, padtype=padtype)

    def forward(self, x):
        return self.network(x)


class ConvNet(nn.Module):

    def __init__(self, in_channels, out_channels, inner_channels, kernel_size, stride=1,
                 padding=0, factor=2, num_inner_layers=3, dropout=0.5, norm="BatchNorm2d",
                 activation="Tanh", inner_activation="PReLu"):
        super(ConvNet, self).__init__()

        self.network = nn.Sequential()

        self.network.add_module("input_Conv", Conv2dBlock(
            in_channels, inner_channels, kernel_size, stride, padding, dropout, norm, inner_activation))

        for i in range(num_inner_layers):
            self.network.add_module("inner_Conv_%d" % (i), Conv2dBlock(
                inner_channels, inner_channels*factor,  kernel_size, stride, padding, dropout, norm, inner_activation))
            inner_channels *= factor
        self.network.add_module("output_Conv", Conv2dBlock(
            inner_channels, out_channels, kernel_size, stride, padding, 0, None, activation))

    def forward(self, x):
        return self.network(x)


class ResNet(nn.Module):

    def __init__(self, in_channels, out_channels, inner_channels, kernel_size, stride=1,
                 padding=0, factor=2, num_layers=2, num_res_blocks=3, dropout=0.5, norm="BatchNorm2d",
                 activation="Tanh", inner_activation="PReLu", padtype="replicate"):
        super(ResNet, self).__init__()

        self.network = nn.Sequential()
        self.network.add_module("in_Conv2dBlock", Conv2dBlock(in_channels, inner_channels, kernel_size=kernel_size,
                                                              stride=stride, padding=padding, dropout=dropout, norm=norm, activation=inner_activation))

        ngf = inner_channels
        for i in range(num_layers-1):
            self.network.add_module("Conv2dBlock_%d" % (i), Conv2dBlock(ngf, ngf*factor, kernel_size=kernel_size,
                                                                        stride=stride, padding=padding, dropout=dropout, norm=norm, activation=inner_activation))
            ngf *= factor

        for i in range(num_res_blocks):
            self.network.add_module("ResBlock_%d" % (i), ResnetBlock(ngf, norm=norm, activation=inner_activation,
                                                                     inner_activation=inner_activation, padtype=padtype))

        for i in range(num_layers-1):
            self.network.add_module("ConvTranspose2dBlock_%d" % (i), ConvTranspose2dBlock(ngf, ngf//factor, kernel_size=kernel_size,
                                                                                          stride=stride, padding=padding, dropout=dropout, norm=norm, activation=inner_activation))
            ngf //= factor
        self.network.add_module("out_ConvTranspose2dBlock", ConvTranspose2dBlock(inner_channels, out_channels, kernel_size=kernel_size,
                                                                                 stride=stride, padding=padding, dropout=0, norm=None, activation=activation))

    def forward(self, x):
        return self.network(x)


class ConvNetworkFactory(NetworkFactory):

    def __init__(self):
        self.exists_model_name = ["ConvNet", "UNet", "ResNet",
                                  "Conv2dBlock", "ConvTanspose2dBlock", "ResnetBlock"]

    def define_model(self, param, in_channels, out_channels):
        result = []
        for key in param.keys():
            module_type = param[key]["type"]
            if module_type in self.exists_model_name:

                id = param[key]["id"]

                factor = param[key]["factor"] if "factor" in param[key] else 2
                sub_in_channels = param[key]["in_channels"] if "in_channels" in param[key] else in_channels
                sub_out_channels = param[key]["out_channels"] if "out_channels" in param[key] else out_channels
                inner_channels = param[key]["inner_channels"] if "inner_channels" in param[key] else 128
                num_inner_layers = param[key]["num_inner_layers"] if "num_inner_layers" in param[key] else 3
                num_layers = param[key]["num_layers"] if "num_layers" in param[key] else 2
                num_res_blocks = param[key]["num_res_blocks"] if "num_res_blocks" in param[key] else 3
                kernel_size = param[key]["kernel_size"] if "kernel_size" in param[key] else 3
                stride = param[key]["stride"] if "stride" in param[key] else 1
                padding = param[key]["padding"] if "padding" in param[key] else 0
                output_padding = param[key]["output_padding"] if "output_padding" in param[key] else 0
                norm = param[key]["norm"] if "norm" in param[key] else 1
                inner_activation = param[key]["inner_activation"] if "inner_activation" in param[key] else "BatchNorm2d"
                activation = param[key]["activation"] if "activation" in param[key] else "PReLu"
                dropout = param[key]["dropout"] if "dropout" in param[key] else 0
                padtype = param[key]["padtype"] if "padtype" in param[key] else "replicate"

                if module_type == "ConvNet":
                    result.append([id, ConvNet(sub_in_channels, sub_out_channels, inner_channels, kernel_size, stride,
                                               padding, factor, num_inner_layers, dropout, norm, activation, inner_activation)])
                elif module_type == "UNet":
                    result.append([id, UNet(sub_in_channels, sub_out_channels, inner_channels, kernel_size, stride,
                                            padding, factor, num_inner_layers, dropout, norm, activation, inner_activation, padtype)])
                elif module_type == "ResNet":
                    result.append([id, ResNet(sub_in_channels, sub_out_channels, inner_channels, kernel_size, stride,
                                              padding, factor, num_layers, num_res_blocks, dropout, norm, activation, inner_activation, padtype)])
                elif module_type == "Conv2dBlock":
                    result.append(
                        [id, Conv2dBlock(sub_in_channels, sub_out_channels, kernel_size, stride, padding, dropout,  norm, activation)])
                elif module_type == "ConvTanspose2dBlock":
                    result.append(
                        [id, ConvTranspose2dBlock(sub_in_channels, sub_out_channels, kernel_size, stride, padding, output_padding, dropout, norm, activation)])
                elif module_type == "ResNetBlock":
                    result.append([id, ResNetBlock(sub_in_channels, sub_out_channels, norm, activation,
                                                   inner_activation, padtype)])
            else:
                raise RuntimeError("module not exists!")
        result.sort(key=lambda x: x[0])
        result = [d[1] for d in result]
        result = nn.Sequential(*result)
        return result

    def define_optimizer(self, param, parameters):
        if param["training_type"] == "Adam":
            lr = param["lr"] if "lr" in param else 0.001
            beta1 = param["beta1"] if "beta1" in param else 0.5
            beta2 = param["beta2"] if "beta2" in param else 0.999
            optimizer = torch.optim.Adam(parameters,
                                         lr=param["lr"], betas=(beta1, beta2))
        elif param["training_type"] == "SGD":
            lr = param["lr"] if "lr" in param else 0.001
            momentum = param["momentum"] if "momentum" in param else 0.5
            optimizer = torch.optim.SGD(
                parameters, lr=lr, momentum=momentum)
        else:
            optimizer = None
        return optimizer

    def define_loss(self, loss_type):
        if loss_type == "BCELoss":
            return nn.BCELoss()
        elif loss_type == "L1Loss":
            return nn.L1Loss()
        elif loss_type == "MSELoss":
            return nn.MSELoss()
        else:
            return None


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
