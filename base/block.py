import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvTranspose2dBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dropout=0, norm="BatchNorm2d", activation="PReLu", negative_slope=0.0, inplace=True, bias=True):
        super(ConvTranspose2dBlock, self).__init__()
        self.network = nn.Sequential()
        if dropout:
            self.network.add_module("Dropout", nn.Dropout(dropout))

        self.network.add_module("ConvTranspose2d", nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias))

        if norm == "BatchNorm2d":
            self.network.add_module("Norm_layer", nn.BatchNorm2d(out_channels))
        elif norm == "InstanceNorm2d":
            self.network.add_module(
                "Norm_layer", nn.InstanceNorm2d(out_channels))

        if activation == "PReLu":
            self.network.add_module(
                "Activation_layer", nn.PReLU(num_parameters=out_channels))
        elif activation == "ReLu":
            self.network.add_module(
                "Activation_layer", nn.ReLU(inplace=inplace))
        elif activation == "LeakyReLu":
            self.network.add_module(
                "Activation_layer", nn.LeakyReLU(negative_slope, inplace=inplace))
        elif activation == "Sigmoid":
            self.network.add_module(
                "Activation_layer", nn.Sigmoid())
        elif activation == "Tanh":
            self.network.add_module(
                "Activation_layer", nn.Tanh())

    def forward(self, x):
        return self.network(x)


class Conv2dBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dropout=0, norm="BatchNorm2d", activation="PReLu", negative_slope=0.0, inplace=True, bias=True):
        super(Conv2dBlock, self).__init__()
        self.network = nn.Sequential()

        self.network.add_module("Conv2d", nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias))

        if norm == "BatchNorm2d":
            self.network.add_module("Norm_layer", nn.BatchNorm2d(out_channels))
        elif norm == "InstanceNorm2d":
            self.network.add_module(
                "Norm_layer", nn.InstanceNorm2d(out_channels))

        if activation == "PReLu":
            self.network.add_module(
                "Activation_layer", nn.PReLU(num_parameters=out_channels))
        elif activation == "ReLu":
            self.network.add_module(
                "Activation_layer", nn.ReLU(inplace=inplace))
        elif activation == "LeakyReLu":
            self.network.add_module(
                "Activation_layer", nn.LeakyReLU(negative_slope, inplace=inplace))
        elif activation == "Sigmoid":
            self.network.add_module(
                "Activation_layer", nn.Sigmoid())
        elif activation == "Tanh":
            self.network.add_module(
                "Activation_layer", nn.Tanh())

        if dropout:
            self.network.add_module("Dropout", nn.Dropout(dropout))

    def forward(self, x):
        return self.network(x)


class UNetBlock(nn.Module):

    def __init__(self, inner_nc, outer_nc, kernel_size, stride=1,
                 padding=0, dropout=0.5, input_nc=None, outermost=False,
                 innermost=False, subblock=None, norm="BatchNorm2d", activation="PReLu",
                 inner_activation="PReLu", padtype="replicate", negative_slope=0.0, inner_negative_slope=0.0, inplace=True, bias=True):
        super(UNetBlock, self).__init__()
        self.network = nn.Sequential()

        if input_nc is None:
            input_nc = outer_nc
        self.network.add_module("Input_conv2dblock", Conv2dBlock(
            input_nc, inner_nc, kernel_size=kernel_size, stride=stride, padding=padding, dropout=dropout, norm=norm, activation=inner_activation, negative_slope=inner_negative_slope, inplace=inplace, bias=bias))

        if not innermost:
            self.network.add_module("subblock", subblock)

        if innermost:
            self.network.add_module("Output_convTranspose2dblock", ConvTranspose2dBlock(
                inner_nc, outer_nc, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=0, dropout=0, norm=norm, activation=activation, negative_slope=negative_slope, inplace=inplace, bias=bias))
        else:
            self.network.add_module("Output_convTranspose2dblock", ConvTranspose2dBlock(
                inner_nc*2, outer_nc, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=0, dropout=dropout, norm=norm, activation=activation, negative_slope=negative_slope, inplace=inplace, bias=bias))

        self.outermost = outermost
        self.padtype = padtype

    def forward(self, x):
        out = self.network(x)
        if self.outermost:
            return out
        else:
            width = x.size(-1)
            height = x.size(-2)
            out_width = out.size(-1)
            out_height = out.size(-2)
            pad = ((height-out_height)//2, (height-out_height)-(height-out_height) //
                   2, (width-out_width)//2, (width-out_width)-(width-out_width)//2)
            if pad[0] != 0 or pad[1] != 0 or pad[2] != 0 or pad[3] != 0:
                out = F.pad(out, pad, mode=self.padtype)
            out = torch.cat([x, out], 1)
            return out


class ResnetBlock(nn.Module):

    def __init__(self, in_channels, inner_channels=None, dropout=0.5, norm="BatchNorm2d", activation="PReLu",
                 inner_activation="PReLu", padtype="replicate", mode="BottleNeck", negative_slope=0.0, inner_negative_slope=0.0, inplace=True, bias=True):
        super(ResnetBlock, self).__init__()

        if inner_channels is None:
            inner_channels = in_channels

        self.network = nn.Sequential()
        if mode == "BottleNeck":
            self.network.add_module("in_Conv2dBlock", Conv2dBlock(
                in_channels, inner_channels, kernel_size=1, stride=1, padding=0, dropout=dropout, norm=norm, activation=inner_activation, negative_slope=inner_negative_slope, inplace=inplace, bias=bias))

            p = 0
            if padtype == "replicate":
                self.network.add_module("Pad_Layer", nn.ReplicationPad2d(1))
            elif padtype == "reflection":
                self.network.add_module("Pad_Layer", nn.ReflectionPad2d(1))
            elif padtype == "zero":
                p = 1

            self.network.add_module("middle_Conv2dBlock", Conv2dBlock(
                inner_channels, inner_channels, kernel_size=3, stride=1, padding=p, dropout=dropout, norm=norm, activation=inner_activation, negative_slope=inner_negative_slope, inplace=inplace, bias=bias))

            self.network.add_module("out_Conv2dBlock", Conv2dBlock(
                inner_channels, in_channels, kernel_size=1, stride=1, padding=0, dropout=0, norm=norm, activation=activation, negative_slope=negative_slope, inplace=inplace, bias=bias))
        else:
            p = 0
            if padtype == "replicate":
                self.network.add_module("in_Pad_Layer", nn.ReplicationPad2d(1))
            elif padtype == "reflection":
                self.network.add_module("in_Pad_Layer", nn.ReflectionPad2d(1))
            elif padtype == "zero":
                p = 1

            self.network.add_module("in_Conv2dBlock", Conv2dBlock(
                in_channels, inner_channels, kernel_size=3, stride=1, padding=p, dropout=dropout, norm=norm, activation=inner_activation, negative_slope=inner_negative_slope, inplace=inplace, bias=bias))

            p = 0
            if padtype == "replicate":
                self.network.add_module(
                    "out_Pad_Layer", nn.ReplicationPad2d(1))
            elif padtype == "reflection":
                self.network.add_module("out_Pad_Layer", nn.ReflectionPad2d(1))
            elif padtype == "zero":
                p = 1

            self.network.add_module("out_Conv2dBlock", Conv2dBlock(
                inner_channels, in_channels, kernel_size=3, stride=1, padding=p, dropout=0, norm=norm, activation=activation, negative_slope=negative_slope, inplace=inplace, bias=bias))

    def forward(self, x):
        out = self.network(x)
        return out+x
