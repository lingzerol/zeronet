
from ..base.container import *
from ..base.block import *
from ..base.factory import *
from ..network.network import *
from .factory import *


class ConvNetworkFactory(NetworkFactory):

    def __init__(self):
        super(ConvNetworkFactory, self).__init__()

        self.exists_model_name = ["ConvNet", "ConvTransposeNet", "UNet", "ResNet",
                                  "Conv2dBlock", "ConvTranspose2dBlock", "ResnetBlock", "UNetBlock"]

    def define(self, param, in_channels=None, out_channels=None, subblock=None):
        module_type = param["type"]

        if module_type not in self.exists_model_name:
            raise RuntimeError("module not exists!")

        factor = param["factor"] if "factor" in param else 2
        sub_in_channels = param["in_channels"] if "in_channels" in param else in_channels
        sub_out_channels = param["out_channels"] if "out_channels" in param else out_channels
        inner_channels = param["inner_channels"] if "inner_channels" in param else in_channels*factor
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
        negative_slope = param["negative_slope"] if "negative_slope" in param else 0.0
        inner_negative_slope = param["inner_negative_slope"] if "inner_negative_slope" in param else 0.0
        inplace = param["inplace"] if "inplace" in param else True
        bias = param["bias"] if "bias" in param else True

        if module_type == "ConvNet":
            return ConvNet(sub_in_channels, sub_out_channels, inner_channels, kernel_size, stride=stride,
                           padding=padding, factor=factor, num_inner_layers=num_inner_layers, dropout=dropout, norm=norm, activation=activation, inner_activation=inner_activation, negative_slope=negative_slope, inner_negative_slope=inner_negative_slope, inplace=inplace, bias=bias)
        elif module_type == "ConvTransposeNet":
            return ConvTransposeNet(sub_in_channels, sub_out_channels, inner_channels, kernel_size, stride=stride,
                                    padding=padding, output_padding=output_padding, factor=factor, num_inner_layers=num_inner_layers, dropout=dropout, norm=norm, activation=activation, inner_activation=inner_activation, negative_slope=negative_slope, inner_negative_slope=inner_negative_slope, inplace=inplace, bias=bias)
        elif module_type == "UNet":
            return UNet(sub_in_channels, sub_out_channels, inner_channels, kernel_size, stride=stride,
                        padding=padding, factor=factor, num_inner_layers=num_inner_layers, dropout=dropout, norm=norm, activation=activation, inner_activation=inner_activation, padtype=padtype, negative_slope=negative_slope, inner_negative_slope=inner_negative_slope, inplace=inplace, bias=bias)
        elif module_type == "ResNet":
            return ResNet(sub_in_channels, sub_out_channels, inner_channels, kernel_size, stride=stride,
                          padding=padding, output_padding=output_padding, factor=factor, num_layers=num_layers, num_res_blocks=num_res_blocks, dropout=dropout, norm=norm, activation=activation, inner_activation=inner_activation, padtype=padtype, mode=mode, negative_slope=negative_slope, inner_negative_slope=inner_negative_slope, inplace=inplace, bias=bias)
        elif module_type == "Conv2dBlock":
            return Conv2dBlock(sub_in_channels, sub_out_channels, kernel_size, stride=stride, padding=padding, dropout=dropout, norm=norm, activation=activation, negative_slope=negative_slope, inplace=inplace, bias=bias)
        elif module_type == "ConvTranspose2dBlock":
            return ConvTranspose2dBlock(sub_in_channels, sub_out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, dropout=dropout, norm=norm, activation=activation, negative_slope=negative_slope, inplace=inplace, bias=bias)
        elif module_type == "ResNetBlock":
            return ResNetBlock(sub_in_channels, sub_out_channels, norm=norm, activation=activation,
                               inner_activation=inner_activation, padtype=padtype, mode=mode, negative_slope=negative_slope, inner_negative_slope=inner_negative_slope, inplace=inplace, bias=bias)
        elif module_type == "UNetBlock":
            return UNetBlock(inner_channels, out_channels, kernel_size, stride=stride,
                             padding=padding, dropout=dropout, input_nc=in_channels, outermost=outermost,
                             innermost=innermost, subblock=subblock, norm=norm, activation=activation,
                             inner_activation=inner_activation, padtype=padtype, negative_slope=negative_slope, inner_negative_slope=inner_negative_slope, inplace=inplace, bias=bias)


class ConvArchitectFactory(BaseArchitectFactory):

    def __init__(self):
        super(ConvArchitectFactory, self).__init__()
        self.conv_network_factory = ConvNetworkFactory()

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
        return Interative_Sequential(*result)


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
