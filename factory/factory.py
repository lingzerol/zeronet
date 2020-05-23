import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base.factory import *
from ..base.block import *
from ..base.container import *
from ..base.function import *


class BaseNetworkFactory(NetworkFactory):
    def __init__(self):
        super(BaseNetworkFactory, self).__init__()
        self.exists_model_name = ["Linear", "Flatten", "Conv2d", "ConvTranspose2d", "Dropout", "Dropout2d", "BatchNorm1d",
                                  "BatchNorm2d", "InstanceNorm1d", "InstanceNorm2d", "ReflectionPad2d", "ReplicationPad2d",
                                  "ZeroPad2d", "Interpolate", "Reshape", "Sigmoid", "PReLu", "LeakyReLu", "ReLu", "Tanh", "Softmax"]

    def define(self, param):
        module_type = param["type"]

        if module_type not in self.exists_model_name:
            raise RuntimeError("module not exists!")

        if module_type == "Conv2d" or module_type == "ConvTranspose2d":
            in_channels = param["in_channels"]
            out_channels = param["out_channels"]
            kernel_size = param["kernel_size"]
            stride = param["stride"] if "stride" in param else 1
            padding = param["padding"] if "padding" in param else 0
            output_padding = param["output_padding"] if "output_padding" in param else 0
            dilation = param["dilation"] if "dilation" in param else 1
            groups = param["groups"] if "groups" in param else 1
            bias = param["bias"] if "bias" in param else True
            padding_mode = param["padding_mode"] if "padding_mode" in param else 'zeros'

        if module_type == "Linear":
            in_features = param["in_features"]
            out_features = param["out_features"]
            bias = param["bias"] if "bias" in param else True
            return nn.Linear(in_features, out_features, bias)
        elif module_type == "Flatten":
            return nn.Flatten()
        elif module_type == "Conv2d":
            return nn.Conv2d(in_channels, out_channels, kernel_size,
                             stride, padding, dilation, groups, bias, padding_mode)
        elif module_type == "ConvTranspose2d":
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                      stride, padding, output_padding, dilation, groups, bias, padding_mode)
        elif module_type == "Dropout":
            dropout = param["dropout"]
            return nn.Dropout(dropout)
        elif module_type == "Dropout2d":
            dropout = param["dropout"]
            return nn.Dropout2d(dropout)
        elif module_type == "BatchNorm1d":
            num_features = param["num_features"]
            return nn.BatchNorm1d(num_features)
        elif module_type == "BatchNorm2d":
            num_features = param["num_features"]
            return nn.BatchNorm2d(num_features)
        elif module_type == "InstanceNorm1d":
            num_features = param["num_features"]
            return nn.InstanceNorm1d(num_features)
        elif module_type == "InstanceNorm2d":
            num_features = param["num_features"]
            return nn.InstanceNorm2d(num_features)
        elif module_type == "ReflectionPad2d":
            padding = param["padding"]
            return nn.ReflectionPad2d(padding)
        elif module_type == "ReplicationPad2d":
            padding = param["padding"]
            return nn.ReplicationPad2d(padding)
        elif module_type == "ZeroPad2d":
            padding = param["padding"]
            return nn.ZeroPad2d(padding)
        elif module_type == "Interpolate":
            size = param["size"] if "size" in param else None
            scale_factor = param["scale_factor"] if "scale_factor" in param else None
            mode = param["mode"] if "mode" in param else "nearest"
            align_corners = param["align_corners"] if "align_corners" in param else None
            return Interpolate(size, scale_factor, mode, align_corners)
        elif module_type == "Reshape":
            reshape_size = param["reshape_size"]
            return Reshape(reshape_size)
        elif module_type == "Sigmoid":
            return nn.Sigmoid()
        elif module_type == "PReLu":
            num_parameters = param["num_parameters"] if "num_parameters" in param else 1
            init = param["init"] if "init" in param else 0.25
            return nn.PReLU(num_parameters, init)
        elif module_type == "LeakyReLu":
            negative_slope = param["negative_slope"] if "negative_slope" in param else 1e-2
            inplace = param["inplace"] if "inplace" in param else False
            return nn.LeakyReLU(negative_slope, inplace)
        elif module_type == "ReLu":
            return nn.ReLU()
        elif module_type == "Tanh":
            return nn.Tanh()
        elif module_type == "Softmax":
            dim = param["dim"] if "dim" in param else None
            return nn.Softmax(dim=dim)


class BaseLossFactory(LossFactory):

    def __init__(self):
        super(BaseLossFactory, self).__init__()
        self.exists_loss_name = ["BCELoss", "L1Loss", "MSELoss"]

    def define(self, loss_type):
        if loss_type not in self.exists_loss_name:
            raise RuntimeError("Loss not exists!")

        if loss_type == "BCELoss":
            return nn.BCELoss()
        elif loss_type == "L1Loss":
            return nn.L1Loss()
        elif loss_type == "MSELoss":
            return nn.MSELoss()


class BaseOptimizerFactory(OptimizerFactory):

    def __init__(self):
        super(BaseOptimizerFactory, self).__init__()
        self.exists_optimizer_name = ["Adam", "SGD"]

    def define(self, param, parameters):
        if param["type"] not in self.exists_optimizer_name:
            raise RuntimeError("Optimizer not exists!")

        if param["type"] == "Adam":
            lr = param["lr"] if "lr" in param else 0.001
            beta1 = param["beta1"] if "beta1" in param else 0.5
            beta2 = param["beta2"] if "beta2" in param else 0.999
            optimizer = torch.optim.Adam(parameters,
                                         lr=param["lr"], betas=(beta1, beta2))
        elif param["type"] == "SGD":
            lr = param["lr"] if "lr" in param else 0.001
            momentum = param["momentum"] if "momentum" in param else 0.5
            optimizer = torch.optim.SGD(
                parameters, lr=lr, momentum=momentum)
        return optimizer


class BaseSchedularFactory(SchedularFactory):

    def __init__(self):
        super(BaseSchedularFactory, self).__init__()
        self.exists_schedular_name = ["Linear", "Step", "Plateau", "Cosine"]

    def define(self, param, parameters):
        if param["type"] not in self.exists_schedular_name:
            raise RuntimeError("Schedular not exists!")

        if param["type"] == "Linear":
            def lambda_rule(epoch):
                lr_l = 1.0 - \
                    min(1.0, max(0, epoch -
                                 param["start"]) / float(param["num_epoch_decay"] + 1))
                return lr_l
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                parameters, lr_lambda=lambda_rule)
        elif param["type"] == "Step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                parameters, step_size=self.param["step_size"],
                gamma=self.param["decay_gamma"])
        elif param["type"] == "Plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                parameters, mode=param["mode"], factor=param["factor"], threshold=param["threshold"], patience=param["patience"])
        elif param["type"] == "Cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                parameters, T_max=param["T_max"], eta_min=param["eta_min"])
        return scheduler


class BaseArchitectFactory(ArchitectFactory):

    def __init__(self, network_factory=BaseNetworkFactory(), optmizer_factory=BaseOptimizerFactory(), loss_factory=BaseLossFactory(), scheduler_factory=BaseSchedularFactory()):
        super(BaseArchitectFactory, self).__init__()
        self.network_factory = network_factory
        self.optimizer_factor = optmizer_factory
        self.loss_factory = loss_factory
        self.scheduler_factory = scheduler_factory

    def define_network(self, param):
        result = []
        for key in param.keys():
            id = param[key]["id"]
            result.append([id, self.network_factory.define(param)])
        result.sort(key=lambda x: x[0])
        result = [r[1] for r in result]
        return Interative_Sequential(*result)

    def define_optimizer(self, param, parameters):
        return self.optimizer_factor.define(param, parameters)

    def define_loss(self, loss_type):
        return self.loss_factory.define(loss_type)

    def define_schedular(self, param, parameters):
        return self.scheduler_factory.define(param, parameters)
