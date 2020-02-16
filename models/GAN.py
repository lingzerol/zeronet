import torch
import torch.nn as nn
import torch.nn.functional as F

from . import networks as net

from abc import ABC, abstractmethod


class GAN(ABC):
    def __init__(self, param):
        self.update_param(param)

    def update_param(self, param):
        self.param = param
        self.real_label = param["parameters"]["real_label"] if "real_label" in param["parameters"] else 1
        self.fake_label = param["parameters"]["fake_label"] if "fake_label" in param["parameters"] else 0
        self.device = param["parameters"]["device"] if "device" in param["parameters"] else "cpu"

        self.device = torch.device(
            "cuda" if self.device and torch.cuda.is_available() else "cpu")


    def D_loss(self, netD, real_data, fake_data, D_criterion=nn.BCELoss()):
        b_size = real_data.size(0)

        real_labels_t = None
        fake_labels_t = None
        real_output = netD(real_data).view(-1)

        fake_output = netD(fake_data.detach()).view(-1)

        out_size = fake_output.size(-1)
        real_labels_t = torch.full(
            (out_size,), self.real_label, device=self.device)
        fake_labels_t = torch.full(
            (out_size,), self.fake_label, device=self.device)

        errD_fake = D_criterion(fake_output, fake_labels_t)
        errD_real = D_criterion(real_output, real_labels_t)
        errD = errD_fake + errD_real
        return errD

    def G_loss(self, netD, fake_data, G_criterion=nn.BCELoss()):
        fake_output = netD(fake_data).view(-1)

        out_size = fake_output.size(-1)
        real_labels_t = torch.full(
            (out_size,), self.real_label, device=self.device)

        errG = G_criterion(fake_output, real_labels_t)

        return errG

    @abstractmethod
    def train(self, *args):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass


