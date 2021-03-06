import torch
import torch.nn as nn
import torch.nn.functional as F

class Interpolate(nn.Module):

    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(Interpolate, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(x, self.size, self.scale_factor, self.mode, self.align_corners)


class Reshape(nn.Module):

    def __init__(self, reshape_size):
        super(Reshape, self).__init__()
        self.reshape_size = reshape_size

    def forward(self, x):
        return torch.reshape(x, self.reshape_size)


class Sine(nn.Module):

    def __init__(self):
        super(Sine, self).__init__()
        
    def forward(self, x):
        return torch.sin(x)