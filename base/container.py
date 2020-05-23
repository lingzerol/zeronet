import torch
import torch.nn as nn
import torch.nn.functional as F

class Interative_Sequential(nn.Sequential):

    def forward(self, input, layer=-1, every=False):
        output = []
        for i, module in enumerate(self._modules.values()):
            input = module(input)
            if every:
                output.append(input)
            if layer > 0 and i >= layer:
                break
        if every:
            return output
        else:
            return input
