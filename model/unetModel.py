from turtle import forward
from unicodedata import name
import torch.nn.functional as func

from .downSampling import *
from .upSampling import *

class UNet(nn.Module):
    def __init__(self, inputChannels, outputClasses):
        super(UNet, self).__init__()
        self.inputChannels = inputChannels
        self.outputClasses = outputClasses

        self.downSampling = downSa(inputChannels, 1024)
        self.upSampling = upSa(1024, outputClasses)

    def forward(self, x):
        xBottom = self.downSampling(x)
        xOutput = self.upSampling(xBottom)
        return xOutput

if __name__ == '__main__':
    net = UNet()