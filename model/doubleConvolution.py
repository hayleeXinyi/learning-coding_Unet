import torch
import torch.nn as nn
import torch.nn.functional as func

class doubleConvFunc(nn.Module):

    def __init__(self, inputChannels, outputChannels):
        super().__init__()
        self.doubleConv = nn.Sequential(
            nn.Conv2d(inputChannels, outputChannels, kernel_size=3, padding=0), #haylee HyperParameters
            nn.BatchNorm2d(outputChannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(outputChannels, outputChannels, kernel_size=3, padding=0),#haylee HyperParameters
            nn.BatchNorm2d(outputChannels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.doubleConv(x)