from .doubleConvolution import *

class downSamplingOnce(nn.Module):

    def __init__(self, inputChannels, outChannels):
        super().__init__()
        self.downSampleOnce = nn.Sequential(
            doubleConvFunc(inputChannels, outChannels),
            nn.MaxPool2d(2), #haylee HyperParameters
        )

    def forward(self, x):
        return self.downSampleOnce(x)

class downSamolingProcess(nn.Module):
    
    def __init__(self, inputChannels):
        super().__init__()
        self.dS1 = downSamplingOnce(inputChannels, 64)#haylee HyperParameters
        self.dS2 = downSamplingOnce(64, 128)#haylee HyperParameters
        self.dS3 = downSamplingOnce(128, 256)#haylee HyperParameters
        self.ds4 = downSamplingOnce(256, 512)#haylee HyperParameters


    def forward(self, x):
        x1 = self.dS1(x)
        x2 = self.dS2(x1)
        x3 = self.dS3(x2)
        x4 = self.ds4(x3)
        return x4