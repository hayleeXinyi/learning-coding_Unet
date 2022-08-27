from .doubleConvolution import *

class upSamplingOnce(nn.Module):

    def __init__(self, inputChannels, outChannels):
        super().__init__()
        self.upSamplePool = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)#haylee HyperParameters
        
        self.doubleConv =doubleConvFunc(inputChannels, outChannels)


    def forward(self, x, x1):
        x = self.upSamplePool(x)

        #concat process. Because set padding=1, then size will be same. If keep padding=0, Then need some extra action.
        x2 = torch.cat([x1,x], dim=1)
        return self.doubleConv(x2)

class upSamolingProcess(nn.Module):
    
    def __init__(self, outputClasses, ):
        super().__init__()
        self.dS1 = upSamplingOnce(1024, 512)
        self.dS2 = upSamplingOnce(512, 256)
        self.dS3 = upSamplingOnce(256, 128)
        self.ds4 = upSamplingOnce(128, 64)


    def forward(self, x):
        #Because concat,maybe the design now need to be changed
        return x