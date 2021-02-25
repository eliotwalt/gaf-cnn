import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary

class ResNeXt(nn.Module):
    
    '''
    Source:
    https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035
    '''
    
    def __init__(self, in_channels, out_channels, width):
        
        pass
    
    def forward(self, x):
        
        return x
    
class DenseNetLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.bn = nn.BatchNorm2d(num_features=in_channels)
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=3,
                              padding=1)
        
    def forward(self, x):
        
        y = F.relu(self.bn(x))
        y = self.conv(y)
        
        return torch.cat([y, x], dim=1)

class ResidualDenseNet(nn.Module):

    '''
    Source:
    https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035
    '''
    
    def __init__(self, channels, depth):
        
        super().__init__()
        
        self.channels = channels
        self.depth = depth
        self.net = self.build_net()
        
    def build_net(self):
        
        net = nn.Sequential()
        
        for i in range(self.depth):
            cin = (i+1)*self.channels
            print(f'\tAdding DenseNetLayer({cin}, {self.channels})')
            net = nn.Sequential(
                net,
                DenseNetLayer(in_channels=cin, out_channels=self.channels)
            )
            
        return net
    
    def forward(self, x):
        
        return self.net(x)