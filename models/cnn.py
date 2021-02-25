import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary

class ConstantShapeReductionCNN(nn.Module):

    '''
    ConstantShapeReductionCNN:

    Keeps the image size constant and reduce the number of channels from
    the input number of channels to the output number of channels
    '''

    def __init__(self, in_channels, out_channels, num_blocks, dropout=0.2):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        if dropout < 0:
            raise ValueError(f'Dropout must be between 0 and 1 (both included).')
        self.dropout = dropout

        self.net = self.build_net()

    def ConvBlock(self, in_channels, out_channels, output):

        '''
        A ConvBlock is composed of:
        1. Conv2d
        2. ReLU
        3. BatchNorm2d (if self.batch_norm)
        3. Conv2d
        4. ReLU
        3. BatchNorm2d (if not output)
        4. Dropout (if not output)
        '''

        block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        padding=1,
                        stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_channels),
            nn.Conv2d(in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        padding=1,
                        stride=1),
            nn.ReLU()
        )

        if not output:
            block = nn.Sequential(
                block,
                nn.BatchNorm2d(num_features=out_channels),
                nn.Dropout(self.dropout)
            )

        return block

    def build_net(self):

        net = nn.Sequential()
        for n in range(self.num_blocks):
            if n == 0:
                cin = self.in_channels
            else:
                cin = int((self.out_channels-self.in_channels)/self.num_blocks*n+self.in_channels)
            if n == self.num_blocks-1:
                cout = self.out_channels
                net = nn.Sequential(
                    net,
                    self.ConvBlock(
                        in_channels=cin,
                        out_channels=cout,
                        output=True)
                )
            else:
                cout = int((self.out_channels-self.in_channels)/self.num_blocks*(n+1)+self.in_channels)
                net = nn.Sequential(
                    net,
                    self.ConvBlock(
                        in_channels=cin,
                        out_channels=cout,
                        output=False)
                )

        return net

    def forward(self, x):

        return self.net(x)

class ConstantShapeParametricCNN(nn.Module):

    '''
    ConstantShapeParametricCNN:

    Keeps the image size constant and apply convolution 
    to get number of channels according to self.channels
    '''

    def __init__(self, channels, dropout=0.2):

        super(ConstantShapeParametricCNN, self).__init__()

        self.in_channels = channels[:-1]
        self.out_channels = channels[1:]
        if dropout < 0:
            raise ValueError(f'Dropout must be between 0 and 1 (both included).')
        self.dropout = dropout

        self.net = self.build_net()

    def ConvBlock(self, in_channels, out_channels, output):

        '''
        A ConvBlock is composed of:
        1. Conv2d
        2. ReLU
        3. BatchNorm2d (if self.batch_norm)
        3. Conv2d
        4. ReLU
        3. BatchNorm2d (if not output)
        4. Dropout (if not output)
        '''

        block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        padding=1,
                        stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_channels),
            nn.Conv2d(in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        padding=1,
                        stride=1),
            nn.ReLU()
        )

        if not output:
            block = nn.Sequential(
                block,
                nn.BatchNorm2d(num_features=out_channels),
                nn.Dropout(self.dropout)
            )

        return block

    def build_net(self):

        net = nn.Sequential()
        for n, (cin, cout) in enumerate(zip(self.in_channels, self.out_channels)):
            if n == len(self.in_channels)-1:
                net = nn.Sequential(
                    net,
                    self.ConvBlock(
                        in_channels=cin,
                        out_channels=cout,
                        output=True)
                )
            else:
                net = nn.Sequential(
                    net,
                    self.ConvBlock(
                        in_channels=cin,
                        out_channels=cout,
                        output=False)
                )

        return net

    def forward(self, x):

        return self.net(x)

