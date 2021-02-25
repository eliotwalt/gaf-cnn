import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary

class MoxCNN(nn.Module):

    def __init__(self, in_channels, out_channels, width):

        super().__init__()

        self.c_in = in_channels
        self.c_out = out_channels

        

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=self.c_in,  out_channels=self.c_in*6, 
                      kernel_size=(3,3), stride=1),#, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=self.c_in*6,  out_channels=self.c_in*8, 
                      kernel_size=(3,3), stride=1),#, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=self.c_in*8, out_channels=self.c_in*10, 
                      kernel_size=(3,3), stride=1),#, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=self.c_in*10, out_channels=self.c_in*12, 
                      kernel_size=(3,3), stride=1),#, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.c_in*12, out_channels=self.c_out*12,
                               kernel_size=(3,3), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=self.c_out*12, out_channels=self.c_out*8, 
                               kernel_size=(2,2), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=self.c_out*8, out_channels=self.c_out*6, 
                               kernel_size=(2,2), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=self.c_out*6, out_channels=self.c_out*3, 
                               kernel_size=(3,3), stride=2),
            nn.ConvTranspose2d(in_channels=self.c_out*3, out_channels=self.c_out, 
                               kernel_size=(2,2), stride=2),
        )

    def forward(self, x):

        return self.decoder(self.encoder(x))