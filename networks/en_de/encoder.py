import torch
import torch.nn as nn
from torch import sigmoid
from torch.nn.init import xavier_uniform_, zeros_


def conv(in_channels, out_channels, kernel_size=3, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=stride),
        nn.ReLU(inplace=True)
    )


def conv_block(in_channels,out_channels):
    return nn.Sequential(
        conv(in_channels,out_channels/2,kernel_size=3,stride=2),
        conv(out_channels/2,out_channels/2,kernel_size=3,stride=1)
    )

class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        self.in_channels = in_channels

        list_size = [self.in_channels,4,8,16,32,64,128]

        self.block1 = conv_block(list_size[0],list_size[1])
        self.block2 = conv_block(list_size[1],list_size[2])
        self.block3 = conv_block(list_size[2],list_size[3])
        self.block4 = conv_block(list_size[3],list_size[4])
        self.block5 = conv_block(list_size[4],list_size[5])
        self.block6 = conv_block(list_size[5],list_size[6])

    def forward(self,input):
        out_block1 = self.block1(input)
        out_block2 = self.block2(out_block1)
        out_block3 = self.block3(out_block2)
        out_block4 = self.block4(out_block3)
        out_block5 = self.block5(out_block4)
        out_block6 = self.block6(out_block5)

        results = [out_block6,out_block5,out_block4,out_block3,out_block2,out_block1]
        return results