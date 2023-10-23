import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import Encoder
from refine_net import RefineNet
class Encoder_Decoder(nn.Module):
    def __init__(self,input_channel = 3):
        super(Encoder_Decoder,self)

        self.input_channel = input_channel
        self.Encoder = Encoder(self.input_channel)
        list_size = [self.in_channels,4,8,16,32,64,128]

        self.Decoder1 = RefineNet()
        self.Decoder2 = RefineNet()
    def forward(self,input):
        encoded_features = self.Encoder(input)
        decoded1_features = self.Decoder1(encoded_features)
        decoded2_features = self.Decoder2(encoded_features)

        return decoded1_features, decoded2_features