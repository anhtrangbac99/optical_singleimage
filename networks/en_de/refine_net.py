import torch
import torch.nn as nn
import torch.nn.functional as F
from stn import STN
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)

class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)
    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes+i*growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)


def upconv(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True)
    )


class RefineNet(nn.Module):
    def __init__(self, n = 5, growth_rate=12,
                 reduction=0.5, bottleneck=False, dropRate=0.0):
        super(RefineNet, self).__init__()
        
        
        if bottleneck == True:
            n = n/2
            block = BottleneckBlock
        else:
            block = BasicBlock


        list_size = [4,8,16,32,64,128]
   
        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(list_size[-1], list_size[-1] , kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block6 = DenseBlock(n, list_size[-1] , growth_rate, block, dropRate)
        in_planes = int(list_size[-1]+n*growth_rate)
        self.trans6 = TransitionBlock(in_planes , list_size[-1], dropRate=dropRate)
        self.upconv6 = upconv(list_size[-1],list_size[-2])
        self.stn6 = STN()

        self.block5 = DenseBlock(n, list_size[-2], growth_rate, block, dropRate)
        in_planes = int(list_size[-2]+n*growth_rate)
        self.trans5 = TransitionBlock(in_planes , list_size[-2], dropRate=dropRate)
        self.upconv5 = upconv(list_size[-2],list_size[-3])
        self.stn5 = STN()

        self.block4 = DenseBlock(n, list_size[-3], growth_rate, block, dropRate)
        in_planes = int(list_size[-3]+n*growth_rate)
        self.trans4 = TransitionBlock(in_planes , list_size[-3], dropRate=dropRate)
        self.upconv4 = upconv(list_size[-3],list_size[-4])
        self.stn4 = STN()

        self.block3 = DenseBlock(n, list_size[-4], growth_rate, block, dropRate)
        in_planes = int(list_size[-4]+n*growth_rate)
        self.trans3 = TransitionBlock(in_planes , list_size[-4], dropRate=dropRate)
        self.upconv3 = upconv(list_size[-4],list_size[-5])
        self.stn3 = STN()

        self.block2 = DenseBlock(n, list_size[-5] , growth_rate, block, dropRate)
        in_planes = int(list_size[-5]+n*growth_rate)
        self.trans2 = TransitionBlock(in_planes , list_size[-5], dropRate=dropRate)
        self.upconv2 = upconv(list_size[-5],list_size[-6])
        self.stn2 = STN()

        self.block1 = DenseBlock(n, list_size[-6] , growth_rate, block, dropRate)
        in_planes = int(list_size[-6]+n*growth_rate)
        self.trans1 = TransitionBlock(in_planes , list_size[-6], dropRate=dropRate)
        self.stn1 = STN()

        self.block_list = [self.block6,self.block5,self.block4,self.block2,self.block2,self.block1]
        self.trans_list = [self.trans6,self.trans5,self.trans4,self.trans3,self.trans2,self.trans1]
        self.stn_list = [self.stn6,self.stn5,self.stn4,self.stn3,self.stn2,self.stn1]

        self.upconv_list = [self.upconv6,self.upconv5,self.upconv4,self.upconv3,self.upconv2]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self,encoded_features):
        decoded_features = []

        for idx,encoded_feature in enumerate(encoded_features):
            stn_feature = self.stn_list[idx](encoded_feature)
            if idx != 0: 
                upscale_feature = self.upconv_list[idx-1](decoded_features[-1])
                concatenated_input = torch.cat((stn_feature,encoded_feature,upscale_feature),dim=-1)

            else:
                concatenated_input = torch.cat((stn_feature,encoded_feature),dim= -1)
            
            decoded_features.append(trans_list[idx](self.block_list[idx](concatenated_input)))
        
        return decoded_features



