from __future__ import print_function
from torch import nn
import math
import numpy as np

class Bottleneck(nn.Module):

    expansion = 4
    # output plane/ input plane

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        # if stride == 1: no reduction in dimension
        # if stride == 2: divivde dimension by 2

        self.downsample = downsample
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1,
            bias=False) # initial input 74*66
        # this is a bottleneck layer that reduces no. of planes/channels
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 3), stride=stride, 
            padding=(1, 1), bias=False) 
        
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, 
            kernel_size=1, bias=False) 
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        # the second bottleneck, expanding the no. of planes



    def forward(self, x):

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetEncoder(nn.Module):

    def __init__(self, original_size, vocab_size, emb_size, expansion=4):
        super(ResNetEncoder, self).__init__()

        self.inplanes = 32
        self.emb = nn.Embedding(vocab_size, emb_size) 
        # initial regular conv layers

        # (W - F + 2P)/S + 1
        # http://cs231n.github.io/convolutional-networks/

        self.conv1 = nn.Conv2d(1, 8, kernel_size=(3,2), stride=(1, 2), 
            padding=(0, 0), bias=False) # raw input is 84*64 --> 80*32
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3,2), stride=(1, 1),
            padding=(0, 1)) # 78*65

        self.conv2 = nn.Conv2d(8, 32, kernel_size=(3,2), stride=(1, 1), 
            padding=(0, 1), bias=False) # 76*66
        self.bn2 = nn.BatchNorm2d(32)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1),
            padding=(0, 0)) # 74*66

        # blocks of bottleneck layers
        self.block1 = self._make_layer(Bottleneck, 32, 3)
        self.block2 = self._make_layer(Bottleneck, 64, 4, stride=2)
        self.block3 = self._make_layer(Bottleneck, 128, 6, stride=2)
        self.block4 = self._make_layer(Bottleneck, 256, 3, stride=2)

        # 19*17

        self.avgpool = nn.AvgPool2d(kernel_size=(5, 5), stride=1)

        # 15*13
        # TODO: add conditional statement that if width is > threshold, add another block to reduce dimension
        width = int(self._get_correct_dim(self._get_correct_dim(original_size - 4 - 2 - 2)))
        width  = width - 4
        height = int(self._get_correct_dim(self._get_correct_dim(emb_size/2 + 1 + 1)))
        height = height - 4
        # print(512*width*height, emb_size)
        self.lin = nn.Linear(512*width*height, emb_size)
        self.bn = nn.BatchNorm1d(emb_size, momentum=0.01)

        # initialize original params
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _get_correct_dim(self, height):
        return np.floor((height-1)/2 + 1)

    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion,
                    kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.emb(x).unsqueeze(1)
        # print(x.size())
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        # print(x.size())

        x = self.conv2(x)
        # print(x.size())
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        # print(x.size())


        x = self.block1(x)
        # print(x.size())
        x = self.block2(x)
        # print(x.size())
        x = self.block3(x)
        # print(x.size())
        # x = self.block4(x)

        x = self.avgpool(x)
        # print(x.size())
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.bn(self.lin(x))

        return x














