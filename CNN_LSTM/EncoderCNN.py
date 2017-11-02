from torch import nn

class Bottleneck(nn.Module):

	expansion = 4
	# output plane/ input plane

	def __init__(self, inplanes, planes):
		super(Bottleneck, self).__init__()

		# (W - F + 2P)/S + 1
		# http://cs231n.github.io/convolutional-networks/

		# each bottleneck module reduces height by 4,
		# while keeping width as 1

		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, 
			bias=False) # initial input size 74*1
		# this is a bottleneck layer that reduces no. of planes/channels
		self.bn1 = nn.BatchNorm2d(planes)

		self.conv2 = nn.Conv2d(planes, planes, kernel_size=(5, 1), 
			stride=(1, 1), padding=(0, 0), bias=False) 
			# first run: 70*1
		self.bn2 = nn.BatchNorm2d(planes)

		self.conv3 = nn.Conv2d(planes, planes * 4, 
			kernel_size=1, bias=False) # 72*1
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

        out += residual
        out = self.relu(out)

        return out


class ResNetEncoder(nn.Module)

	def __init__(self):
		super(ResNetEncoder, self).__init__()

		self.inplanes = 32

		# initial regular conv layers
		self.conv1 = nn.Conv2d(1, 8, kernel_size=(3,2), stride=(1, 2), 
			padding=(0, 0), bias=False) # raw input is 82*22 --> 80*11
		self.bn1 = nn.BatchNorm2d(8)
		self.relu = nn.ReLu(inplace=True)
		self.maxpool1 = nn.MaxPool2d(kernel_size=(3,3), stride=(1, 2),
			padding=(0, 0), bias=False) # 78*5

		self.conv2 = nn.Conv2d(8, 32, kernel_size=(3,3), stride=(1, 1), 
			padding=(0, 0), bias=False) # 76*3
		self.bn2 = nn.BatchNorm2d(32)
		self.maxpool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1),
			bias=False) # 74*1

		# blocks of bottleneck layers
		self.block1 = self._make_layer(Bottleneck, 32, 3)
		self.block2 = self._make_layer(Bottleneck, 64, 4)
		self.block2 = self._make_layer(Bottleneck, 128, 6)
		self.block2 = self._make_layer(Bottleneck, 256, 3)

		self.avgpool = nn.AvgPool2d(kernel_size=(3, 1), stride=1)

		# initialize original params
		for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

	def _make_layer(self, block, planes, blocks):

		layers = []
        layers.append(block(self.inplanes, planes))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool2(x)


        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x














