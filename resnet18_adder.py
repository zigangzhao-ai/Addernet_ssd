# 2020.01.10-Replaced conv with adder
'''
code by zzg 2020-12-17
'''
import torch
import torch.nn as nn
#from AdderNetCuda import adder
from AdderNet import adder


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return adder.adder2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv3x3_p(in_planes, out_planes, stride, padding=0):
    """3x3 convolution with padding"""
    return adder.adder2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return adder.adder2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
 
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Conv2d(512 * block.expansion, num_classes, 1)
        # self.bn2 = nn.BatchNorm2d(num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = self.fc(x)
        # x = self.bn2(x)
        # return x.view(x.size(0), -1)
        return x


def resnet18(**kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def conv_bn(inp, oup, stride, padding):
    return nn.Sequential(
        #nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        conv3x3_p(inp, oup, stride, padding),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv1_bn(inp, oup, stride):
    return nn.Sequential(
        #nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        conv1x1(inp, oup, 1),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )

class Extra_layers(nn.Module):
    ''' add extra layers
    '''

    def __init__(self, input_channels=512):
        super(Extra_layers, self).__init__()
        
        self.conv1 = conv1_bn(input_channels, 256, 1)
        self.conv2 = conv_bn(256, 512, 2, 1)

        self.conv3 = conv1_bn(512, 128, 1)
        self.conv4 = conv_bn(128, 256, 2, 1)

        self.conv5 = conv1_bn(256, 128, 1)
        self.conv6 = conv_bn(128, 256, 2, 0)


    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)

        x = self.conv3(x)
        x = self.conv4(x)

        x = self.conv1(x)
        x = self.conv2(x)
       
        return x


if __name__ == "__main__":
    """Testing
    """
    resnet = resnet18().cuda()

    print("=====starting======")
    x = torch.zeros(32, 3, 300, 300)
    x = x.cuda()
    x = resnet.conv1(x)
    print(x.shape) ##75*75
    x = resnet.bn1(x)
    x = resnet.relu(x)
    x = resnet.maxpool(x)
    print(x.shape) ##38*38

    x = resnet.layer1(x)
    print(1, x.shape)
    x = resnet.layer2(x)
    print(2, x.shape)
    x = resnet.layer3(x)
    print(3, x.shape) ##19*19
    x = resnet.layer4(x)
    print(4, x.shape) #10*10
    # x = resnet(x)
    
