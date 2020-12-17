import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco
import os
import resnet18_adder as resnet
from resnet18_adder import Extra_layers, conv1x1
from AdderNet import adder


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes

        self.cfg = voc   #(coco, voc)[num_classes == 4]

        self.priorbox = PriorBox(self.cfg)

        #self.priors = Variable(self.priorbox.forward(), volatile=True)
        with torch.no_grad():
            self.priors = Variable(self.priorbox.forward())
            #print(self.priors.size())

        self.size = size

        # SSD network
        self.resnet = base
        #print(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.extras = extras
        print(self.extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect()

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        # apply resnet18-->ssd
        x = self.resnet.conv1(x)  ## 75*75
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
         
        x = self.resnet.maxpool(x) # 75*75
        #print(x.shape)

        x = self.resnet.layer1(x) #75*75
        x = self.resnet.layer2(x) #38*38
        sources.append(x)
        x = self.resnet.layer3(x) #19*19
        sources.append(x)
        x = self.resnet.layer4(x) #10*10
        sources.append(x)
        #print(x.shape)

        #extra layers
        x = self.extras.conv1(x)
        x = self.extras.conv2(x) #5*5
        sources.append(x)
        x = self.extras.conv3(x)
        x = self.extras.conv4(x) #3*3
        sources.append(x)
        x = self.extras.conv5(x)
        x = self.extras.conv6(x) #1*1
        sources.append(x)

        # for x in sources:
        #     print(x.shape)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        
        if self.phase == "test":

            output = self.detect.apply(self.num_classes, 0, 200, 0.01, 0.45,
                loc.view(loc.size(0), -1, 4),                  # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )

        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output



def multibox_adder(resnet, extra_layers, num_classes):
    '''
    detection head --> use addernet
    '''
    loc_layers = []
    conf_layers = []

    ### may be have bug ###
    loc_layers += [conv1x1(128, 4 * 4, stride=1)]
    conf_layers += [conv1x1(128, 4 * num_classes, stride=1)]

    loc_layers += [conv1x1(256, 6 * 4, stride=1)]
    conf_layers += [conv1x1(256, 6 * num_classes, stride=1)]

    loc_layers += [conv1x1(512, 6 * 4, stride=1)]
    conf_layers += [conv1x1(512, 6 * num_classes, stride=1)]

    loc_layers += [conv1x1(512, 6 * 4, stride=1)]
    conf_layers += [conv1x1(512, 6 * num_classes, stride=1)]

    loc_layers += [conv1x1(256, 4 * 4, stride=1)]
    conf_layers += [conv1x1(256, 4 * num_classes, stride=1)]

    loc_layers += [conv1x1(256, 4 * 4, stride=1)]
    conf_layers += [conv1x1(256, 4 * num_classes, stride=1)]

    return resnet, extra_layers, (loc_layers, conf_layers)


def multibox(resnet, extra_layers, num_classes):
    '''
    detection head --> use nn.conv2d()
    '''
    loc_layers = []
    conf_layers = []

    loc_layers += [nn.Conv2d(128, 4 * 4, kernel_size=1)]
    conf_layers += [nn.Conv2d(128, 4 * num_classes, kernel_size=1)]

    loc_layers += [nn.Conv2d(256, 6 * 4, kernel_size=1)]
    conf_layers += [nn.Conv2d(256, 6 * num_classes, kernel_size=1)]

    loc_layers += [nn.Conv2d(512, 6 * 4, kernel_size=1)]
    conf_layers += [nn.Conv2d(512, 6 * num_classes, kernel_size=1)]

    loc_layers += [nn.Conv2d(512, 6 * 4, kernel_size=1)]
    conf_layers += [nn.Conv2d(512, 6 * num_classes, kernel_size=1)]

    loc_layers += [nn.Conv2d(256, 4 * 4, kernel_size=1)]
    conf_layers += [nn.Conv2d(256, 4 * num_classes, kernel_size=1)]

    loc_layers += [nn.Conv2d(256, 4 * 4, kernel_size=1)]
    conf_layers += [nn.Conv2d(256, 4 * num_classes, kernel_size=1)]

    return resnet, extra_layers, (loc_layers, conf_layers) 



mbox = {
    '300':[4, 6, 6, 6, 4, 4],
    '512': [],
}

def build_ssd(phase, size=300, num_classes=21):

    # add, no use
    size = 300

    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    
    extra = Extra_layers(512)

    base_, extras_, head_ = multibox(resnet.resnet18(), extra, num_classes)

    return SSD(phase, size, base_, extras_, head_, num_classes)


if __name__ =="__main__":

    torch.backends.cudnn.enabled = False
    ssd = build_ssd("train")
    x = torch.zeros((32, 96, 19, 19))
    x = ssd.loc[0](x)
    print(x.size())
    x = torch.zeros((32, 1280, 10, 10))
    x = ssd.loc[1](x)
    print(x.size())
    x = torch.zeros((32, 512, 5, 5))
    x = ssd.loc[2](x)
    print(x.size())
    x = torch.zeros((32, 256, 3, 3))
    x = ssd.loc[3](x)
    print(x.size())
    x = torch.zeros((32, 256, 2, 2))
    x = ssd.loc[4](x)
    print(x.size())
    x = torch.zeros((32, 128, 1, 1))
    x = ssd.loc[5](x)
    print(x.size())

