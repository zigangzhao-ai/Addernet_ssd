from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss

# from ssd_vgg16_adder import build_ssd
from ssd_resnet50_adder import build_ssd

import os
import sys
import time
import math
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse

import warnings
warnings.filterwarnings('ignore')

from attention import WarmUpLR
from apex import amp

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

# CUDA_LAUNCH_BLOCKING = 1

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

# torch.cuda.set_device(1)

#add  parameters
parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default= "ResNet50-AdderNet.pth",
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=4, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=8, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float, #default=1e-3
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--pretrained_model', default='pretrained_model/',
                    help='Directory for pretrained_model')
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():
    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        cfg = coco
        dataset = COCODetection(root=args.dataset_root,
                                transform=SSDAugmentation(cfg['min_dim'],
                                                          MEANS))
    elif args.dataset == 'VOC':
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))


    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    net = ssd_net

    if args.cuda:
        #net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)

    else:
        '''
        resnet50_weights = torch.load(args.pretrained_model + args.basenet)
        print('Loading base network...')
        model_dict = ssd_net.resnet.state_dict()
        ##remove classification layer
        pretrained_dict = {k: v for k, v in resnet50_weights.items() if (k in model_dict and 'Prediction' not in k)}

        model_dict.update(pretrained_dict)

        ssd_net.resnet.load_state_dict(model_dict)
        '''
        resnet50_weights = torch.load(args.pretrained_model + args.basenet)
        model = ssd_net.resnet
        model.load_state_dict(resnet50_weights, strict=False)

    if args.cuda:
        net = net.cuda()

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)
        
        

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)

    optimizer1 = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999),eps=1e-8,  
                          weight_decay=0)
    #train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8000, 10000, 12000], gamma=0.2) #learning rate decay
    iter_per_epoch = len(dataset) // args.batch_size
    print("iter_per_epoch=", iter_per_epoch, "==starting warmup==")
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * 5)

    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)

    
    net, optimizer = amp.initialize(net, optimizer, opt_level="O1")

    if torch.cuda.device_count() > 1:
    #     print(torch.cuda.device_count())
        net = torch.nn.DataParallel(net)
        
    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0


    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True,drop_last=True)
    # create batch iterator
    batch_iterator = iter(data_loader)
    for iteration in range(args.start_iter, cfg['max_iter']):
        

        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        # load train data
        try:
            images, targets = next(batch_iterator)
        except:
            batch_iterator = iter(data_loader)  
            images, targets = next(batch_iterator)
        
       # [volatile=True] replace by  with torch.no_grad() 

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]
        # forward
        t0 = time.time()
        
        # with torch.no_grad():
        out = net(images)
        # backprop
        optimizer.zero_grad()
        
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        #loss.backward()
        #nn.utils.clip_grad_norm_(net.parameters(), max_norm=2, norm_type=2) ##add by zzg
        #print(iteration)
        if iteration <= iter_per_epoch * 5:
            warmup_scheduler.step()
        else:
            optimizer.step()
      
        #optimizer.step()
        #train_scheduler.step(step_index)
        t1 = time.time()
        loc_loss += loss_l.item()
        conf_loss += loss_c.item()

        if iteration % 2 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||'%(loss.item()),end=' ')


        if iteration >= 10000 and iteration % 500 == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), 'weights/ssd300_COCO_' +
                       repr(iteration) + '.pth')
    torch.save(ssd_net.state_dict(),
               args.save_folder + '' + args.dataset + '.pth')


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    # if isinstance(m, nn.Conv2d):
    #     xavier(m.weight.data)
    #     m.bias.data.zero_()

def focalloss_weights_init(m):
    ##initialize the bias for focal loss
    prior_prob = 0.01
    bias_value = -math.log((1 - prior_prob) / prior_prob)
    if isinstance(m, nn.Conv2d):  
      nn.init.constant_(m.weight.data, bias_value)



if __name__ == '__main__':
    train()
