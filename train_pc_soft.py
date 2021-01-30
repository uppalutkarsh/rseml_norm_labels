from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
import pandas as pd
from datetime import datetime
import numpy as np
import cv2
import os
import sys
import time
import argparse
import random
from torch.autograd import Variable
from nloss import normLSFLoss
import torchvision.models as models

mean_x, std_x = (0.500548956397424, 0.464450589729188, 0.5005489563974249), (0.2534465549796417, 0.2510691153272786, 0.2534465549796417)
def seed_everything(seed=13):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class Conv2d_partial(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, partial=False):
        super(Conv2d_partial, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.partial = partial
    def forward(self, input):
        if self.partial:
            self.padding = 0

            pad_val = (self.kernel_size[0] - 1) // 2
            if pad_val > 0:
                if (self.kernel_size[0] - self.stride[0]) % 2 == 0:
                    pad_top = pad_val
                    pad_bottom = pad_val
                    pad_left = pad_val
                    pad_right = pad_val
                else:
                    pad_top = pad_val
                    pad_bottom = self.kernel_size[0] - self.stride[0] - pad_top
                    pad_left = pad_val
                    pad_right = self.kernel_size[0] - self.stride[0] - pad_left
                
                p0 = torch.ones_like(input) 
                p0 = p0.sum()
                                
                input = F.pad(input, (pad_left, pad_right, pad_top, pad_bottom) , mode='constant', value=0)
                
                p1 = torch.ones_like(input) 
                p1 = p1.sum()

                ratio = torch.div(p1, p0 + 1e-8) 
                input = torch.mul(input, ratio)  
            
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv2d_partial(in_planes, planes, kernel_size=1, bias=False, partial=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2d_partial(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, partial=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = Conv2d_partial(planes, self.expansion*planes, kernel_size=1, bias=False, partial=True)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                Conv2d_partial(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, partial=True),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2d_partial(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, partial=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2d_partial(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, partial=True)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                Conv2d_partial(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, partial=True),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
#class ResNet(object):
    def __init__(self, block, num_blocks, num_classes=3):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = Conv2d_partial(3, 64, kernel_size=7, stride=2, padding=3, bias=False, partial=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


args = {
        'num_gpus': 1,
        'ckpt_dir': 'ckpt/lsce/1pc_resnet50',
        'epochs': 50,
        'batch_size': 32,
        'val_batch_size': 100,
        'lr' : 0.1,
        'lr_schedule': 2,
        'momentum': 0.9,
        'nesterov': False,
        'weight_decay': 0.000001,
        }

if not os.path.exists(args['ckpt_dir']):
    os.makedirs(args['ckpt_dir'])

def adjust_learning_rate(optimizer, epoch):
    if args['lr_schedule'] == 0:
        lr = args['lr'] * ((0.2 ** int(epoch >= 20)) * (0.2 ** int(epoch >= 80)) * (0.2 ** int(epoch >= 150) * (0.2 ** int(epoch >= 180))))
    elif args['lr_schedule'] == 1:
        lr = args['lr'] * ((0.1 ** int(epoch >= 150)) * (0.1 ** int(epoch >= 225)))
    elif args['lr_schedule'] == 2:
        lr = args['lr'] * ((0.1 ** int(epoch >= 20)) * (0.1 ** int(epoch >= 40)))
    else:
        raise Exception("Invalid learning rate schedule!")
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

# Training
def train(train_loader, model, criterion, optimizer):
    model.train()
    lossl= []
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        lossl.append(loss.item())
    tloss= np.mean(np.array(lossl))
    return tloss

# Evaluating
def eval(test_loader, model, epoch, lr, best_acc):
    model.eval()
    correct,total = 0,0
    lossl=[]
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            loss = criterion(outputs, targets)
            lossl.append(loss.item())
    print('Epoch:',epoch, 'Accuracy: %f %%' % (100 * correct / total), 'Best Accuracy:', best_acc, 'lr:', lr)

    vloss= np.mean(np.array(lossl))
    return (float(100 * correct / total)), vloss

if __name__ == '__main__':

    seed_everything()
    print('To train and eval on climate dataset......')
    num_classes = 3
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean_x, std_x),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean_x, std_x),
    ])
    train_set = torchvision.datasets.ImageFolder(root='../new_dataset/train', transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args['batch_size'], shuffle=True,
                                               num_workers=4)

    test_set = torchvision.datasets.ImageFolder(root='../new_dataset/valid',  transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args['val_batch_size'], shuffle=False, num_workers=4)

    print('==> Building model..',args['ckpt_dir'][5:] )
    #Here
    #model = ResNet(BasicBlock, [2, 2, 2, 2], 3) #ResNet-18
    #model = ResNet(BasicBlock, [3, 4, 6, 3], 3) # 34
    model = ResNet(Bottleneck, [3, 4, 6, 3], 3) # 50
    model = model.cuda()

    #criterion = nn.CrossEntropyLoss()
    criterion =normLSFLoss(gamma=0.0, smoothing=0.1, reduction='mean')
    optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'], nesterov=args['nesterov'],weight_decay=args['weight_decay'])
    best_acc = 0
    start_epoch = 0
    tlossl, vlossl = [], []
    for epoch in range(start_epoch, args['epochs']):
        lr = adjust_learning_rate(optimizer, epoch + 1)
        tloss = train(train_loader, model, criterion, optimizer)
        acc, vloss = eval(test_loader, model, epoch, lr, best_acc)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(args['ckpt_dir'], 'best_epoch' + '.pth.tar'))
        tlossl.append(tloss)
        vlossl.append(vloss)
    dict = {'train_loss': tlossl, 'val_loss': vlossl}

    df = pd.DataFrame(dict)

    df.to_csv(args['ckpt_dir'] + "/loss.csv") 
