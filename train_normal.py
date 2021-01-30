from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

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
#from nloss import normLSFLoss
import torchvision.models as models
resnet50 = models.resnet18(pretrained=False)
# densenet = models.densenet161(pretrained=False)
# mobilenet = models.mobilenet_v2(pretrained=False)
#from efficientnet_pytorch import EfficientNet
#efficientnet = EfficientNet.from_name('efficientnet-b0')
mean_cifar10 = (0.4914, 0.4822, 0.4465)
std_cifar10 = (0.2023, 0.1994, 0.2010)
def seed_everything(seed=13):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



args = {
        'num_gpus': 1,
        'ckpt_dir': 'ckpt/lce/resnet18/',
        'epochs': 100,
        'batch_size': 64,
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
    num_classes = 10
    transform_train = transforms.Compose([
        transforms.Resize((32, 32), padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean_cifar10, std_cifar10),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((32, 32), padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean_cifar10, std_cifar10),
    ])
    # train_set = torchvision.datasets.ImageFolder(root='new_dataset/train', transform=transform_train)
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=args['batch_size'], shuffle=True,
    #                                            num_workers=4)

    train_set = torchvision.datasets.CIFAR10(root = 'new_dataset/train', train=True, transform=transform_train, download = True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args['batch_size'], shuffle=True,
                                               num_workers=4)

    # test_set = torchvision.datasets.ImageFolder(root='new_dataset/valid',  transform=transform_test)
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=args['val_batch_size'], shuffle=False, num_workers=4)

    test_set = torchvision.datasets.CIFAR10(root='new_dataset/valid', train=False,  transform=transform_test, download = True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args['val_batch_size'], shuffle=False, num_workers=4)

    print('==> Building model..',args['ckpt_dir'][5:] )
    #Here
    # # new final layer with 3 classes
    # num_ftrs = resnet50.fc.in_features

    # efficientnet._fc = nn.Linear(1280, num_classes, bias=True)
    # model  = efficientnet.cuda()
    num_ftrs = resnet50.fc.in_features
    resnet50.fc = nn.Linear(num_ftrs, num_classes)
    model = resnet50.cuda()

    criterion =normLSFLoss(gamma=0.0, smoothing=0.1, reduction='mean', isNorm= True)
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
