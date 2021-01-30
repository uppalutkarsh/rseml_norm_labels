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
from nloss import normLSFLoss
import torchvision.models as models

#just rename models.resnet50 to models.resnet18 similar...
resnet50 = models.resnet50(pretrained=False)
densenet = models.densenet121(pretrained=False)

mobilenet = models.mobilenet_v2(pretrained=False)

from efficientnet_pytorch import EfficientNet
efficientnet = EfficientNet.from_name('efficientnet-b0')


mean_x, std_x = (0.500548956397424, 0.464450589729188, 0.5005489563974249), (0.2534465549796417, 0.2510691153272786, 0.2534465549796417)
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
        'ckpt_dir': 'ckpt/lce/1resnet50/',
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

    seed_everything(13)
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

    # For Efficient Net
    # efficientnet._fc = nn.Linear(1280, num_classes, bias=True)
    # model  = efficientnet.cuda()
    
    #For Resnet Waale koi bhi
    num_ftrs = resnet50.fc.in_features
    resnet50.fc = nn.Linear(num_ftrs, num_classes)
    model = resnet50.cuda()
    
    #For Mobilenet
    # mobilenet.classifier = nn.Linear(1280, num_classes)
    # model  = mobilenet.cuda()
    
    
    # densenet.classifier = nn.Linear(1024, num_classes)
    # model  = densenet.cuda()
    
    #CE

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