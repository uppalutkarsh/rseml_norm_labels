
from __future__ import print_function

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
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

resnet34 = models.resnet50(pretrained=False)
#resnet34 = models.resnet34(pretrained=False)
# densenet = models.densenet121(pretrained=False)
# mobilenet = models.mobilenet_v2(pretrained=False)
# from efficientnet_pytorch import EfficientNet
# efficientnet = EfficientNet.from_name('efficientnet-b0')
#from train_pc import *
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
        'val_batch_size': 100,
        }

if not os.path.exists(args['ckpt_dir']):
    os.makedirs(args['ckpt_dir'])

# Evaluating
def eval(test_loader, model):
    model.eval()
    correct,total = 0,0
    p, r, f1 = [], [], []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            y_pred = predicted.cpu()
            y_true = targets.cpu()
            prec = precision_score(y_true, y_pred, average='weighted')
            rec = recall_score(y_true, y_pred, average='weighted')
            f1s = f1_score(y_true, y_pred, average='weighted')
            p.append(prec)
            r.append(rec)
            f1.append(f1s)
    return (float(100 * correct / total)), sum(p)/3.0, sum(r)/3.0, sum(f1)/3.0

if __name__ == '__main__':

    seed_everything()
    print('To train and eval on climate dataset......')
    num_classes = 3
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean_x, std_x),
    ])
    test_set = torchvision.datasets.ImageFolder(root='../new_dataset/valid',  transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args['val_batch_size'], shuffle=False, num_workers=4)

    print('==> Building model..',args['ckpt_dir'][5:] )
    path =args['ckpt_dir']+"/best_epoch.pth.tar"
    #Here
    # # new final layer with 3 classes
    # efficientnet._fc = nn.Linear(1280, num_classes, bias=True)
    # model  = efficientnet.cuda()
    num_ftrs = resnet34.fc.in_features
    resnet34.fc = nn.Linear(num_ftrs, num_classes)
    model = resnet34.cuda()
    # densenet.classifier = nn.Linear(1024, num_classes)
    # model  = densenet.cuda()
    # mobilenet.classifier = nn.Linear(1280, num_classes)
    # model  = mobilenet.cuda()
    #model = ResNet(Bottleneck, [3, 4, 6, 3], 3).cuda()
    
    states = torch.load(path)
    model.load_state_dict(states)
    acc, p, r, f1 = eval(test_loader, model)
    print(acc, p, r, f1)
