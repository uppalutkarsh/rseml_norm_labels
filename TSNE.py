import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torchvision.models as models
#model = models.mobilenet_v2(pretrained=False)
mean_x, std_x = (0.500548956397424, 0.464450589729188, 0.5005489563974249), (0.2534465549796417, 0.2510691153272786, 0.2534465549796417)
path = './ckpt/lce/densenet121/best_epoch.pth.tar'
title = 'Test_lce_dense'
num_classes = 3
#model = models.resnet34(pretrained=False)
model = models.densenet121(pretrained=False)
#model.classifier = nn.Linear(1280, num_classes)

# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, num_classes)
#model = resnet50.cuda()
#model.classifier = nn.Linear(1280, num_classes)
#model  = mobilenet.cuda()
model.classifier = nn.Linear(1024, num_classes)

states = torch.load(path)
model.load_state_dict(states)
model.linear = nn.Flatten()

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean_x, std_x),
])


test_set = torchvision.datasets.ImageFolder(root='../new_dataset/valid',  transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=30, shuffle=False, num_workers=4)

extract = model
extract.cuda()
extract.eval()

out_target = []
out_output = []

for batch_idx, (inputs, targets) in enumerate(test_loader):
    inputs, targets = inputs.cuda(), targets.cuda()
    outputs = extract(inputs)
    output_np = outputs.data.cpu().numpy()
    target_np = targets.data.cpu().numpy()
    out_output.append(output_np)
    out_target.append(target_np[:,np.newaxis])
    
output_array = np.concatenate(out_output, axis=0)
target_array = np.concatenate(out_target, axis=0)

print('Pred shape :',output_array.shape)
print('Target shape :',target_array.shape)

tsne = TSNE(n_components=2, init='pca', random_state=0)
output_array = tsne.fit_transform(output_array)
plt.rcParams['figure.figsize'] = 10,10
plt.scatter(output_array[:, 0], output_array[:, 1], c= target_array[:,0])
plt.title(title)
plt.savefig('./'+title+'.png', bbox_inches='tight')

