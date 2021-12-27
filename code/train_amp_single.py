#!/usr/bin/env python
# coding: utf-8

# # Import modules
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import os

# # Set Hyperparameter
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

torch.manual_seed(777)
if device == "cuda:1":
    torch.cuda.manual_seed_all(777)
print(f"using {device}")

batch_size = 100
learning_rate = 1e-3
num_epoch = 2000


# # Load Data
def get_alphabet(root: str, batch_size: int):
    
    train_path = os.path.join(root, 'train')
    test_path = os.path.join(root, 'test')
    
    train1_rotation = ImageFolder(root = train_path,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Grayscale(1),
                                   #  transforms.RandomRotation(5)
                                 ]),
                                 target_transform=None)
    
    train2_default = ImageFolder(root = train_path,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Grayscale(1)
                                 ]),
                                 target_transform=None)
    
    test_default = ImageFolder(root = test_path,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Grayscale(1)
                                 ]),
                                 target_transform=None)

    train_loader = DataLoader(train1_rotation,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True,
                              num_workers=8)
   
    test_loader = DataLoader(test_default,
                             batch_size=batch_size,
                             shuffle=False,
                             drop_last=False,
                             num_workers=8) 

    return (train_loader, test_loader)


data_path = "/home/r320/wooseok/MNISTClassification/dataset/processed_data/"

train_loader, test_loader = get_alphabet(data_path, batch_size)

# # CNN Class
    
## rotate, dropout04, optim_adam, kernel all 5, out_feature = 512 ---> 99.56
## rotate, dropout04, optim_adam, kernel 3, out_feature = 512 ---> 99.48
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(            # 28x28
            nn.Conv2d(1, 128, 5, padding=2),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2,2),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(            # 14x14
            nn.Conv2d(128, 256, 5, padding=2),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2,2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4)
        )
        self.layer3 = nn.Sequential(            # 7x7

            nn.Conv2d(256, 512, 5, padding=2),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2, padding=1),  
            nn.ReLU(inplace=True),
            nn.Dropout(0.4)
        )
        self.layer4 = nn.Sequential(            # 4x4
            nn.Conv2d(512, 26, 5, padding=2),
            nn.BatchNorm2d(26),
            nn.AdaptiveMaxPool2d(1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4)
        )
        

        
    def forward(self, x):
        out = self.layer1(x) 
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        return out

# # Define model
#model = CNN().to(device)
#model_path = "Finetune_lr04_GAP_adam_reduceLR_5555kernel123ep_acc99.82498168945312acc297.24085998535156.pth"
model_path = "AMP077ep_acc99.7812271118164.pth"
model = torch.load("/home/r320/wooseok/MNISTClassification/weights/"+model_path).to(device)


loss_func = nn.CrossEntropyLoss().to(device)
#optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)


# # Training

def test_acc(model, test_loader=test_loader):
    correct = 0
    total = 0
    model = model.eval()
    with torch.no_grad():
        for image, label in test_loader:
            x = image.to(device)
            y = label.to(device)

            output = model.forward(x)
            _, output_index = torch.max(output, 1)

            total += label.size(0)
            correct += (output_index == y ).sum().float()

        return f"{100.0*correct/total}"

max_accuracy =0.
from tqdm import tqdm

for i in range(1, num_epoch+1):
    model = model.train()
    for _, [image, label] in tqdm(enumerate(train_loader), total= len(train_loader)):
        x = image.to(device)
        y = label.to(device)
        
        optimizer.zero_grad()
        output = model.forward(x)
        loss = loss_func(output, y)
        loss.backward()
        optimizer.step()
        
    accuracy = float(test_acc(model))
    if (accuracy> max_accuracy and accuracy>99.6):
        max_accuracy = accuracy 
        model_path = f"/home/r320/wooseok/MNISTClassification/weights/AMP{i:03d}ep_acc{max_accuracy}.pth"
        torch.save(model, model_path)

    #scheduler.step(accuracy)
    print(f"Epoch: {i}, Loss: {loss.item()}, LR: {scheduler.optimizer.state_dict()['param_groups'][0]['lr']}, Acc: {accuracy}, Best: {max_accuracy}")
