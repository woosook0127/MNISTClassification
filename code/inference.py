# # Import modules
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import time

# # Set Hyperparameter
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

torch.manual_seed(777)
if device == "cuda:1":
    torch.cuda.manual_seed_all(777)
print(f"using {device}")

batch_size = 100

start_time = time.perf_counter()
# # Load Data
def get_alphabet(root: str, batch_size: int):
    
    train_path = os.path.join(root, 'train')
    test_path = os.path.join(root, 'test')
    
    test_default = ImageFolder(root = test_path,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Grayscale(1),
                                     transforms.RandomRotation(15)
                                 ]),
                                 target_transform=None)
    
    test_kaggle = ImageFolder(root = "/home/r320/wooseok/MNISTClassification/dataset/alpha_small/test",
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Resize(28),
                                     transforms.Grayscale(1),
                                     transforms.RandomInvert(1),
                                     transforms.RandomRotation(15)
                                 ]),
                                 target_transform=None)

    test_loader = DataLoader(test_default,
                             batch_size=batch_size,
                             shuffle=False,
                             drop_last=False,
                             num_workers=8) 
        
    test_loader2 = DataLoader(test_kaggle,
                             batch_size=batch_size,
                             shuffle=False,
                             drop_last=False,
                             num_workers=8) 

    return (test_loader, test_loader2)


data_path = "/home/r320/wooseok/MNISTClassification/dataset/processed_data/"

test_loader, test_loader2 = get_alphabet(data_path, batch_size)


# # CNN Class
## rotate, dropout04, optim_adam, kernel all 5, out_feature = 512 ---> 99.56
## rotate, dropout04, optim_adam, kernel 3, out_feature = 512 ---> 99.48
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(         # 28x28
            nn.Conv2d(1, 128, 5, padding=2),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2,2),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(          # 28x28      

            nn.Conv2d(128, 256, 5, padding=2),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2,2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4)
        )
        self.layer3 = nn.Sequential(         #14x14

            nn.Conv2d(256, 512, 5, padding=2),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(512, 26, kernel_size=5, padding=2),
            nn.BatchNorm2d(26),
            nn.AdaptiveAvgPool2d(1),
            nn.ReLU(),
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

model_arr = []
model_name_arr = []

def import_model(model_name):
    model = torch.load("/home/r320/wooseok/MNISTClassification/weights/"+ model_name).to(device)
    model_arr.append(model)
    model_name_arr.append(model_name)


#import_model("FT_9989_best_kaggle_up748ep_acc99.32181549072266acc297.05443572998047.pth")
#import_model("FT_9989_best_kaggle_up749ep_acc99.23430633544922acc297.01715087890625.pth")
#import_model("FT_BEST_9989to_acc99.84686279296875acc297.4645767211914.pth")
#import_model("FT_Round_9989_with_kaggle_cos001ep_acc99.82498168945312acc20.0.pth")
#import_model("FT_Round_9989_with_kaggle_cos031ep_acc99.84686279296875acc297.4645767211914.pth")
import_model("FT_Round_9989_with_kaggle_cos031ep_acc99_84686279296875_97_4645767211914.pth")
# import_model("FT_forOurDataACC_with_kaggle_cos025ep_acc99.80310821533203acc297.4645767211914.pth")
# import_model("Finetune_lr04_GAP_adam_reduceLR_5555kernel101ep_acc99.80310821533203acc297.59507751464844.pth")
# import_model("Finetune_lr04_GAP_adam_reduceLR_5555kernel123ep_acc99.82498168945312acc297.24085998535156.pth")
# import_model("Finetune_mergeddata_GAP_adam_reduceLR_5555kernel123ep_acc99.82498168945312.pth")
import_model("Training_original_byrotation_lr03_GAP_adam_reduceLR_5555kernel007ep_acc99.84686279296875.pth")
# import_model("acc99.80310821533203acc297.53914642333984.pth")
import_model("double_traind_cos_tunning_lr03_GAP_adamw_5555kernel1120ep_acc99.89061737060547.pth")
# import_model("originaldata_GAP_adam_reduceLR_5553kernel072ep_acc99.82498168945312.pth")
# import_model("processed_Rotate_GAP_adam_reduceLR_all5kernel070ep_acc99.86873626708984.pth")
# import_model("whatis.pth")

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

for i in range(len(model_arr)):
    accuracy = float(test_acc(model_arr[i]))
    accuracy2 = float(test_acc(model_arr[i], test_loader2))

    print(f"model: {model_name_arr[i]}")
    print(f"Hand written data Acc: {accuracy},\t Kaggle data Acc: {accuracy2}\n")

