import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import cv2
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import albumentations
import argparse

from zmq import device
import models

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.utils import save_image
from sklearn.model_selection import train_test_split


# constructing the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=100, help='number of epochs')
args = vars(parser.parse_args())


#helper functions
image_dir = '../output/saved_images/'
os.makedirs(image_dir, exist_ok=True)


def save_decoded_image(img, name):
    img = img.view(img.size(0), 3, 224, 224)
    save_image(img, name)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print(device)


batch_size = 2

gauss_blur = os.listdir('./input/gaussian_blurred')
gauss_blur.sort()
sharp = os.listdir('./input/sharp')
sharp.sort()

x_blur = []

for i in range(len(gauss_blur)):
    x_blur.append(gauss_blur[i])

y_sharp = []
for i in range(len(sharp)):
    y_sharp.append(sharp[i])


print(x_blur[10])
print(y_sharp[10])


(x_train, x_val, y_train, y_val) = train_test_split(x_blur, y_sharp, test_size=0.25)

print(len(x_train))
print(len(x_val))



# define transforms

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Resize((224, 224))
])


class DeblurDataset(Dataset):
    def __init__(self, blur_paths, sharp_paths=None, transforms=None):
        self.X = blur_paths
        self.y = sharp_paths
        self.transforms = transforms


    def __len__(self):
        return (len(self.X))

    def __getitem__(self, i):
        blur_image = cv2.imread(f'./input/gaussian_blurred/{self.X[i]}')

        if self.transforms:
            blur_image = self.transforms(blur_image)

        if self.y is not None:
            sharp_image = cv2.imread(f'./input/sharp/{self.y[i]}')
            sharp_image = self.transforms(sharp_image)
            return (blur_image, sharp_image)
        else:
            return blur_image


train_data = DeblurDataset(x_train, y_train, transform)
val_data = DeblurDataset(x_val, y_val, transform)


trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)


model = models.CNN().to(device)
print(model)