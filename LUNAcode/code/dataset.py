import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import config as co
import matplotlib.image as mpimg
import torchvision.transforms.functional as TF
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
class ImageData(Dataset):
    def __init__(self,is_train=True):
        self.is_train = is_train

        if is_train == True:
            self.img_list = os.listdir(co.train_img_dir)
            self.img_dir = co.train_img_dir
            print(len(self.img_list))

            self.transform = transforms.Compose([transforms.RandomCrop(size=(int(co.IMG_HEIGHT),int(co.IMG_WIDTH)),pad_if_needed=True),transforms.ToTensor()])
        else:
            self.img_dir = co.val_img_dir
            self.img_list = os.listdir(co.val_img_dir)
            print(len(self.img_list))
            self.transform = transforms.Compose([transforms.ToTensor()])
        # self.train_index = int(co.valid_ratio * len(self.img_list))
        self.train_index = len(self.img_list)

    def __len__(self):
        if self.is_train:
            return self.train_index
        else:
            return self.train_index
    def __getitem__(self, index):
#         print("hey  "*4 + str(index))
        img = Image.open(self.img_dir+self.img_list[index])
        img = np.array(img)
        # plt.imshow(img)
        # plt.show()
        img = TF.to_pil_image(img)
        img = self.transform(img)
        # plt.imshow(img.numpy().transpose(1,2,0))
        # plt.show()
        img = (img-0.5) /0.5
#         img = (img - 255.0) / 255.0
        return img

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = np.array(sample)
        plt.imshow(image)
        plt.show()
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        image= torch.from_numpy(image)
        return image
