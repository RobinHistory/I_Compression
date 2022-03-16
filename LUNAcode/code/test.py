# !/usr/bin/env python
# -*- coding:utf-8 -*-
# Project Name:code_szt 
# File Name:test 
# User:szt
# DateTime:2020/3/23 下午10:55
# Software: PyCharm
# CopyRight: szt CC 4.0 BY 
# Waseda Uni.
# Code Start Here

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import ImageData
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
import torchvision.utils as vutils
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import copy
import time
import cv2 as cv
from tqdm import tqdm_notebook as tqdm
import  PSNR
import config as co
import network
import math
import MSSIM
from skimage import io, transform
print("Evaluating the model ...")

tot_img_size = co.IMG_WIDTH * co.IMG_HEIGHT * 3

valid_dataset = ImageData(is_train=False)
batch_size=1
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,num_workers=0)


netE16,netD16,netG16 = network.create_3network()
netE16.to(co.device)
netG16.to(co.device)
netE16 = torch.nn.DataParallel(netE16).cuda()
netG16 = torch.nn.DataParallel(netG16).cuda()

netG16.load_state_dict(torch.load(co.output_model+"netG_fin32.model"))
netE16.load_state_dict(torch.load(co.output_model+"netE_fin32.model"))

netE16.eval()
netG16.eval()
for i, (images) in enumerate(valid_dataloader, 0):

    encoded_img = netE16(images)
    reconstructed_img = netG16(encoded_img)
    images_t = images.clone()
    rec_img_t= reconstructed_img.clone()
    rec_img = (reconstructed_img.squeeze().cpu().detach().permute(1, 2, 0) * 0.5) + 0.5
    images = (images.squeeze().cpu().detach().permute(1, 2, 0) * 0.5) + 0.5

    rec_img = transform.resize(rec_img.numpy(), output_shape=images.shape, preserve_range=True, anti_aliasing=True,
                              order=3).squeeze()
    plt.subplot(1,2,1)
    plt.imshow(images)
    plt.title('img')
    plt.subplot(1,2,2)
    plt.imshow(rec_img)
    plt.title('rec')
    plt.show()

    images = images.squeeze().numpy()


    psnr = PSNR.psnr1(images,rec_img)
    print('img_psnr ' + str(i) + ': ' + str(psnr))

    images = torch.autograd.Variable(
        torch.from_numpy(images).unsqueeze(dim=0)).double().permute(0,3, 1,2)

    rec_img = torch.autograd.Variable(
        torch.from_numpy(rec_img).unsqueeze(dim=0)).double().permute(0,3, 1,2)
    mssim =  MSSIM.MSSSIM()(images,rec_img)

    print('img_mssim ' + str(i) + ': ' + str(mssim))
