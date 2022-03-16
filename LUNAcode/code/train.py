# !/usr/bin/env python
# -*- coding:utf-8 -*-
# Project Name:code_szt 
# File Name:train 
# User:szt
# DateTime:2020/3/23 下午3:43
# Software: PyCharm
# CopyRight: szt CC 4.0 BY 
# Waseda Uni.
# Code Start Here
# This Python 3 environment comes with many helpful analytics libraries installed
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

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

import config as co
import network
import math
import SSIM

batch_size=co.batch_size
dataset = ImageData(is_train=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,num_workers = co.numworks)
device = 'cuda'


lr = co.LR
# Initialize BCELoss function
criterion = nn.BCELoss()
msecriterion = nn.MSELoss()
l1criterion = nn.L1Loss()
# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

netE,netD,netG = network.create_3network()
if co.IF_mutiGPU:
    netE=nn.DataParallel(netE)
    netG = nn.DataParallel(netG)
    netD = nn.DataParallel(netD)
# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerE = optim.Adam(netE.parameters(), lr=lr, betas=(0.5, 0.999))
lambda1 = lambda epoch: math.pow(1 - epoch / co.num_epochs,co.poly)
scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizerD, lr_lambda=lambda1,last_epoch=-1)
scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizerG, lr_lambda=lambda1,last_epoch=-1)
scheduler_E = torch.optim.lr_scheduler.LambdaLR(optimizerE, lr_lambda=lambda1,last_epoch=-1)
valid_dataset = ImageData(is_train=False)
num_images_to_show = 1
valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=True)
valid_batch = next(iter(valid_dataloader)).to(device)


# Lists to keep track of progress
G_losses = []
D_losses = []
E_losses = []
iters = 0
num_epochs = co.num_epochs

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):

    # For each batch in the dataloader
    for i, (images) in enumerate(dataloader, 0):
        netG.train()
        netD.train()
        netE.train()

        netD.zero_grad()

        images = images.to(device)
        fake_images = netG(netE(images))
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################

        ## Create a fake pair batch --

        inp_x = {}
        inp_x['img'] = images
        inp_x['encoded'] = netE(images)

        #         label = torch.full((images.size(0),), real_label, device=device)
        label = torch.FloatTensor(np.random.uniform(low=0.855, high=0.999, size=(images.size(0)))).to(device)
        output = netD(inp_x).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward(retain_graph=True)
        D_x = output.mean().item()

        inp_x_fake = {}
        inp_x_fake['img'] = fake_images
        inp_x_fake['encoded'] = netE(images)
        label = torch.FloatTensor(np.random.uniform(low=0.005, high=0.155, size=(images.size(0)))).to(device)
        #         label.fill_(fake_label)
        output = netD(inp_x_fake).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward(retain_graph=True)
        D_G_z1 = output.mean().item()

        errD = errD_real + errD_fake

        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################

        ##############################
        loss_G_SSIM = 0
        ssim_loss =SSIM.SSIM()
        loss_G_SSIM = co.W_ssimloss*ssim_loss(images, fake_images)
        ##############################
        netG.zero_grad()
        inp_x_fake = {}
        inp_x_fake['img'] = fake_images
        inp_x_fake['encoded'] = netE(images)

        label = torch.FloatTensor(np.random.uniform(low=0.895, high=0.999, size=(images.size(0)))).to(device)
        #         label.fill_(real_label)
        output = netD(inp_x_fake).view(-1)

        errG = criterion(output, label) + co.L1Loss* l1criterion(images, fake_images)+loss_G_SSIM
        errG.backward(retain_graph=True)
        D_G_z2 = output.mean().item()
        optimizerG.step()

        netE.zero_grad()
        inp_x_fake = {}
        inp_x_fake['img'] = fake_images
        inp_x_fake['encoded'] = netE(images)

        label = torch.FloatTensor(np.random.uniform(low=0.895, high=0.999, size=(images.size(0)))).to(device)
        output = netD(inp_x_fake).view(-1)

        errE = criterion(output, label) + co.L1Loss* l1criterion(images, fake_images)+loss_G_SSIM
        #I add one line
        errE=errE.detach_().requires_grad_(True)
        errE.backward(retain_graph=True)
        E_G_z2 = output.mean().item()
        optimizerE.step()

        #################################_______STATS________###########################################
        # Output training stats
        if i % co.plt_times == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tLoss_E: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), errE.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        E_losses.append(errE.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        #         if (iters % 50 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
        #             netG.eval()
        #             with torch.no_grad():
        #                 fake = netG(fixed_noise).detach().cpu()
        #                 fake[:] = fake[:]*0.5 + 0.5
        #             img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        del images
        del inp_x_fake
        del inp_x
        del label
        del output
        torch.cuda.empty_cache()
        iters += 1

        if (epoch %co.plt_times == 0) and (i % co.plt_times == 0):
            netE.eval()
            netG.eval()
            encoded_img = netE(valid_batch)
            reconstructed_img = netG(encoded_img)
            for i in range(1):
                validimg = (valid_batch[i].cpu().detach().permute(1, 2, 0) * 0.5) + 0.5
                rec_img = (reconstructed_img[i].cpu().detach().permute(1, 2, 0) * 0.5) + 0.5
                validimg=validimg*255
                rec_img=rec_img*255
                rec_img=rec_img.numpy().astype(np.uint8)
                validimg=validimg.numpy().astype(np.uint8)
                plt.imsave(str(epoch) + "_" + str(i) + 'rec.png',rec_img)
                plt.imsave(str(epoch) + "_" + str(i) + 'val.png', validimg)
                torch.save(netE.state_dict(), "netE" + str(epoch)+"_" +str(co.num_channels_in_encoder) + ".model")
                torch.save(netG.state_dict(), "netG" + str(epoch)+"_" +str(co.num_channels_in_encoder) + ".model")
                torch.save(netD.state_dict(), "netD" + str(epoch) +"_"+ str(co.num_channels_in_encoder) + ".model")
    scheduler_D.step()
    scheduler_E.step()
    scheduler_G.step()
print("finshed")
torch.save(netE.state_dict(), "netE_fin" + str(co.num_channels_in_encoder) + ".model")
torch.save(netG.state_dict(), "netG_fin" + str(co.num_channels_in_encoder) + ".model")
torch.save(netD.state_dict(), "netD_fin" + str(co.num_channels_in_encoder) + ".model")