# !/usr/bin/env python
# -*- coding:utf-8 -*-
# Project Name:code_szt 
# File Name:config 
# User:szt
# DateTime:2020/3/23 下午3:43
# Software: PyCharm
# CopyRight: szt CC 4.0 BY 
# Waseda Uni.
# Code Start Here

IF_mutiGPU = True
batch_size = 14
device ='cuda'
LR = 0.0002
train_img_dir = './data/'#train dataset
val_img_dir = './kodak/'

valid_ratio = 0.8
num_images_to_show = 5
W_ssimloss = -0.5
L1Loss=5
IMG_WIDTH = 202
IMG_HEIGHT = 202
poly = 0.4
latent_size = 200
num_epochs = 1000
num_channels_in_encoder = 32
numworks = 8
plt_times = 20

output_model = './output_m/'
output_img = './output_i/'