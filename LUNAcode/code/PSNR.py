# !/usr/bin/env python
# -*- coding:utf-8 -*-
# Project Name:code_szt 
# File Name:PSNR 
# User:szt
# DateTime:2020/3/25 下午3:53
# Software: PyCharm
# CopyRight: szt CC 4.0 BY 
# Waseda Uni.
# Code Start Here
import math
import numpy as np

def psnr1(Original, Distorted):
    Original = Original * 255
    # Original = Original.astype(np.uint8)

    Distorted = Distorted * 255
    # Distorted = Distorted.astype(np.uint8)

    pixel_value_Ori = Original.flatten().astype(float)
    pixel_value_Dis = Distorted.flatten().astype(float)

    #画素情報の取得
    imageHeight, imageWidth,BPP = Original.squeeze().shape

    #画素数
    N = imageHeight * imageWidth

    #1画素あたりRGB3つの情報がある.
    addr = N * BPP

    #RGB画素値の差の2乗の総和
    sumR=0
    sumG=0
    sumB=0

    #差の2乗の総和を計算
    for i in range(addr):
        if(i%3==0):
            sumB += pow ( (pixel_value_Ori[i]-pixel_value_Dis[i]), 2 )
        elif(i%3==1):
            sumG += pow ( (pixel_value_Ori[i]-pixel_value_Dis[i]), 2 )
        else:
            sumR += pow ( (pixel_value_Ori[i]-pixel_value_Dis[i]), 2 )

    #PSNRを求める
    MSE =(sumR + sumG + sumB) / (3 * N )
    PSNR = 10 * math.log(255*255/MSE,10)
    return PSNR

def psnr2(img1, img2):
    img1= img1 *255
    img2= img2 *255
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))