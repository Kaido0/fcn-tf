# -*- coding: utf-8 -*-
from __future__ import division
import scipy as scp
import os
import scipy.misc as misc
import numpy as np
import matplotlib.pyplot as plt

temp_picN = '225.bmp'
temp_loss = '225_loss_0.6728.bmp'


def caculate_dice(image_label,image_val):

    img1 = scp.misc.imread(os.path.join(os.getcwd(), image_label)) #label图像
    img1_loss = scp.misc.imread(os.path.join(os.getcwd(), image_val))#自己分割出来的图像

    img_test=np.array(img1_loss[:,:,1])
    img_test=np.where(img_test==58,85,np.where(img_test==67,170,np.where(img_test==141,255,0)))
    #替换图像，成标签格式

    misc.imsave(temp_picN+"_save.bmp",img_test)

    print "图像尺寸：",img1.shape
    width = img1.shape[1]  # 267
    height = img1.shape[0]  # 210
    # print img1.shape
    # print img1_loss.shape
    # print img_test.shape
    # print np.unique(img1) # 类似set，输出不重复的像素
    # print np.unique(img1_loss[:,:,1])
    # print np.unique(img_test)

    sum_csf=(img1==85).sum()
    sum_gm=(img1==170).sum()
    sum_wm=(img1==255).sum()

    A_csf=(img_test==85).sum()
    A_gm = (img_test==170).sum()
    A_wm = (img_test==255).sum()

    num_csf=0
    num_gm=0
    num_wm=0
    for i in range(0,height):
        for j in range(0,width):
            if img_test[i,j]==img1[i,j]:
                if img_test[i,j]==85:
                    num_csf+=1
                if img_test[i,j]==170:
                    num_gm+=1
                if img_test[i,j]==255:
                    num_wm+=1

    dice_CSF = 2 * num_csf / (A_csf + sum_csf)
    dice_GM = 2 * num_gm / (A_gm + sum_gm)
    dice_WM = 2 * num_wm / (A_wm + sum_wm)
    print "dice_csf:",dice_CSF
    print "dice_gm:", dice_GM
    print "dice_wm:", dice_WM
    return dice_CSF,dice_GM,dice_WM


caculate_dice(temp_picN,temp_loss)
