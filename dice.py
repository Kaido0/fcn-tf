# -*- coding: utf-8 -*-
from __future__ import division
import scipy as scp
import os
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt

temp_picN = '225.bmp'
temp_loss = '225_loss_0.6728.bmp'


def caculate_dice(image_label,image_val):

    img1 = scp.misc.imread(os.path.join(os.getcwd(), image_label)) #原图像
    img1_loss = scp.misc.imread(os.path.join(os.getcwd(), image_val))
    img1_test=img1_loss[:,:,1] #输出的图像



    width = img1.shape[1]  # 267
    height = img1.shape[0]  # 210
    print('img1.shape=(%d,%d)' % (height, width))  # (210*267)

    img_test=np.array(img1_loss[:,:,1])
    img_test=np.where(img_test==58,85,np.where(img_test==67,170,np.where(img_test==141,255,0)))
    #转成

    print img1.shape
    print img1_loss.shape
    print img_test.shape
    print np.unique(img1)
    print np.unique(img1_loss[:,:,1])
    print np.unique(img_test)


    total=(img1>0).sum()
    print total
    diff=(img1!=img_test).sum()
    print diff
    dice=(total-diff)/total
    print dice

caculate_dice(temp_picN,temp_loss)