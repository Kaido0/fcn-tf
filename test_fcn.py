# -*- coding:utf8 -*-

import os
import scipy as scp
import scipy.misc

import numpy as np
import logging
import tensorflow as tf
import sys

import fcn8_vgg
import utils
#import loss

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


root_path_x = '/home/kaido/workspace/5_3/trainImage/'
root_path_l = '/home/kaido/workspace/5_3/SegTXT/'
train_txt = '/home/kaido/workspace/5_3/train.txt'
test_txt = '/home/kaido/workspace/5_3/test.txt'

model_dir = "model"
model_name = "model1.ckpt"
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
    
def test():

    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)

    with tf.Session() as sess:
        images = tf.placeholder("float")
        batch_images = tf.expand_dims(images, 0)
        labels = tf.placeholder("int32")
        labels2 = tf.expand_dims(labels, 0)
        
        vgg_fcn = fcn8_vgg.FCN8VGG()
        vgg_fcn.build(batch_images, debug=True)
        print('Finished building Network.')

        logits = vgg_fcn.upscore32
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels2, logits=logits, )
        cross_entropy = tf.reduce_mean(cross_entropy)
        
        saver = tf.train.Saver()
        print('loading model........' + model_name)
        saver.restore(sess,os.path.join(model_dir,model_name)) #注意这个地方不能写sess=saver.restore....
        print('model restored')

        test_file = open(test_txt)
        pic_name = test_file.readline()
        while pic_name:
            temp_picN = root_path_x + pic_name[0:-1] + '.bmp'
            img1 = scp.misc.imread(temp_picN)
            temp_lab = root_path_l + pic_name[0:-1] + '.txt'
            y_ = np.loadtxt(temp_lab)
            feed_dict = {images: img1,labels:y_}
            prediction,loss=sess.run([vgg_fcn.pred_up,cross_entropy],feed_dict=feed_dict)
            print loss
            up_color = utils.color_image(prediction[0])
            img_save_path = '/home/kaido/workspace/5_3/my_test/' + pic_name[0:-1] + '.bmp'
            scp.misc.imsave(img_save_path, up_color)
            pic_name = test_file.readline()

test()
