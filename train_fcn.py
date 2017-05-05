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

def main(train=False,test=False):

    with tf.Session() as sess:
        images = tf.placeholder("float")
        labels=tf.placeholder("int32")

        batch_images = tf.expand_dims(images, 0)  # 最左边加一个维度
        labels2 = tf.expand_dims(labels, 0)

        vgg_fcn = fcn8_vgg.FCN8VGG()
        vgg_fcn.build(batch_images, train=True,debug=True)
        print('Finished building Network.')
        
        saver=tf.train.Saver()
        print('Running the Network')
        logits=vgg_fcn.upscore32
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels2,logits=logits,)
        with tf.name_scope('loss'):
            cross_entropy=tf.reduce_mean(cross_entropy)
            tf.summary.scalar('loss', cross_entropy)
        with tf.name_scope('train'):  # 训练过程
            optimizer = tf.train.GradientDescentOptimizer(0.000001)
            train_op = optimizer.minimize(cross_entropy)

        init = tf.global_variables_initializer()
        sess.run(init)

        merged = tf.summary.merge_all()  # 将图形、训练过程等数据合并在一起
        writer = tf.summary.FileWriter('logs/', sess.graph)  # 将训练日志写入到logs文件夹下

        train_file = open(train_txt)
        pic_name = train_file.readline()
        while pic_name:
            temp_picN = root_path_x + pic_name[0:-1]+'.bmp'
            temp_lab = root_path_l + pic_name[0:-1]+'.txt'
            img1 = scp.misc.imread(temp_picN)
            y_ = np.loadtxt(temp_lab)
            feed_dict = {images: img1, labels: y_}
            for step in range(100):
                _,loss,up,result=sess.run([train_op,cross_entropy,vgg_fcn.pred_up,merged], feed_dict=feed_dict)
                writer.add_summary(result, step)  # 将日志数据写入文件
                if(step%20==0):
                    print('pic_name=%s,step=%d,loss:%0.04f'%(pic_name,step,loss))        
            print('loss:%0.04f' % loss)   #up:(1,210,267)
            up_color = utils.color_image(up[0])
            img_save_path='/home/kaido/workspace/5_3/my_train/'+pic_name[0:-1]+'_loss_'+u'%0.04f'%(loss)+'.bmp'
            scp.misc.imsave(img_save_path, up_color)
            pic_name = train_file.readline()
        save_path = saver.save(sess, os.path.join(model_dir,model_name))
    print('Done!')
