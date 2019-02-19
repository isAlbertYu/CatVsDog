# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 23:22:46 2018

@author: Albert
"""

#coding=utf-8
from __future__ import print_function
import tensorflow as tf

class AlexNet:
    '''
    输入图片尺寸：100*100*3
    参数：
    
    '''
    def __init__(self, X, dropout, n_classes):
        self.__x = tf.reshape(X, shape=[-1, 100, 100, 3])
        self.__dropout = dropout
        self.__n_classes = n_classes
        
        # 存储所有的网络参数
        self.__weights = {
            'wc1': tf.Variable(tf.random_normal([11, 11, 3, 96])),
            'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256])),
            'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384])),
            'wc4': tf.Variable(tf.random_normal([3, 3, 384, 384])),
            'wc5': tf.Variable(tf.random_normal([3, 3, 384, 256])),
            'wd1': tf.Variable(tf.random_normal([2*2*256, 4096])),
            'wd2': tf.Variable(tf.random_normal([4096, 4096])),
            'out': tf.Variable(tf.random_normal([4096, self.__n_classes]))
        }
        self.__biases = {
            'bc1': tf.Variable(tf.random_normal([96])),
            'bc2': tf.Variable(tf.random_normal([256])),
            'bc3': tf.Variable(tf.random_normal([384])),
            'bc4': tf.Variable(tf.random_normal([384])),
            'bc5': tf.Variable(tf.random_normal([256])),
            'bd1': tf.Variable(tf.random_normal([4096])),
            'bd2': tf.Variable(tf.random_normal([4096])),
            'out': tf.Variable(tf.random_normal([self.__n_classes]))
        }
    # 私有方法    
    # 卷积操作
    def __conv2d(self, l_input, w, b, stride, padding='SAME'):
        return tf.nn.relu(
                tf.nn.bias_add(
                        tf.nn.conv2d(l_input, w, strides=[1, stride, stride, 1], padding=padding), 
                        b))
     
    # 最大下采样操作
    def __max_pool(self, l_input, k, stride, padding='SAME'):
        return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, stride, stride, 1], padding=padding)
     
    # 归一化操作
    def __norm(self, l_input, lsize=4):
        return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
     
    
    def forward(self, name):
        # 第一层卷积
        # 卷积
        conv1 = self.__conv2d(self.__x, self.__weights['wc1'], self.__biases['bc1'], stride=4, padding='VALID') #28*28*64
        pool1 = self.__max_pool(conv1, k=3, stride=2,padding='VALID')#14*14*64
        norm1 = self.__norm(pool1, lsize=4)
     
        # 第二层卷积
        # 卷积
        conv2 = self.__conv2d(norm1, self.__weights['wc2'], self.__biases['bc2'], stride=1,padding='SAME')#14*14*192
        pool2 = self.__max_pool(conv2, k=3, stride=2, padding='VALID')#7*7*192
        norm2 = self.__norm(pool2, lsize=4)
     
        # 第三层卷积
        # 卷积
        conv3 = self.__conv2d(norm2, self.__weights['wc3'], self.__biases['bc3'], stride=1)#7*7*384
        norm3 = self.__norm(conv3, lsize=4)
     
        # 第四层卷积
        # 卷积
        conv4 = self.__conv2d(norm3, self.__weights['wc4'], self.__biases['bc4'], stride=1)#7*7*384
        norm4 = self.__norm(conv4, lsize=4)
     
        # 第五层卷积
        # 卷积
        conv5 = self.__conv2d(norm4, self.__weights['wc5'], self.__biases['bc5'], stride=1)#7*7*256
        pool5 = self.__max_pool(conv5, k=3, stride=2, padding='VALID')#4*4*256
        norm5 = self.__norm(pool5, lsize=4)
        
        print('conv1大小： ', conv1.get_shape())
        print('pool1大小： ', pool1.get_shape())
        print('conv2大小： ', conv2.get_shape())
        print('pool2大小： ', pool2.get_shape())
        print('conv3大小： ', conv3.get_shape())
        print('conv4大小： ', conv4.get_shape())
        print('conv5大小： ', conv5.get_shape())
        print('pool5大小： ', pool5.get_shape())
    
        
        # 全连接层1，先把特征图转为向量
        dense1 = tf.reshape(norm5, [-1, self.__weights['wd1'].get_shape().as_list()[0]])
        print('dense1大小： ', dense1.get_shape())
        dense1 = tf.nn.relu(tf.matmul(dense1, self.__weights['wd1']) + self.__biases['bd1'], name='fc1')
        print('全连接层1---结束')
        dense1 = tf.nn.dropout(dense1, self.__dropout)
     
        # 全连接层2
        dense2 = tf.reshape(dense1, [-1, self.__weights['wd2'].get_shape().as_list()[0]])
        dense2 = tf.nn.relu(tf.matmul(dense1, self.__weights['wd2']) + self.__biases['bd2'], name='fc2') # Relu activation
        dense2 = tf.nn.dropout(dense2, self.__dropout)
        print('全连接层2---结束')
        # 网络输出层
        out = tf.add(tf.matmul(dense2, self.__weights['out']),
                     self.__biases['out'],
                     name = name)
        print('网络输出层---结束')
        return out

