# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 14:53:57 2018

@author: Albert
"""
import os
import tensorflow as tf
import glob
import random
from PIL import Image

class dataset(object):
    '''
    读取原始图片，将之resize成尺寸为image_size*image_size的图片，
    然后再将之封装为tfrecord格式的数据集文件存储在本地
    filename: tfrecord训练集的路径
    imagefile: 图片数据集的原始储存路径
    '''
    def __init__(self, filename, imagefile=None, image_size=100):
        self.__filename = filename
        self.__image_size = image_size

    def write_records(self, shuffle=True):
        '''
        imagefile:图片数据集的原始储存路径
        record_location:TFrecord储存的路径
        shuffle:默认洗牌,对图片list洗牌后写入
        '''
        writer = tf.python_io.TFRecordWriter(self.__filename)
        
        os.chdir(self.__filename)
        image_filenames = glob.glob('*.jpg')
        if shuffle:     
            random.shuffle(image_filenames)
            
        for image in image_filenames:
            
            img_type = os.path.split(image)[-1].split('.')[0]
            if img_type == 'cat':
                index = 0
            else:
                index = 1
            
            img = Image.open(image).convert('RGB')
            img = img.resize((self.__image_size, self.__image_size))
            img_raw = img.tobytes()
            
            
            example = tf.train.Example(
                    features=tf.train.Features(feature={
                            'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                            'img_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                            }))
            writer.write(example.SerializeToString())  #序列化为字符串
        
        writer.close()
        print('write_records 结束')
        
    def read_records(self):
        filename_queue = tf.train.string_input_producer([self.__filename])
        
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'label': tf.FixedLenFeature([], tf.int64),
                                               'img_raw' : tf.FixedLenFeature([], tf.string),
                                           })
    
        img = tf.decode_raw(features['img_raw'], tf.uint8)
    #    img = tf.reshape(img, [image_size, image_size, 3])
        img.set_shape([self.__image_size * self.__image_size * 3])
        img = tf.cast(img, tf.float32) * (1./255)
        label = tf.cast(features['label'], tf.int32)
        n_class = 2
        label = tf.one_hot(label, n_class, 1,0)
    
        print('img: ', img)
        print('label: ', label)
        return img, label
    
    
    def next_batch(self, batch_size):
        
        img, label = self.read_records()
        img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                        batch_size=batch_size,
                                                        num_threads=4,
                                                        capacity=2000,
                                                        min_after_dequeue=1000)
    
        print('read_records 结束')
        return img_batch, label_batch
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    