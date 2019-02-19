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
#import matplotlib.pyplot as plt
#image_filenames = glob.glob('D:\\MyProgramma\\myPy\\TF_Google\\AlexNet\\cat/*.jpg')


def write_records(imagefile, record_location, shuffle=True):
    '''
    image_filenames:图片数据集的原始储存路径
    record_location:TFrecord储存的路径
    shuffle:默认洗牌
    '''
    writer = tf.python_io.TFRecordWriter(record_location)
    
    os.chdir(imagefile)
    image_filenames = glob.glob('*.jpg')
    if shuffle:     
        random.shuffle(image_filenames)
        
    image_size = 100
    for image in image_filenames:
        
        img_type = os.path.split(image)[-1].split('.')[0]
        if img_type == 'cat':
            index = 0
        else:
            index = 1
        
        img = Image.open(image).convert('RGB')
        img = img.resize((image_size,image_size))
        img_raw = img.tobytes()
        
        
        example = tf.train.Example(
                features=tf.train.Features(feature={
                        'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                        'img_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                        }))
        writer.write(example.SerializeToString())  #序列化为字符串
    
    writer.close()
    print('write_records 结束')
    
def read_records(filename, batch_size):
    filename_queue = tf.train.string_input_producer([filename])
    
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })
    image_size = 100
    img = tf.decode_raw(features['img_raw'], tf.uint8)
#    img = tf.reshape(img, [image_size, image_size, 3])
    img.set_shape([image_size*image_size*3])
    img = tf.cast(img, tf.float32) * (1./255)
    label = tf.cast(features['label'], tf.int32)
    n_class = 2
    label = tf.one_hot(label, n_class, 1,0)

    print('img: ', img)
    print('label: ', label)

    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=batch_size,
                                                    num_threads=4,
                                                    capacity=2000,
                                                    min_after_dequeue=1000)

    print('read_records 结束')
    return img_batch, label_batch
#    return img, label

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    