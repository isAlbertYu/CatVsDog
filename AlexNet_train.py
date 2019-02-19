# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 10:53:08 2018

@author: Albert
"""

#coding=utf-8
from __future__ import print_function
from tensorflow.python.framework import graph_util
import tensorflow as tf
from datagenerator import dataset
from AlexNet import AlexNet

# 训练集和测试集文件的路径
train_set = 'my_tfrecord/my_tfrecord_train'
validate_set =  'my_tfrecord/my_tfrecord_validatet'


# 定义网络的训练超参数
learning_rate = 0.00005
training_iters = 5000
BATCH_SIZE = 64
n_batch = 10000 // BATCH_SIZE

# 定义网络参数
image_size = 100
n_input = image_size*image_size*3 # 输入的维度
n_classes = 2 # 标签的维度
dropout = 0.75 # Dropout 的概率
 
# 占位符输入
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

data_tr = dataset(train_set)
image_batch, label_batch = data_tr.next_batch(BATCH_SIZE)# 读取训练集

data_va = dataset(validate_set)
image_valid, label_valid = data_va.next_batch(BATCH_SIZE)# 读取验证集

pred = AlexNet(x, keep_prob, n_classes).forward(name='pred')

# 定义损失函数和优化器
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=tf.argmax(y, 1)))

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 1e-4
learning_rate = tf.train.exponential_decay(learning_rate=starter_learning_rate, global_step=global_step,
                                           decay_steps=100, decay_rate=0.95, staircase=True)


train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
#train_op = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss)
output = tf.argmax(pred, 1, name='output')
# 评价函数
correct_pred = tf.equal(output, tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


print('结束0...')
saver = tf.train.Saver()
with tf.Session() as sess:
    print('结束1...')
    init = tf.global_variables_initializer()
    sess.run(init)
    coord=tf.train.Coordinator()
    threads= tf.train.start_queue_runners(sess=sess, coord=coord)
    for step in range(training_iters):# 每轮训练都把所有批次的数据训练一遍
        print('训练轮数...', str(step))
        for batch in range(n_batch):
            img, lab = sess.run([image_batch, label_batch])
            _, acc = sess.run([train_op, accuracy], feed_dict={x: img, y: lab, keep_prob: dropout})
        
        print ("Iter " + str(step) + ", Training Accuracy = " + "{:.5f}".format(acc))  
        # 保存模型
        saver.save(sess,'dogs_cats_model/dog-cat.ckpt', global_step=step)

    print('结束2...')
    
    # 计算测试精度
    image_f, label_f = sess.run([image_valid, label_valid])
    final_acc = sess.run(accuracy, feed_dict={x:image_f, y:label_f, keep_prob: 1.})
    print ("Final Validating Accuracy:", final_acc)
    
    coord.request_stop()
    coord.join(threads)
    
    # 保存训练好的模型
    #形参output_node_names用于指定输出的节点名称,output_node_names=['output']对应pre_num=tf.argmax(y,1,name="output"),
    output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=['output'])
    f = tf.gfile.FastGFile('dogs_cats_model/curing.pb', mode='wb')#’wb’中w代表写文件，b代表将数据以二进制方式写入文件。
    f.write(output_graph_def.SerializeToString())
    f.close() # 关流以写入本地磁盘
    sess.close()
    
    
# 优化点：
    '''
    1.数据增强
    2.变学习率
    3.梯度下降函数
    
    
    '''
    
    
