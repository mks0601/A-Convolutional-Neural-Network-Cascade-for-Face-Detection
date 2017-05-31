import tensorflow as tf
import numpy as np

import param

def weight_variable(shape,name=None):
    initial = tf.truncated_normal(shape=shape, stddev=param.w_std)
    return tf.Variable(initial,name=name)

def bias_variable(shape,name=None):
    initial = tf.constant(value=param.b_init, shape=shape)
    return tf.Variable(initial,name=name)

def conv2d(x, W, stride, pad = 'SAME'):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=pad)

def max_pool(x, kernelSz, stride, pad = 'SAME'):
    return tf.nn.max_pool(x, ksize=[1, kernelSz, kernelSz, 1], strides=[1, stride, stride, 1], padding=pad)

class detect_12Net:

    def __init__(self,inputs,targets):
        
        #12-net
        with tf.variable_scope("12det_"):
            #conv layer 1
            self.w_conv1 = weight_variable([3,3,param.input_channel,16],"w1")
            self.b_conv1 = bias_variable([16],"b1")
            self.h_conv1 = tf.nn.relu(conv2d(inputs, self.w_conv1, 1) + self.b_conv1)

            #pooling layer 1
            self.h_pool1 =  max_pool(self.h_conv1, 3, 2)
            
            #fully conv layer 1
            self.w_conv2 = weight_variable([param.img_size_12//2,param.img_size_12//2,16,16],"w2")
            self.b_conv2 = bias_variable([16],"b2")
            self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.w_conv2, param.window_stride/2, pad="VALID") + self.b_conv2)

            #fully conv layer 2
            self.w_conv3 = weight_variable([1,1,16,1],"w3")
            self.b_conv3 = bias_variable([1],"b3")
            self.h_conv3 = tf.nn.sigmoid(conv2d(self.h_conv2, self.w_conv3, 1) + self.b_conv3)

            self.conv2_shape = tf.concat(0,[[-1],[tf.reduce_prod(tf.slice(tf.shape(self.h_conv2),[1],[3]),0)]])
            self.h_conv2_reshaped = tf.reshape(self.h_conv2,self.conv2_shape)

            self.conv3_shape = tf.concat(0,[[-1],[tf.reduce_prod(tf.slice(tf.shape(self.h_conv3),[1],[3]),0)]])
            self.h_conv3_reshaped = tf.reshape(self.h_conv3,self.conv3_shape)
        
        self.from_12 = self.h_conv2_reshaped
        self.prediction = self.h_conv3
        self.prediction_flatten = self.h_conv3_reshaped
        self.loss = tf.reduce_mean(tf.add(-tf.reduce_sum(targets * tf.log(self.prediction_flatten + 1e-9),1), -tf.reduce_sum((1-targets) * tf.log(1-self.prediction_flatten + 1e-9),1)))
        self.train_step = tf.train.GradientDescentOptimizer(param.lr).minimize(self.loss)  


class detect_24Net:

    def __init__(self,inputs,targets,from_12):

        #24-net
        with tf.variable_scope("24det_"):
            #conv layer 1
            self.w_conv1 = weight_variable([5,5,param.input_channel,64],"w1")
            self.b_conv1 = bias_variable([64],"b1")
            self.h_conv1 = tf.nn.relu(conv2d(inputs, self.w_conv1, 1) + self.b_conv1)

            #pooling layer 1
            self.h_pool1 =  max_pool(self.h_conv1, 3, 2)

            #fc layer 1
            self.w_fc1 =  weight_variable([param.img_size_24//2 * param.img_size_24//2 * 64, 128],"w2")
            self.b_fc1 =  bias_variable([128],"b2")
            self.h_pool1_reshaped = tf.reshape(self.h_pool1, [-1, param.img_size_24//2 * param.img_size_24//2 * 64])
            self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool1_reshaped, self.w_fc1) + self.b_fc1)

            #fc layer2
            self.w_fc2 =  weight_variable([128+16, 1],"w3")
            self.b_fc2 =  bias_variable([1],"b3")
            self.h_fc1_concat = tf.concat(1,[self.h_fc1,from_12])
            self.h_fc2 = tf.nn.sigmoid(tf.matmul(self.h_fc1_concat, self.w_fc2) + self.b_fc2)
       
        self.from_24 = self.h_fc1_concat
        self.prediction = self.h_fc2
        self.loss = tf.reduce_mean(tf.add(-tf.reduce_sum(targets * tf.log(self.prediction + 1e-9),1), -tf.reduce_sum((1-targets) * tf.log(1-self.prediction + 1e-9),1)))
        self.train_step = tf.train.GradientDescentOptimizer(param.lr).minimize(self.loss)  


class detect_48Net:

    def __init__(self,inputs,targets,from_24):
        
        #48-net
        with tf.variable_scope("48det_"):
            #conv layer 1
            self.w_conv1 = weight_variable([5,5,param.input_channel,64],"w1")
            self.b_conv1 = bias_variable([64],"b1")
            self.h_conv1 = tf.nn.relu(conv2d(inputs, self.w_conv1, 1) + self.b_conv1)

            #pooling layer 1
            self.h_pool1 =  max_pool(self.h_conv1, 3, 2)
            
            #conv layer 2
            self.w_conv2 = weight_variable([5,5,64,64],"w2")
            self.b_conv2 = bias_variable([64],"b2")
            self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.w_conv2, 1) + self.b_conv2)

            #pooling layer 2
            self.h_pool2 =  max_pool(self.h_conv2, 3, 2)

            #fc layer 1
            self.w_fc1 =  weight_variable([param.img_size_48//4 * param.img_size_48//4 * 64, 256],"w3")
            self.b_fc1 =  bias_variable([256],"b3")
            self.h_pool2_reshaped = tf.reshape(self.h_pool2, [-1, param.img_size_48//4 * param.img_size_48//4 * 64])
            self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_reshaped, self.w_fc1) + self.b_fc1)

            #fc layer2
            self.w_fc2 =  weight_variable([256+128+16, 1],"w4")
            self.b_fc2 =  bias_variable([1],"b4")
            self.h_fc1_concat = tf.concat(1,[self.h_fc1,from_24])
            self.h_fc2 = tf.nn.sigmoid(tf.matmul(self.h_fc1_concat, self.w_fc2) + self.b_fc2)
       
        self.prediction = self.h_fc2
        self.loss = tf.reduce_mean(tf.add(-tf.reduce_sum(targets * tf.log(self.prediction + 1e-9),1), -tf.reduce_sum((1-targets) * tf.log(1-self.prediction + 1e-9),1)))
        self.train_step = tf.train.GradientDescentOptimizer(param.lr).minimize(self.loss)  


class calib_12Net:

    def __init__(self,inputs,targets):
        
        #12-net
        with tf.variable_scope("12calib_"):
            #conv layer 1
            self.w_conv1 = weight_variable([3,3,param.input_channel,16],"w1")
            self.b_conv1 = bias_variable([16],"b1")
            self.h_conv1 = tf.nn.relu(conv2d(inputs, self.w_conv1, 1) + self.b_conv1)

            #pooling layer 1
            self.h_pool1 =  max_pool(self.h_conv1, 3, 2)

            #fc layer 1
            self.w_fc1 =  weight_variable([param.img_size_12//2 * param.img_size_12//2 * 16, 128],"w2")
            self.b_fc1 =  bias_variable([128],"b2")
            self.h_pool1_reshaped = tf.reshape(self.h_pool1, [-1, param.img_size_12//2 * param.img_size_12//2 * 16])
            self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool1_reshaped, self.w_fc1) + self.b_fc1)

            #fc layer2
            self.w_fc2 =  weight_variable([128, param.cali_patt_num],"w3")
            self.b_fc2 =  bias_variable([param.cali_patt_num],"b3")
            self.h_fc2 = tf.nn.softmax(tf.matmul(self.h_fc1, self.w_fc2) + self.b_fc2)
       
        self.prediction = self.h_fc2
        self.loss = tf.reduce_mean(-tf.reduce_sum(targets * tf.log(self.prediction + 1e-9),1))
        self.train_step = tf.train.GradientDescentOptimizer(param.lr).minimize(self.loss)    


class calib_24Net:

    def __init__(self,inputs,targets):
        
        #24-net
        with tf.variable_scope("24calib_"):
            #conv layer 1
            self.w_conv1 = weight_variable([5,5,param.input_channel,32],"w1")
            self.b_conv1 = bias_variable([32],"b1")
            self.h_conv1 = tf.nn.relu(conv2d(inputs, self.w_conv1, 1) + self.b_conv1)

            #pooling layer 1
            self.h_pool1 =  max_pool(self.h_conv1, 3, 2)

            #fc layer 1
            self.w_fc1 =  weight_variable([param.img_size_24//2 * param.img_size_24//2 * 32, 64],"w2")
            self.b_fc1 =  bias_variable([64],"b2")
            self.h_pool1_reshaped = tf.reshape(self.h_pool1, [-1, param.img_size_24//2 * param.img_size_24//2 * 32])
            self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool1_reshaped, self.w_fc1) + self.b_fc1)

            #fc layer2
            self.w_fc2 =  weight_variable([64, param.cali_patt_num],"w3")
            self.b_fc2 =  bias_variable([param.cali_patt_num],"b3")
            self.h_fc2 = tf.nn.softmax(tf.matmul(self.h_fc1, self.w_fc2) + self.b_fc2)
       
        self.prediction = self.h_fc2
        self.loss = tf.reduce_mean(-tf.reduce_sum(targets * tf.log(self.prediction + 1e-9),1))
        self.train_step = tf.train.GradientDescentOptimizer(param.lr).minimize(self.loss)    


class calib_48Net:

    def __init__(self,inputs,targets):
        
        #24-net
        with tf.variable_scope("48calib_"):
            #conv layer 1
            self.w_conv1 = weight_variable([5,5,param.input_channel,64],"w1")
            self.b_conv1 = bias_variable([64],"b1")
            self.h_conv1 = tf.nn.relu(conv2d(inputs, self.w_conv1, 1) + self.b_conv1)

            #pooling layer 1
            self.h_pool1 =  max_pool(self.h_conv1, 3, 2)
            
            #conv layer 2
            self.w_conv2 = weight_variable([5,5,64,64],"w2")
            self.b_conv2 = bias_variable([64],"b2")
            self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.w_conv2, 1) + self.b_conv2)

            #fc layer 1
            self.w_fc1 =  weight_variable([param.img_size_48//2 * param.img_size_48//2 * 64, 256],"w3")
            self.b_fc1 =  bias_variable([256],"b3")
            self.h_conv2_reshaped = tf.reshape(self.h_conv2, [-1, param.img_size_48//2 * param.img_size_48//2 * 64])
            self.h_fc1 = tf.nn.relu(tf.matmul(self.h_conv2_reshaped, self.w_fc1) + self.b_fc1)

            #fc layer2
            self.w_fc2 =  weight_variable([256, param.cali_patt_num],"w4")
            self.b_fc2 =  bias_variable([param.cali_patt_num],"b4")
            self.h_fc2 = tf.nn.softmax(tf.matmul(self.h_fc1, self.w_fc2) + self.b_fc2)
       
        self.prediction = self.h_fc2
        self.loss = tf.reduce_mean(-tf.reduce_sum(targets * tf.log(self.prediction + 1e-9),1))
        self.train_step = tf.train.GradientDescentOptimizer(param.lr).minimize(self.loss)    


