import numpy as np
import tensorflow as tf
import time
import random
import math
from skimage.util.shape import view_as_windows
from skimage.transform import pyramid_gaussian
import Image
import ImageDraw
from compiler.ast import flatten
import sys

import etc
import load_db



train_db = load_db.load_db_cali_train()

sess = tf.InteractiveSession()

y_target = tf.placeholder("float", [None,etc.cali_patt_num])

#12-net
#conv layer 1
x_12 = tf.placeholder("float", [None, etc.img_size_12 * etc.img_size_12 * etc.input_channel])
W_conv1_12_cali = etc.weight_variable([3,3,3,16],'calib_wc1_12')
b_conv1_12_cali = etc.bias_variable([16],'calib_bc1_12')
x_12_reshaped = tf.reshape(x_12, [-1, etc.img_size_12, etc.img_size_12, etc.input_channel])
h_conv1_12 = tf.nn.relu(etc.conv2d(x_12_reshaped, W_conv1_12_cali) + b_conv1_12_cali)

#pooling layer 1
h_pool1_12 = etc.max_pool_3x3(h_conv1_12)

#fully layer 1
W_fc1_12_cali = etc.weight_variable([6 * 6 * 16, 128],'calib_wfc1_12')
b_fc1_12_cali = etc.bias_variable([128],'calib_bfc1_12')
h_pool1_12_reshaped = tf.reshape(h_pool1_12, [-1, 6 * 6 * 16])
h_fc1_12 = tf.nn.relu(tf.matmul(h_pool1_12_reshaped, W_fc1_12_cali) + b_fc1_12_cali)

#fully layer2
W_fc2_12_cali = etc.weight_variable([128, etc.cali_patt_num],'calib_wfc2_12')
b_fc2_12_cali = etc.bias_variable([etc.cali_patt_num],'calib_bfc2_12')
h_fc2_12 = tf.nn.softmax(tf.matmul(h_fc1_12, W_fc2_12_cali) + b_fc2_12_cali)



#24-net
x_24 = tf.placeholder("float", [None, etc.img_size_24 * etc.img_size_24 * etc.input_channel])
#conv layer 1
W_conv1_24_cali = etc.weight_variable([5,5,3,32],'calib_wc1_24')
b_conv1_24_cali = etc.bias_variable([32],'calib_bc1_24')
x_24_reshaped = tf.reshape(x_24, [-1, etc.img_size_24, etc.img_size_24, etc.input_channel])
h_conv1_24 = tf.nn.relu(etc.conv2d(x_24_reshaped, W_conv1_24_cali) + b_conv1_24_cali)

#pooling layer 1
h_pool1_24 = etc.max_pool_3x3(h_conv1_24)

#fully layer 1
W_fc1_24_cali = etc.weight_variable([12 * 12 * 32, 64],'calib_wfc1_24')
b_fc1_24_cali = etc.bias_variable([64],'calib_bfc1_24')
h_pool1_24_reshaped = tf.reshape(h_pool1_24, [-1, 12 * 12 * 32])
h_fc1_24 = tf.nn.relu(tf.matmul(h_pool1_24_reshaped, W_fc1_24_cali) + b_fc1_24_cali)

#fully layer 2
W_fc2_24_cali = etc.weight_variable([64,etc.cali_patt_num],'calib_wfc2_24')
b_fc2_24_cali = etc.bias_variable([etc.cali_patt_num],'calib_bfc2_24')
h_fc2_24 = tf.nn.softmax(tf.matmul(h_fc1_24, W_fc2_24_cali) + b_fc2_24_cali)


#48-net
x_48 = tf.placeholder("float", [None, etc.img_size_48 * etc.img_size_48 * etc.input_channel])
#conv layer 1
W_conv1_48_cali = etc.weight_variable([5,5,3,64],'calib_wc1_48')
b_conv1_48_cali = etc.bias_variable([64],'calib_bc1_48')
x_48_reshaped = tf.reshape(x_48, [-1, etc.img_size_48, etc.img_size_48, etc.input_channel])
h_conv1_48 = tf.nn.relu(etc.conv2d(x_48_reshaped, W_conv1_48_cali) + b_conv1_48_cali)

#pooling layer 1
h_pool1_48 = etc.max_pool_3x3(h_conv1_48)

#normalization layer 1

#conv layer 2
W_conv2_48_cali = etc.weight_variable([5,5,64,64],'calib_wc2_48')
b_conv2_48_cali = etc.bias_variable([64],'calib_bc2_48')
h_conv2_48 = tf.nn.relu(etc.conv2d(h_pool1_48, W_conv2_48_cali) + b_conv2_48_cali)

#normalization layer 2


#fully layer 1
W_fc1_48_cali = etc.weight_variable([24 * 24 * 64, 256],'calib_wfc1_48')
b_fc1_48_cali = etc.bias_variable([256],'calib_bfc1_48')
h_conv2_48_reshaped = tf.reshape(h_conv2_48, [-1, 24 * 24 * 64])
h_fc1_48 = tf.nn.relu(tf.matmul(h_conv2_48_reshaped, W_fc1_48_cali) + b_fc1_48_cali)

#fully layer 2
W_fc2_48_cali = etc.weight_variable([256, etc.cali_patt_num],'calib_wfc2_48')
b_fc2_48_cali = etc.bias_variable([etc.cali_patt_num],'calib_bfc2_48')
h_fc2_48 = tf.nn.softmax(tf.matmul(h_fc1_48, W_fc2_48_cali) + b_fc2_48_cali)


loss_12 = tf.reduce_mean(-tf.reduce_sum(y_target * tf.log(h_fc2_12 + 1e-9),1))
train_step_12 = tf.train.GradientDescentOptimizer(etc.lr).minimize(loss_12)    
accuracy_12 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(h_fc2_12,1), tf.argmax(y_target,1)), "float"))

loss_24 = tf.reduce_mean(-tf.reduce_sum(y_target * tf.log(h_fc2_24 + 1e-9),1))
train_step_24 = tf.train.GradientDescentOptimizer(etc.lr).minimize(loss_24)    
accuracy_24 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(h_fc2_24,1), tf.argmax(y_target,1)), "float"))

loss_48 = tf.reduce_mean(-tf.reduce_sum(y_target * tf.log(h_fc2_48 + 1e-9),1))
train_step_48 = tf.train.GradientDescentOptimizer(etc.lr).minimize(loss_48)    
accuracy_48 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(h_fc2_48,1), tf.argmax(y_target,1)), "float"))

sess.run(tf.initialize_all_variables())

dim_12 = etc.input_channel * etc.img_size_12 * etc.img_size_12
dim_24 = etc.input_channel * etc.img_size_24 * etc.img_size_24
dim_48 = etc.input_channel * etc.img_size_48 * etc.img_size_48

db_12 = np.zeros((etc.mini_batch,dim_12), np.float32)
db_24 = np.zeros((etc.mini_batch,dim_24), np.float32)
db_48 = np.zeros((etc.mini_batch,dim_48), np.float32)
lb = np.zeros((etc.mini_batch, etc.cali_patt_num), np.float32)

acc_db_12 = np.zeros((etc.acc_bench_num,dim_12),np.float32)
acc_db_24 = np.zeros((etc.acc_bench_num,dim_24),np.float32)
acc_db_48 = np.zeros((etc.acc_bench_num,dim_48),np.float32)
acc_lb = np.zeros((etc.acc_bench_num,etc.cali_patt_num),np.float32)

for k in tf.all_variables():
    print k.name

for cascade_lv in xrange(etc.cascade_level):

    print "Training start!"
    train_start = time.time()
    fp_loss = open("./result/loss_cali_" + str(cascade_lv) + "_.txt", "w")
     
    for e in xrange(etc.epoch_calib_num[cascade_lv]):
        
        fp_conf_mat = open("./result/conf_mat_cali_" + str(e) + "_.txt","w")  
        epoch_start = time.time()
        average_loss = 0
        
        for b_iter in xrange(etc.batch_iter):
            
            db_id = random.sample(xrange(len(train_db)),etc.mini_batch)
    
            if cascade_lv == 0:
                
                lb[:,:] = 0.0
                for id_,did in enumerate(db_id):
                    patch = etc.img2array(train_db[did][0],etc.img_size_12)
                    calib_idx = train_db[did][1]

                    db_12[id_,:] = patch
                    lb[id_,calib_idx] = 1.0

                zipped = zip(db_12, lb)
                np.random.shuffle(zipped)
                X_12 = [elem[0] for elem in zipped]
                Y = [elem[1] for elem in zipped]
                X_12 = np.asarray(X_12)
                Y = np.reshape(np.asarray(Y),(np.shape(X_12)[0],etc.cali_patt_num))
                                
                average_loss += loss_12.eval(feed_dict = {x_12:X_12, y_target:Y})
                train_step_12.run(feed_dict = {x_12:X_12, y_target:Y})
            elif cascade_lv == 1:
                lb[:,:] = 0.0
                for id_,did in enumerate(db_id):
                    patch = etc.img2array(train_db[did][0],etc.img_size_24)
                    calib_idx = train_db[did][1]

                    db_24[id_,:] = patch
                    lb[id_,calib_idx] = 1.0

                zipped = zip(db_24,lb)
                np.random.shuffle(zipped)
                X_24 = [elem[0] for elem in zipped]
                Y = [elem[1] for elem in zipped]
                X_24 = np.asarray(X_24)
                Y = np.reshape(np.asarray(Y),(np.shape(X_24)[0],etc.cali_patt_num))

                average_loss += loss_24.eval(feed_dict = {x_24:X_24, y_target:Y})
                train_step_24.run(feed_dict = {x_24:X_24, y_target:Y})
            elif cascade_lv == 2:
                lb[:,:] = 0.0
                for id_,did in enumerate(db_id):
                    patch = etc.img2array(train_db[did][0],etc.img_size_48)
                    calib_idx = train_db[did][1]

                    db_48[id_,:] = patch
                    lb[id_,calib_idx] = 1.0

                zipped = zip(db_48, lb)
                np.random.shuffle(zipped)
                X_48 = [elem[0] for elem in zipped]
                Y = [elem[1] for elem in zipped]
                X_48 = np.asarray(X_48)
                Y = np.reshape(np.asarray(Y),(np.shape(X_48)[0],etc.cali_patt_num))

                average_loss += loss_48.eval(feed_dict = {x_48: X_48, y_target:Y})
                train_step_48.run(feed_dict = {x_48:X_48, y_target:Y})

            if b_iter > 0 and b_iter % etc.result_interval == 0: 
                print "cas_lv: " +  str(cascade_lv) + " epoch: " + str(e) + " db_num: " + str(b_iter) + "/" + str(etc.batch_iter) + " avg_loss: " + str(average_loss / b_iter)
              

        epoch_finish = time.time()
        average_loss /= etc.batch_iter
        
        db_id = random.sample(xrange(len(train_db)),etc.acc_bench_num)

        if cascade_lv == 0:
            acc_lb[:,:] = 0.0
            for id_,did in enumerate(db_id):
                patch = etc.img2array(train_db[did][0],etc.img_size_12)
                calib_idx = train_db[did][1]

                acc_db_12[id_,:] = patch
                acc_lb[id_,calib_idx] = 1.0

            X_12 = acc_db_12
            Y = acc_lb
            accuracy_eval = accuracy_12.eval(feed_dict={x_12:X_12, y_target:Y})
            output = h_fc2_12.eval(feed_dict={x_12:X_12})
            for n in xrange(len(output)):
                fp_conf_mat.write("(" + str(output[n]) + ", " + str(Y[n,:]) + ")\n")
        elif cascade_lv == 1:
            acc_lb[:,:] = 0.0
            for id_,did in enumerate(db_id):
                patch = etc.img2array(train_db[did][0],etc.img_size_24)
                calib_idx = train_db[did][1]

                acc_db_24[id_,:] = patch
                acc_lb[id_,calib_idx] = 1.0

            X_24 = acc_db_24
            Y = acc_lb

            accuracy_eval = accuracy_24.eval(feed_dict={x_24:X_24, y_target:Y})
            output = h_fc2_24.eval(feed_dict={x_24:X_24})
            for n in xrange(len(output)):
                fp_conf_mat.write("(" + str(output[n]) + ", " + str(Y[n,:]) + ")\n")
        else:
            acc_lb[:,:] = 0.0
            for id_,did in enumerate(db_id):
                patch = etc.img2array(train_db[did][0],etc.img_size_48)
                calib_idx = train_db[did][1]

                acc_db_48[id_,:] = patch
                acc_lb[id_,calib_idx] = 1.0

            X_48 = acc_db_48
            Y = acc_lb


            accuracy_eval = accuracy_48.eval(feed_dict={x_48:X_48, y_target:Y})
            output = h_fc2_48.eval(feed_dict={x_48:X_48})
            for n in xrange(len(output)):
                fp_conf_mat.write("(" + str(output[n]) + ", " + str(Y[n,:]) + ")\n")
            
        
        print "Accuracy: ", accuracy_eval
        print "Time: ", epoch_finish - epoch_start
        print "Loss: ", average_loss

        fp_loss.write(str(average_loss)+"\n")
    
    train_finish = time.time()
    print train_finish - train_start, "secs for training"
    fp_loss.close()
    
    saver = tf.train.Saver()
    
    if cascade_lv == 0:
        saver.save(sess, etc.save_dir + "12-net_calib.ckpt")
    elif cascade_lv == 1:
        saver.save(sess, etc.save_dir + "24-net_calib.ckpt")
    else:
        saver.save(sess, etc.save_dir + "48-net_calib.ckpt")
     
    
