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



train_db = load_db.load_db_detect_train()

pos_db = train_db[0]
neg_db = train_db[1]
neg_img = train_db[2]

sess = tf.InteractiveSession()

thr = tf.placeholder("float")
y_target = tf.placeholder("float", [None,1])

#12-net
#conv layer 1
x_12 = tf.placeholder("float", [None, etc.img_size_12 * etc.img_size_12 * etc.input_channel])
W_conv1_12 = etc.weight_variable([3,3,3,16])
b_conv1_12 = etc.bias_variable([16])
x_12_reshaped = tf.reshape(x_12, [-1, etc.img_size_12, etc.img_size_12, etc.input_channel])
h_conv1_12 = tf.nn.relu(etc.conv2d(x_12_reshaped, W_conv1_12) + b_conv1_12)

#pooling layer 1
h_pool1_12 = etc.max_pool_3x3(h_conv1_12)

#fully layer 1
W_fc1_12 = etc.weight_variable([6 * 6 * 16, 16])
b_fc1_12 = etc.bias_variable([16])
h_pool1_12_reshaped = tf.reshape(h_pool1_12, [-1, 6 * 6 * 16])
h_fc1_12 = tf.nn.relu(tf.matmul(h_pool1_12_reshaped, W_fc1_12) + b_fc1_12)

#fully layer2
W_fc2_12 = etc.weight_variable([16, 1])
b_fc2_12 = etc.bias_variable([1])
h_fc2_12 = tf.nn.sigmoid(tf.matmul(h_fc1_12, W_fc2_12) + b_fc2_12)



#24-net
x_24 = tf.placeholder("float", [None, etc.img_size_24 * etc.img_size_24 * etc.input_channel])
#conv layer 1
W_conv1_24 = etc.weight_variable([5,5,3,64])
b_conv1_24 = etc.bias_variable([64])
x_24_reshaped = tf.reshape(x_24, [-1, etc.img_size_24, etc.img_size_24, etc.input_channel])
h_conv1_24 = tf.nn.relu(etc.conv2d(x_24_reshaped, W_conv1_24) + b_conv1_24)

#pooling layer 1
h_pool1_24 = etc.max_pool_3x3(h_conv1_24)

#fully layer 1
W_fc1_24 = etc.weight_variable([12 * 12 * 64, 128])
b_fc1_24 = etc.bias_variable([128])
h_pool1_24_reshaped = tf.reshape(h_pool1_24, [-1, 12 * 12 * 64])
h_fc1_24 = tf.nn.relu(tf.matmul(h_pool1_24_reshaped, W_fc1_24) + b_fc1_24)

#fully layer 2
from_12 = tf.placeholder("float", [None, 16])
W_fc2_24 = etc.weight_variable([128 + 16,1])
b_fc2_24 = etc.bias_variable([1])
h_fc1_24_concat = tf.concat(1, [h_fc1_24, from_12])
h_fc2_24 = tf.nn.sigmoid(tf.matmul(h_fc1_24_concat, W_fc2_24) + b_fc2_24)


#48-net
x_48 = tf.placeholder("float", [None, etc.img_size_48 * etc.img_size_48 * etc.input_channel])
#conv layer 1
W_conv1_48 = etc.weight_variable([5,5,3,64])
b_conv1_48 = etc.bias_variable([64])
x_48_reshaped = tf.reshape(x_48, [-1, etc.img_size_48, etc.img_size_48, etc.input_channel])
h_conv1_48 = tf.nn.relu(etc.conv2d(x_48_reshaped, W_conv1_48) + b_conv1_48)

#pooling layer 1
h_pool1_48 = etc.max_pool_3x3(h_conv1_48)

#normalization layer 1

#conv layer 2
W_conv2_48 = etc.weight_variable([5,5,64,64])
b_conv2_48 = etc.bias_variable([64])
h_conv2_48 = tf.nn.relu(etc.conv2d(h_pool1_48, W_conv2_48) + b_conv2_48)

#normalization layer 2

#pooling layer 2
h_pool2_48 = etc.max_pool_3x3(h_conv2_48)

#fully layer 1
W_fc1_48 = etc.weight_variable([12 * 12 * 64, 256])
b_fc1_48 = etc.bias_variable([256])
h_pool2_48_reshaped = tf.reshape(h_pool2_48, [-1, 12 * 12 * 64])
h_fc1_48 = tf.nn.relu(tf.matmul(h_pool2_48_reshaped, W_fc1_48) + b_fc1_48)

#fully layer 2
from_24= tf.placeholder("float", [None,128])
W_fc2_48 = etc.weight_variable([256 + 128 + 16, 1])
b_fc2_48 = etc.bias_variable([1])
h_fc1_48_concat = tf.concat(1, [from_24, from_12, h_fc1_48])
h_fc2_48 = tf.sigmoid(tf.matmul(h_fc1_48_concat, W_fc2_48) + b_fc2_48)


loss_12 = tf.reduce_mean(tf.div(tf.add(-tf.reduce_sum(y_target * tf.log(h_fc2_12 + 1e-9),1), -tf.reduce_sum((1-y_target) * tf.log(1-h_fc2_12 + 1e-9),1)),2))
train_step_12 = tf.train.GradientDescentOptimizer(etc.lr).minimize(loss_12)    
thresholding_12 = tf.cast(tf.greater(h_fc2_12, thr), "float")
recall_12 = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(thresholding_12, tf.constant([1.0])), tf.equal(y_target, tf.constant([1.0]))), "float")) / tf.reduce_sum(y_target)
accuracy_12 = tf.reduce_mean(tf.cast(tf.equal(thresholding_12, y_target), "float"))

loss_24 = tf.reduce_mean(tf.div(tf.add(-tf.reduce_sum(y_target * tf.log(h_fc2_24 + 1e-9),1), -tf.reduce_sum((1-y_target) * tf.log(1-h_fc2_24 + 1e-9),1)),2))
train_step_24 = tf.train.GradientDescentOptimizer(etc.lr).minimize(loss_24)    
thresholding_24 = tf.cast(tf.greater(h_fc2_24, thr), "float")
recall_24 = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(thresholding_24, tf.constant([1.0])), tf.equal(y_target, tf.constant([1.0]))), "float")) / tf.reduce_sum(y_target)
accuracy_24 = tf.reduce_mean(tf.cast(tf.equal(thresholding_24, y_target), "float"))

loss_48 = tf.reduce_mean(tf.div(tf.add(-tf.reduce_sum(y_target * tf.log(h_fc2_48 + 1e-9),1), -tf.reduce_sum((1-y_target) * tf.log(1-h_fc2_48 + 1e-9),1)),2))
train_step_48 = tf.train.GradientDescentOptimizer(etc.lr).minimize(loss_48)    
thresholding_48 = tf.cast(tf.greater(h_fc2_48, thr), "float")
recall_48 = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(thresholding_48, tf.constant([1.0])), tf.equal(y_target, tf.constant([1.0]))), "float")) / tf.reduce_sum(y_target)
accuracy_48 = tf.reduce_mean(tf.cast(tf.equal(thresholding_48, y_target), "float"))


#########calib-net
#12-net
#conv layer 1
x_12_cali = tf.placeholder("float", [None, etc.img_size_12 * etc.img_size_12 * etc.input_channel])
W_conv1_12_cali = etc.weight_variable([3,3,3,16],'calib_wc1_12')
b_conv1_12_cali = etc.bias_variable([16],'calib_bc1_12')
x_12_reshaped_cali = tf.reshape(x_12_cali, [-1, etc.img_size_12, etc.img_size_12, etc.input_channel])
h_conv1_12_cali = tf.nn.relu(etc.conv2d(x_12_reshaped_cali, W_conv1_12_cali) + b_conv1_12_cali)

#pooling layer 1
h_pool1_12_cali = etc.max_pool_3x3(h_conv1_12_cali)

#fully layer 1
W_fc1_12_cali = etc.weight_variable([6 * 6 * 16, 128],'calib_wfc1_12')
b_fc1_12_cali = etc.bias_variable([128],'calib_bfc1_12')
h_pool1_12_reshaped_cali = tf.reshape(h_pool1_12_cali, [-1, 6 * 6 * 16])
h_fc1_12_cali = tf.nn.relu(tf.matmul(h_pool1_12_reshaped_cali, W_fc1_12_cali) + b_fc1_12_cali)

#fully layer2
W_fc2_12_cali = etc.weight_variable([128, etc.cali_patt_num],'calib_wfc2_12')
b_fc2_12_cali = etc.bias_variable([etc.cali_patt_num],'calib_bfc2_12')
h_fc2_12_cali = tf.nn.softmax(tf.matmul(h_fc1_12_cali, W_fc2_12_cali) + b_fc2_12_cali)



#24-net
x_24_cali = tf.placeholder("float", [None, etc.img_size_24 * etc.img_size_24 * etc.input_channel])
#conv layer 1
W_conv1_24_cali = etc.weight_variable([5,5,3,32],'calib_wc1_24')
b_conv1_24_cali = etc.bias_variable([32],'calib_bc1_24')
x_24_reshaped_cali = tf.reshape(x_24_cali, [-1, etc.img_size_24, etc.img_size_24, etc.input_channel])
h_conv1_24_cali = tf.nn.relu(etc.conv2d(x_24_reshaped_cali, W_conv1_24_cali) + b_conv1_24_cali)

#pooling layer 1
h_pool1_24_cali = etc.max_pool_3x3(h_conv1_24_cali)

#fully layer 1
W_fc1_24_cali = etc.weight_variable([12 * 12 * 32, 64],'calib_wfc1_24')
b_fc1_24_cali = etc.bias_variable([64],'calib_bfc1_24')
h_pool1_24_reshaped_cali = tf.reshape(h_pool1_24_cali, [-1, 12 * 12 * 32])
h_fc1_24_cali = tf.nn.relu(tf.matmul(h_pool1_24_reshaped_cali, W_fc1_24_cali) + b_fc1_24_cali)

#fully layer 2
W_fc2_24_cali = etc.weight_variable([64,etc.cali_patt_num],'calib_wfc2_24')
b_fc2_24_cali = etc.bias_variable([etc.cali_patt_num],'calib_bfc2_24')
h_fc2_24_cali = tf.nn.softmax(tf.matmul(h_fc1_24_cali, W_fc2_24_cali) + b_fc2_24_cali)


sess.run(tf.initialize_all_variables())

thr_12 = 5e-3
thr_24 = 1e-9

db_12 = np.zeros((etc.mini_batch,etc.dim_12), np.float32)
db_24 = np.zeros((etc.mini_batch,etc.dim_24), np.float32)
db_48 = np.zeros((etc.mini_batch,etc.dim_48), np.float32)
lb = np.zeros((etc.mini_batch, 1), np.float32)


for cascade_lv in xrange(1,etc.cascade_level):
    
    if cascade_lv > 1:
        print "Training start!"
        train_start = time.time()
        fp_loss = open("./result/loss_" + str(cascade_lv) + "_.txt", "w")
         
        for e in xrange(etc.epoch_num[cascade_lv]):
            
            fp_conf_mat = open("./result/conf_mat_" + str(e) + "_.txt","w")  
            epoch_start = time.time()
            average_loss = 0
            
            for b_iter in xrange(etc.batch_iter):

                pos_id = random.sample(xrange(len(pos_db)),etc.pos_batch)
                neg_id = random.sample(xrange(len(neg_db)),etc.neg_batch)
        
                if cascade_lv == 0:
                    db_12[:etc.pos_batch,:] = pos_db[pos_id,:etc.dim_12]
                    db_12[etc.pos_batch:,:] = neg_db[neg_id,:etc.dim_12]
                    lb[:etc.pos_batch,:] = np.ones((etc.pos_batch,1),np.float32)
                    lb[etc.pos_batch:,:] = np.zeros((etc.neg_batch,1),np.float32)
                    zipped = zip(db_12, lb)
                    np.random.shuffle(zipped)
                    X_12 = [elem[0] for elem in zipped]
                    Y = [elem[1] for elem in zipped]
                    X_12 = np.asarray(X_12)
                    Y = np.reshape(np.asarray(Y),(np.shape(X_12)[0],1))
                    average_loss += loss_12.eval(feed_dict = {x_12:X_12, y_target:Y})
                    train_step_12.run(feed_dict = {x_12:X_12, y_target:Y})
                elif cascade_lv == 1:
                    db_12[:etc.pos_batch,:] = pos_db[pos_id,:etc.dim_12]
                    db_24[:etc.pos_batch,:] = pos_db[pos_id,etc.dim_12:etc.dim_12+etc.dim_24]
                    for eid,id_ in enumerate(neg_id):
                        neg_12 = etc.img2array(neg_db[id_],etc.img_size_12)
                        neg_24 = etc.img2array(neg_db[id_],etc.img_size_24)
                        db_12[etc.pos_batch+eid,:] = neg_12
                        db_24[etc.pos_batch+eid,:] = neg_24
                    lb[:etc.pos_batch,:] = np.ones((etc.pos_batch,1),np.float32)
                    lb[etc.pos_batch:,:] = np.zeros((etc.neg_batch,1),np.float32)
                    zipped = zip(db_12, db_24,lb)
                    np.random.shuffle(zipped)
                    X_12 = [elem[0] for elem in zipped]
                    X_24 = [elem[1] for elem in zipped]
                    Y = [elem[2] for elem in zipped]
                    X_12 = np.asarray(X_12)
                    X_24 = np.asarray(X_24)
                    Y = np.reshape(np.asarray(Y),(np.shape(X_12)[0],1))

                    FROM_12 = h_fc1_12.eval(feed_dict = {x_12:X_12})
                    average_loss += loss_24.eval(feed_dict = {from_12:FROM_12, x_24:X_24, y_target:Y})
                    train_step_24.run(feed_dict = {from_12:FROM_12, x_24:X_24, y_target:Y})
                elif cascade_lv == 2:
                    db_12[:etc.pos_batch,:] = pos_db[pos_id,:etc.dim_12]
                    db_24[:etc.pos_batch,:] = pos_db[pos_id,etc.dim_12:etc.dim_12+etc.dim_24]
                    db_48[:etc.pos_batch,:] = pos_db[pos_id,etc.dim_12+etc.dim_24:]
                    for eid,id_ in enumerate(neg_id):
                        neg_12 = etc.img2array(neg_db[id_],etc.img_size_12)
                        neg_24 = etc.img2array(neg_db[id_],etc.img_size_24)
                        neg_48 = etc.img2array(neg_db[id_],etc.img_size_48)
                        db_12[etc.pos_batch+eid,:] = neg_12
                        db_24[etc.pos_batch+eid,:] = neg_24
                        db_48[etc.pos_batch+eid,:] = neg_48
                    lb[:etc.pos_batch,:] = np.ones((etc.pos_batch,1),np.float32)
                    lb[etc.pos_batch:,:] = np.zeros((etc.neg_batch,1),np.float32)
                    zipped = zip(db_12, db_24, db_48, lb)
                    np.random.shuffle(zipped)
                    X_12 = [elem[0] for elem in zipped]
                    X_24 = [elem[1] for elem in zipped]
                    X_48 = [elem[2] for elem in zipped]
                    Y = [elem[3] for elem in zipped]
                    X_12 = np.asarray(X_12)
                    X_24 = np.asarray(X_24)
                    X_48 = np.asarray(X_48)
                    Y = np.reshape(np.asarray(Y),(np.shape(X_12)[0],1))

                    FROM_12 = h_fc1_12.eval(feed_dict = {x_12:X_12})
                    FROM_24 = h_fc1_24.eval(feed_dict = {x_24:X_24})
                    average_loss += loss_48.eval(feed_dict = {from_12:FROM_12, from_24:FROM_24, x_48: X_48, y_target:Y})
                    train_step_48.run(feed_dict = {from_12:FROM_12, from_24:FROM_24, x_48:X_48, y_target:Y})

                if b_iter > 0 and b_iter % etc.result_interval == 0: 
                    print "cas_lv: " +  str(cascade_lv) + " epoch: " + str(e) + " db_num: " + str(b_iter) + "/" + str(etc.batch_iter) + " avg_loss: " + str(average_loss / b_iter)
                  

            epoch_finish = time.time()
            average_loss /= etc.batch_iter
            

            #test each epoch
            pos_id = random.sample(xrange(len(pos_db)),etc.acc_bench_num)
            neg_id = random.sample(xrange(len(neg_db)),etc.acc_bench_num)

            if cascade_lv == 0:
                X_12 = np.append(pos_db[pos_id,:etc.dim_12], neg_db[neg_id,:etc.dim_12], axis = 0)
                Y = np.reshape(np.append(np.ones((etc.acc_bench_num),np.float32), np.zeros((etc.acc_bench_num),np.float32), axis = 0), (np.shape(X_12)[0],1))
                accuracy_eval = accuracy_12.eval(feed_dict={x_12:X_12, y_target:Y, thr:0.5})
                output = h_fc2_12.eval(feed_dict={x_12:X_12})
                for n in xrange(len(output)):
                    fp_conf_mat.write("(" + str(output[n]) + ", " + str(Y[n,0]) + ")\n")
            elif cascade_lv == 1:
                neg_acc_12 = np.zeros((etc.acc_bench_num,etc.dim_12),np.float32)
                neg_acc_24 = np.zeros((etc.acc_bench_num,etc.dim_24),np.float32)
                for eid,id_ in enumerate(neg_id):
                        neg_12 = etc.img2array(neg_db[id_],etc.img_size_12)
                        neg_24 = etc.img2array(neg_db[id_],etc.img_size_24)
                        neg_acc_12[eid,:] = neg_12
                        neg_acc_24[eid,:] = neg_24
                X_12 = np.append(pos_db[pos_id,:etc.dim_12], neg_acc_12, axis = 0)
                X_24 = np.append(pos_db[pos_id,etc.dim_12:etc.dim_12+etc.dim_24], neg_acc_24, axis = 0)
                Y = np.reshape(np.append(np.ones((etc.acc_bench_num),np.float32), np.zeros((etc.acc_bench_num),np.float32), axis = 0), (np.shape(X_12)[0],1))

                FROM_12 = h_fc1_12.eval(feed_dict = {x_12:X_12})
                accuracy_eval = accuracy_24.eval(feed_dict={from_12:FROM_12, x_24:X_24, y_target:Y, thr:0.5})
                output = h_fc2_24.eval(feed_dict={from_12:FROM_12,x_24:X_24})
                for n in xrange(len(output)):
                    fp_conf_mat.write("(" + str(output[n]) + ", " + str(Y[n,0]) + ")\n")
            else:
                neg_acc_12 = np.zeros((etc.acc_bench_num,etc.dim_12),np.float32)
                neg_acc_24 = np.zeros((etc.acc_bench_num,etc.dim_24),np.float32)
                neg_acc_48 = np.zeros((etc.acc_bench_num,etc.dim_48),np.float32)
                for eid,id_ in enumerate(neg_id):
                        neg_12 = etc.img2array(neg_db[id_],etc.img_size_12)
                        neg_24 = etc.img2array(neg_db[id_],etc.img_size_24)
                        neg_48 = etc.img2array(neg_db[id_],etc.img_size_48)
                        neg_acc_12[eid,:] = neg_12
                        neg_acc_24[eid,:] = neg_24
                        neg_acc_48[eid,:] = neg_48
                
                X_12 = np.append(pos_db[pos_id,:etc.dim_12], neg_acc_12, axis = 0)
                X_24 = np.append(pos_db[pos_id,etc.dim_12:etc.dim_12+etc.dim_24], neg_acc_24, axis = 0)
                X_48 = np.append(pos_db[pos_id,etc.dim_12+etc.dim_24:], neg_acc_48,axis = 0)
                Y = np.reshape(np.append(np.ones((etc.acc_bench_num),np.float32), np.zeros((etc.acc_bench_num),np.float32), axis = 0), (np.shape(X_12)[0],1))

                FROM_12 = h_fc1_12.eval(feed_dict = {x_12:X_12})
                FROM_24 = h_fc1_24.eval(feed_dict = {x_24:X_24})
                accuracy_eval = accuracy_48.eval(feed_dict={from_12:FROM_12, from_24:FROM_24, x_48:X_48, y_target:Y, thr:0.5})
                output = h_fc2_48.eval(feed_dict={from_12:FROM_12, from_24:FROM_24,x_48:X_48})
                for n in xrange(len(output)):
                    fp_conf_mat.write("(" + str(output[n]) + ", " + str(Y[n,0]) + ")\n")
                
            
            print "Accuracy: ", accuracy_eval
            print "Time: ", epoch_finish - epoch_start
            print "Loss: ", average_loss

            fp_loss.write(str(average_loss)+"\n")
        
        
        train_finish = time.time()
        print train_finish - train_start, "secs for training"
        fp_loss.close()
        
        saver = tf.train.Saver()
        if cascade_lv == 0:
            saver.save(sess, etc.save_dir + "12-net.ckpt")
        elif cascade_lv == 1:
            saver.save(sess, etc.save_dir + "24-net.ckpt")
        else:
            saver.save(sess, etc.save_dir + "48-net.ckpt")
        
    saver_detect = tf.train.Saver({"Variable":W_conv1_12, "Variable_1":b_conv1_12, "Variable_2":W_fc1_12, "Variable_3":b_fc1_12, "Variable_4":W_fc2_12, "Variable_5":b_fc2_12, "Variable_6": W_conv1_24, "Variable_7":b_conv1_24, "Variable_8":W_fc1_24, "Variable_9":b_fc1_24, "Variable_10":W_fc2_24, "Variable_11":b_fc2_24, "Variable_12":W_conv1_48, "Variable_13":b_conv1_48, "Variable_14":W_conv2_48, "Variable_15":b_conv2_48, "Variable_16":W_fc1_48, "Variable_17":b_fc1_48, "Variable_18":W_fc2_48,"Variable_19":b_fc2_48})
    
    saver_calib_12 = tf.train.Saver({"calib_wc1_12":W_conv1_12_cali, "calib_bc1_12":b_conv1_12_cali, "calib_wfc1_12":W_fc1_12_cali, "calib_bfc1_12":b_fc1_12_cali, "calib_wfc2_12":W_fc2_12_cali, "calib_bfc2_12":b_fc2_12_cali})
    
    saver_calib_24 = tf.train.Saver({"calib_wc1_24": W_conv1_24_cali, "calib_bc1_24":b_conv1_24_cali, "calib_wfc1_24":W_fc1_24_cali, "calib_bfc1_24":b_fc1_24_cali, "calib_wfc2_24":W_fc2_24_cali, "calib_bfc2_24":b_fc2_24_cali})
 
    if cascade_lv == 0:
        saver_detect.restore(sess,etc.save_dir+"12-net.ckpt")
    elif cascade_lv == 1:
        saver_detect.restore(sess,etc.save_dir+"24-net.ckpt")

    saver_calib_12.restore(sess,etc.save_dir+"12-net_calib.ckpt")
    saver_calib_24.restore(sess,etc.save_dir+"24-net_calib.ckpt")

   
    if cascade_lv == 2:
        print "Training finished"
        break

       
    print "Negative sample mining"
    neg_db = [0 for _ in xrange(len(neg_img))]
    neg_db_num = 0 
    for nid, img in enumerate(neg_img):
        print "cas_lv ", cascade_lv, " ", nid+1,"/",len(neg_img), "th image...", "neg_db size: ", neg_db_num, "thr_12: ", thr_12, "thr_24: ", thr_24
        
        #12-net
        result_box = etc.slidingW_Test(img,thr_12,x_12,h_fc2_12)


        #12-calib
        if len(result_box) > 0:
            neg_db_tmp = np.zeros((len(result_box),etc.dim_12),np.float32)
            for id_,box in enumerate(result_box):
                resized_img = etc.img2array(box[5],etc.img_size_12)
                neg_db_tmp[id_,:] = resized_img

            result = h_fc2_12_cali.eval(feed_dict={x_12_cali: neg_db_tmp})
            result_box = etc.calib_run(result_box,result,img) 
                

            #NMS for each scale
            scale_cur = 0
            scale_box = []
            final_box = []
            for id_,box in enumerate(result_box):
                if box[6] == scale_cur:
                    scale_box.append(box)
                if box[6] != scale_cur or id_ == len(result_box)-1:
                    scale_box.sort(key=lambda x :x[4])
                    scale_box.reverse()
                    supp_box_id = etc.NMS(scale_box)
                    final_box += [scale_box[i] for i in supp_box_id]
                    scale_cur += 1
                    scale_box = [box]

            result_box = final_box
            final_box = []                     

            if cascade_lv == 1 and len(result_box) > 0:
                #24-net
                test_db_12 = np.zeros((len(result_box),etc.dim_12),np.float32)
                test_db_24 = np.zeros((len(result_box),etc.dim_24),np.float32)
                for id_,box in enumerate(result_box):
                    original_patch = box[5]
                    resized_img_12 = etc.img2array(original_patch,etc.img_size_12)
                    resized_img_24 = etc.img2array(original_patch,etc.img_size_24)
                    
                    test_db_12[id_,:] = resized_img_12
                    test_db_24[id_,:] = resized_img_24
                

                final_box = []
                test_db = np.empty((0,etc.dim_24),np.float32)
                for tit in xrange(len(test_db_12)/etc.test_bench_num+1):
                    if tit == len(test_db_12)/etc.test_bench_num:
                        test_12 = test_db_12[tit*etc.test_bench_num:,:]
                        test_24 = test_db_24[tit*etc.test_bench_num:,:]
                        box_tmp = result_box[tit*etc.test_bench_num:]
                    else:
                        test_12 = test_db_12[tit*etc.test_bench_num:(tit+1)*etc.test_bench_num,:]
                        test_24 = test_db_24[tit*etc.test_bench_num:(tit+1)*etc.test_bench_num,:]
                        box_tmp = result_box[tit*etc.test_bench_num:(tit+1)*etc.test_bench_num]
                    
                    FROM_12 = h_fc1_12.eval(feed_dict={x_12:test_12})
                    result = h_fc2_24.eval(feed_dict={from_12: FROM_12, x_24: test_24})
                    result_id = np.where(result > thr_24)[0]
                    test_db = np.append(test_db,test_24[result_id,:],axis=0)
                    final_box += [box_tmp[i] for i in result_id]
                result_box = final_box
                final_box = []
                
                if len(result_box) > 0:
                    #24-net_calib
                    test_db_24 = test_db
                    result = h_fc2_24_cali.eval(feed_dict={x_24_cali: test_db_24})
                    result_box = etc.calib_run(result_box,result,img)
                    
                    #NMS for each scale
                    scale_cur = 0
                    scale_box = []
                    final_box = []
                    for id_,box in enumerate(result_box):
                        if box[6] == scale_cur:
                            scale_box.append(box)
                        if box[6] != scale_cur or id_ == len(result_box)-1:
                            scale_box.sort(key=lambda x :x[4])
                            scale_box.reverse()
                            supp_box_id = etc.NMS(scale_box)
                            final_box += [scale_box[i] for i in supp_box_id]
                            scale_cur += 1
                            scale_box = [box]

                    result_box = final_box
                    final_box = []                     

        if len(result_box)>0:
            neg_db_num += len(result_box)
            if cascade_lv == 0:
                neg_db[nid] = [result_box[i][5].resize((etc.img_size_24,etc.img_size_24)) for i in xrange(len(result_box))]
            elif cascade_lv == 1:
                neg_db[nid] = [result_box[i][5].resize((etc.img_size_48,etc.img_size_48)) for i in xrange(len(result_box))]

   
    neg_db = [elem for elem in neg_db if type(elem) != int]
    neg_db = flatten(neg_db)

    print "neg_db size: ", len(neg_db)
