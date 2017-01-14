import numpy as np
import tensorflow as tf
import random
import sys

import param
import util
import data
import model

input_node = tf.placeholder("float")
target_node = tf.placeholder("float", [None,param.cali_patt_num])

if sys.argv[1] == str(param.img_size_12):
    dim = param.img_size_12
    folder_name = "12calib"
    model_name = "12-calib-net.ckpt"
    net = model.calib_12Net(input_node,target_node)

elif sys.argv[1] == str(param.img_size_24):
    dim = param.img_size_24
    folder_name = "24calib"
    model_name = "24-calib-net.ckpt"
    net = model.calib_24Net(input_node,target_node)

elif sys.argv[1] == str(param.img_size_48):
    dim = param.img_size_48
    folder_name = "48calib"
    model_name = "48-calib-net.ckpt"
    net = model.calib_48Net(input_node,target_node)


train_db = data.load_db_calib_train(dim)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


print "Training start!"
fp_loss = open("./result/" + folder_name + "/loss.txt", "w")
data_id = range(len(train_db)) 
for epoch in xrange(param.epoch_num):
    
    loss = 0
    inputs = np.zeros((param.mini_batch,dim,dim,param.input_channel), np.float32)
    targets = np.zeros((param.mini_batch, param.cali_patt_num), np.float32)
    random.shuffle(data_id)

    for it,bid in enumerate(xrange(0,len(train_db),param.mini_batch)):

        if bid+param.mini_batch <= len(train_db):
            for i in xrange(bid,bid+param.mini_batch):
                did = data_id[i]
                inputs[i-bid,:] = train_db[did][0]
                targets[i-bid,:] = np.zeros((param.cali_patt_num),np.float32)
                targets[i-bid,train_db[did][1]] = 1
        else:
            inputs = np.zeros((len(train_db)-bid,dim,dim,param.input_channel), np.float32)
            targets = np.zeros((len(train_db)-bid,param.cali_patt_num), np.float32)
            for i in xrange(bid,len(train_db)):
                did = data_id[i]
                inputs[i-bid,:] = train_db[did][0]
                targets[i-bid,:] = np.zeros((param.cali_patt_num),np.float32)
                targets[i-bid,train_db[did][1]] = 1

        loss += net.loss.eval(feed_dict = {input_node:inputs, target_node:targets})
        net.train_step.run(feed_dict = {input_node:inputs, target_node:targets})

        if it > 0 and it % 3000 == 0: 
            print "epoch: " + str(epoch) + " iter: " + str(bid) + "/" + str(len(train_db)) + " loss: " + str(loss / it)

    loss /= it
    fp_loss.write(str(loss)+"\n")
     
    saver = tf.train.Saver()
    saver.save(sess, param.model_dir + model_name)

    #test each epoch
    test_score = 0
    test_inputs = np.zeros((param.mini_batch,dim,dim,param.input_channel), np.float32)
    test_targets = np.zeros((param.mini_batch, param.cali_patt_num), np.float32)
    for bid in xrange(0,len(train_db),param.mini_batch):

        if bid+param.mini_batch <= len(train_db):
            for did in xrange(bid,bid+param.mini_batch):
                test_inputs[did-bid,:] = train_db[did][0]
                test_targets[did-bid,:] = np.zeros((param.cali_patt_num),np.float32)
                test_targets[did-bid,train_db[did][1]] = 1
        else:
            test_inputs = np.zeros((len(train_db)-bid,dim,dim,param.input_channel), np.float32)
            test_targets = np.zeros((len(train_db)-bid, param.cali_patt_num), np.float32)
            for did in xrange(bid,len(train_db)):
                test_inputs[did-bid,:] = train_db[did][0]
                test_targets[did-bid,:] = np.zeros((param.cali_patt_num),np.float32)
                test_targets[did-bid,train_db[did][1]] = 1

        output = net.prediction.eval(feed_dict = {input_node:test_inputs, target_node:test_targets})
        test_score += np.sum(np.argmax(output,1) == np.argmax(test_targets,1))

    test_score /= float(len(train_db))
    print "Accuracy: ", test_score
    print 

fp_loss.close()
    
        
    
 
