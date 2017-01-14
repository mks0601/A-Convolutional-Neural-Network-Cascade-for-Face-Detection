import numpy as np
import tensorflow as tf
import random
from PIL import Image

import param
import util
import data
import model


[pos_db_12,pos_db_24,pos_db_48],neg_db_12,neg_db_24,neg_db_48 = data.load_db_detect_train(param.img_size_48)

#48net
input_48_node = tf.placeholder("float", [None, param.img_size_48, param.img_size_48, param.input_channel])
from_24_node = tf.placeholder("float",[None,128+16])
target_48_node = tf.placeholder("float", [None,1])
inputs_48 = np.zeros((param.mini_batch,param.img_size_48,param.img_size_48,param.input_channel), np.float32)
targets_48 = np.zeros((param.mini_batch, 1), np.float32)

net_48 = model.detect_48Net(input_48_node,target_48_node,from_24_node)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

#12net
input_12_node = tf.placeholder("float")
target_12_node = tf.placeholder("float", [None,1])
inputs_12 = np.zeros((param.mini_batch,param.img_size_12,param.img_size_12,param.input_channel), np.float32)

net_12 = model.detect_12Net(input_12_node,target_12_node)
restorer_12 = tf.train.Saver([v for v in tf.global_variables() if "12det_" in v.name])
restorer_12.restore(sess, param.model_dir + "12-net.ckpt")
net_12_calib = model.calib_12Net(input_12_node,target_12_node)
restorer_12_calib = tf.train.Saver([v for v in tf.global_variables() if "12calib_" in v.name])
restorer_12_calib.restore(sess, param.model_dir + "12-calib-net.ckpt")

#24net
input_24_node = tf.placeholder("float", [None, param.img_size_24, param.img_size_24, param.input_channel])
from_12_node = tf.placeholder("float",[None,16])
target_24_node = tf.placeholder("float", [None,1])
inputs_24 = np.zeros((param.mini_batch,param.img_size_24,param.img_size_24,param.input_channel), np.float32)

net_24 = model.detect_24Net(input_24_node,target_24_node,from_12_node)
net_24_calib = model.calib_24Net(input_24_node,target_24_node)
restorer_24 = tf.train.Saver([v for v in tf.global_variables() if "24det_" in v.name])
restorer_24.restore(sess, param.model_dir + "24-net.ckpt")
restorer_24_calib = tf.train.Saver([v for v in tf.global_variables() if "24calib_" in v.name])
restorer_24_calib.restore(sess, param.model_dir + "24-calib-net.ckpt")


print "Training start!"
fp_loss = open("./result/48net/loss.txt", "w")
 
for epoch in xrange(param.epoch_num):
    
    loss = 0
    
    for it in xrange(param.batch_iter):

        pos_id = random.sample(xrange(len(pos_db_48)),param.pos_batch)
        neg_id = random.sample(xrange(len(neg_db_48)),param.neg_batch)
        
        inputs_48[:param.pos_batch,:] = pos_db_48[pos_id,:]
        inputs_48[param.pos_batch:,:] = neg_db_48[neg_id,:]

        inputs_24[:param.pos_batch,:] = pos_db_24[pos_id,:]
        inputs_24[param.pos_batch:,:] = neg_db_24[neg_id,:]
        
        inputs_12[:param.pos_batch,:] = pos_db_12[pos_id,:]
        inputs_12[param.pos_batch:,:] = neg_db_12[neg_id,:]

        targets_48[:param.pos_batch,:] = np.ones((param.pos_batch,1),np.float32)
        targets_48[param.pos_batch:,:] = np.zeros((param.neg_batch,1),np.float32)
       
        from_12 = net_12.from_12.eval(feed_dict = {input_12_node:inputs_12})
        from_24 = net_24.from_24.eval(feed_dict = {input_24_node:inputs_24, from_12_node:from_12}) 

        loss += net_48.loss.eval(feed_dict = {input_48_node:inputs_48, target_48_node:targets_48, from_24_node:from_24})
        net_48.train_step.run(feed_dict = {input_48_node:inputs_48, target_48_node:targets_48, from_24_node:from_24})

        if it > 0 and it % 3000 == 0: 
            print "epoch: " + str(epoch) + " iter: " + str(it) + "/" + str(param.batch_iter) + " loss: " + str(loss / it)

    loss /= param.batch_iter
    fp_loss.write(str(loss)+"\n")
     
    saver = tf.train.Saver([v for v in tf.global_variables() if "48det_" in v.name])
    saver.save(sess, param.model_dir + "48-net.ckpt")
       
    #test each epoch
    test_score = 0
    for bid in xrange(0,len(pos_db_48),param.mini_batch):

        if bid+param.mini_batch <= len(pos_db_48):
            test_inputs_48 = pos_db_48[bid:bid+param.mini_batch,:]
            test_inputs_24 = pos_db_24[bid:bid+param.mini_batch,:]
            test_inputs_12 = pos_db_12[bid:bid+param.mini_batch,:]
            test_targets_48 = np.ones((param.mini_batch,1),np.float32)
        else:
            test_inputs_48 = pos_db_48[bid:,:]
            test_inputs_24 = pos_db_24[bid:,:]
            test_inputs_12 = pos_db_12[bid:,:]
            test_targets_48 = np.ones((param.mini_batch,1),np.float32)
        
        from_12 = net_12.from_12.eval(feed_dict = {input_12_node:test_inputs_12})
        from_24 = net_24.from_24.eval(feed_dict = {input_24_node:test_inputs_24, from_12_node:from_12})

        output = net_48.prediction.eval(feed_dict = {input_48_node:test_inputs_48, from_24_node:from_24})
        test_score += np.sum(output > 0.5)

    for bid in xrange(0,len(neg_db_48),param.mini_batch):

        if bid+param.mini_batch <= len(neg_db_48):
            test_inputs_48 = neg_db_48[bid:bid+param.mini_batch,:]
            test_inputs_24 = neg_db_24[bid:bid+param.mini_batch,:]
            test_inputs_12 = neg_db_12[bid:bid+param.mini_batch,:]
            test_targets_48 = np.ones((param.mini_batch,1),np.float32)
        else:
            test_inputs_48 = neg_db_48[bid:,:]
            test_inputs_24 = neg_db_24[bid:,:]
            test_inputs_12 = neg_db_12[bid:,:]
            test_targets_48 = np.ones((param.mini_batch,1),np.float32)
        
        from_12 = net_12.from_12.eval(feed_dict = {input_12_node:test_inputs_12})
        from_24 = net_24.from_24.eval(feed_dict = {input_24_node:test_inputs_24, from_12_node:from_12})

        output = net_48.prediction.eval(feed_dict = {input_48_node:test_inputs_48, from_24_node:from_24})
        test_score += np.sum(output < 0.5)

    test_score /= float(len(pos_db_48)+len(neg_db_48))
    print "Accuracy: ", test_score
    print 

            
fp_loss.close()
    
        
    
 
