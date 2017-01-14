import numpy as np
import tensorflow as tf
import random
from PIL import Image

import param
import util
import data
import model


[pos_db_12,pos_db_24,_],neg_db_12,neg_db_24 = data.load_db_detect_train(param.img_size_24)

#24net
input_24_node = tf.placeholder("float", [None, param.img_size_24, param.img_size_24, param.input_channel])
from_12_node = tf.placeholder("float",[None,16])
target_24_node = tf.placeholder("float", [None,1])
inputs_24 = np.zeros((param.mini_batch,param.img_size_24,param.img_size_24,param.input_channel), np.float32)
targets_24 = np.zeros((param.mini_batch, 1), np.float32)

net_24 = model.detect_24Net(input_24_node,target_24_node,from_12_node)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

#12net
input_12_node = tf.placeholder("float")
target_12_node = tf.placeholder("float", [None,1])
inputs_12 = np.zeros((param.mini_batch,param.img_size_12,param.img_size_12,param.input_channel), np.float32)

net_12 = model.detect_12Net(input_12_node,target_12_node)
restorer_12 = tf.train.Saver([v for v in tf.global_variables() if "12det_" in v.name])
restorer_12.restore(sess, param.model_dir + "12-net.ckpt")

print "Training start!"
fp_loss = open("./result/24net/loss.txt", "w")
 
for epoch in xrange(param.epoch_num):
    
    loss = 0
    
    for it in xrange(param.batch_iter):

        pos_id = random.sample(xrange(len(pos_db_24)),param.pos_batch)
        neg_id = random.sample(xrange(len(neg_db_24)),param.neg_batch)

        inputs_24[:param.pos_batch,:] = pos_db_24[pos_id,:]
        inputs_24[param.pos_batch:,:] = neg_db_24[neg_id,:]
        
        inputs_12[:param.pos_batch,:] = pos_db_12[pos_id,:]
        inputs_12[param.pos_batch:,:] = neg_db_12[neg_id,:]

        targets_24[:param.pos_batch,:] = np.ones((param.pos_batch,1),np.float32)
        targets_24[param.pos_batch:,:] = np.zeros((param.neg_batch,1),np.float32)
       
        from_12 = net_12.from_12.eval(feed_dict = {input_12_node:inputs_12}) 
        loss += net_24.loss.eval(feed_dict = {input_24_node:inputs_24, target_24_node:targets_24, from_12_node:from_12})
        net_24.train_step.run(feed_dict = {input_24_node:inputs_24, target_24_node:targets_24, from_12_node:from_12})

        if it > 0 and it % 3000 == 0: 
            print "epoch: " + str(epoch) + " iter: " + str(it) + "/" + str(param.batch_iter) + " loss: " + str(loss / it)

    loss /= param.batch_iter
    fp_loss.write(str(loss)+"\n")
     
    saver = tf.train.Saver([v for v in tf.global_variables() if "24det_" in v.name])
    saver.save(sess, param.model_dir + "24-net.ckpt")
       
    
    #test each epoch
    test_score = 0
    for bid in xrange(0,len(pos_db_24),param.mini_batch):

        if bid+param.mini_batch <= len(pos_db_24):
            test_inputs_24 = pos_db_24[bid:bid+param.mini_batch,:]
            test_inputs_12 = pos_db_12[bid:bid+param.mini_batch,:]
            test_targets_24 = np.ones((param.mini_batch,1),np.float32)
        else:
            test_inputs_24 = pos_db_24[bid:,:]
            test_inputs_12 = pos_db_12[bid:,:]
            test_targets_24 = np.ones((param.mini_batch,1),np.float32)
        
        from_12 = net_12.from_12.eval(feed_dict = {input_12_node:test_inputs_12})
        output = net_24.prediction.eval(feed_dict = {input_24_node:test_inputs_24, from_12_node:from_12})
        test_score += np.sum(output > 0.5)

    for bid in xrange(0,len(neg_db_24),param.mini_batch):

        if bid+param.mini_batch <= len(neg_db_24):
            test_inputs_24 = neg_db_24[bid:bid+param.mini_batch,:]
            test_inputs_12 = neg_db_12[bid:bid+param.mini_batch,:]
            test_targets_24 = np.ones((param.mini_batch,1),np.float32)
        else:
            test_inputs_24 = neg_db_24[bid:,:]
            test_inputs_12 = neg_db_12[bid:,:]
            test_targets_24 = np.ones((param.mini_batch,1),np.float32)
        
        from_12 = net_12.from_12.eval(feed_dict = {input_12_node:test_inputs_12})
        output = net_24.prediction.eval(feed_dict = {input_24_node:test_inputs_24, from_12_node:from_12})
        test_score += np.sum(output < 0.5)

    test_score /= float(len(pos_db_24)+len(neg_db_24))
    print "Accuracy: ", test_score
    print 

            
fp_loss.close()
    
        
    
 
