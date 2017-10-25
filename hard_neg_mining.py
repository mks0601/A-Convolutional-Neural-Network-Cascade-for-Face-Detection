import numpy as np
import tensorflow as tf
from PIL import Image
from compiler.ast import flatten
import os
import sys
import math

import param
import util
import model

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

if sys.argv[1] == str(param.img_size_48):
    
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

neg_file_list = [f for f in os.listdir(param.neg_dir) if f.endswith(".jpg")]

#hard neg mining
neg_db_sz = 0
neg_db = [0 for _ in range(1000)]
for nid,img_name in enumerate(neg_file_list):
    
    img = Image.open(param.neg_dir + img_name)
    
    #check if gray
    if len(np.shape(img)) != param.input_channel:
        img = np.asarray(img)
        img = np.reshape(img,(np.shape(img)[0],np.shape(img)[1],1))
        img = np.concatenate((img,img,img),axis=2)
        img = Image.fromarray(img)

    #12-net
    #box: xmin, ymin, xmax, ymax, score, cropped_img, scale
    neg_box = util.sliding_window(img, param.thr_12, net_12, input_12_node)

    #12-calib
    neg_db_tmp = np.zeros((len(neg_box),param.img_size_12,param.img_size_12,param.input_channel),np.float32)
    for id_,box in enumerate(neg_box):
        neg_db_tmp[id_,:] = util.img2array(box[5],param.img_size_12)

    calib_result = net_12_calib.prediction.eval(feed_dict={input_12_node: neg_db_tmp})
    neg_box = util.calib_box(neg_box,calib_result,img)

    #NMS for each scale
    scale_cur = 0
    scale_box = []
    suppressed = []
    for id_,box in enumerate(neg_box):
        if box[6] == scale_cur:
            scale_box.append(box)
        if box[6] != scale_cur or id_ == len(neg_box)-1:
            suppressed += util.NMS(scale_box)
            scale_cur = box[6]
            scale_box = [box]

    neg_box = suppressed
    suppressed = []    
    
    if sys.argv[1] == str(param.img_size_48):
        #24-net
        result_db_12 = np.zeros((len(neg_box),param.img_size_12,param.img_size_12,param.input_channel),np.float32)
        result_db_24 = np.zeros((len(neg_box),param.img_size_24,param.img_size_24,param.input_channel),np.float32)
        for bid,box in enumerate(neg_box):
            resized_img_12 = util.img2array(box[5],param.img_size_12)
            resized_img_24 = util.img2array(box[5],param.img_size_24)

            result_db_12[bid,:] = resized_img_12
            result_db_24[bid,:] = resized_img_24
        
        from_12 = net_12.from_12.eval(feed_dict={input_12_node: result_db_12})
        result = net_24.prediction.eval(feed_dict={input_24_node: result_db_24, from_12_node: from_12})
        result_id = np.where(result > param.thr_24)[0]
        neg_box = [neg_box[i] for i in result_id]
       
        #24-calib
        result_db_tmp = np.zeros((len(neg_box),param.img_size_24,param.img_size_24,param.input_channel),np.float32)
        for id_,box in enumerate(neg_box):
            result_db_tmp[id_,:] = util.img2array(box[5],param.img_size_24)

        calib_result = net_24_calib.prediction.eval(feed_dict={input_24_node: result_db_tmp})
        neg_box = util.calib_box(neg_box,calib_result,img)

        #NMS for each scale
        scale_cur = 0
        scale_box = []
        suppressed = []
        for id_,box in enumerate(neg_box):
            if box[6] == scale_cur:
                scale_box.append(box)
            if box[6] != scale_cur or id_ == len(neg_box)-1:
                suppressed += util.NMS(scale_box)
                scale_cur = box[6]
                scale_box = [box]

        neg_box = suppressed
        suppressed = []    


    neg_db_ = [0 for _ in range(len(neg_box))]
    for bid,box in enumerate(neg_box):
        neg_cropped_img = box[5]
        neg_db_[bid] = neg_cropped_img

    neg_db_sz += len(neg_box)
    neg_db[nid%1000] = neg_db_
    img.close()

    if (nid+1) % 1000 == 0 or nid == len(neg_file_list)-1:

        neg_db = flatten(neg_db)
        
        if sys.argv[1] == str(param.img_size_24):
            neg_db_12 = np.zeros((len(neg_db),param.img_size_12,param.img_size_12,param.input_channel),np.float32)
            neg_db_24 = np.zeros((len(neg_db),param.img_size_24,param.img_size_24,param.input_channel),np.float32)
            
            for iid,cropped_img in enumerate(neg_db):
                neg_db_12[iid,:] = util.img2array(cropped_img,param.img_size_12)
                neg_db_24[iid,:] = util.img2array(cropped_img,param.img_size_24)

            np.save(param.neg_dir + "neg_hard/24/12_" + str(int(math.ceil(nid/1000))) + ".npy",neg_db_12)
            np.save(param.neg_dir + "neg_hard/24/24_" + str(int(math.ceil(nid/1000))) + ".npy",neg_db_24)

            neg_db_12 = None
            neg_db_24 = None
        elif sys.argv[1] == str(param.img_size_48):
            neg_db_12 = np.zeros((len(neg_db),param.img_size_12,param.img_size_12,param.input_channel),np.float32)
            neg_db_24 = np.zeros((len(neg_db),param.img_size_24,param.img_size_24,param.input_channel),np.float32)
            neg_db_48 = np.zeros((len(neg_db),param.img_size_48,param.img_size_48,param.input_channel),np.float32)
            
            for iid,cropped_img in enumerate(neg_db):
                neg_db_12[iid,:] = util.img2array(cropped_img,param.img_size_12)
                neg_db_24[iid,:] = util.img2array(cropped_img,param.img_size_24)
                neg_db_48[iid,:] = util.img2array(cropped_img,param.img_size_48)

            np.save(param.neg_dir + "neg_hard/48/12_" + str(int(math.ceil(nid/1000))) + ".npy",neg_db_12)
            np.save(param.neg_dir + "neg_hard/48/24_" + str(int(math.ceil(nid/1000))) + ".npy",neg_db_24)
            np.save(param.neg_dir + "neg_hard/48/48_" + str(int(math.ceil(nid/1000))) + ".npy",neg_db_48)

            neg_db_12 = None
            neg_db_24 = None
            neg_db_48 = None
        
        if len(neg_file_list) - (nid+1) < 1000:
            neg_db = [0 for _ in range(len(neg_file_list)-(nid+1))]
        else:
            neg_db = [0 for _ in range(1000)]


    print "neg mining: " + str(nid) + "/" + str(len(neg_file_list)) + " db size: " + str(neg_db_sz)

