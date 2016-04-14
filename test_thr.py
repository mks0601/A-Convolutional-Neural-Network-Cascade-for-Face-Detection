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

import etc
import load_db

sess = tf.InteractiveSession()


##########detection net
"""
#12-net
#conv layer 1
x_12 = tf.placeholder("float")
W_conv1_12 = etc.weight_variable([3,3,etc.input_channel,16])
b_conv1_12 = etc.bias_variable([16])
h_conv1_12 = tf.nn.relu(etc.conv2d(x_12, W_conv1_12) + b_conv1_12)

#pooling layer 1
h_pool1_12 = etc.max_pool_3x3(h_conv1_12)

#fully layer 1
W_fc1_12 = etc.weight_variable([6 * 6 * 16, 16])
W_fc1_12_ = tf.reshape(W_fc1_12, (6,6,16,16))
b_fc1_12 = etc.bias_variable([16])
h_fc1_12 = tf.nn.relu(etc.conv2d(h_pool1_12, W_fc1_12_) + b_fc1_12)

#fully layer2
W_fc2_12 = etc.weight_variable([16, 1])
W_fc2_12_ = tf.reshape(W_fc2_12, (1,1,16,1))
b_fc2_12 = etc.bias_variable([1])
h_fc2_12 = tf.nn.sigmoid(etc.conv2d(h_fc1_12, W_fc2_12_) + b_fc2_12)
h_fc2_12 = tf.reshape(h_fc2_12, [-1])
"""
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

W_fc2_24 = etc.weight_variable([128 + 16,1])
b_fc2_24 = etc.bias_variable([1])
#h_fc1_12_sliced = tf.slice(h_fc1_12,[0,0,0,0],[-1,1,1,16])
#h_fc1_24_concat = tf.concat(1, [h_fc1_24, tf.reshape(h_fc1_12_sliced,(-1,16))])
h_fc1_24_concat = tf.concat(1,[h_fc1_24, tf.reshape(h_fc1_12,(-1,16))])
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
W_fc2_48 = etc.weight_variable([256 + 128 + 16, 1])
b_fc2_48 = etc.bias_variable([1])
h_fc1_48_concat = tf.concat(1, [h_fc1_24_concat, h_fc1_48])
h_fc2_48 = tf.sigmoid(tf.matmul(h_fc1_48_concat, W_fc2_48) + b_fc2_48)




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

#48-net
x_48_cali = tf.placeholder("float", [None, etc.img_size_48 * etc.img_size_48 * etc.input_channel])
#conv layer 1
W_conv1_48_cali = etc.weight_variable([5,5,3,64],'calib_wc1_48')
b_conv1_48_cali = etc.bias_variable([64],'calib_bc1_48')
x_48_reshaped_cali = tf.reshape(x_48_cali, [-1, etc.img_size_48, etc.img_size_48, etc.input_channel])
h_conv1_48_cali = tf.nn.relu(etc.conv2d(x_48_reshaped_cali, W_conv1_48_cali) + b_conv1_48_cali)

#pooling layer 1
h_pool1_48_cali = etc.max_pool_3x3(h_conv1_48_cali)

#normalization layer 1

#conv layer 2
W_conv2_48_cali = etc.weight_variable([5,5,64,64],'calib_wc2_48')
b_conv2_48_cali = etc.bias_variable([64],'calib_bc2_48')
h_conv2_48_cali = tf.nn.relu(etc.conv2d(h_pool1_48_cali, W_conv2_48_cali) + b_conv2_48_cali)

#normalization layer 2


#fully layer 1
W_fc1_48_cali = etc.weight_variable([24 * 24 * 64, 256],'calib_wfc1_48')
b_fc1_48_cali = etc.bias_variable([256],'calib_bfc1_48')
h_conv2_48_reshaped_cali = tf.reshape(h_conv2_48_cali, [-1, 24 * 24 * 64])
h_fc1_48_cali = tf.nn.relu(tf.matmul(h_conv2_48_reshaped_cali, W_fc1_48_cali) + b_fc1_48_cali)

#fully layer 2
W_fc2_48_cali = etc.weight_variable([256, etc.cali_patt_num],'calib_wfc2_48')
b_fc2_48_cali = etc.bias_variable([etc.cali_patt_num],'calib_bfc2_48')
h_fc2_48_cali = tf.nn.softmax(tf.matmul(h_fc1_48_cali, W_fc2_48_cali) + b_fc2_48_cali)

sess.run(tf.initialize_all_variables())


saver = tf.train.Saver({"Variable":W_conv1_12, "Variable_1":b_conv1_12, "Variable_2":W_fc1_12, "Variable_3":b_fc1_12, "Variable_4":W_fc2_12, "Variable_5":b_fc2_12, "Variable_6": W_conv1_24, "Variable_7":b_conv1_24, "Variable_8":W_fc1_24, "Variable_9":b_fc1_24, "Variable_10":W_fc2_24, "Variable_11":b_fc2_24, "Variable_12":W_conv1_48, "Variable_13":b_conv1_48, "Variable_14":W_conv2_48, "Variable_15":b_conv2_48, "Variable_16":W_fc1_48, "Variable_17":b_fc1_48, "Variable_18":W_fc2_48,"Variable_19":b_fc2_48})

saver.restore(sess,etc.save_dir+"24-net.ckpt")


saver = tf.train.Saver({"calib_wc1_12":W_conv1_12_cali, "calib_bc1_12":b_conv1_12_cali, "calib_wfc1_12":W_fc1_12_cali, "calib_bfc1_12":b_fc1_12_cali, "calib_wfc2_12":W_fc2_12_cali, "calib_bfc2_12":b_fc2_12_cali})
saver.restore(sess, etc.save_dir + "12-net_calib.ckpt")

saver = tf.train.Saver({"calib_wc1_24": W_conv1_24_cali, "calib_bc1_24":b_conv1_24_cali, "calib_wfc1_24":W_fc1_24_cali, "calib_bfc1_24":b_fc1_24_cali, "calib_wfc2_24":W_fc2_24_cali, "calib_bfc2_24":b_fc2_24_cali})
saver.restore(sess, etc.save_dir + "24-net_calib.ckpt")


test_db = load_db.load_db_test()
test_img = test_db[0]
test_annot = test_db[1]
test_fn = test_db[2]

test_start = time.time()
img_num = len(test_img)
face_num = sum(test_fn)
correct_count = 0
win_num = 0

thr_12 = 12e-2
thr_24 = 1e-3

for tid,img_dir in enumerate(test_img):
    
    print tid, "/", img_num,  "th image is in testing..." 
    
    img = Image.open(etc.db_dir + img_dir + ".jpg")
    
    if len(np.shape(np.asarray(img))) < 3:
        arr = np.asarray(img)
        img = np.zeros((img.size[1],img.size[0],etc.input_channel),np.uint8)
        img[:,:,0] = arr
        img[:,:,1] = arr
        img[:,:,2] = arr
        img = Image.fromarray(img)


    pyramid = tuple(pyramid_gaussian(img, downscale = etc.downscale))
    detected_list = [0 for _ in xrange(len(pyramid))]
    for scale in xrange(etc.pyramid_num):
        X = pyramid[scale]

        resized = Image.fromarray(np.uint8(X*255)).resize((int(np.shape(X)[1] * float(etc.img_size_12)/float(etc.face_minimum)), int(np.shape(X)[0]*float(etc.img_size_12)/float(etc.face_minimum))))
        X = np.asarray(resized).astype(np.float32)/255

        img_row = np.shape(X)[0]
        img_col = np.shape(X)[1]
        
        if img_row < etc.img_size_12 or img_col < etc.img_size_12:
            break
        
        """
        feature_row = int(math.ceil(float(img_row)/2))
        feature_col = int(math.ceil(float(img_col)/2))
        no_pad_w_row = ((img_row-3)+1 -3)/2 + 1
        no_pad_w_col = ((img_col-3)+1 -3)/2 + 1
        result = h_fc2_12.eval(feed_dict = {x_12:np.reshape(X, (1, img_row, img_col, etc.input_channel))})

        result_id = np.where(result > thr_12)[0]
        
        detected_list_scale = np.zeros((len(result_id),5),np.float32)
        
        detected_list_scale[:,0] = (result_id%feature_col)
        detected_list_scale[:,1] = (result_id/feature_col)
        detected_list_scale[:,2] = detected_list_scale[:,0] + etc.img_size_12/2 - 1
        detected_list_scale[:,3] = detected_list_scale[:,1] + etc.img_size_12/2 - 1

        detected_list_scale[:,0] = detected_list_scale[:,0] / (no_pad_w_col-1 + etc.img_size_12/2-1) * img.size[0]
        detected_list_scale[:,1] = detected_list_scale[:,1] / (no_pad_w_row-1 + etc.img_size_12/2-1) * img.size[1]
        detected_list_scale[:,2] = detected_list_scale[:,2] / (no_pad_w_col-1 + etc.img_size_12/2-1) * img.size[0]
        detected_list_scale[:,3] = detected_list_scale[:,3] / (no_pad_w_row-1 + etc.img_size_12/2-1) * img.size[1]
        detected_list_scale[:,4] = result[result_id]

        detected_list_scale = detected_list_scale.tolist()
        
        detected_list_scale = [elem + [img.crop((int(elem[0]),int(elem[1]),int(elem[2]),int(elem[3]))), scale, False] for id_,elem in enumerate(detected_list_scale) if result_id[id_]%feature_col < no_pad_w_col and result_id[id_]/feature_col < no_pad_w_row and result_id[id_]%feature_col%2 == 0 and result_id[id_]/feature_col%2 == 0]
        """        

        
        windows = view_as_windows(X, (etc.img_size_12,etc.img_size_12,etc.input_channel),4)
        feature_col = np.shape(windows)[1]
        feature_row = np.shape(windows)[0]
        result = h_fc2_12.eval(feed_dict={x_12:np.reshape(windows,(-1,etc.img_size_12*etc.img_size_12*etc.input_channel))})

        result_id = np.where(result > thr_12)[0]
        
        detected_list_scale = np.zeros((len(result_id),5),np.float32)
        
        detected_list_scale[:,0] = (result_id%feature_col)*4
        detected_list_scale[:,1] = (result_id/feature_col)*4
        detected_list_scale[:,2] = detected_list_scale[:,0] + etc.img_size_12 - 1
        detected_list_scale[:,3] = detected_list_scale[:,1] + etc.img_size_12 - 1

        detected_list_scale[:,0] = detected_list_scale[:,0] / img_col * img.size[0]
        detected_list_scale[:,1] = detected_list_scale[:,1] / img_row * img.size[1]
        detected_list_scale[:,2] = detected_list_scale[:,2] / img_col * img.size[0]
        detected_list_scale[:,3] = detected_list_scale[:,3] / img_row * img.size[1]
        detected_list_scale[:,4] = np.reshape(result[result_id], (-1))

        detected_list_scale = detected_list_scale.tolist()
        
        detected_list_scale = [elem + [img.crop((int(elem[0]),int(elem[1]),int(elem[2]),int(elem[3]))), scale, False] for id_,elem in enumerate(detected_list_scale)]
        
        
        if len(detected_list_scale) > 0:
            detected_list[scale] = detected_list_scale

    detected_list = [elem for elem in detected_list if type(elem) != int]
    result_box = [detected_list[i][j] for i in xrange(len(detected_list)) for j in xrange(len(detected_list[i]))]
    
    #12-calib net
    test_db_12 = np.zeros((len(result_box),etc.dim_12),np.float32)
    for id_,box in enumerate(result_box):
        resized_img_12 = etc.img2array(box[5],etc.img_size_12)
        test_db_12[id_,:] = resized_img_12

    result = h_fc2_12_cali.eval(feed_dict={x_12_cali: test_db_12})
    result_box = etc.calib_run(result_box,result,img) 
    
    #NMS for each scale
    scale_cur = result_box[0][6]
    scale_box = []
    final_box = []
    for id_,box in enumerate(result_box):
        if box[6] == scale_cur:
            scale_box.append(box)
        if box[6] != scale_cur or id_ == len(result_box)-1:
            scale_box.sort(key=lambda x :x[4])
            scale_box.reverse()
            supp_box_id = etc.NMS_fast(scale_box)
            final_box += [scale_box[i] for i in supp_box_id]
            scale_cur += 1
            scale_box = [box]
    result_box = final_box
    final_box = []                     
    
    if etc.thr_lv == 1:
        #24-net
        test_db_12 = np.zeros((len(result_box),etc.dim_12),np.float32)
        test_db_24 = np.zeros((len(result_box),etc.dim_24),np.float32)
        for id_,box in enumerate(result_box):
            resized_img_12 = etc.img2array(box[5],etc.img_size_12)
            resized_img_24 = etc.img2array(box[5],etc.img_size_24)
           
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

            result = h_fc2_24.eval(feed_dict={x_12: test_12, x_24: test_24})
            result_id = np.where(result > thr_24)[0]
            test_db = np.append(test_db,test_24[result_id,:],axis=0)
            final_box += [box_tmp[i] for i in result_id]
        result_box = final_box
        final_box = [] 
        
        """
        #24-calib-net
        test_db_24 = test_db
        result = h_fc2_24_cali.eval(feed_dict={x_24_cali: test_db_24})
        result_box = etc.calib_run(result_box,result,img)
       

        #NMS for each scale
        scale_cur = result_box[0][6]
        scale_box = []
        final_box = []
        for id_,box in enumerate(result_box):
            if box[6] == scale_cur:
                scale_box.append(box)
            if box[6] != scale_cur or id_ == len(result_box)-1:
                scale_box.sort(key=lambda x :x[4])
                scale_box.reverse()
                supp_box_id = etc.NMS_fast(scale_box)
                final_box += [scale_box[i] for i in supp_box_id]
                scale_cur += 1
                scale_box = [box]

        result_box = final_box
        final_box = []
        """
    for box in result_box:
        h = box[3]-box[1]
        box[1] -= 0.2*h
        box[1] = max(box[1],0)

    check_bb = np.zeros((len(test_annot[tid])),np.int32)
    for box in result_box:
        for gid,gt in enumerate(test_annot[tid]):
            left = max(box[0],gt[0])
            right = min(box[2],gt[2])
            upper = max(box[1],gt[1])
            lower = min(box[3],gt[3])

            inter_area = (right-left) * (lower-upper)
            union_area = (box[2]-box[0])*(box[3]-box[1]) + (gt[2]-gt[0])*(gt[3]-gt[1]) - inter_area
            if right>left and lower>upper and inter_area > 0 and union_area > 0 and float(inter_area)/float(union_area) >= 0.5:
                check_bb[gid] = 1
                box[7] = True
    
    for id_,box in enumerate(result_box):
        if box[6] == 10:
            ImageDraw.Draw(img).rectangle((box[0], box[1], box[2], box[3]), outline = "red")

    img.save(etc.result_dir + str(tid) + ".jpg")

    correct_count += sum(check_bb)
    win_num += len(result_box)
    print "thr_lv: ",etc.thr_lv, "thr_12: ", thr_12, "thr_24",thr_24, "recall: ", correct_count/float(sum(test_fn[:tid+1])), "win_num: ", float(win_num)/float(tid+1)



