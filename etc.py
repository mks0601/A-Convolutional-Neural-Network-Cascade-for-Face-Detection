import math
import tensorflow as tf
import numpy as np
import Image

#dir
db_dir = "/media/sda1/Study/Data/FDDB/"
save_dir = db_dir + "var_save/"
pos_dir = db_dir + "AFLW/aflw/data/flickr/"
neg_dir = db_dir + "neg_train/"
test_dir = db_dir + "test/"
result_dir = db_dir + "result/"

#db parameters
input_channel = 3
img_size_12 = 12
img_size_24 = 24
img_size_48 = 48
neg_per_img = 35
fold_num = 10
dim_12 = img_size_12 * img_size_12 * input_channel
dim_24 = img_size_24 * img_size_24 * input_channel
dim_48 = img_size_48 * img_size_48 * input_channel

#network parameters
b_intial = 0.0
w_stddev = 0.1
lr = 5e-2
epoch_num = [100,100,100]
epoch_calib_num = [50,50,50]
pos_batch = 50
neg_batch = 50
mini_batch = 100
batch_iter = int(1e4)

#result parameters
thr_start = -0.1
thr_end = 0.1
thr_num = 1
thr = 0.5
result_interval = 500
test_fold = 10


#training parameters
cascade_level = 3
window_stride = 4
acc_bench_num = int(1e2)
test_bench_num = int(5e3)
face_minimum = 27
downscale = 1.18
pyramid_num = 16
thr_lv = 1

cali_scale = [0.83, 0.91, 1.0, 1.10, 1.21]
cali_off_x = [-0.17, 0., 0.17]
cali_off_y = [-0.17, 0., 0.17]
cali_patt_num = len(cali_scale) * len(cali_off_x) * len(cali_off_y)
cali_thr = 0.5

def img2array(img,dim):
     
    if dim == img_size_12:    
        if img.size[0] != img_size_12 or img.size[1] != img_size_12:
            img = img.resize((img_size_12,img_size_12))
        img = np.asarray(img).astype(np.float32)/255 
        img = np.reshape(img,(1,dim_12))
    elif dim == img_size_24:
        if img.size[0] != img_size_24 or img.size[1] != img_size_24:
            img = img.resize((img_size_24,img_size_24))
        img = np.asarray(img).astype(np.float32)/255
        img = np.reshape(img,(1,dim_24))
    elif dim == img_size_48:
        if img.size[0] != img_size_48 or img.size[1] != img_size_48:
            img = img.resize((img_size_48,img_size_48))
        img = np.asarray(img).astype(np.float32)/255
        img = np.reshape(img,(1,dim_48))
    return img

def calib_run(result_box,result,img):
    

    for id_,cid in enumerate(np.argmax(result,axis=1).tolist()):
        s = cid / (len(cali_off_x) * len(cali_off_y))
        x = cid % (len(cali_off_x) * len(cali_off_y)) / len(cali_off_y)
        y = cid % (len(cali_off_x) * len(cali_off_y)) % len(cali_off_y) 
                
        s = cali_scale[s]
        x = cali_off_x[x]
        y = cali_off_y[y]
    
        
        new_ltx = result_box[id_][0] + x*(result_box[id_][2]-result_box[id_][0])
        new_lty = result_box[id_][1] + y*(result_box[id_][3]-result_box[id_][1])
        new_rbx = new_ltx + s*(result_box[id_][2]-result_box[id_][0])
        new_rby = new_lty + s*(result_box[id_][3]-result_box[id_][1])
        
        result_box[id_][0] = int(max(new_ltx,0))
        result_box[id_][1] = int(max(new_lty,0))
        result_box[id_][2] = int(min(new_rbx,img.size[0]-1))
        result_box[id_][3] = int(min(new_rby,img.size[1]-1))
        result_box[id_][5] = img.crop((result_box[id_][0],result_box[id_][1],result_box[id_][2],result_box[id_][3]))

    return result_box 


def weight_variable(shape,name=None):
    initial = tf.truncated_normal(shape, stddev=w_stddev)
    return tf.Variable(initial,name=name)

def bias_variable(shape,name=None):
    initial = tf.constant(b_intial, shape=shape)
    return tf.Variable(initial,name=name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_3x3(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                          strides=[1, 2, 2, 1], padding='SAME')



def NMS_helper(lid, detected_list, is_supp):
    
    for i in range(lid+1,len(detected_list)):
        if is_supp[i] == 0:
            left = max([detected_list[lid][0], detected_list[i][0]])
            right = min([detected_list[lid][2], detected_list[i][2]])
            upper = max([detected_list[lid][1], detected_list[i][1]])
            lower = min([detected_list[lid][3], detected_list[i][3]])

            inter_area = (right-left) * (lower-upper)
            union_area = (detected_list[lid][2] - detected_list[lid][0]) * (detected_list[lid][3] - detected_list[lid][1])
            union_area += (detected_list[i][2] - detected_list[i][0]) * (detected_list[i][3] - detected_list[i][1])
            union_area -= inter_area
            
            if left >= right or upper >= lower or inter_area <= 0 or union_area <= 0:
                continue
            overlap = float(inter_area) / float(union_area)
            if overlap < 1 and overlap >= 0.5:  # or inter_area >= 0.9*(detected_list[lid][2]-detected_list[lid][0])*(detected_list[lid][3]-detected_list[lid][1]):
                is_supp[i] = 1
    
    return lid, is_supp

def NMS(detected_list):
    
    is_supp = np.zeros(len(detected_list))
    result = []
    for i in range(len(detected_list)):
        if is_supp[i] == 0:
            is_supp[i] = 1
            lid, is_supp = NMS_helper(i, detected_list, is_supp)
            result.append(lid)
    return result














def NMS_fast(detected_list):

    if len(detected_list) == 0:
        return []
    
    pick = []
    lt_x = np.array([detected_list[i][0] for i in range(len(detected_list))],np.float32)
    lt_y = np.array([detected_list[i][1] for i in range(len(detected_list))],np.float32)
    rb_x = np.array([detected_list[i][2] for i in range(len(detected_list))],np.float32)
    rb_y = np.array([detected_list[i][3] for i in range(len(detected_list))],np.float32)

    area = (rb_x-lt_x)*(rb_y-lt_y)
    idxs = np.array(range(len(detected_list)))

    while len(idxs) > 0:
        i = idxs[0]
        pick.append(i)

        xx1 = np.maximum(lt_x[i],lt_x[idxs[1:]])
        yy1 = np.maximum(lt_y[i],lt_y[idxs[1:]])
        xx2 = np.minimum(rb_x[i],rb_x[idxs[1:]])
        yy2 = np.minimum(rb_y[i],rb_y[idxs[1:]])

        w = np.maximum(xx2-xx1,0)
        h = np.maximum(yy2-yy1,0)

        overlap = (w*h)/(area[idxs[1:]] + area[i] - w*h)
        included = w*h / (0.9 * area[idxs[1:]])

        idxs = np.delete(idxs, np.concatenate(([0],np.where(((overlap >= 0.5) & (overlap < 1)) |(included > 0.9)  )[0]+1)))
    
    return pick
