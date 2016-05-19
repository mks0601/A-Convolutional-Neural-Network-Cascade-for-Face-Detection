import math
import tensorflow as tf
import numpy as np
import Image
from skimage.util.shape import view_as_windows
from skimage.transform import pyramid_gaussian

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

def slidingW_Test(img, thr_12, x_12, h_fc2_12):

    pyramid = tuple(pyramid_gaussian(img, downscale = downscale))
    detected_list = [0 for _ in xrange(len(pyramid))]
    for scale in xrange(pyramid_num):
        X = pyramid[scale]


        resized = Image.fromarray(np.uint8(X*255)).resize((int(np.shape(X)[1] * float(img_size_12)/float(face_minimum)), int(np.shape(X)[0]*float(img_size_12)/float(face_minimum))))
        X = np.asarray(resized).astype(np.float32)/255

        img_row = np.shape(X)[0]
        img_col = np.shape(X)[1]
        
        if img_row < img_size_12 or img_col < img_size_12:
            break

        if img_row%2 == 1:
            img_row -= 1
            X = X[:img_row,:]
        if img_col%2 == 1:
            img_col -= 1
            X = X[:,:img_col]
        
        windows = view_as_windows(X, (img_size_12,img_size_12,input_channel),4)
        feature_col = np.shape(windows)[1]
        result = h_fc2_12.eval(feed_dict={x_12:np.reshape(windows,(-1,img_size_12*img_size_12*input_channel))})

        result_id = np.where(result > thr_12)[0]
        
        detected_list_scale = np.zeros((len(result_id),5),np.float32)
        
        detected_list_scale[:,0] = (result_id%feature_col)*4
        detected_list_scale[:,1] = (result_id/feature_col)*4
        detected_list_scale[:,2] = detected_list_scale[:,0] + img_size_12 - 1
        detected_list_scale[:,3] = detected_list_scale[:,1] + img_size_12 - 1

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
    
    return result_box
