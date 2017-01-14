import tensorflow as tf
import numpy as np
import math
import Image
from skimage.util.shape import view_as_windows
from skimage.transform import pyramid_gaussian

import param

def img2array(img,dim):
     
    if dim == param.img_size_12:    
        if img.size[0] != param.img_size_12 or img.size[1] != param.img_size_12:
            img = img.resize((param.img_size_12,param.img_size_12))
        img = np.asarray(img).astype(np.float32)/255 
    elif dim == param.img_size_24:
        if img.size[0] != param.img_size_24 or img.size[1] != param.img_size_24:
            img = img.resize((param.img_size_24,param.img_size_24))
        img = np.asarray(img).astype(np.float32)/255
    elif dim == param.img_size_48:
        if img.size[0] != param.img_size_48 or img.size[1] != param.img_size_48:
            img = img.resize((param.img_size_48,param.img_size_48))
        img = np.asarray(img).astype(np.float32)/255
    return img

def calib_box(result_box,result,img):
    

    for id_,cid in enumerate(np.argmax(result,axis=1).tolist()):
        s = cid / (len(param.cali_off_x) * len(param.cali_off_y))
        x = cid % (len(param.cali_off_x) * len(param.cali_off_y)) / len(param.cali_off_y)
        y = cid % (len(param.cali_off_x) * len(param.cali_off_y)) % len(param.cali_off_y) 
                
        s = param.cali_scale[s]
        x = param.cali_off_x[x]
        y = param.cali_off_y[y]
    
        
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

def NMS(box):
    
    if len(box) == 0:
        return []
    
    #xmin, ymin, xmax, ymax, score, cropped_img, scale
    box.sort(key=lambda x :x[4])
    box.reverse()

    pick = []
    x_min = np.array([box[i][0] for i in range(len(box))],np.float32)
    y_min = np.array([box[i][1] for i in range(len(box))],np.float32)
    x_max = np.array([box[i][2] for i in range(len(box))],np.float32)
    y_max = np.array([box[i][3] for i in range(len(box))],np.float32)

    area = (x_max-x_min)*(y_max-y_min)
    idxs = np.array(range(len(box)))

    while len(idxs) > 0:
        i = idxs[0]
        pick.append(i)

        xx1 = np.maximum(x_min[i],x_min[idxs[1:]])
        yy1 = np.maximum(y_min[i],y_min[idxs[1:]])
        xx2 = np.minimum(x_max[i],x_max[idxs[1:]])
        yy2 = np.minimum(y_max[i],y_max[idxs[1:]])

        w = np.maximum(xx2-xx1,0)
        h = np.maximum(yy2-yy1,0)

        overlap = (w*h)/(area[idxs[1:]] + area[i] - w*h)

        idxs = np.delete(idxs, np.concatenate(([0],np.where(((overlap >= 0.5) & (overlap <= 1)))[0]+1)))
    
    return [box[i] for i in pick]

def sliding_window(img, thr, net, input_12_node):

    pyramid = tuple(pyramid_gaussian(img, downscale = param.downscale))
    detected_list = [0 for _ in xrange(len(pyramid))]
    for scale in xrange(param.pyramid_num):
        
        X = pyramid[scale]

        resized = Image.fromarray(np.uint8(X*255)).resize((int(np.shape(X)[1] * float(param.img_size_12)/float(param.face_minimum)), int(np.shape(X)[0]*float(param.img_size_12)/float(param.face_minimum))))
        X = np.asarray(resized).astype(np.float32)/255

        img_row = np.shape(X)[0]
        img_col = np.shape(X)[1]

        X = np.reshape(X,(1,img_row,img_col,param.input_channel))
        
        if img_row < param.img_size_12 or img_col < param.img_size_12:
            break
        
        #predict and get rid of boxes from padding
        win_num_row = math.floor((img_row-param.img_size_12)/param.window_stride)+1
        win_num_col = math.floor((img_col-param.img_size_12)/param.window_stride)+1

        result = net.prediction.eval(feed_dict={input_12_node:X})
        result_row = np.shape(result)[1]
        result_col = np.shape(result)[2]

        result = result[:,\
                int(math.floor((result_row-win_num_row)/2)):int(result_row-math.ceil((result_row-win_num_row)/2)),\
                int(math.floor((result_col-win_num_col)/2)):int(result_col-math.ceil((result_col-win_num_col)/2)),\
                :]

        feature_col = np.shape(result)[2]

        #feature_col: # of predicted window num in width dim
        #win_num_col: # of box(gt)
        assert(feature_col == win_num_col)

        result = np.reshape(result,(-1,1))
        result_id = np.where(result > thr)[0]
        
        #xmin, ymin, xmax, ymax, score
        detected_list_scale = np.zeros((len(result_id),5),np.float32)
        
        detected_list_scale[:,0] = (result_id%feature_col)*param.window_stride
        detected_list_scale[:,1] = np.floor(result_id/feature_col)*param.window_stride
        detected_list_scale[:,2] = np.minimum(detected_list_scale[:,0] + param.img_size_12 - 1, img_col-1)
        detected_list_scale[:,3] = np.minimum(detected_list_scale[:,1] + param.img_size_12 - 1, img_row-1)

        detected_list_scale[:,0] = detected_list_scale[:,0] / (img_col-1) * (img.size[0]-1)
        detected_list_scale[:,1] = detected_list_scale[:,1] / (img_row-1) * (img.size[1]-1)
        detected_list_scale[:,2] = detected_list_scale[:,2] / (img_col-1) * (img.size[0]-1)
        detected_list_scale[:,3] = detected_list_scale[:,3] / (img_row-1) * (img.size[1]-1)
        detected_list_scale[:,4] = result[result_id,0]

        detected_list_scale = detected_list_scale.tolist()
       
        #xmin, ymin, xmax, ymax, score, cropped_img, scale
        detected_list_scale = [elem + [img.crop((int(elem[0]),int(elem[1]),int(elem[2]),int(elem[3]))), scale] for id_,elem in enumerate(detected_list_scale)]
        
        if len(detected_list_scale) > 0:
            detected_list[scale] = detected_list_scale 
            
    detected_list = [elem for elem in detected_list if type(elem) != int]
    result_box = [detected_list[i][j] for i in xrange(len(detected_list)) for j in xrange(len(detected_list[i]))]
    
    return result_box
