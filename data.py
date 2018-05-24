import numpy as np
import math
import os
from random import randint
from PIL import Image

import param
import util

def load_db_detect_train(dim):
    
    print "Loading positive training db..."

    annot_dir = param.db_dir + "AFLW/aflw/data/"
    annot_fp = open(annot_dir + "annot", "r")
    raw_data = annot_fp.readlines()

    #pos image cropping
    pos_db_12 = [0 for _ in xrange(len(raw_data))]
    pos_db_24 = [0 for _ in xrange(len(raw_data))]
    pos_db_48 = [0 for _ in xrange(len(raw_data))]

    for i,line in enumerate(raw_data):
        
        parsed_line = line.split(',')

        filename = parsed_line[0][3:-1]
        xmin = int(parsed_line[1])
        ymin = int(parsed_line[2])
        xmax = xmin + int(parsed_line[3])
        ymax = ymin + int(parsed_line[4][:-2])

        img = Image.open(param.pos_dir+filename)
        
        #for debugging
        #img.save(str(i)  + ".jpg")
        
        #truncated image(error)
        if i == 8160 or i == 14884 or i == 14886:
            continue

        #check if gray
        if len(np.shape(img)) != param.input_channel:
            img = np.asarray(img)
            img = np.reshape(img,(np.shape(img)[0],np.shape(img)[1],1))
            img = np.concatenate((img,img,img),axis=2)
            img = Image.fromarray(img)

        pos_db_line_12 = np.zeros((2,param.img_size_12,param.img_size_12,param.input_channel), np.float32)
        pos_db_line_24 = np.zeros((2,param.img_size_24,param.img_size_24,param.input_channel), np.float32)
        pos_db_line_48 = np.zeros((2,param.img_size_48,param.img_size_48,param.input_channel), np.float32)

        if xmax >= img.size[0]:
            xmax = img.size[0]-1
        if ymax >= img.size[1]:
            ymax = img.size[1]-1
        
        cropped_img = img.crop((xmin, ymin, xmax, ymax))
        flipped_img = cropped_img.transpose(Image.FLIP_LEFT_RIGHT)

        cropped_arr_12 = util.img2array(cropped_img,param.img_size_12)
        flipped_arr_12 = util.img2array(flipped_img,param.img_size_12)
        cropped_arr_24 = util.img2array(cropped_img,param.img_size_24)
        flipped_arr_24 = util.img2array(flipped_img,param.img_size_24)
        cropped_arr_48 = util.img2array(cropped_img,param.img_size_48)
        flipped_arr_48 = util.img2array(flipped_img,param.img_size_48)
        
        #for debugging
        #cropped_img.save(param.pos_dir + str(i)  + ".jpg")
        
        pos_db_line_12[0,:] = cropped_arr_12
        pos_db_line_24[0,:] = cropped_arr_24
        pos_db_line_48[0,:] = cropped_arr_48
        
        
        pos_db_line_12[1,:] = flipped_arr_12
        pos_db_line_24[1,:] = flipped_arr_24
        pos_db_line_48[1,:] = flipped_arr_48
        
        pos_db_12[i] = pos_db_line_12
        pos_db_24[i] = pos_db_line_24
        pos_db_48[i] = pos_db_line_48

        img.close()

    pos_db_12 = [elem for elem in pos_db_12 if type(elem) != int]   
    pos_db_24 = [elem for elem in pos_db_24 if type(elem) != int]
    pos_db_48 = [elem for elem in pos_db_48 if type(elem) != int]

    pos_db_12 = np.vstack(pos_db_12)
    pos_db_24 = np.vstack(pos_db_24) 
    pos_db_48 = np.vstack(pos_db_48) 
    print "Loading negative training db..."
    if dim == param.img_size_12:
        
        #neg image cropping
        nid = 0
        neg_file_list = [f for f in os.listdir(param.neg_dir) if f.endswith(".jpg")]
        neg_db_12 = [0 for n in xrange(len(neg_file_list))]

        for filename in neg_file_list:
            
            img = Image.open(param.neg_dir + filename)
            
            #check if gray
            if len(np.shape(np.asarray(img))) != param.input_channel:
                img = np.asarray(img)
                img = np.reshape(img,(np.shape(img)[0],np.shape(img)[1],1))
                img = np.concatenate((img,img,img),axis=2)
                img = Image.fromarray(img)

            neg_db_line = np.zeros((param.neg_per_img,param.img_size_12,param.img_size_12,param.input_channel), np.float32)
            for neg_iter in xrange(param.neg_per_img):
                         
                rad_rand = randint(0,min(img.size[0],img.size[1])-1)
                while(rad_rand <= param.face_minimum):
                    rad_rand = randint(0,min(img.size[0],img.size[1])-1)
                
                x_rand = randint(0, img.size[0] - rad_rand - 1)
                y_rand = randint(0, img.size[1] - rad_rand - 1)
                
                neg_cropped_img = img.crop((x_rand, y_rand, x_rand + rad_rand, y_rand + rad_rand))
                neg_cropped_arr = util.img2array(neg_cropped_img,param.img_size_12)
                
                #for debugging
                #neg_cropped_img.save(param.neg_dir + str(fid) + "_" + str(r) + ".jpg")
                       
                neg_db_line[neg_iter,:] = neg_cropped_arr
            
            neg_db_12[nid] = neg_db_line
            nid += 1
       
        neg_db_12 = [elem for elem in neg_db_12 if type(elem) != int]
        neg_db_12 = np.vstack(neg_db_12)
        return [pos_db_12,pos_db_24,pos_db_48], neg_db_12

    elif dim == param.img_size_24:
        
        neg_db_12 = np.empty((0,param.img_size_12,param.img_size_12,param.input_channel),np.float32)
        neg_file_list = [f for f in os.listdir(param.neg_dir + "neg_hard/24/") if f.startswith("12_") and f.endswith(".npy")]
        for nid,db_name in enumerate(neg_file_list):

            tmp = np.load(param.neg_dir + "neg_hard/24/" + db_name)
            neg_db_12 = np.concatenate((neg_db_12,tmp),axis=0)


        neg_db_24 = np.empty((0,param.img_size_24,param.img_size_24,param.input_channel),np.float32)
        neg_file_list = [f for f in os.listdir(param.neg_dir + "neg_hard/24/") if f.startswith("24_") and f.endswith(".npy")]
        for nid,db_name in enumerate(neg_file_list):

            tmp = np.load(param.neg_dir + "neg_hard/24/" + db_name)
            neg_db_24 = np.concatenate((neg_db_24,tmp),axis=0)

        return [pos_db_12,pos_db_24,pos_db_48], neg_db_12, neg_db_24
    
    elif dim == param.img_size_48:
        
        neg_db_12 = np.empty((0,param.img_size_12,param.img_size_12,param.input_channel),np.float32)
        neg_file_list = [f for f in os.listdir(param.neg_dir + "neg_hard/48/") if f.startswith("12_") and f.endswith(".npy")]
        for nid,db_name in enumerate(neg_file_list):

            tmp = np.load(param.neg_dir + "neg_hard/48/" + db_name)
            neg_db_12 = np.concatenate((neg_db_12,tmp),axis=0)


        neg_db_24 = np.empty((0,param.img_size_24,param.img_size_24,param.input_channel),np.float32)
        neg_file_list = [f for f in os.listdir(param.neg_dir + "neg_hard/48/") if f.startswith("24_") and f.endswith(".npy")]
        for nid,db_name in enumerate(neg_file_list):

            tmp = np.load(param.neg_dir + "neg_hard/48/" + db_name)
            neg_db_24 = np.concatenate((neg_db_24,tmp),axis=0)

        
        neg_db_48 = np.empty((0,param.img_size_48,param.img_size_48,param.input_channel),np.float32)
        neg_file_list = [f for f in os.listdir(param.neg_dir + "neg_hard/48/") if f.startswith("48_") and f.endswith(".npy")]
        for nid,db_name in enumerate(neg_file_list):

            tmp = np.load(param.neg_dir + "neg_hard/48/" + db_name)
            neg_db_48 = np.concatenate((neg_db_48,tmp),axis=0)

        return [pos_db_12,pos_db_24,pos_db_48], neg_db_12, neg_db_24, neg_db_48

     
    
def load_db_calib_train(dim):
   
    print "Loading calibration training db..."

    annot_dir = param.db_dir + "AFLW/aflw/data/"
    annot_fp = open(annot_dir + "annot", "r")
    raw_data = annot_fp.readlines()
    
    #pos image cropping
    x_db = [0 for _ in xrange(len(raw_data))]
    for i,line in enumerate(raw_data):
        
        parsed_line = line.split(',')

        filename = parsed_line[0][3:-1]
        xmin = int(parsed_line[1])
        ymin = int(parsed_line[2])
        xmax = xmin + int(parsed_line[3])
        ymax = ymin + int(parsed_line[4][:-2])

        img = Image.open(param.pos_dir+filename)
        
        #truncated image(error)
        if i == 8160 or i == 14884 or i == 14886:
            continue

        #check if gray
        if len(np.shape(np.asarray(img))) != param.input_channel:
            img = np.asarray(img)
            img = np.reshape(img,(np.shape(img)[0],np.shape(img)[1],1))
            img = np.concatenate((img,img,img),axis=2)
            img = Image.fromarray(img)

        if xmax >= img.size[0]:
            xmax = img.size[0]-1
        if ymax >= img.size[1]:
            ymax = img.size[1]-1
        
        x_db_list = [0 for _ in xrange(param.cali_patt_num)]

        for si,s in enumerate(param.cali_scale):
            for xi,x in enumerate(param.cali_off_x):
                for yi,y in enumerate(param.cali_off_y):
                    
                    new_xmin = xmin - x*float(xmax-xmin)/s
                    new_ymin = ymin - y*float(ymax-ymin)/s
                    new_xmax = new_xmin+float(xmax-xmin)/s
                    new_ymax = new_ymin+float(ymax-ymin)/s
                    
                    new_xmin = int(new_xmin)
                    new_ymin = int(new_ymin)
                    new_xmax = int(new_xmax)
                    new_ymax = int(new_ymax)


                    if new_xmin < 0 or new_ymin < 0 or new_xmax >= img.size[0] or new_ymax >= img.size[1]:
                        continue
                    
                    cropped_img = util.img2array(img.crop((new_xmin, new_ymin, new_xmax, new_ymax)),dim)
                    calib_idx = si*len(param.cali_off_x)*len(param.cali_off_y)+xi*len(param.cali_off_y)+yi

                    #for debugging
                    #cropped_img.save(param.pos_dir + str(i)  + ".jpg")

                    x_db_list[calib_idx] = [cropped_img,calib_idx]

            
        x_db_list = [elem for elem in x_db_list if type(elem) != int]
        if len(x_db_list) > 0:
            x_db[i] = x_db_list

    x_db = [elem for elem in x_db if type(elem) != int]    
    x_db = [x_db[i][j] for i in xrange(len(x_db)) for j in xrange(len(x_db[i]))]
    
    return x_db


def load_db_test():
    
    print "Loading test db..."
    
    annot_dir = param.test_dir + "FDDB-folds/"
    test_img_name = [0 for _ in range(param.fold_num)]
    test_annot = [0 for _ in range(param.fold_num)]
       
    
    for fid in xrange(1,param.fold_num+1):
        
        print fid, "/", 10, "folds is loading..."
        fold_img_name = []
        fold_annot = []

        index = str(fid).zfill(2)
        annot_file = annot_dir + "FDDB-fold-" + index + "-ellipseList.txt"
        
        fp = open(annot_file)
        raw_data = fp.readlines()
        
        stage = 0
        for parsed_data in raw_data:
                                
            if stage == 0:
                file_name = parsed_data.rstrip()
                stage = 1

            elif stage == 1:
                face_num = int(parsed_data)
                it = 0
                
                fold_img_name.append(file_name)
                fold_annot_line = [0 for r in xrange(face_num)]
                stage = 2

            elif stage == 2:
                splitted = parsed_data.split()
                y_rad = max([float(splitted[0]) * math.cos(abs(float(splitted[2]))), float(splitted[0]) * math.sin(abs(float(splitted[2])))])
                x_rad = max([float(splitted[1]) * math.sin(abs(float(splitted[2]))), float(splitted[1]) * math.cos(abs(float(splitted[2])))])
                
                x_min = float(splitted[3]) - x_rad
                x_max = float(splitted[3]) + x_rad
                y_min = float(splitted[4]) - y_rad
                y_max = float(splitted[4]) + y_rad
                

                fold_annot_line[it] = [x_min, y_min, x_max, y_max]

                it += 1
                face_num -= 1
                if face_num == 0:
                    fold_annot.append(fold_annot_line)
                    stage = 0

        fp.close()

        test_img_name[fid-1] = fold_img_name
        test_annot[fid-1] = fold_annot
    
    
    return test_img_name, test_annot



