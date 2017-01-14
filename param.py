
#dir
db_dir = "/media/sda1/Data/Face_detection/"
pos_dir = db_dir + "AFLW/aflw/data/flickr/"
neg_dir = db_dir + "neg_train/"
test_dir = db_dir + "FDDB/"
fig_dir = db_dir + "result/fig/"
model_dir = db_dir + "result/model/"

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
b_init = 0.0
w_std = 0.1
lr = 5e-2
epoch_num = 100
pos_batch = 32
neg_batch = 96
mini_batch = 128
batch_iter = int(1e4)

#result parameters
thr_12 = 3e-3
thr_24 = 1e-9
thr_48 = 1e-2

#training parameters
window_stride = 4
face_minimum = 20
downscale = 1.18
pyramid_num = 16

cali_scale = [0.83, 0.91, 1.0, 1.10, 1.21]
cali_off_x = [-0.17, 0., 0.17]
cali_off_y = [-0.17, 0., 0.17]
cali_patt_num = len(cali_scale) * len(cali_off_x) * len(cali_off_y)
