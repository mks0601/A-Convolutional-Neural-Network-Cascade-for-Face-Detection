# A Convolutional Neural Network Cascade for Face Detection

This repo is re-implementation of the [paper](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Li_A_Convolutional_Neural_2015_CVPR_paper.pdf) in TensorFlow.

## Start
### Preparing data
1. Download AFLW dataset (positive) and COCO dataset (negative) for training. Any other dataset can be used instead of COCO datset for negative one.

2. Download FDDB dataset for testing.

3. Run `data_parse.py` in the `dataGen` folder before training and testing network.

### Training classification net
```
12-net: python train_12net.py

24-net: python train_24net.py

48-net: python train_48net.py
```
### Training calibration net
```
12-calib net: python train_calib.py 12

24-calib net: python train_calib.py 24

48-calib net: python train_calib.py 48
```
### Hard negative mining(save hard neg db to disk in neg_train/neg_hard/)
```
hard neg db to train 24-net: python hard_neg_mining.py 24

hard neg db to train 48-net: python hard_neg_mining.py 48
```
### Test
```
python test.py
```

## Implementation
Implemented with TensorFlow and yields similar result with paper

training set: AFLW dataset(positive), COCO dataset(negative)

test set: FDDB dataset


## Result(gren: GT, blue: detected face)
![face](https://cloud.githubusercontent.com/assets/13601723/15349050/a8192776-1d0a-11e6-86be-243175c22ba4.png)

![2](https://cloud.githubusercontent.com/assets/13601723/15348767/4d1d0768-1d08-11e6-99d9-07785c131c4b.png)
![5](https://cloud.githubusercontent.com/assets/13601723/15348775/5137c464-1d08-11e6-85d8-f4323ce7cf4b.png)
![3](https://cloud.githubusercontent.com/assets/13601723/15348777/513f09fe-1d08-11e6-8cf2-751ea470aecd.png)
![1](https://cloud.githubusercontent.com/assets/13601723/15348745/24f5f362-1d08-11e6-957d-e41718122426.png)
![8](https://cloud.githubusercontent.com/assets/13601723/15348772/5118bb3c-1d08-11e6-8274-be075da97d6f.png)
![7](https://cloud.githubusercontent.com/assets/13601723/15348773/512d4002-1d08-11e6-87be-2ac486923a07.png)
![6](https://cloud.githubusercontent.com/assets/13601723/15348774/512fb3be-1d08-11e6-9c77-d7a445d94f7e.png)
![4](https://cloud.githubusercontent.com/assets/13601723/15348776/513dc3d2-1d08-11e6-804f-37fd3299bf74.png)
![9](https://cloud.githubusercontent.com/assets/13601723/15348771/50f0e6e8-1d08-11e6-99af-c26a3428bbe2.png)
![10](https://cloud.githubusercontent.com/assets/13601723/15349097/f515f11c-1d0a-11e6-96dd-38c07d3cc05c.png)

