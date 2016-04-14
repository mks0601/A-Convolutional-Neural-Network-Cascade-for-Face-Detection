import numpy as np
import tensorflow as tf
from skimage.util.shape import view_as_windows
import os
import random
import Image
import etc
from compiler.ast import flatten
import sys
import copy
import math



a = [[1,2,3],[4,5,6]]

result_box_copy = [] 
result_box_copy += [copy.deepcopy([box[0], box[1]])  + copy.deepcopy([box[2]]) for box in a]

print a
            

