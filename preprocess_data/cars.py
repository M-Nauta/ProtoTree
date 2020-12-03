import os
import shutil
import scipy.io
import numpy as np
import time
from PIL import Image

path = './data/cars/dataset/'

if not os.path.isdir(path):
    os.mkdir(path)

time_start = time.time()

# convert ids to class names
class_names = scipy.io.loadmat('./data/cars/devkit/cars_meta.mat')
class_ids_to_names = dict()
for row in range(len(class_names['class_names'][0])):
    name = class_names['class_names'][0][row][0]
    name = name.replace("/", "") #remove slash to prevent directory errors
    name = name.replace(" ", "_") #replace space
    class_ids_to_names[row+1] = name

# adapted from https://github.com/tonylaioffer/cnn_car_classification/blob/master/data_prepare.py
mat = scipy.io.loadmat('./data/cars/devkit/cars_train_annos.mat')
# print("annotations: ", mat['annotations'])
training_class = mat['annotations']['class']
training_fname = mat['annotations']['fname']
training_x1 = mat['annotations']['bbox_x1']
training_y1 = mat['annotations']['bbox_y1']
training_x2 = mat['annotations']['bbox_x2']
training_y2 = mat['annotations']['bbox_y2']

mat = scipy.io.loadmat('./data/cars/devkit/cars_test_annos_withlabels.mat')
# print(mat['annotations'])
testing_class = mat['annotations']['class']
testing_fname = mat['annotations']['fname']

training_source = './data/cars/cars_train/' # specify source training image path
training_output = path+'train/' # specify target trainig image path (trainig images need to be orgnized to specific structure)
if not os.path.exists(training_output):
    os.mkdir(training_output)

for idx, cls in enumerate(training_class[0]):
    cls = cls[0][0]
    fname = training_fname[0][idx][0]
    print(cls, class_ids_to_names[cls], fname)
    output_path = os.path.join(training_output, class_ids_to_names[cls])
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    shutil.copy(os.path.join(training_source, fname), os.path.join(output_path, fname))

testing_source = './data/cars/cars_test/' # specify source testing image path
testing_output = path+'test/' # specify target testing image path (testing images need to be orgnized to specific structure)
if not os.path.exists(testing_output):
        os.mkdir(testing_output)
for idx, cls in enumerate(testing_class[0]):
    cls = cls[0][0]
    fname = testing_fname[0][idx][0]
    output_path = os.path.join(testing_output, class_ids_to_names[cls])
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    shutil.copy(os.path.join(testing_source, fname), os.path.join(output_path, fname))
 
time_end = time.time()
print('Cars dataset processed, %s!' % (time_end - time_start))