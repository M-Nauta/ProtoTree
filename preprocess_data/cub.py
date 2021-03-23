import os
import shutil
import numpy as np
import time
from PIL import Image

path = './data/CUB_200_2011/'

time_start = time.time()

path_images = os.path.join(path,'images.txt')
path_split = os.path.join(path,'train_test_split.txt')
train_save_path = os.path.join(path,'dataset/train_crop/')
test_save_path = os.path.join(path,'dataset/test_crop/')
bbox_path = os.path.join(path, 'bounding_boxes.txt')
 
images = []
with open(path_images,'r') as f:
    for line in f:
        images.append(list(line.strip('\n').split(',')))
print("Images: ", images)
split = []
with open(path_split, 'r') as f_:
    for line in f_:
        split.append(list(line.strip('\n').split(',')))

bboxes = dict()
with open(bbox_path, 'r') as bf:
    for line in bf:
        id, x, y, w, h = tuple(map(float, line.split(' ')))
        bboxes[int(id)]=(x, y, w, h)

num = len(images)
for k in range(num):
    id, fn = images[k][0].split(' ')
    id = int(id)
    file_name = fn.split('/')[0]
    if int(split[k][0][-1]) == 1:
        
        if not os.path.isdir(train_save_path + file_name):
            os.makedirs(os.path.join(train_save_path, file_name))
        img = Image.open(os.path.join(os.path.join(path, 'images'),images[k][0].split(' ')[1])).convert('RGB')
        x, y, w, h = bboxes[id]
        cropped_img = img.crop((x, y, x+w, y+h))
        cropped_img.save(os.path.join(os.path.join(train_save_path,file_name),images[k][0].split(' ')[1].split('/')[1]))
        print('%s' % images[k][0].split(' ')[1].split('/')[1])
    else:
        if not os.path.isdir(test_save_path + file_name):
            os.makedirs(os.path.join(test_save_path,file_name))
        img = Image.open(os.path.join(os.path.join(path, 'images'),images[k][0].split(' ')[1])).convert('RGB')
        x, y, w, h = bboxes[id]
        cropped_img = img.crop((x, y, x+w, y+h))
        cropped_img.save(os.path.join(os.path.join(test_save_path,file_name),images[k][0].split(' ')[1].split('/')[1]))
        print('%s' % images[k][0].split(' ')[1].split('/')[1])


train_save_path = os.path.join(path,'dataset/train_corners/')
test_save_path = os.path.join(path,'dataset/test_full/')

num = len(images)
for k in range(num):
    id, fn = images[k][0].split(' ')
    id = int(id)
    file_name = fn.split('/')[0]
    if int(split[k][0][-1]) == 1:
        
        if not os.path.isdir(train_save_path + file_name):
            os.makedirs(os.path.join(train_save_path, file_name))
        img = Image.open(os.path.join(os.path.join(path, 'images'),images[k][0].split(' ')[1])).convert('RGB')
        x, y, w, h = bboxes[id]
        width, height = img.size
        
        hmargin = int(0.1*h)
        wmargin = int(0.1*w)
        
        cropped_img = img.crop((0, 0, min(x+w+wmargin, width), min(y+h+hmargin, height)))
        cropped_img.save(os.path.join(os.path.join(train_save_path,file_name),"upperleft_"+images[k][0].split(' ')[1].split('/')[1]))
        cropped_img = img.crop((0, max(y-hmargin, 0), min(x+w+wmargin, width), height))
        cropped_img.save(os.path.join(os.path.join(train_save_path,file_name),"lowerleft_"+images[k][0].split(' ')[1].split('/')[1]))
        cropped_img = img.crop((max(x-wmargin,0), 0, width, min(y+h+hmargin, height)))
        cropped_img.save(os.path.join(os.path.join(train_save_path,file_name),"upperright_"+images[k][0].split(' ')[1].split('/')[1]))
        cropped_img = img.crop(((max(x-wmargin,0), max(y-hmargin, 0), width, height)))
        cropped_img.save(os.path.join(os.path.join(train_save_path,file_name),"lowerright_"+images[k][0].split(' ')[1].split('/')[1]))
       
        img.save(os.path.join(os.path.join(train_save_path,file_name),"normal_"+images[k][0].split(' ')[1].split('/')[1]))
        
        print('%s' % images[k][0].split(' ')[1].split('/')[1])
    else:
        if not os.path.isdir(test_save_path + file_name):
            os.makedirs(os.path.join(test_save_path,file_name))
        shutil.copy(path + 'images/' + images[k][0].split(' ')[1], os.path.join(os.path.join(test_save_path,file_name),images[k][0].split(' ')[1].split('/')[1]))
        print('%s' % images[k][0].split(' ')[1].split('/')[1])
time_end = time.time()
print('CUB200, %s!' % (time_end - time_start))
