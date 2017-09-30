'''
Created on Sep 27, 2017

@author: hiroki
'''

import cv2
import numpy as np

datalist = np.empty((0, 32*32 + 62), float)

for i in range(0, 62):
    dir_num_str = str(i+1)
    label = np.zeros(62)
    label[i] = 1.0
    for j in range(1, 30):
        filename = './GoodImg/Bmp/Sample0'
        if i < 9:
            filename += ('0' + dir_num_str + '/img00' + dir_num_str)
        else:
            filename += (dir_num_str + '/img0' + dir_num_str)
        if j < 10:
            filename += ('-0000'+str(j) + '.png')
        else:
            filename += ('-000'+ str(j) + '.png')
        img = cv2.imread(filename)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        border_width = max([0, int((gray_img.shape[0]-gray_img.shape[1])/2)])
        border_height = max([0, int((gray_img.shape[1]-gray_img.shape[0])/2)])
        bordered = cv2.copyMakeBorder(gray_img, border_height, border_height, border_width, border_width
                                      , borderType = cv2.BORDER_REFLECT_101)
        resized = cv2.resize(bordered, (32, 32))
        image_data = np.array([])
        image_data = np.append(image_data, resized)
        image_data = np.append(image_data, label)
        datalist = np.row_stack([datalist, image_data])

data_file = open('data', 'wb')
np.save(data_file, datalist)
data_file.close()