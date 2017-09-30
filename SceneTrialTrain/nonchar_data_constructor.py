'''
Created on Sep 29, 2017

@author: hiroki
'''
import os
import cv2
import numpy as np
import random

datalist = np.empty((0, 32*32), float)

image_num = 0
for file in os.listdir('./apanar_06.08.2002'):
    img = cv2.imread('./apanar_06.08.2002/'+file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resize_rate = random.randrange(50, 100)/100
    gray_resized = cv2.resize(gray, None, fx = resize_rate, fy = resize_rate)
    for i in range(200):
        x_pos = random.randrange(gray_resized.shape[0] - 32)
        y_pos = random.randrange(gray_resized.shape[1] - 32)
        patch = gray_resized[x_pos:x_pos+32, y_pos:y_pos+32]
        cv2.imwrite(r'./patches/'+ str(image_num) + '_' + str(i) + '.png', patch)
    image_num += 1