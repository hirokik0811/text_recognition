'''
Created on Sep 27, 2017

@author: hiroki
'''
import cv2
import numpy as np
import xml.etree.ElementTree

datalist = np.empty((0, 32*32 + 62), float)
char_list = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
print(len(char_list))
e = xml.etree.ElementTree.parse('char.xml').getroot()
for image in e:
    img = cv2.imread(image.get('file'))
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray_img, (32, 32))
    label = np.zeros(62)
    label[char_list.find(image.get('tag'))] = 1.0
    image_data = np.array([])
    image_data = np.append(image_data, resized)
    image_data = np.append(image_data, label)
    datalist = np.row_stack([datalist, image_data])
    
data_file = open('data', 'wb')
np.save(data_file, datalist)
data_file.close()