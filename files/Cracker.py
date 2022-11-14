#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 16:35:16 2022

@author: archquin
"""


import os
import cv2
import qrcode as qr
import numpy as np
import tensorflow as tf
from tensorflow import keras


 
XeroX = tf.keras.models.load_model('Mi6.h5')
XeroX.summary()


path = "Wordscr/"

    
# https://stackoverflow.com/questions/30230592/loading-all-images-using-imread-from-a-given-folder

def load_images_from_folder(folder):
    images = []
    gomu = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        gomu.append(filename)
        if img is not None:
            images.append(img)
    return gomu

X = load_images_from_folder(path)
Y,Z= [],[]
for x,i in enumerate(X):
    j = i.replace('line_','').replace("_img_",',').replace("_letter_",',').replace(".jpg",'')
    Y.append([j])
    j = j.split(',')
    Z.append(j)
    
Z.sort()


D1,D2,D3=[],[],[]
for i in Z:
  #  print(len(i))
    if len(i)==1:
        D1.append(i)
    elif len(i)==2:
        D2.append(i)
    elif len(i)==3:
        D3.append(i)

for i in D1:
    for j in D2:
        if  i[0] == j[0]:
            for k in D3:
                if j == k[0:2]:
                    filename = 'line_'+i[0]+'_img_'+j[1]+'_letter_'+k[2]+'.jpg'
                    im = cv2.imread(path+filename)
                    cv2.imshow('Im',im)
                    cv2.waitKey()
cv2.destroyAllWindows()
