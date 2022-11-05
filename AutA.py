#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 16:12:22 2022

@author: archquin
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2

import os

#https://www.folkstalk.com/tech/draw-bounding-box-on-image-python-opencv-with-code-examples/

jpeg = 'imageocr.jpg'
fix = 1
me = 1

img  = cv2.imread(jpeg)
img = cv2.resize(img, None, fx=fix  , fy=me , interpolation=5 )
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)

contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


cv2.imshow("Show",img)
cv2.waitKey()  
    

def canny(image):
    return cv2.Canny(image, 500, 1200) 
    
# https://stackoverflow.com/questions/47627182/detecting-interword-space-in-ocr-using-python-and-opencv 
# idea for spaces
img = canny(img)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 2))
img = cv2.dilate(img, kernel)
#https://stackoverflow.com/questions/43053923/replace-black-by-white-and-white-by-black-in-images
img = cv2.subtract(255, img) 

contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

img2  = cv2.imread(jpeg)
img2 = cv2.resize(img2, None, fx=fix  , fy=me , interpolation=5)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img2 = cv2.adaptiveThreshold(img2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31,11  )

cv2.imshow("Show",img)
cv2.waitKey()  




lie = 0
Y,U = [],[]
for c in contours:
    rect = cv2.boundingRect(c)
    if rect[2] < 50 or rect[3] < 50: continue
   # print (cv2.contourArea(c))
    x,y,w,h = rect
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    Y.append(y)   
    U.append(y)
    img3 = img2[y:y+h+3,x:x+w]
    lie += 1
  #  cv2.imshow("cropped", img3)
    cv2.imwrite("Wordscr/line_"+str(lie)+".jpg", img3)


    cv2.imshow("Show",img)
    cv2.waitKey()  

cv2.destroyAllWindows()

