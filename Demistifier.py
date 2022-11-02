#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 14:13:29 2022

@author: archquin
"""


import cv2 
import os
import numpy as np
import qrcode as qr
#https://www.folkstalk.com/tech/draw-bounding-box-on-image-python-opencv-with-code-examples/

img  = cv2.imread('imageocr.jpg')
img = cv2.resize(img, None, fx=1  , fy=1 , interpolation=2 )
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)

contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


X = []
for c in contours:
    rect = cv2.boundingRect(c)
    if rect[2] < 12 or rect[3] < 15 : continue
   # print (cv2.contourArea(c))
    x,y,w,h = rect
    #cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,0),2)
  #  cv2.putText(img,'('+str(x)+','+str(y)+')'+'x'+'('+str(x+w)+','+str(y+h)+')',(x+w+10,y+h),0,0.3,(0,255,0))
    Y = '('+str(x)+','+str(y)+')'+'x'+'('+str(x+w)+','+str(y+h)+')'
    X.append(Y)
    #cv2.imshow("Show",img)
    #cv2.waitKey()  
    
#cv2.destroyAllWindows()

def canny(image):
    return cv2.Canny(image, 200, 200) 
    
# https://stackoverflow.com/questions/47627182/detecting-interword-space-in-ocr-using-python-and-opencv 
# idea for spaces
img = canny(img)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 9))
img = cv2.dilate(img, kernel)
#https://stackoverflow.com/questions/43053923/replace-black-by-white-and-white-by-black-in-images
img = cv2.subtract(255, img) 

#cv2.imshow("Show",img)
#cv2.waitKey()  

contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

img2  = cv2.imread('imageocr.jpg')
img2 = cv2.resize(img2, None, fx=1  , fy=1 , interpolation=5 )
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img2 = cv2.adaptiveThreshold(img2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31,11 )


def greater(a, b):
    momA = cv2.moments(a)        
    (xa,ya) = int(momA['m10']/momA['m00']), int(momA['m01']/momA['m00'])

    momB = cv2.moments(b)        
    (xb,yb) = int(momB['m10']/momB['m00']), int(momB['m01']/momB['m00'])
    if xa > xb:
        return 1

    if xa == xb:
        return 0
    else:
        return -1

def sort_contours(contours, x_axis_sort='LEFT_TO_RIGHT', y_axis_sort='TOP_TO_BOTTOM'):
    # initialize the reverse flag
    x_reverse = False
    y_reverse = False
    if x_axis_sort == 'RIGHT_TO_LEFT':
        x_reverse = True
    if y_axis_sort == 'BOTTOM_TO_TOP':
        y_reverse = True
    
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    
    # sorting on x-axis 
    sortedByX = zip(*sorted(zip(contours, boundingBoxes),
    key=lambda b:b[1][0], reverse=x_reverse))
    
    # sorting on y-axis 
    (contours, boundingBoxes) = zip(*sorted(zip(*sortedByX),
    key=lambda b:b[1][1], reverse=y_reverse))
    # return the list of sorted contours and bounding boxes
    return (contours, boundingBoxes)
    


import cv2 
import os
import numpy as np

def canny(image):
    return cv2.Canny(image, 200, 200) 
  






Y  = []

      
#https://www.folkstalk.com/tech/draw-bounding-box-on-image-python-opencv-with-code-examples/
for i in range(1,10):
    img  = cv2.imread('Wordscr/line_'+str(i)+'.jpg')
    img = cv2.resize(img, None, fx=1   , fy=1 , interpolation=2 )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)
    
    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
  
    
    # idea for spaces
    img = canny(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10 , 8)) 
    img = cv2.dilate(img, kernel)
    #https://stackoverflow.com//questions/43053923/replace-black-by-white-and-white-by-black-in-images
    img = cv2.subtract(255, img) 
    
    #cv2.imshow("Show",img)
    #cv2.waitKey()  
    
    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    img2  = cv2.imread('Wordscr/line_'+str(i)+'.jpg')
    img2 = cv2.resize(img2, None, fx=1  , fy=1 , interpolation=2 )
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2 = cv2.adaptiveThreshold(img2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31,11 )
    
    
    #contours, hierarchy = cv2.findContours(img , cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #contours, boundingBoxes = sort_contours(contours, x_axis_sort='LEFT_TO_RIGHT', y_axis_sort='TOP_TO_BOTTOM')
    
    
    X,Z = [],[]
   
    for l,c in enumerate(contours):
        rect = cv2.boundingRect(c)
        if rect[2] < 20 or rect[3] < 20  : continue
       # print (cv2.contourArea(c))
        x,y,w,h = rect
      
        X.append(x)
        Z.append(x) 
    Z.sort()
    for l,c in enumerate(contours):
        rect = cv2.boundingRect(c)
        if rect[2] < 20 or rect[3] < 20 : continue
       # print (cv2.contourArea(c))
        x,y,w,h = rect
      
        img5 = img2[y:y+h,x:x+w]

        cv2.imshow("Show",img5)
        if x != 0:
            cv2.imwrite("Wordscr/line_"+str(i)+"_img_"+str(Z.index(x))+".jpg", img5)
        else : 
            Y.append(len(Z))

        cv2.waitKey()  
cv2.destroyAllWindows()
    
'''    
    lie = 0
    for c in contours:
        rect = cv2.boundingRect(c)
        if rect[2] < 20 or rect[3] < 20 : continue
       # print (cv2.contourArea(c))
        x,y,w,h = rect
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        img3 = img2[y:y+h,x:x+w]
        lie += 1
      #  cv2.imshow("cropped", img3)
        cv2.imwrite("Wordscr/line_"+str(i)+"_img_"+str(lie)+".jpg", img3)
    
    
        cv2.imshow("Show",img)
        cv2.waitKey()  
    
    cv2.destroyAllWindows()
'''

mg = qr.make(Y)

mg.save('Y.jpg')