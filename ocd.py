 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 17:24:06 2022

@author: archquin
"""




import cv2 
import os
import numpy as np
import qrcode
path = 'Wordscr/'


Y = cv2.imread('Y.jpg')
detect = cv2.QRCodeDetector()
value, points, straight_qrcode = detect.detectAndDecode(Y)
value = value.replace('[','').replace(']','').split(",")
dlib = []
for i in value:
    dlib.append(int(i))
    
    
    

for i in range(2,8):
    for j in range(1,dlib[i]):
        img4  = cv2.imread(path+'line_'+str(i)+"_img_"+str(j)+'.jpg')
        img4 = cv2.resize(img4, None, fx=0.6  , fy=.5 , interpolation=1 )
        img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
    
        img4 = cv2.adaptiveThreshold(img4, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
        contours, _ = cv2.findContours(img4, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        X,Z = [],[]
       
        for l,c in enumerate(contours):
            rect = cv2.boundingRect(c)
            if rect[2] < 7 or rect[3] < 7  : continue
            x,y,w,h = rect
          
            X.append(x)
            Z.append(x)  
            img5 = img4[y:y+h,x:x+w]
        Z.sort() 
        for l,c in enumerate(contours):
            rect = cv2.boundingRect(c)
            if rect[2] < 7 or rect[3] < 7 : continue
           # print (cv2.contourArea(c))
            x,y,w,h = rect
          
            img5 = img4[y:y+h,x:x+w]
    
            cv2.imshow("Show",img5)
            if x != 0: 
                cv2.imwrite("Wordscr/line_"+str(i)+"_img_"+str(j)+"_letter_"+str(Z.index(x))+".jpg", img5)
    
            cv2.waitKey()  
cv2.destroyAllWindows()
