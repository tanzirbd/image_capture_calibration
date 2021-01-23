import cv2
import numpy as np
import glob

img_array = []
time=12.1
for x in range (122,124):
    x=str(x)
    #print(x)
    time=str(time) 
    path='17_3d_image_new/'
    img = cv2.imread(path+'image_'+x+'_3d_image_time-'+time+'.png')
    #print(img.shape)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append (img)
    time=float(time)
    time+=0.1
    time=round(time,2)
       #print(time)
out = cv2.VideoWriter ('3d_skel_moving.avi', cv2.VideoWriter_fourcc (*'DIVX'), 4, size)

for i in range (len (img_array)):
    out.write (img_array[i])

out.release ()
