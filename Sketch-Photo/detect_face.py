import cv2
import os
import numpy as np
from tqdm import tqdm




def main(path):
    #os.makedirs('crop')
    j = 1
    for i in tqdm(range(1, 200000)):
        #a = path + str(i) + ".jpg"
        s=str(i)
        s=s.zfill(6)
        a = path + s + ".jpg"
        #print(a)
        #img = cv2.imread(a)
        img = cv2.imread(a)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 15)
        for (x,y,h,w) in faces:
            roi_gray = gray[y:y+h+15, x:x+w+15]
            roi_color = img[y:y+h+15, x:x+w+15]

            cv2.imwrite('crop/' + str(j) + '.jpg', roi_color)
            if(j==100000):
            	return
            j = j+1
        

if __name__ == '__main__':
    
    path = '/Users/chandana/Downloads/celebA/'
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    main(path)
    
    
        

