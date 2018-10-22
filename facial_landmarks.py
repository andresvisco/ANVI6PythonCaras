# USAGE
# python facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg 

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
from imutils.video import VideoStream


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=False,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=False,
	help="path to input image")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
#predictor = dlib.shape_predictor(args["shape_predictor"])
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# detect faces in the grayscale image
vs=VideoStream(src=0).start()
xaLimiteL=36
xaLimiteU=40
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    frame=vs.read()
    frame = imutils.resize(frame, width=500)
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    rects = detector(gray, 0)
    xaP=[]
    xa=0
    suma=0
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        #busco cada punto en el shape
        for(x,y) in shape:                
            #aislo los 2 puntos correspondientes al ojo izquierdo (parpado arriba y abajo)
            if (xa>=xaLimiteL and xa<=xaLimiteU):
                if(xa%2!=0):                
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
                    suma=x/y
                    
                    xaP+=[y]
                    
                    
                    if len(xaP)==2:
                        diferencia = abs(xaP[0] - xaP[1])
                        
                        if diferencia>4:
                            cv2.putText(frame,"Ojo abierto",(0,20),font, 0.6,(0,255,0),1, cv2.LINE_AA)                            
                        else:
                            cv2.putText(frame,"Ojo cerrado",(0,20),font, 0.6,(0,255,0),1, cv2.LINE_AA)                            


            else:
                xaP=[]
                
            xa+=1
        cv2.imshow("frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        cv2.destroyAllWindows()
        vs.stop()
        break

# show the output image with the face detections + facial landmarks


##while True:
##    frame = vs.read()
##    frame = imutils.resize(frame, width=600)
##    (h, w) = frame.shape[:2]
#    
##    imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300))
##    , 1.0, (300, 300),	(104.0, 177.0, 123.0), swapRB=False, crop=False)
#    
##    detector.run(predictor, imageBlob)
#
#for (i, rect) in enumerate(rects):
#    shape = predictor(gray, rect)
#    shape = face_utils.shape_to_np(shape)
#    (x, y, w, h) = face_utils.rect_to_bb(rect)
#    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#    cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10)
#    ,cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#    for (x, y) in shape:
#        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)