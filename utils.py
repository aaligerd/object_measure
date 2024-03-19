import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS


def getContours(img,cannyThresold=[50,150],showCanny=False,minArea=9000,filter=0,draw=False):
    imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur=cv2.GaussianBlur(imgGray,(5,5),5)
    imgCanny=cv2.Canny(imgBlur,cannyThresold[0],cannyThresold[1])
    kernel=np.ones((5,5))
    imgDial=cv2.dilate(imgCanny,kernel,iterations=3)
    imgThre=cv2.erode(imgDial,kernel=kernel)
    if showCanny:
        cv2.imshow("Canny",imgCanny)
        cv2.waitKey(0)
    contours,heiarchy=cv2.findContours(imgThre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    finalcontours=[]
    for contour in contours:
        area=cv2.contourArea(contour)
        if area>minArea:
            peri=cv2.arcLength(contour,True)
            approx=cv2.approxPolyDP(contour,0.02*peri,True)
            bounding_box=cv2.boundingRect(approx)
            if filter>0:
                if len(approx)==filter:
                    finalcontours.append([len(approx),area,approx,bounding_box,contour])
            else:
                finalcontours.append([len(approx),area,approx,bounding_box,contour])
    finalcontours=sorted(finalcontours,key=lambda x:x[1],reverse=True)
    if draw:
        for contour in finalcontours:
            cv2.drawContours(img,contour[4],-1,(0,0,255),3)
    return img,finalcontours
    
def findReorder(points):
    new_points=np.zeros_like(points)
    points=points.reshape((4,2))
    add=points.sum(1)
    new_points[0]=points[np.argmin(add)]
    new_points[3]=points[np.argmax(add)]
    diff=np.diff(points,axis=1)
    new_points[1]=points[np.argmin(diff)]
    new_points[2]=points[np.argmax(diff)]
    print(new_points)
    return new_points

def wrapImg(img,points,w,h):
    pts1=np.float32(points)
    pts2=np.float32([[0,0],[w,0],[0,h],[w,h]])
    matrix=cv2.getPerspectiveTransform(pts1,pts2)
    imgWrap=cv2.warpPerspective(img,matrix,(w,h))
    return imgWrap

def calculatedistance(point1,point2):
    return (((point2[0]-point1[0])**2+(point2[1]-point1[1])**2)**0.5)

