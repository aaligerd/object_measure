import cv2 as cv
import cvzone
import numpy as np
import pyautogui as pygui
import utils

imagepath='billboard.png'
s_width,s_height=pygui.size()
dpi=96
img=cv.imread(imagepath)
cv.imshow("Image",img)
imgWidth=img.shape[0]
imgHeight=img.shape[1]
# if imgWidth>s_width or imgHeight>s_height:
#     img=cv.resize(img,(int(imgWidth/2),int(imgHeight/2)))
img2,contours=utils.getContours(img,draw=True,minArea=900)
if len(contours)!=0:
    biggest=contours[0][2]
    for contour in contours:
        cv.polylines(img2,[contour[2]],True,(1,1,255),2)
        nPoints=utils.findReorder(contour[2])
        mW=(round(utils.calculatedistance(nPoints[0][0],nPoints[1][0])/dpi,1))
        mH=(round(utils.calculatedistance(nPoints[0][0],nPoints[2][0])/dpi,1))
    print(mW,mH)
    cv.imshow("Main Image",img2)
    cv.waitKey(0)
else:
    print("No Contours")
cv.waitKey(0)

