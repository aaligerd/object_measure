import cv2 as cv
import cvzone
import numpy as np
import pyautogui as pygui


s_width,s_height=pygui.size()
img=cv.imread('pen1.jpg')
imgWidth=img.shape[0]
imgHeight=img.shape[1]
if imgWidth>s_width or imgHeight>s_height:
    img=cv.resize(img,(int(imgWidth/2),int(imgHeight/2)))
imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
imgCanny=cv.Canny(imgGray,100,200)
kernel=np.ones((5,5),np.uint8)
imgDilated=cv.dilate(imgCanny,kernel,1)
imgErro=cv.erode(imgDilated,kernel,1)
imgContours,contours=cvzone.findContours(img,imgErro,filter=[4])
# Iterate over all contours
# for contour in contours:
#     # Approximate the contour to a polygon
#     perimeter = cv.arcLength(contour, True)
#     approx = cv.approxPolyDP(contour, 0.02 * perimeter, True)
#     print(approx)
    # If the contour has four vertices, it is likely a rectangle
    # if len(approx) == 4:
        # Compute the bounding box of the contour
        # x, y, w, h = cv.boundingRect(contour)
        # cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv.putText(img, f'Width: {w}', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # cv.putText(img, f'Height: {h}', (x, y - 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
cv.imshow("Dilated",imgDilated)
cv.imshow("Canny",imgCanny)
cv.imshow("Errosion",imgErro)
cv.imshow("Countours",imgContours)
cv.waitKey(0)
