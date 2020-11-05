
import cv2
import numpy as np
 
path = 'Big.png'
img = cv2.imread(path)
imgContour = img.copy()
 
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray,(7,7),1)
imgCanny = cv2.Canny(imgBlur,50,50)

hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#Blue color
lower_blue = np.array([100,110,110])
upper_blue = np.array([130,255,255])
blue_mask = cv2.inRange(hsv_img, lower_blue, upper_blue)
blue = cv2.bitwise_and(img, img, mask = blue_mask)

# Red Color
lower_red = np.array([0,100,20])
upper_red = np.array([8,255,255])
lower_red2 = np.array([175,100,20])
upper_red2 = np.array([179,255,255])

red_mask = cv2.inRange(hsv_img, lower_red, upper_red)
red_mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
mask_red = cv2.add(red_mask, red_mask2)
red = cv2.bitwise_and(img, img, mask = mask_red)
med = cv2.medianBlur(mask_red, 5)
cv2.imwrite("ref_red.png", med)

#White color
lower_white = np.array([250,250,250])
upper_white = np.array([255,255,255])
white_mask = cv2.inRange(hsv_img, lower_white, upper_white)
white = cv2.bitwise_and(img, img, mask = white_mask)

def getContours(img):

    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    tri=0
    sqr=0
    cir=0
    area_tri=0
    area_sqr=0
    area_cir=0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        #print(area)

        if area>10:
            cv2.drawContours(imgContour, cnt, -1, (0, 255, 0), 1)
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            #print(len(approx))
            x, y, w, h = cv2.boundingRect(approx)
 
            #if objCor >=3 and objCor <=3.5:
            if len(approx)==3:  
              tri+=1
              area_tri+=area+area
              objectType="Tri"

            elif len(approx)==4:
              sqr+=1
              area_sqr=area+area
              objectType="Squ"

            elif len(approx)>5: 
              cir+=1
              area_cir=area+area
              objectType="Cir"

            else:objectType="None"
 
            cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),2)
            #Put text
            cv2.putText(imgContour,objectType,
                        (x+(w//2)-10,y+(h//2)-10),cv2.FONT_HERSHEY_COMPLEX,0.4,
                        (0,0,0),2)

    totaltri=(tri)
    totalsqr=(sqr)
    totalcir=(cir)
    totalfig=tri+sqr+cir
    total_area_tri=area_tri
    total_area_sqr=area_sqr
    total_area_cir=area_cir
    total_area = total_area_tri+total_area_sqr+total_area_cir
    print("--------------------------------- Number of figures ---------------------------------")
    print("Total Number of Triangles: ", totaltri)
    print("Total Number of Squares: ", totalsqr)
    print("Total Number of Circles: ", totalcir)
    print("Total Number of Figures", totalfig)
    print("--------------------------------- Number of areas ---------------------------------")
    print("Total Area of Triangles: ", total_area_tri)
    print("Total Area of Squares: ", total_area_sqr)
    print("Total Area of Circles: ", total_area_cir)
    print("Total Area of Figures: ", total_area)


cv2.imshow('Color Segmentation: Red', red)
cv2.imshow('Color Segmentation: Blue', blue)

getContours(imgCanny)
imgBlank = np.zeros_like(img)
cv2.imshow("Final", imgContour)

#print(tri)
#print(sqr)
#print(cir)
cv2.waitKey(0)
