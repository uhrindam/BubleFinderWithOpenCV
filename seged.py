import cv2
import numpy as np
import math
from random import randint
import sys

"""
-----------Argumentum lista-----------
    argv[0] --> file neve
    argv[1] --> a feldolgozandó kép elérési útvonala
    argv[2] --> a file neve kiterjesztéssel együtt
    argv[3] --> a feldolgozott képek mentési helyének elérési útvonala
    argv[4] --> ha van szükség a feldolgozási lépések megjelenítésére, akkor ez megadja, hogy melyik képen
    argv[5] --> a kép sorszáma. (A feldolgozandó képek között hányadik.)
--------------------------------------
"""

TRESHHOLDMIN = 210 #Ez a treshold érték volt a leghatékonyabb

#------------------------------------------------------------------
img_input = cv2.imread("C:\\Users\\Adam\\Desktop\\samples\\hulk1.jpg") #1--> greyscale
#img_input = cv2.imread(sys.argv[1]) #1--> greyscale
#------------------------------------------------------------------

ROWS,COLLUMS,_ = img_input.shape

LETTERHIGHT = ROWS / 200 # áltában egy betű mérete így viszonyul magának az oldalnak a méretéhez

img_greyscaled = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)


th, im_th = cv2.threshold(img_greyscaled, TRESHHOLDMIN, 255, cv2.THRESH_BINARY_INV);
#cv2.imwrite("02.jpg", im_th)-----------------------------------------------------------------------------------------------------

#Mask a fillhez
mask = np.zeros((ROWS+2, COLLUMS+2), np.uint8)

im_frame = im_th.copy()
cv2.floodFill(im_frame, mask, (0,0), 128)
cv2.floodFill(im_frame, mask, (COLLUMS-1,0), 128)
cv2.floodFill(im_frame, mask, (0,ROWS-1), 128)
cv2.floodFill(im_frame, mask, (COLLUMS-1,ROWS-1), 128)
#cv2.imwrite("03.0.jpg", im_frame)-----------------------------------------------------------------------------------------------------

#--------------------------------------
#a négy sarokról kitöltöm a keretet
cv2.floodFill(im_th, mask, (0,0), 255)
cv2.floodFill(im_th, mask, (COLLUMS-1,0), 255)
cv2.floodFill(im_th, mask, (0,ROWS-1), 255)
cv2.floodFill(im_th, mask, (COLLUMS-1,ROWS-1), 255)
#cv2.imwrite("03.jpg", im_th)-----------------------------------------------------------------------------------------------------
#--------------------------------------

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
grad = cv2.morphologyEx(im_th, cv2.MORPH_CLOSE, kernel)
#cv2.imwrite("04.jpg", grad)-----------------------------------------------------------------------------------------------------

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
connected = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, kernel)
#cv2.imwrite("05.jpg", connected)-----------------------------------------------------------------------------------------------------

_,contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)

keretezett = img_input.copy()
m = np.zeros(im_th.shape, dtype=np.uint8) #mask a konturokhoz

im_floodfill = im_th.copy()

for idx in range(len(contours)):
    x, y, w, h = cv2.boundingRect(contours[idx])
    m[y:y+h, x:x+w] = 0
    cv2.drawContours(m, contours, idx, (255, 255, 255), -1)
    r = float(cv2.countNonZero(m[y:y+h, x:x+w])) / (w * h)

    if w > LETTERHIGHT/2 and h > LETTERHIGHT and h < LETTERHIGHT*5: #0.7 8 8 30    r > 0.7 and
        cv2.rectangle(keretezett, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)
        ax = math.floor( x+(w/2))
        ay = math.floor(y+(h/2))
        while(im_floodfill[ay,ax] == 255):
            a = randint(0, 3)
            if a == 0:
                ax = ax - 1
            elif a == 1:
                ax = ax + 1
            elif a == 1:
                ay = ay + 1
            else:
                ay = ay + 1
        cv2.floodFill(im_floodfill, mask, (ax, ay), 128)
#----------------------------------------------------------------------------
#cv2.imwrite("06.jpg", keretezett)-----------------------------------------------------------------------------------------------------
#cv2.imwrite("07.jpg", im_floodfill)-----------------------------------------------------------------------------------------------------

im_floodfill_inv = cv2.bitwise_not(im_floodfill)

im_out = im_th | im_floodfill_inv
#cv2.imwrite("08.jpg", im_out)-----------------------------------------------------------------------------------------------------

th, im_th2 = cv2.threshold(im_out, TRESHHOLDMIN, 255, cv2.THRESH_BINARY)
#cv2.imwrite("09.jpg", im_th2)-----------------------------------------------------------------------------------------------------

im_th3 = cv2.bitwise_not(im_th2)
#cv2.imwrite("10.jpg", im_th3)-----------------------------------------------------------------------------------------------------
_,contour,hier = cv2.findContours(im_th3,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

for cnt in contour:
    cv2.drawContours(im_th3,[cnt],0,255,-1)
#cv2.imwrite("11.jpg", im_th3)-----------------------------------------------------------------------------------------------------

segm = cv2.bitwise_not(im_th3)
#cv2.imwrite("12.jpg", segm)-----------------------------------------------------------------------------------------------------


#---------------------------------------------------
im_frame_inv = cv2.bitwise_not(im_frame)

im_keret = im_th | im_frame_inv
#cv2.imwrite("13.1.jpg", im_keret)-----------------------------------------------------------------------------------------------------

th, im_th2 = cv2.threshold(im_keret, TRESHHOLDMIN, 255, cv2.THRESH_BINARY)
#cv2.imwrite("13.2.jpg", im_th2)-----------------------------------------------------------------------------------------------------

im_th3 = cv2.bitwise_not(im_th2)
#cv2.imwrite("13.3.jpg", im_th3)-----------------------------------------------------------------------------------------------------
_,contour,hier = cv2.findContours(im_th3,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

for cnt in contour:
    cv2.drawContours(im_th3,[cnt],0,255,2)
#cv2.imwrite("13.4.jpg", im_th3)-----------------------------------------------------------------------------------------------------

segm2 = cv2.bitwise_not(im_th3)
#cv2.imwrite("13.5.jpg", segm2)-----------------------------------------------------------------------------------------------------


sss = segm & segm2

#cv2.imwrite("13.jpg", sss)-----------------------------------------------------------------------------------------------------

talan = img_input.copy()

for i in range(ROWS):
    for j in range(COLLUMS):
        if sss[i,j] == 0:
            talan[i,j] = 255



cv2.imwrite("completed.jpg", talan)
#---------------------------------------------------
# cv2.imwrite(sys.argv[3]+"\\"+sys.argv[2], talan)
#---------------------------------------------------

"""
blurred = cv2.GaussianBlur(talan, (5, 5), 0)
edges = cv2.Canny(blurred,250,250);
cv2.imwrite("15.jpg", edges)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
grad = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
cv2.imwrite("16.jpg", grad)

e = cv2.SuperpixelSEEDS.getLabelContourMask(talan, 2)
"""


#---------------------------------------------------------------------------------------------------------------------------
"""
csakfekete = cv2.cvtColor(talan, cv2.COLOR_BGR2GRAY)

black = np.zeros(csakfekete.shape, dtype=np.uint8)
black = cv2.bitwise_not(black);

for i in range(ROWS):
    for j in range(COLLUMS):
        if csakfekete[i,j] < 60:
            black[i,j] = 0

cv2.imwrite("15.jpg", black)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
grad = cv2.morphologyEx(black, cv2.MORPH_CLOSE, kernel)
cv2.imwrite("16.jpg", grad)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
connected = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, kernel)
cv2.imwrite("17.jpg", connected)

#---------------------------------------------------csak a fekete elmosva
_,contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)

keretezett = talan.copy()
m = np.zeros(black.shape, dtype=np.uint8) #mask a konturokhoz

im_floodfill = black.copy()

for idx in range(len(contours)):
    x, y, w, h = cv2.boundingRect(contours[idx])
    m[y:y+h, x:x+w] = 0
    cv2.drawContours(m, contours, idx, (255, 255, 255), -1)
    r = float(cv2.countNonZero(m[y:y+h, x:x+w])) / (w * h)

    if r > 0.7 and w > 4 and h > 4 and h < 30:
        cv2.rectangle(keretezett, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)
        ax = math.floor( x+(w/2))
        ay = math.floor(y+(h/2))
        cv2.floodFill(im_floodfill, mask, (ax, ay), 128)

cv2.imwrite("18.jpg", keretezett)
cv2.imwrite("19.jpg", im_floodfill)
#----------------------------------------------------------------------------

im_floodfill_inv = cv2.bitwise_not(im_floodfill)

im_out = black | im_floodfill_inv
cv2.imwrite("20.jpg", im_out)

th, im_th2 = cv2.threshold(im_out, TRESHHOLDMIN, 255, cv2.THRESH_BINARY)
cv2.imwrite("21.jpg", im_th2)

"""
"""
im_th3 = cv2.bitwise_not(im_th2)
cv2.imwrite("22.jpg", im_th3)


_,contour,hier = cv2.findContours(im_th3,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
for cnt in contour:
    cv2.drawContours(im_th3,[cnt], 0, 255, -1)
cv2.imwrite("23.jpg", im_th3)

segm = cv2.bitwise_not(im_th3)
cv2.imwrite("24.jpg", segm)
"""
"""

talan2 = talan.copy()
for i in range(ROWS):
    for j in range(COLLUMS):
        if im_th2[i,j] == 0:
            talan2[i,j] = 0

cv2.imwrite("22.jpg", talan2)
"""

# if sys.argv[4] >=-1 and sys.argv[4] == sys.argv[5]:
#     save = True
# else:
#     save = False
saveIndex = 0

def Inc():
    global saveIndex
    saveIndex +=1
    return saveIndex

#-------------------------------------
save = True
#-------------------------------------
if save:
    cv2.imwrite("{} - Input image.jpg".format(Inc()), img_input)
    cv2.imwrite("{} - Greyscale image.jpg".format(Inc()), img_greyscaled)


