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

------------Kiegészítés---------------
    " _ " nevű változó egy "szemét változó", ebben tárolom azokat a visszaadott értékeket, amelyeket nem használok
--------------------------------------

"""
#------------------------------------------------------------------
img_input = cv2.imread("C:\\Users\\Adam\\Desktop\\samples\\hulk1.jpg") #1--> greyscale
#img_input = cv2.imread(sys.argv[1]) #1--> greyscale
#------------------------------------------------------------------

#konstansok definiálása, valamint a maszkok létrehozása
TRESHHOLDMIN = 210 #Ez a treshold érték volt a leghatékonyabb
ROWS,COLLUMS,_ = img_input.shape
maskForTheFill = np.zeros((ROWS+2, COLLUMS+2), np.uint8) #Mask a fillhez
maskForTheContours = np.zeros((ROWS, COLLUMS), dtype=np.uint8) #mask a conturokhoz
LETTERHIGHT = ROWS / 200 # áltában egy betű mérete így viszonyul magának az oldalnak a méretéhez

#Szürkeárnyalatossá alakítás, majd binarizálás
img_greyscaled = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
_, img_thresholded = cv2.threshold(img_greyscaled, TRESHHOLDMIN, 255, cv2.THRESH_BINARY_INV);

#morphologyTransformation --> előbb ellipsevel, majd rectvel
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
img_morphologyWithEllipse = cv2.morphologyEx(img_thresholded, cv2.MORPH_CLOSE, kernel)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
img_morphologyWithRect = cv2.morphologyEx(img_morphologyWithEllipse, cv2.MORPH_CLOSE, kernel)

#A feldolgozandó képregényoldalaknak általában fehér kerete van.. Ez problémát okoz akkor, amikor a képregénybuborék
# "összeér" a kép keretével, ezért a keretet kitöltöm a négy sarkánál.
img_frame = img_thresholded.copy()
cv2.floodFill(img_frame, maskForTheFill, (0, 0), 128)
cv2.floodFill(img_frame, maskForTheFill, (COLLUMS - 1, 0), 128)
cv2.floodFill(img_frame, maskForTheFill, (0, ROWS - 1), 128)
cv2.floodFill(img_frame, maskForTheFill, (COLLUMS - 1, ROWS - 1), 128)

#Az átalakított képen megkeresem a kontúrokat
_,contours, hierarchy = cv2.findContours(img_morphologyWithRect.copy(), cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)

#másolatok készítése
img_signed = img_input.copy()
img_filled = img_thresholded.copy()

for idx in range(len(contours)):
    #Az adott kontúr köré írható téglalap adatait kigyűjtöm
    x, y, w, h = cv2.boundingRect(contours[idx])

    #Ezeket a részeket megjelölöm a maskon
    maskForTheContours[y:y + h, x:x + w] = 0
    cv2.drawContours(maskForTheContours, contours, idx, (255, 255, 255), -1)

    #Ha egy adott kontúr megfelel a feltételnek, akkor az olyan tulajdonságokkal rendelkezik mint amiket egy
    #szövegbuborék is, ezért ezeket a részeket eltárolom
    if w > LETTERHIGHT/2 and h > LETTERHIGHT and h < LETTERHIGHT*5:



        #TODO







        #Egy zöld keretet teszek a megtalált rész köré
        cv2.rectangle(img_signed, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)
        #Kiszámítom a megtalált rész középpontjának a koordinátáját
        ax = math.floor( x+(w/2))
        ay = math.floor(y+(h/2))

        #Ha a megtaláltalakzat közepén található pixel fekete színű, aklkor az valószínűleg egy betű, része, ezért
        #addig változtatom a pozíciót amíg el nem érek egy fehér részhez, amely a szövegbuborék hátterére mutat.
        while(img_filled[ay,ax] == 255):
            a = randint(0, 4)
            if a == 0:
                ax = ax - 1
            elif a == 1:
                ax = ax + 1
            elif a == 2:
                ay = ay + 1
            else:
                ay = ay - 1
        #A binarizált kép másolatán a megtalált alakokat kitöltöm szürke színnel
        cv2.floodFill(img_filled, maskForTheFill, (ax, ay), 128)

#Invertálom a képet.
img_filled_inv = cv2.bitwise_not(img_filled)
#Mivel a megtalált részeket kitöltöttem azzal a szürke színnel, amely a skála középső eleme, ezért ez az eredeti,
#valamint az invertált képen is ugyanazzal a színnel rendelkezik. Ahhoz, hogy csak ezeket a részeket tartsam meg,
#A két képből csak azokat a pixeleket másolom át, amelyek megegyeznek mindkét képen.
img_foundPartsInGrey = img_thresholded | img_filled_inv

#A "kivonás" után a megtalált részek szürkével jelennek meg, ezeket a részeket átalakítom feketévé, valamint invertálom
_, img_foundParts = cv2.threshold(img_foundPartsInGrey, TRESHHOLDMIN, 255, cv2.THRESH_BINARY_INV)

img_foundPartsFilled = img_foundParts.copy()
#a kapott képen kigyűjtöm a kontúrokat, majd a talált kontúrokat kitöltöm a köríven belül.
_,contour,hier = cv2.findContours(img_foundParts,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
for cnt in contour:
    cv2.drawContours(img_foundPartsFilled,[cnt],0,255,-1)

#Ezt követően a keret feldolgozása következik. Hasonló módon mint korábban, a keret szürke színnel lett megjelölve,
#ezért "kivonva" belőle az inverzét, csak a szürke részt kapjuk eredményül, ami maga a keret.
img_frame_inv = cv2.bitwise_not(img_frame)
img_frameInGrey = img_thresholded | img_frame_inv

#inverz binarizálom
_, img_frameAfterTH = cv2.threshold(img_frameInGrey, TRESHHOLDMIN, 255, cv2.THRESH_BINARY_INV)

img_frameWithUpgrade = img_frameAfterTH.copy();
_,contour,hier = cv2.findContours(img_frameAfterTH,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
for cnt in contour:
    #itt a vonalvastagság 2, amely eltávolít néhány zajt, valamint kiegyenesíti a blokkok határait.
    cv2.drawContours(img_frameWithUpgrade,[cnt],0,255,2)

#A képek invertálása után, elvágzem rajtuk a logikai AND műveletet
img_removableParts = cv2.bitwise_not(img_foundPartsFilled )
img_removableFrame = cv2.bitwise_not(img_frameWithUpgrade )
img_merged = img_removableParts & img_removableFrame

img_colorWithoutTheParts = img_input.copy()
for i in range(ROWS):
    for j in range(COLLUMS):
        if img_merged[i,j] == 0:
            img_colorWithoutTheParts[i,j] = 255

cv2.imwrite("completed.jpg", img_colorWithoutTheParts)
#---------------------------------------------------
# cv2.imwrite(sys.argv[3]+"\\"+sys.argv[2], talan)
#---------------------------------------------------


# if sys.argv[4] >=-1 and sys.argv[4] == sys.argv[5]:
#     save = True
# else:
#     save = False
saveIndex = -1

def Inc():
    global saveIndex
    saveIndex +=1
    if saveIndex < 10:
        return "0{}".format(saveIndex)
    else:
        return "{}".format(saveIndex)

#-------------------------------------
save = True
#-------------------------------------

if save:
    dest = "Steps\\"
    cv2.imwrite(dest + "{} - Input image.jpg".format(Inc()), img_input)
    cv2.imwrite(dest + "{} - Greyscale image.jpg".format(Inc()), img_greyscaled)
    cv2.imwrite(dest + "{} - Thresholded image.jpg".format(Inc()), img_thresholded)
    cv2.imwrite(dest + "{} - Morphology Transformation with ellipse image.jpg".format(Inc()), img_morphologyWithEllipse)
    cv2.imwrite(dest + "{} - Morphology Transformation with rectangle image.jpg".format(Inc()), img_morphologyWithRect)
    cv2.imwrite(dest + "{} - The found parts are signed with green rects.jpg".format(Inc()), img_signed)
    cv2.imwrite(dest + "{} - The found parts are filled with grey colon in the binaryzed image.jpg".format(Inc()), img_filled)
    cv2.imwrite(dest + "{} - Just the found parts in grey color.jpg".format(Inc()), img_foundPartsInGrey)
    cv2.imwrite(dest + "{} - Just the found parts, after invert.jpg".format(Inc()), img_foundParts)
    cv2.imwrite(dest + "{} - The found parts are filled.jpg".format(Inc()), img_foundPartsFilled)
    cv2.imwrite(dest + "{} - The frame is gray.jpg".format(Inc()), img_frame)
    cv2.imwrite(dest + "{} - Just the frame.jpg".format(Inc()), img_frameInGrey)
    cv2.imwrite(dest + "{} - The frme after inverz binaryzing.jpg".format(Inc()), img_frameAfterTH)
    cv2.imwrite(dest + "{} - The frame after processing.jpg".format(Inc()), img_frameWithUpgrade)
    cv2.imwrite(dest + "{} - Merged removable parts.jpg".format(Inc()), img_merged)
    cv2.imwrite(dest + "{} - The color image without the found parts.jpg".format(Inc()), img_colorWithoutTheParts)
