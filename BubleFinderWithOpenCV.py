import cv2
import numpy as np
import math
from random import randint
import sys
import os

import pytesseract
tessdata_dir_config = '--tessdata-dir "c:\\Program Files (x86)\\Tesseract-OCR\\tessdata"'
# import pyocr.builders
# from PIL import Image
# pytesseract.tesseract_cmd = 'c:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'

"""
-----------Argumentum lista-----------
    argv[0] --> a python file neve
    argv[1] --> a feldolgozandó kép elérési útvonala
    argv[2] --> a file neve
    argv[3] --> a feldolgozott képek mentési helyének elérési útvonala
    argv[4] --> ha van szükség a feldolgozási lépések megjelenítésére, akkor ez megadja, hogy melyik képen 
                (a kép útvonala) ha nincs szükség a lépések mentésére akkor üres string
    argv[5] --> felhasználói mód
--------------------------------------

-----------Felhasználói módok---------
    argv[5] = 0 --> A kép minőségének változatlanul hagyása, a szövegbuborékok üresen hagyásával
    argv[5] = 1 --> A kép minőségének változatlanul hagyása, a szövegbuborékok feltöltése az eredet szöveggel
    argv[5] = 2 --> A kép minőségének feljavítása, a szövegbuborékok üresen hagyásával
    argv[5] = 3 --> A kép minőségének feljavítása, a szövegbuborékok feltöltése az eredeti szöveggel
    
    
    argv[5] = 4 --> A kép minőségének feljavítása, a szövegbuborékok feltöltése a lefordított szöveggel
--------------------------------------
    
------------Kiegészítés---------------
    " _ " nevű változó egy "szemét változó", ebben tárolom azokat a visszaadott értékeket, amelyeket nem használok
--------------------------------------
"""
# ------------------------------------------------------------------
readPath = "xmen2.jpg"
fileName = "xmen2_proc.jpg"
writePath = ""
save = True
mode = 0

if len(sys.argv) > 1:
    readPath = sys.argv[1]
    fileName = sys.argv[2]
    writePath = sys.argv[3]
    if sys.argv[4] == "-":
        save = False
    mode = sys.argv[5]

# A kép beolvasása
img_input = cv2.imread(readPath)

# konstansok definiálása, valamint a maszkok létrehozása
TRESHHOLDMIN = 210  # Ez a treshold érték volt a leghatékonyabb
ROWS, COLLUMS, _ = img_input.shape
maskForTheFill = np.zeros((ROWS + 2, COLLUMS + 2), np.uint8)  # Mask a fillhez
maskForTheContours = np.zeros((ROWS, COLLUMS), dtype=np.uint8)  # mask a conturokhoz
LETTERHIGHT = ROWS / 200  # áltában egy betű mérete így viszonyul magának az oldalnak a méretéhez

# Szürkeárnyalatossá alakítás, majd binarizálás
img_greyscaled = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
_, img_thresholded = cv2.threshold(img_greyscaled, TRESHHOLDMIN, 255, cv2.THRESH_BINARY_INV)

# morphologyTransformation --> előbb ellipsevel, majd rectvel
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
img_morphologyWithEllipse = cv2.morphologyEx(img_thresholded, cv2.MORPH_CLOSE, kernel)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
img_morphologyWithRect = cv2.morphologyEx(img_morphologyWithEllipse, cv2.MORPH_CLOSE, kernel)

# A feldolgozandó képregényoldalaknak általában fehér kerete van.. Ez problémát okoz akkor, amikor a képregénybuborék
# "összeér" a kép keretével, ezért a keretet kitöltöm a négy sarkánál.
img_frame = img_thresholded.copy()
cv2.floodFill(img_frame, maskForTheFill, (0, 0), 128)
cv2.floodFill(img_frame, maskForTheFill, (COLLUMS - 1, 0), 128)
cv2.floodFill(img_frame, maskForTheFill, (0, ROWS - 1), 128)
cv2.floodFill(img_frame, maskForTheFill, (COLLUMS - 1, ROWS - 1), 128)

# Az átalakított képen megkeresem a kontúrokat
_, contours, hierarchy = cv2.findContours(img_morphologyWithRect.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# másolatok készítése
img_signed = img_input.copy()
img_filled = img_thresholded.copy()

for idx in range(len(contours)):
    # Az adott kontúr köré írható téglalap adatait kigyűjtöm
    x, y, w, h = cv2.boundingRect(contours[idx])

    # Ezeket a részeket megjelölöm a maskon
    maskForTheContours[y:y + h, x:x + w] = 0
    cv2.drawContours(maskForTheContours, contours, idx, (255, 255, 255), -1)

    # Ha egy adott kontúr megfelel a feltételnek, akkor az olyan tulajdonságokkal rendelkezik mint amiket egy
    # szövegbuborék is, ezért ezeket a részeket eltárolom
    if w > LETTERHIGHT / 2 and h > LETTERHIGHT and h < LETTERHIGHT * 5:
        # Egy zöld keretet teszek a megtalált rész köré
        cv2.rectangle(img_signed, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 2)
        # Kiszámítom a megtalált rész középpontjának a koordinátáját
        ax = math.floor(x + (w / 2))
        ay = math.floor(y + (h / 2))

        # Ha a megtaláltalakzat közepén található pixel fekete színű, aklkor az valószínűleg egy betű, része, ezért
        # addig változtatom a pozíciót amíg el nem érek egy fehér részhez, amely a szövegbuborék hátterére mutat.
        while (img_filled[ay, ax] == 255):
            a = randint(0, 4)
            if a == 0 and ax - 1 > 0:
                ax = ax - 1
            elif a == 1 and ax + 1 < ROWS:
                ax = ax + 1
            elif a == 2 and ay - 1 > 0:
                ay = ay - 1
            elif a == 3 and ay + 1 < COLLUMS:
                ay = ay + 1
        # A binarizált kép másolatán a megtalált alakokat kitöltöm szürke színnel
        cv2.floodFill(img_filled, maskForTheFill, (ax, ay), 128)

# Invertálom a képet.
img_filled_inv = cv2.bitwise_not(img_filled)
# Mivel a megtalált részeket kitöltöttem azzal a szürke színnel, amely a skála középső eleme, ezért ez az eredeti,
# valamint az invertált képen is ugyanazzal a színnel rendelkezik. Ahhoz, hogy csak ezeket a részeket tartsam meg,
# A két képből csak azokat a pixeleket másolom át, amelyek megegyeznek mindkét képen.
img_foundPartsInGrey = img_thresholded | img_filled_inv

# A "kivonás" után a megtalált részek szürkével jelennek meg, ezeket a részeket átalakítom feketévé, valamint invertálom
_, img_foundParts = cv2.threshold(img_foundPartsInGrey, TRESHHOLDMIN, 255, cv2.THRESH_BINARY_INV)

img_foundPartsFilled = img_foundParts.copy()
# a kapott képen kigyűjtöm a kontúrokat, majd a talált kontúrokat kitöltöm a köríven belül.
_, contour, _ = cv2.findContours(img_foundParts, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contour:
    cv2.drawContours(img_foundPartsFilled, [cnt], 0, 255, -1)
# -----------------------------------------------------------------------------------------------------------------------------------

_, contours, _ = cv2.findContours(img_foundPartsFilled, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

parts = []
partsBIGTH = []
indexsForTheTextts = []
for i in range(len(contours)):
    # Az alakzat köré írható tégalalp adatait kigyűjtöm
    x, y, w, h = cv2.boundingRect(contours[i])

    # a kis képek pozíciójának és méretének elmentése, későbbi felhasznlás céljából
    indexsForTheTextts.append([])
    indexsForTheTextts[i].append(x)
    indexsForTheTextts[i].append(y)
    indexsForTheTextts[i].append(w)
    indexsForTheTextts[i].append(h)

    helperMask = np.zeros((ROWS, COLLUMS), dtype=np.uint8)
    for j in range(len(contours[i])):
        posX = contours[i][j][0][1]
        posY = contours[i][j][0][0]
        helperMask[posX, posY] = 255
    cv2.drawContours(helperMask, contours, i, (255, 255, 255), -1)

    helperPartOfTheHelperMask = helperMask[y:y + h, x:x + w]
    littleROWS, littleCOLLUMS = helperPartOfTheHelperMask.shape

    helperPartOfTheOriginalColorImage = np.zeros((littleROWS, littleCOLLUMS, 3), dtype=np.uint8)
    helperPartOfTheOriginalColorImage.fill(255)

    for pixX in range(littleROWS):
        for pixY in range(littleCOLLUMS):
            if helperPartOfTheHelperMask[pixX, pixY] == 255:
                helperPartOfTheOriginalColorImage[pixX, pixY] = img_input[pixX + y, pixY + x]

    partsBIGTH.append(helperMask)
    parts.append(helperPartOfTheOriginalColorImage)

img_RealBubbles = np.zeros((ROWS, COLLUMS), dtype=np.uint8)
img_Onlytexts = np.zeros((ROWS, COLLUMS), dtype=np.uint8)
for i in range(len(parts)):
    textGrey = cv2.cvtColor(parts[i], cv2.COLOR_BGR2GRAY)
    _, textThresholded = cv2.threshold(textGrey, 140, 255, cv2.THRESH_BINARY_INV)

    # Itt nézem meg, hogy az adott kis kivágott képen található-e szöveg
    text = pytesseract.image_to_string(textThresholded, lang='eng', config=tessdata_dir_config)

    # cv2.imwrite("asd.jpg", textThresholded)
    #
    #
    #
    # img = Image.open("asd.jpg")
    # img = img.convert('L')
    #
    # tools = pyocr.get_available_tools()[0]
    # text = tools.image_to_string(img)







    if text.__len__() > 4:
        img_RealBubbles = img_RealBubbles | partsBIGTH[i]
        if mode == 3:
            _, textThresholdedforTexts = cv2.threshold(textGrey, 170, 255, cv2.THRESH_BINARY_INV)
            for w in range(indexsForTheTextts[i][2]):
                for h in range(indexsForTheTextts[i][3]):
                    if textThresholdedforTexts[h][w] == 255:
                        img_Onlytexts[indexsForTheTextts[i][1] + h][indexsForTheTextts[i][0] + w] = 255

# -----------------------------------------------------------------------------------------------------------------------------------
# Ezt követően a keret feldolgozása következik. Hasonló módon mint korábban, a keret szürke színnel lett megjelölve,
# ezért "kivonva" belőle az inverzét, csak a szürke részt kapjuk eredményül, ami maga a keret.
img_frame_inv = cv2.bitwise_not(img_frame)
img_frameInGrey = img_thresholded | img_frame_inv

# inverz binarizálom
_, img_frameAfterTH = cv2.threshold(img_frameInGrey, TRESHHOLDMIN, 255, cv2.THRESH_BINARY_INV)

img_frameWithUpgrade = img_frameAfterTH.copy()
_, contour, hier = cv2.findContours(img_frameAfterTH, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contour:
    # itt a vonalvastagság 2, amely eltávolít néhány zajt, valamint kiegyenesíti a blokkok határait.
    cv2.drawContours(img_frameWithUpgrade, [cnt], 0, 255, 2)
# -----------------------------------------------------------------------------------------------------------------------------------

# A képek invertálása után, elvágzem rajtuk a logikai AND műveletet
img_removableParts = cv2.bitwise_not(img_RealBubbles)
img_removableFrame = cv2.bitwise_not(img_frameWithUpgrade)
img_merged = img_removableParts & img_removableFrame

img_colorWithoutTheParts = img_input.copy()
for i in range(ROWS):
    for j in range(COLLUMS):
        if img_merged[i, j] == 0:
            img_colorWithoutTheParts[i, j] = 255
        if mode == 3:
            if img_Onlytexts[i, j] == 255:
                img_colorWithoutTheParts[i, j] = 0

# Ha a kiválasztott mentési mappa nem ugyanaz mint a feldolgozási mappa akkor az eredeti névvel, különben
# módosított névvel menti
if readPath != writePath + fileName + ".jpg":
    cv2.imwrite(writePath + fileName + ".jpg", img_colorWithoutTheParts)
else:
    cv2.imwrite(writePath + fileName + "_Processed.jpg", img_colorWithoutTheParts)

saveIndex = -1


def Inc():
    global saveIndex
    saveIndex += 1
    if saveIndex < 10:
        return "0{}".format(saveIndex)
    else:
        return "{}".format(saveIndex)


if save:
    dest = writePath + "Steps\\"
    if not os.path.exists(dest):
        os.makedirs(dest)
    cv2.imwrite(dest + "\\{} - {} - Input image.jpg".format(fileName, Inc()), img_input)
    cv2.imwrite(dest + "\\{} - {} - Greyscale image.jpg".format(fileName, Inc()), img_greyscaled)
    cv2.imwrite(dest + "\\{} - {} - Thresholded image.jpg".format(fileName, Inc()), img_thresholded)
    cv2.imwrite(dest + "\\{} - {} - Morphology Transformation with ellipse image.jpg"
                .format(fileName, Inc()), img_morphologyWithEllipse)
    cv2.imwrite(dest + "\\{} - {} - Morphology Transformation with rectangle image.jpg"
                .format(fileName, Inc()), img_morphologyWithRect)
    cv2.imwrite(dest + "\\{} - {} - The found parts are signed with green rects.jpg"
                .format(fileName, Inc()), img_signed)
    cv2.imwrite(dest + "\\{} - {} - The found parts are filled with grey colon in the binaryzed image.jpg"
                .format(fileName, Inc()), img_filled)
    cv2.imwrite(dest + "\\{} - {} - Just the found parts in grey color.jpg"
                .format(fileName, Inc()), img_foundPartsInGrey)
    cv2.imwrite(dest + "\\{} - {} - Just the found parts, after invert.jpg"
                .format(fileName, Inc()), img_foundParts)
    cv2.imwrite(dest + "\\{} - {} - The found parts are filled.jpg".format(fileName, Inc()), img_foundPartsFilled)
    cv2.imwrite(dest + "\\{} - {} - Only the real bubbles.jpg".format(fileName, Inc()), img_RealBubbles)
    cv2.imwrite(dest + "\\{} - {} - The frame is gray.jpg".format(fileName, Inc()), img_frame)
    cv2.imwrite(dest + "\\{} - {} - Just the frame.jpg".format(fileName, Inc()), img_frameInGrey)
    cv2.imwrite(dest + "\\{} - {} - The frame after inverz binaryzing.jpg"
                .format(fileName, Inc()), img_frameAfterTH)
    if mode == 3:
        cv2.imwrite(dest + "\\{} - {} - Only the text.jpg"
                    .format(fileName, Inc()), img_Onlytexts)
    cv2.imwrite(dest + "\\{} - {} - The frame after processing.jpg".format(fileName, Inc()), img_frameWithUpgrade)
    cv2.imwrite(dest + "\\{} - {} - Merged removable parts.jpg".format(fileName, Inc()), img_merged)
    cv2.imwrite(dest + "\\{} - {} - The color image without the found parts.jpg"
                .format(fileName, Inc()), img_colorWithoutTheParts)
