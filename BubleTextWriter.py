import cv2
import numpy as np
import math
from random import randint
import sys
import os

"""
    argv[2] --> a file neve
    argv[3] --> a feldolgozott képek mentési helyének elérési útvonala
    
"""

# fileName = sys.argv[2]
# writePath = sys.argv[3]
# readPath = writePath + fileName
# tempPath = writePath + "Temp\\" + fileName

fileName = "x.jpg"
writePath = "Improve\\"
readPath = writePath + fileName
tempPath = writePath + "Temp\\" + fileName

img_input = cv2.imread(readPath)
img_onlyText = cv2.imread(tempPath,cv2.IMREAD_GRAYSCALE)

ROWS, COLLUMS, _ = img_input.shape

img_output = img_input.copy()
for i in range(ROWS):
    for j in range(COLLUMS):
        if img_onlyText[i, j] == 255:
            img_output[i, j] = 0

cv2.imwrite(writePath + fileName, img_output)

