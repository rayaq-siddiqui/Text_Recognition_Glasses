import cv2
import pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'/usr/local/Cellar/tesseract/4.1.1/bin/tesseract'

# Load image, convert to HSV, color threshold to get mask
image = cv2.imread('test_folder/image.jpeg')
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower = np.array([0, 0, 0])
upper = np.array([100, 175, 110])
mask = cv2.inRange(hsv, lower, upper)

# Invert image and OCR
invert = 255 - mask
data = pytesseract.image_to_string(invert, lang='eng', config='--psm 6')
print(data)

cv2.imshow('mask', mask)
cv2.imshow('invert', invert)
cv2.waitKey()
