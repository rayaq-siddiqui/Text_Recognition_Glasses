# https://github.com/rayaq-siddiqui/Text_Recognition_Glasses
# ********************************************************************************************************
# IMPORTS

# imports for cv
import cv2 as cv

# imports for pytesseract
import pytesseract

# imports for Skew Correction
import numpy as np

# imports from google translate
import googletrans
from googletrans import Translator

# working with dropbox
import dropbox

# ********************************************************************************************************
# CONSTANTS YOU HAVE TO DEFINE
filename = ''  # 'img/input/image5.jpeg'
filelang = 'ara'
# ita eng ara rus

# where it will be saved on dropbox and where it is being saved locally
dropbox_path = "/input/in.jpg"           # on dropbox
computer_path = "dropbox/input.jpeg"     # local

# ********************************************************************************************************
# INITIALIZE CODE & INITIALIZE DROPBOX & EDIT IMAGE COLOURS

# Linking with dropbox account
dropbox_access_token = "HyKnLMnTXVMAAAAAAAAAARiPgdCGYaDC8-ne9zsm3VbXxXlLSkbihvSjsmiFAqlW"
dbx = dropbox.Dropbox(dropbox_access_token)
print("[SUCCESS] dropbox account linked")

# this will save the code on my computer
_, f = dbx.files_download(dropbox_path)
out = open("dropbox/input.jpeg", 'wb')   # as a writable copy
out.write(f.content)
out.close()

# inoitalizing the actual name of the file
filename = "dropbox/input.jpeg"

# Reading the given file and opening it with CV
img = cv.imread(filename)

# Establishing connection with pytesseract
pytesseract.pytesseract.tesseract_cmd = r'/usr/local/Cellar/tesseract/4.1.1/bin/tesseract'

# Adaptive thresholding methods to make text more visible
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
adaptive = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 101, 4)  # 81 4
img = adaptive

cv.imwrite("img/process images/binary.jpeg", img)

# ADJUSTING THE SKEW OF THE IMAGE
# define gray and thresh
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = cv.bitwise_not(img)  # normally 'cv.bitwise_not(gray)'
thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

# grabs the (x,y) coordinates of all of the pixels where thresh greater than 0 (essentially black
coords = np.column_stack(np.where(thresh > 0))

# determine the angle which the image has to flip
angle = cv.minAreaRect(coords)[-1]
if angle < -45:
    angle = -(90 + angle)
else:
    angle = -angle

# actually rotating the object - essentially rotating the angle determined above
(h, w) = img.shape[:2]
center = (w // 2, h // 2)
M = cv.getRotationMatrix2D(center, angle, 1.0)
rotated = cv.warpAffine(img, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)

img = rotated

cv.imwrite("img/process images/skew.jpeg", img)

# printing the information
print("[INFO] angle: {:.3f}".format(angle))
# NOTE: adding the rotation text at the bottom

# ********************************************************************************************************
# HEAVY PROCESSING

# Word recognition methods used in pytesseract
# initializing values (nn = not necessary)
hImg, wImg = img.shape
# I have to run a code to make it chose any specific language in terminal
# export TESSDATA_PREFIX=PATH TO THE tessdata DIRECTORY FROM Users/... -> no quotes
# export TESSDATA_PREFIX=/Users/rayaq/Desktop/1A\ SE\ 101/Testing\ Image\ Recognition/tessdata
# YOU HAVE TO MANUALLY ADD THE TESSDATA TO THE TESSDATA ROOT DIRECTORY LOCATED AT usr/local/share WHICH IS LOWER
# THAN USERS
boxes = pytesseract.image_to_data(img, lang=filelang, timeout=10)
print(boxes)

# creating a translator object
translator = Translator()
text = []
before_trans = []

# iterates through the given information
# box[6] = x, box[7] = y, box[8] = width, box[9] = height, box[11] = content
for i, box in enumerate(boxes.splitlines()):
    # the first line is just labeling info
    if i != 0:
        # breaks the data string into a data array
        box = box.split()
        if len(box) == 12:
            # finds where to put in the boxes and what to label them as
            x, y, width, height = int(box[6]), int(box[7]), int(box[8]), int(box[9])
            cv.rectangle(img, (x, y), (width + x, height + y), (0, 0, 255), 1)
            
            # will translate if no error occurs
            try:
                before_trans.append(str(box[11]))
                result = translator.translate(str(box[11]), dest='english')
                result = result.text
            except:
                result = box[11]
                print("Did not work")
            
            text.append(str(result))
            
            cv.putText(img, str(result), (x, y - 5), cv.FONT_ITALIC, 0.4, (50, 50, 255), 1)

print(before_trans)
print(text)

# OUTPUTTING A TEXT FILE
g = open("dropbox/output.txt", "w")   # clear the text file first
g.write("TEXT FILE \n")
g.close()
g = open("dropbox/output.txt", "a")   # append to it after

# writing to the text file
for t in text:
    g.write(str(t) + "\n")
    
g.close()    # cause we will open again later with different type

# writing the angle the object rotated
cv.putText(rotated, "Angle: {:.2f} degrees".format(angle), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)

# ********************************************************************************************************
# DISPLAYING THE OUTPUT & UPLOADING TEXT TO DROPBOX

# Uploading to dropbox
file_to = '/output/out.txt'

with open('dropbox/output.txt', 'rb') as g:
    dbx.files_upload(f=g.read(), path=file_to, mute=True, mode=dropbox.files.WriteMode.overwrite)

g.close()    # close it from before

# Displaying the Image - Required by OpenCV
cv.imshow('Result', img)
cv.imwrite('img/output.jpeg', img)
cv.waitKey(0)
cv.destroyAllWindows()
