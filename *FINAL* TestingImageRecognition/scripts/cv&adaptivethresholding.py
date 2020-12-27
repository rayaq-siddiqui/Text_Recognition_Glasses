# ********************************************************************************************************
# IMPORTS
import cv2 as cv

# Resources
# https://note.nkmk.me/en/python-numpy-opencv-image-binarization/
# https://www.youtube.com/watch?v=Cy4G1F6Io9k

# ********************************************************************************************************
# WORKING CODE
filename = 'test_folder/image10.jpeg'
img = cv.imread(filename)      # essentially collection of pixels from 0 and 255

# Thresholding
# theory of thresholding- every value that is above threshold is
# converted to white and every value above is converted to maxval
'''
# converting colour to gray
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# NON ADAPTIVE THRESHOLDING CODE ***********************
# applying thresholding method
_, result = cv.threshold(img, 70, 255, cv.THRESH_BINARY)

# PART THAT IMPLEMENTS ADAPTIVE THRESHOLDING CODE ********************
adaptive = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 81, 4)

# just so we can work with proper file
# img = result    # if I want non adaptive method
img = adaptive    # for the adaptive method
'''

# **************************************************************************
# Combine both adaptive and non adaptive methods

# first read in file
img = cv.imread(filename)
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# do the non adaptive method
_, binary_threshold = cv.threshold(img, 200, 255, cv.THRESH_BINARY)
img = binary_threshold

# upload file
cv.imwrite(img=img, filename='test_folder/test.jpeg')

# read in file again
img = cv.imread('test_folder/test.jpeg')
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# # adaptive method
bin_AND_adaptive = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 81, 4)
img = bin_AND_adaptive

# upload file
cv.imwrite(img=img, filename='test_folder/test.jpeg')

# Displaying the code at the end
cv.imshow('Result', img)
cv.waitKey(0)
cv.destroyAllWindows()
