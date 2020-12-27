## Press ESC button to get next image

import cv2
import cv2 as cv
import numpy as np

frame = cv2.imread('test_folder/image6.jpeg')
# frame = cv2.imread('extra/c2.png')


## keeping a copy of original
print(frame.shape)
original_frame = frame.copy()
original_frame2 = frame.copy()

## Show the original image
winName = 'Original'
cv.namedWindow(winName, cv.WINDOW_NORMAL)
# cv.resizeWindow(winName, 800, 800)
cv.imshow(winName, frame)
cv.waitKey(0)

## Apply median blur
frame = cv2.medianBlur(frame, 9)

## Show the original image
winName = 'Median Blur'
cv.namedWindow(winName, cv.WINDOW_NORMAL)
# cv.resizeWindow(winName, 800, 800)
cv.imshow(winName, frame)
cv.waitKey(0)

# kernel = np.ones((5,5),np.uint8)
# frame = cv2.dilate(frame,kernel,iterations = 1)


# Otsu's thresholding
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
ret2, thresh_n = cv.threshold(frame, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
frame = thresh_n

## Show the original image
winName = 'Otsu Thresholding'
cv.namedWindow(winName, cv.WINDOW_NORMAL)
# cv.resizeWindow(winName, 800, 800)
cv.imshow(winName, frame)
cv.waitKey(0)

## invert color
frame = cv2.bitwise_not(frame)

## Show the original image
winName = 'Invert Image'
cv.namedWindow(winName, cv.WINDOW_NORMAL)
# cv.resizeWindow(winName, 800, 800)
cv.imshow(winName, frame)
cv.waitKey(0)

## Dilate image
kernel = np.ones((5, 5), np.uint8)
frame = cv2.dilate(frame, kernel, iterations=1)

##
## Show the original image
winName = 'SUB'
cv.namedWindow(winName, cv.WINDOW_NORMAL)
# cv.resizeWindow(winName, 800, 800)
img_gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
cv.imshow(winName, img_gray & frame)
cv.waitKey(0)

## Show the original image
winName = 'Dilate Image'
cv.namedWindow(winName, cv.WINDOW_NORMAL)
# cv.resizeWindow(winName, 800, 800)
cv.imshow(winName, frame)
cv.waitKey(0)

## Get largest contour from contours
contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

## Get minimum area rectangle and corner points
rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
print(rect)
box = cv2.boxPoints(rect)
print(box)

## Sorted points by x and y
## Not used in this code
print(sorted(box, key=lambda k: [k[0], k[1]]))

## draw anchor points on corner
frame = original_frame.copy()
z = 6
for b in box:
    cv2.circle(frame, tuple(b), z, 255, -1)

## show original image with corners
box2 = np.int0(box)
cv2.drawContours(frame, [box2], 0, (0, 0, 255), 2)
cv2.imshow('Detected Corners', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()


## https://stackoverflow.com/questions/11627362/how-to-straighten-a-rotated-rectangle-area-of-an-image-using-opencv-in-python
def subimage(image, center, theta, width, height):
    shape = (image.shape[1], image.shape[0])  # cv2.warpAffine expects shape in (length, height)

    matrix = cv2.getRotationMatrix2D(center=center, angle=theta, scale=1)
    image = cv2.warpAffine(src=image, M=matrix, dsize=shape)

    x = int(center[0] - width / 2)
    y = int(center[1] - height / 2)

    image = image[y:y + height, x:x + width]

    return image


## Show the original image
winName = 'Dilate Image'
cv.namedWindow(winName, cv.WINDOW_NORMAL)
# cv.resizeWindow(winName, 800, 800)


## use the calculated rectangle attributes to rotate and extract it
frame = subimage(original_frame, center=rect[0], theta=int(rect[2]), width=int(rect[1][0]), height=int(rect[1][1]))
original_frame = frame.copy()
cv.imshow(winName, frame)
cv.waitKey(0)

perspective_transformed_image = frame.copy()

## Apply median blur
frame = cv2.medianBlur(frame, 11)

## Show the original image
winName = 'Median Blur'
cv.namedWindow(winName, cv.WINDOW_NORMAL)
# cv.resizeWindow(winName, 800, 800)
cv.imshow(winName, frame)
cv.waitKey(0)

# kernel = np.ones((5,5),np.uint8)
# frame = cv2.dilate(frame,kernel,iterations = 1)


# Otsu's thresholding
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
ret2, thresh_n = cv.threshold(frame, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
frame = thresh_n

## Show the original image
winName = 'Otsu Thresholding'
cv.namedWindow(winName, cv.WINDOW_NORMAL)
# cv.resizeWindow(winName, 800, 800)
cv.imshow(winName, frame)
cv.waitKey(0)

## invert color
frame = cv2.bitwise_not(frame)

## Show the original image
winName = 'Invert Image'
cv.namedWindow(winName, cv.WINDOW_NORMAL)
# cv.resizeWindow(winName, 800, 800)
cv.imshow(winName, frame)
cv.waitKey(0)

## Dilate image
kernel = np.ones((5, 5), np.uint8)
frame = cv2.dilate(frame, kernel, iterations=1)

##
## Show the original image
winName = 'SUB'
cv.namedWindow(winName, cv.WINDOW_NORMAL)
# cv.resizeWindow(winName, 800, 800)
img_gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
frame = img_gray & frame
frame[np.where(frame == 0)] = 255
cv.imshow(winName, frame)
cv.waitKey(0)

hist, bins = np.histogram(frame.flatten(), 256, [0, 256])

cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()
print(cdf)
print(cdf_normalized)
hist_image = frame.copy()

## two decresing range algorithm
low_index = -1
for i in range(0, 256):
    if cdf[i] > 0:
        low_index = i
        break
print(low_index)

tol = 0
tol_limit = 20
broken_index = -1
past_val = cdf[low_index] - cdf[low_index + 1]
for i in range(low_index + 1, 255):
    cur_val = cdf[i] - cdf[i + 1]
    if tol > tol_limit:
        broken_index = i
        break
    if cur_val < past_val:
        tol += 1
    past_val = cur_val

print(broken_index)

##
lower = min(frame.flatten())
upper = max(frame.flatten())
print(min(frame.flatten()))
print(max(frame.flatten()))

# img_rgb_inrange = cv2.inRange(frame_HSV, np.array([lower,lower,lower]), np.array([upper,upper,upper]))
img_rgb_inrange = cv2.inRange(frame, (low_index), (broken_index))
neg_rgb_image = ~img_rgb_inrange
## Show the original image
winName = 'Final'
cv.namedWindow(winName, cv.WINDOW_NORMAL)
# cv.resizeWindow(winName, 800, 800)
cv.imshow(winName, neg_rgb_image)
cv.waitKey(0)

kernel = np.ones((3, 3), np.uint8)
frame = cv2.erode(neg_rgb_image, kernel, iterations=1)
winName = 'Final Dilate'
cv.namedWindow(winName, cv.WINDOW_NORMAL)
# cv.resizeWindow(winName, 800, 800)
cv.imshow(winName, frame)
cv.waitKey(0)

##
winName = 'Final Subtracted'
cv.namedWindow(winName, cv.WINDOW_NORMAL)
img2 = np.zeros_like(perspective_transformed_image)
img2[:, :, 0] = frame
img2[:, :, 1] = frame
img2[:, :, 2] = frame
frame = img2
cv.imshow(winName, perspective_transformed_image | frame)
cv.waitKey(0)

##
import matplotlib.pyplot as plt

plt.plot(cdf_normalized, color='b')
plt.hist(hist_image.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('cdf', 'histogram'), loc='upper left')
plt.show()
