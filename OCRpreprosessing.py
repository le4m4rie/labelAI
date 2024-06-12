import cv2 as cv
import numpy as np
from EasyReader import EasyReader

###################################################
# Testing different preprocessing methods for OCR #
###################################################

#01 Inverted Images
def invert(image):
    image = cv.bitwise_not(image)
    return image


#03 Binarization
def binarize(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresh, im_bw = cv.threshold(gray, 170, 250, cv.THRESH_BINARY)
    image = im_bw
    return image


#04 Noise Removal
def noise_removal(image):
    kernel = np.ones((1,1), np.uint8)
    image = cv.dilate(image, kernel, iterations=1)
    image = cv.erode(image, kernel, iterations=1)
    image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)
    image = cv.medianBlur(image, 3)
    return image


#05 Dilation and Erosion
def thinner_font(image):
    image = cv.bitwise_not(image)
    kernel = np.ones((2,2), np.uint8)
    image = cv.erode(image, kernel, iterations=1)
    image = cv.bitwise_not(image)
    return image


def thicker_font(image):
    image = cv.bitwise_not(image)
    kernel = np.ones((2,2), np.uint8)
    image = cv.dilate(image, kernel, iterations=1)
    image = cv.bitwise_not(image)
    return image

#06 Rotation / Deskewing
#https://becominghuman.ai/how-to-automatically-deskew-straighten-a-text-image-using-opencv-a0c30aed83df


def getSkewAngle(cvImage) -> float:
    # Prep image, copy, convert to gray scale, blur, and threshold
    newImage = cvImage.copy()
    gray = cv.cvtColor(newImage, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (9, 9), 0)
    thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (30, 5))
    dilate = cv.dilate(thresh, kernel, iterations=2)

    # Find all contours
    contours, hierarchy = cv.findContours(dilate, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv.contourArea, reverse = True)
    for c in contours:
        rect = cv.boundingRect(c)
        x,y,w,h = rect
        cv.rectangle(newImage,(x,y),(x+w,y+h),(0,255,0),2)

    # Find largest contour and surround in min area box
    largestContour = contours[0]
    print (len(contours))
    minAreaRect = cv.minAreaRect(largestContour)
    cv.imwrite("temp/boxes.jpg", newImage)
    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle


# Rotate the image around its center
def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv.warpAffine(newImage, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
    return newImage


def deskew(cvImage):
    angle = getSkewAngle(cvImage)
    return rotateImage(cvImage, -1.0 * angle)


def display_all(image):
    image = cv.resize(image, (300, 300))
    inverted_image = invert(image)
    binarized_image = binarize(image)
    binarized_image = cv.cvtColor(binarized_image, cv.COLOR_GRAY2BGR)
    no_noise = noise_removal(image)
    thin_font = thinner_font(no_noise)
    thick_font = thicker_font(no_noise)

    row1 = np.concatenate((image, inverted_image, binarized_image), axis=1)
    row2 = np.concatenate((no_noise, thin_font, thick_font), axis=1)

    composite_image = np.concatenate((row1, row2), axis=0)

    cv.imshow('Multiple Images', composite_image)
    cv.waitKey(0)


def read(image):
    results = easyReader.reader.readtext(image, allowlist='0123456789')
    text_results = [result[1] for result in results]
    for text in text_results:
        print(text)


def read_all(image):
    print('NORMAL IMAGE:')
    read(image)
    print('INVERTED IMAGE:')
    inverted_image = invert(image)
    read(inverted_image)
    print('BINARIZED IMAGE:')
    binarized_image = binarize(image)
    read(binarized_image)
    print('NOISE-FREE IMAGE:')
    no_noise = noise_removal(image)
    read(no_noise)
    print('THIN FONT IMAGE:')
    thin_font = thinner_font(image)
    read(thin_font)
    print('THICK FONT IMAGE:')
    thick_font = thicker_font(image)
    read(thick_font)


def show_contour(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, threshold = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    biggest_contour = max(contours, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(biggest_contour)
    cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv.imshow('Image', image)
    cv.waitKey(0)




    