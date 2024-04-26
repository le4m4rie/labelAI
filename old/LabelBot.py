import cv2 as cv
import numpy as np
from Detector import Detector
from EasyReader import EasyReader


class LabelBot:
    def __init__(self):
        self.detector = Detector()
        self.easyReader = EasyReader()
        self.DETECTED_STRING = '\n' + \
        '-----------------------------------------------\n' + \
             '|              **Label detected**             |\n' + \
             '-----------------------------------------------'
      
        self.NOT_DETECTED_STRING = '\n' + \
        '-----------------------------------------------\n' + \
             '|            **Label not detected**           |\n' + \
             '-----------------------------------------------'


    def rotate_image(self, img):
        rotated_img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
        return rotated_img


    def detected_workflow(self, img, objects):
        """
        Informs user that label got detected.
        For every detected object, cuts out an image according to ROI, shows the cut out and passes it onto the easyReader.
        """
        print(self.DETECTED_STRING)
        for (x, y, w, h) in objects:
            cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            roi = img[y:y+h, x:x+w]
            cv.imshow('Detected label cut out', roi)
            cv.waitKey(0)
            self.easyReader.get_high_confidence_alpha_numeric(roi)


    def search_and_read_labels(self, path: str):
        """
        Tries to detect label in normal image. If no detection, rotate image and detect again. If no detection, apply equalizeHist to normal image and detect again.
        """
        img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        assert img is not None, "file could not be read"
        cv.imshow('Input image', img)
        cv.waitKey(0)
        objects = self.detector.detect_label(img)
        if len(objects) > 0:
            self.detected_workflow(img, objects)
        else:
            rotated_img = self.rotate_image(img)
            objects = self.detector.detect_label(rotated_img)
            if len(objects) > 0:
                self.detected_workflow(rotated_img, objects)
            else:
                equalized_img = cv.equalizeHist(img)
                cv.imshow('Intensified image', equalized_img)
                cv.waitKey(0)
                objects = self.detector.detect_label(equalized_img)
                if len(objects) > 0:
                    self.detected_workflow(equalized_img, objects)
                else:
                    print(self.NOT_DETECTED_STRING)


    def search_with_rotation(self, path: str):
        img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        assert img is not None, "file could not be read"
        cv.imshow('Input image', img)
        cv.waitKey(0)
        objects = self.detector.detect_label(img)
        if len(objects) > 0:
            self.detected_workflow(img, objects)
        else:
            rotated_img = self.rotate_image(img)
            objects = self.detector.detect_label(rotated_img)
            if len(objects) > 0:
                self.detected_workflow(rotated_img, objects) 
            else:
                rotated_img1 = self.rotate_image(rotated_img)
                objects = self.detector.detect_label(rotated_img1)
                if len(objects) > 0:
                    self.detected_workflow(rotated_img1, objects)
                else:
                    rotated_img2 = self.rotate_image(rotated_img1)
                    objects = self.detector.detect_label(rotated_img2)
                    if len(objects) > 0:
                        self.detected_workflow(rotated_img2, objects)
                    else:
                        print(self.NOT_DETECTED_STRING)   


labelBot = LabelBot()
labelBot.search_and_read_labels('etiketten/14.png')