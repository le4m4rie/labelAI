import cv2 as cv
import numpy as np

class Detector:
   def __init__(self):
      self.model = cv.CascadeClassifier('model/cascade190424.xml')


   def detect_label(self, img):
      """
      Calls the openCV cascade detect function.

      Keyword arguments:
      img -- an openCV image object

      Return variables:
      objects -- array of bounding boxes with coordinades x, y, width, height
      """
      objects = self.model.detectMultiScale(img, scaleFactor=1.05, minNeighbors=6, minSize=(100,50))
      return objects


   def show_objects_detected(self, path: str, rescaleFactor=1):
      """
      Display the detected objects.

      Keyword arguments:
      path -- path to the image 
      rescaleFactor -- in case image is too big (DEFAULT 1)

      Return variables:
      none
      """
      image = cv.imread(path, cv.IMREAD_GRAYSCALE)
      original_height, original_width = image.shape[:2]

      new_width = int(original_width * rescaleFactor)
      new_height = int(original_height * rescaleFactor)
      resized_image = cv.resize(image, (new_width, new_height))

      objects = self.detect_label(resized_image)
      for (x, y, w, h) in objects:
         cv.rectangle(resized_image, (x, y), (x+w, y+h), (0, 255, 255), 2)

      cv.imshow('Objects Detected',resized_image)
      cv.waitKey(0)


   def detect_labels_with_weights(self, img):
      """
      Function for getting confidence levels of the detections.

      Keyword arguments:
      img -- an openCV image object

      Return variables:
      objects -- array of bounding boxes with coordinades x, y, width, height
      level_weights -- the confidence of the detection
      """
      objects, reject_levels, level_weights = self.model.detectMultiScale3(img,scaleFactor=1.05,minNeighbors=6,minSize=(100, 50),outputRejectLevels=True)
      return objects, level_weights


detector = Detector()
detector.show_objects_detected('etiketten/4.png')