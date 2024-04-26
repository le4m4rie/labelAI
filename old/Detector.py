import cv2 as cv
import numpy as np

class Detector:
   def __init__(self):
      self.model = cv.CascadeClassifier('model/cascade.xml')

   def detect_label(self, img):
      """
      Reads the image and returns array of detected objects.
      """
      objects = self.model.detectMultiScale(img, scaleFactor=1.1, minNeighbors=20, minSize=(80,80))
      return objects


   def show_objects_detected(self, path: str):
      """
      Draws bounding boxes around detected objects in specified image.
      """
      img_path = path
      objects, img = self.detect_label(img_path)
      for (x, y, w, h) in objects:
         cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
      cv.imshow('Objects Detected', img)
      cv.waitKey(0)
