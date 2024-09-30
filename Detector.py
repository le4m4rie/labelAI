import cv2 as cv
import numpy as np
import math

class Detector:
   def __init__(self, model_path):
      self.model = cv.CascadeClassifier(model_path)


   def detect_label(self, img):
      """
      Calls the OpenCV cascade detect function.

      Parameters:
      img: An openCV image object.

      Returns:
      objects: Array of bounding boxes with coordinades x, y, width, height.
      """
      objects = self.model.detectMultiScale(img, scaleFactor=1.05, minNeighbors=6, minSize=(100, 60), maxSize=(450, 409))
      return objects


   def show_objects_detected(self, path: str, rescaleFactor=1):
      """
      Display the detected objects.

      Parameters:
      path: Path to the image.
      rescaleFactor: In case image is too big (DEFAULT 1).

      Returns:
      none
      """

      image = cv.imread(path)
      original_height, original_width = image.shape[:2]

      new_width = int(original_width * rescaleFactor)
      new_height = int(original_height * rescaleFactor)
      resized_image = cv.resize(image, (new_width, new_height))

      objects = self.detect_label(resized_image)
      for (x, y, w, h) in objects:
         cv.rectangle(resized_image, (x, y), (x+w, y+h), (0, 255, 0), 3)

      cv.imshow('Objects Detected',resized_image)
      cv.waitKey(0)


   def detect_labels_with_weights(self, img):
      """
      Function for getting confidence levels of the detections.

      Parameters:
      img: An OpenCV image object.

      Returns:
      objects: Array of bounding boxes with coordinades x, y, width, height.
      level_weights: The confidence of the detection.
      """
      objects, _, level_weights = self.model.detectMultiScale3(img,scaleFactor=1.05,minNeighbors=6,minSize=(100, 60), maxSize=(450, 409), outputRejectLevels=True)
      return objects, level_weights
   
   
   def detect_labels_weights_variable_multiscale(self, img, scaleFactor, minNeighbours):
      """
      Get confidence levels and objects with changeable parameters for bayesian optimization.

      Parameters:
      img: OpenCV image object.
      scaleFactor: How much the image is scaled in the detection process.
      minNeighbours: Amount of detections in about the same place necessary to count as valid detection.

      Returns:
      objects: Array of bounding boxes with coordinades x, y, width, height.
      level_weights: The confidence of the detection.
      """

      #round minNeighbours to next int
      floored_minNeighbours = math.floor(minNeighbours)

      objects, _, level_weights = self.model.detectMultiScale3(img, scaleFactor, floored_minNeighbours, minSize=(100, 60), maxSize=(450, 409), outputRejectLevels=True)
      return objects, level_weights


   def get_highest_confidence_object(self, img):
      """
      Performs NMS on detected bounding boxes.

      Parameters:
      img: OpenCV image object.

      Returns:
      object_with_highest_confidence: Bounding box after NMS.
      """
      objects, weights = self.detect_labels_with_weights(img)

      indices = cv.dnn.NMSBoxes(objects, weights, 0.0, 0.5)
      if isinstance(indices, tuple):
         nms_box, confidence = 0, 0
      else:
         idx = indices[0]
         nms_box = objects[idx]
         confidence = weights[idx]
      
      return nms_box, confidence
   

   def show_highest_confidence_object(self, img):
      """
      Displays highest confidence box.

      Parameters:
      img: OpenCV image object.

      Returns:
      none
      """
      box , confidence = self.get_highest_confidence_object(img)
      x = int(box[0])
      y = int(box[1])
      xmax = x + int(box[2])
      ymax = y + int(box[3])
      cv.rectangle(img, (x, y), (xmax, ymax), (0, 255, 255), 2)

      confidence = round(confidence, 2)
      text = str(confidence)

      text_x = xmax 
      text_y = ymax

      cv.putText(img, text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
      cv.imshow('final box', img)
      cv.waitKey(0)

