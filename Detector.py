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


   def get_highest_confidence_object(self, img):
      """
      Out of all detected objects, only returns object with highest confidence.

      Keyword arguments:
      img -- openCV image object

      Return variables:
      object_with_highest_confidence -- bounding box with highest confidence score
      """
      objects, weights = self.detect_labels_with_weights(img)
      max_index = np.argmax(weights)
      object_with_highest_confidence = objects[max_index]
      confidence = weights[max_index]
      
      return object_with_highest_confidence, confidence
   

   def show_highest_confidence_object(self, img):
      """
      Displays highest confidence box.

      Keyword arguments:
      img -- openCV image object

      Return variables:
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


detector = Detector()
img = cv.imread('etiketten/4.png')
detector.show_highest_confidence_object(img)