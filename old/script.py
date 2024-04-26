import cv2 as cv
from easyocr import Reader
import re

class Detector:
   def __init__(self):
      self.model = cv.CascadeClassifier('model/cascade.xml')

   def detect_label(self, img):
      """
      Reads the image and returns array of detected objects.
      """
      objects = self.model.detectMultiScale(img, scaleFactor=1.1, minNeighbors=20, minSize=(80,80))
      return objects
   

class EasyReader:
    def __init__(self):
        self.reader = Reader(['en'], gpu=True, model_storage_directory=r'C:\Users\q635556\Desktop\LabelAI\env\Lib\site-packages\easyocr\model', download_enabled=False)


    def get_high_confidence_alpha_numeric(self, image: str):
        results = self.reader.readtext(image, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        numbers = []
        for detection in results:
            if detection[2] > 0.8:
                numbers.append(detection[1])
        my_string = "".join(numbers)
        for i in range(len(my_string)):
            if my_string[i] == 'O':
                my_string = my_string[:i] + '0' + my_string[i+1:]
            elif my_string[i] == 'I':
                my_string = my_string[:i] + '1' + my_string[i+1:]
        pattern = r"[0-9]{4}[0-9A-Z]{7}"
        matches = re.findall(pattern, my_string)
        if matches:
            print('    **Item number detected: ' + matches[0] +  '**\n' +
                          '                                        \n' + 
                          '                                        \n')
        else:
            print('   **Item number not or not fully detected**\n' + 
                          '                                         \n' +
                          '                                           ')


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


easyReader = EasyReader()
detector = Detector()

def find_item_number(file_path: str):
   """
   return parameters:
   0: file path not found
   1: label not found
   2: item number not fully detected
   """
   img = cv.imread(file_path)
   if img is None:
      return 0
   objects = detector.detect_label(img)
   if len(objects) == 0:
      return 1
   else:
      for (x, y, w, h) in objects:
         cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
         roi = img[y:y+h, x:x+w]
         results = easyReader.reader.readtext(roi, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
         numbers = []
         for detection in results:
            if detection[2] > 0.8:
                numbers.append(detection[1])
         my_string = "".join(numbers)
         for i in range(len(my_string)):
            if my_string[i] == 'O':
                my_string = my_string[:i] + '0' + my_string[i+1:]
            elif my_string[i] == 'I':
                my_string = my_string[:i] + '1' + my_string[i+1:]
         pattern = r"[0-9]{4}[0-9A-Z]{7}"
         matches = re.findall(pattern, my_string)
         if matches:
            return matches[0]
         else:
            return 2

print(find_item_number('etiketten/label33.png'))