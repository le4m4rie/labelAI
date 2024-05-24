import cv2 as cv
import numpy as np
from Detector import Detector
from EasyReader import EasyReader
import os


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
        """
        Rotates image 90 degrees clockwise.

        Keyword arguments:
        img -- openCV image object

        Return variables:
        rotated_img -- the rotated image
        """
        rotated_img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
        return rotated_img


    def detected_workflow(self, img, objects):
        """
        Informs user that label got detected.
        For every detected object, cuts out an image according to ROI, shows the cut out and passes it onto the easyReader.

        Keyword arguments:
        img -- openCV image object
        objects -- detected objects in image

        Return variables:
        none
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

        Keyword arguments:
        path -- path to image

        Return variables:
        none
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
        """
        Tries to search for label with rotations.

        Keyword arguments:
        path -- path to image

        Return variables:
        none
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


    def test_item_number_reading(self, test_file, numbers_file):
        """
        Compares ground truth item numbers with predicted item numbers.

        Keyword arguments:
        test_file -- .txt file of positive test images
        numbers_file -- .txt file of ground truth numbers in positive test images

        Return variables:
        none
        """
        paths = []
        gt_numbers = []

        with open(test_file, 'r') as test:
            for line in test:
                values = line.strip().split()
                img_path = values[0]
                paths.append(img_path)

        with open(numbers_file, 'r') as ground_truth:
            for line in ground_truth:
                values = line.strip().split()
                num = values[0]
                gt_numbers.append(num)

        total_numbers = len(gt_numbers)
        detected_labels = 0
        true_preds = 0
        false_preds = 0

        for i, path in enumerate(paths):
            img = cv.imread(path)
            objects = self.detector.detect_label(img)
            if len(objects) == 0:
                total_numbers = total_numbers - 1
            else:
                detected_labels += 1
                box, _ = self.detector.get_highest_confidence_object(img)
                box = [int(coord) for coord in box]

                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]

                roi = img[y:y+h, x:x+w]
                pred_number = self.easyReader.get_high_confidence_alpha_numeric(roi)
                if pred_number == gt_numbers[i]:
                    true_preds += 1
                elif gt_numbers[i] == 0:
                    total_numbers = total_numbers - 1
                elif pred_number == 0:
                    false_preds += 1
                else:
                    false_preds += 1

        print('Total labels detected ' + str(detected_labels) + ' Of which true predictions: ' + str(true_preds))


    def thinner_font(self, image):
        image = cv.bitwise_not(image)
        kernel = np.ones((2,2), np.uint8)
        image = cv.erode(image, kernel, iterations=1)
        image = cv.bitwise_not(image)
        return image

    def thicker_font(self, image):
        image = cv.bitwise_not(image)
        kernel = np.ones((2,2), np.uint8)
        image = cv.dilate(image, kernel, iterations=1)
        image = cv.bitwise_not(image)
        return image
    

    def get_paths_and_numbers(self, test_file, numbers_file):
        """
        Puts image paths of test file and according ground truth numbers into arrays for comparison.

        Keyword arguments:
        test_file -- file of test images
        numbers_file -- file of ground truth numbers

        Return variables:
        paths -- array of paths
        gt_numbers -- array of ground truth item numbers
        """
        paths = []
        gt_numbers = []

        with open(test_file, 'r') as test:
            for line in test:
                values = line.strip().split()
                img_path = values[0]
                paths.append(img_path)

        with open(numbers_file, 'r') as ground_truth:
            for line in ground_truth:
                values = line.strip().split()
                num = values[0]
                gt_numbers.append(num)
        
        if len(paths) == len(gt_numbers):
            return paths, gt_numbers
        else:
            print('Error! Paths and Numbers arrays have different lengths!')


    def test_reading_font_change(self, test_file, numbers_file):
        """
        Tests if more item numbers can be read with font changes in image.

        Keyword arguments:
        test_file -- .txt file of positive test images
        numbers_file -- .txt file of ground truth numbers in positive test images

        Return variables:
        none
        """
        paths, gt_numbers = self.get_paths_and_numbers(test_file, numbers_file)

        true_preds = 0      

        for i, path in enumerate(paths):
            img = cv.imread(path)
            objects = self.detector.detect_label(img)
            if len(objects) == 0:
                total_numbers = total_numbers - 1
            else:
                detected_labels += 1
                box, _ = self.detector.get_highest_confidence_object(img)
                box = [int(coord) for coord in box]

                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]

                roi = img[y:y+h, x:x+w]
                pred_number = self.easyReader.get_high_confidence_alpha_numeric(roi)

                if pred_number != gt_numbers[i]:
                    thin_font_img = self.thinner_font(img)
                    thin_roi = thin_font_img[y:y+h, x:x+w]
                    thin_pred_number = self.easyReader.get_high_confidence_alpha_numeric(thin_roi)
                elif thin_pred_number != gt_numbers[i]:
                    thick_font_img = self.thicker_font(img)
                    thick_roi = thick_font_img[y:y+h, x:x+w]
                    thick_pred_number = self.easyReader.get_high_confidence_alpha_numeric(thick_roi)
                    if thick_pred_number != gt_numbers[i]:
                else:
                    true_preds += 1


labelBot = LabelBot()

labelBot.test_item_number_reading('training/test/fortestingnumbers.txt', 'training/test/test_single_instances_numbers.txt')
