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

#########################
# SEARCHING AND READING #
#########################

    def rotate_image(self, img):
        """
        Rotates image 90 degrees clockwise.

        Parameters:
        img: OpenCV image object.

        Returns:
        rotated_img: The rotated image.
        """
        rotated_img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
        return rotated_img


    def detected_workflow(self, img, objects):
        """
        Informs user that label got detected.
        For every detected object, cuts out an image according to ROI, shows the cut out and passes it onto the easyReader.

        Parameters:
        img: OpenCV image object.
        objects: Detected objects in image.

        Returns:
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

        Parameters:
        path: Path to image.

        Returns:
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

        Parameters:
        path: Path to image.

        Returns:
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


###################
# TESTING READING #
###################

    def test_item_number_reading(self, test_file, numbers_file):
        """
        Compares ground truth item numbers with predicted item numbers.

        Parameters:
        test_file: .txt file of positive test images.
        numbers_file: .txt file of ground truth numbers in positive test images.

        Returns:
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

        detected_labels = 0
        not_readable = 0
        true_preds = 0
        false_preds = 0

        for i, path in enumerate(paths):
            img = cv.imread(path)
            objects = self.detector.detect_label(img)
            if len(objects) > 0:
                detected_labels += 1
                box, _ = self.detector.get_highest_confidence_object(img)
                box = [int(coord) for coord in box]

                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]

                roi = img[y:y+h, x:x+w]
                pred_number = self.easyReader.get_high_confidence_alpha_numeric(roi)
                if gt_numbers[i] == 0:
                    not_readable += 1
                elif pred_number == gt_numbers[i]:
                    true_preds += 1
                elif pred_number != gt_numbers:
                    false_preds += 1
                

        print(f'Total labels detected: {str(detected_labels)}')
        print(f'Of which readable numbers: {str(detected_labels - not_readable)}')
        print(f'Of which correctly predicted numbers: {str(true_preds)}')
    

    def get_paths_and_numbers(self, test_file, numbers_file):
        """
        Puts image paths of test file and according ground truth numbers into arrays for comparison.

        Parameters:
        test_file: File of test images.
        numbers_file: File of ground truth numbers.

        Returns:
        paths: Array of paths.
        gt_numbers: Array of ground truth item numbers.
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
            print('Error! Path and Number arrays have different lengths!')


    def print_pred_and_gt(self, test_file, numbers_file):
        """
        Prints predicted and ground truth item number for comparison.

        Parameters:
        test_file: .txt file of test images.
        numbers_file: .txt file of ground truth item numbers in order of test_file.

        Returns:
        none
        """
        paths, gt_numbers = self.get_paths_and_numbers(test_file, numbers_file)


        for i, path in enumerate(paths):
            img = cv.imread(path)
            objects = self.detector.detect_label(img)
            if len(objects) > 0:
                box, _ = self.detector.get_highest_confidence_object(img)
                box = [int(coord) for coord in box]

                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]

                roi = img[y:y+h, x:x+w]
                pred_number = self.easyReader.get_high_confidence_alpha_numeric(roi)

                print('Actual number: ' + str(gt_numbers[i]) + '. Predicted number: ' + str(pred_number))


    def show_problem_images(self, test_file,  numbers_file):
        """
        Shows images where detection = 0 to analyze the problem.

        Parameters:
        test_file: .txt file of test images.
        numbers_file: .txt file of ground truth item numbers in order of test_file.

        Returns:
        none
        """
        paths, gt_numbers = self.get_paths_and_numbers(test_file, numbers_file)


        for i, path in enumerate(paths):
            img = cv.imread(path)
            objects = self.detector.detect_label(img)
            if len(objects) > 0:
                box, _ = self.detector.get_highest_confidence_object(img)
                box = [int(coord) for coord in box]

                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]

                roi = img[y:y+h, x:x+w]
                pred_number = self.easyReader.get_high_confidence_alpha_numeric(roi)
                if pred_number == 0:
                    cv.imshow('Image', img)
                    cv.waitKey(0)

###################################
# DIFFERENT PREPROCESSING METHODS #
###################################

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
    

    def sharpen_image(self, image):
        kernel = np.array([[-1,-1,-1],
                   [-1,9,-1],
                   [-1,-1,-1]])
        sharpened_image = cv.filter2D(image, -1, kernel)
        return sharpened_image
    

    def test_reading_preprocessing(self, test_file, numbers_file):
        """
        Tests if more item numbers can be read with font changes in image.

        Parameters:
        test_file: .txt file of positive test images.
        numbers_file: .txt file of ground truth numbers in positive test images.

        Returns:
        none
        """
        paths, gt_numbers = self.get_paths_and_numbers(test_file, numbers_file)

        not_readable = 0
        true_preds = 0
        detected_labels = 0     

        for i, path in enumerate(paths):
            img = cv.imread(path)
            objects = self.detector.detect_label(img)
            if len(objects) > 0:
                detected_labels += 1
                box, _ = self.detector.get_highest_confidence_object(img)
                box = [int(coord) for coord in box]

                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]

                roi = img[y:y+h, x:x+w]

                thin_font_img = self.thinner_font(img)
                thin_roi = thin_font_img[y:y+h, x:x+w]

                thick_font_img = self.thicker_font(img)
                thick_roi = thick_font_img[y:y+h, x:x+w]

                sharpened_img = self.sharpen_image(img)
                sharpened_roi = sharpened_img[y:y+h, x:x+w]

                #blurred_img = cv.GaussianBlur(img, (5, 5), 0)
                #blurred_roi = blurred_img[y:y+h, x:x+w]

                #denoised_img = cv.bilateralFilter(img, 9, 75, 75)
                #denoised_roi = denoised_img[y:y+h, x:x+w]

                pred_number = self.easyReader.get_high_confidence_alpha_numeric(roi)
                thin_pred_number = self.easyReader.get_high_confidence_alpha_numeric(thin_roi)
                thick_pred_number = self.easyReader.get_high_confidence_alpha_numeric(thick_roi)
                sharp_pred_number = self.easyReader.get_high_confidence_alpha_numeric(sharpened_roi)
                #blurred_pred_number = self.easyReader.get_high_confidence_alpha_numeric(blurred_roi)
                #desnoised_pred_number = self.easyReader.get_high_confidence_alpha_numeric(denoised_roi)

                if pred_number == gt_numbers[i]:
                    true_preds += 1
                else:
                    if thin_pred_number == gt_numbers[i]:
                        true_preds += 1
                    else:
                        if thick_pred_number == gt_numbers[i]:
                            true_preds += 1
                        else:
                            if sharp_pred_number == gt_numbers[i]:
                                true_preds += 1
            

        print(f'Total labels detected: {str(detected_labels)}')
        print(f'Of which readable numbers: {str(detected_labels - not_readable)}')
        print(f'Of which correctly predicted numbers: {str(true_preds)}')


    def read_with_sharpening(self, test_file, numbers_file):
        paths, gt_numbers = self.get_paths_and_numbers(test_file, numbers_file)

        true_preds = 0
        detected_labels = 0     

        for i, path in enumerate(paths):
            img = cv.imread(path)
            objects = self.detector.detect_label(img)
            if len(objects) > 0:
                detected_labels += 1
                box, _ = self.detector.get_highest_confidence_object(img)
                box = [int(coord) for coord in box]

                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]

                roi = img[y:y+h, x:x+w]
                pred_number = self.easyReader.get_high_confidence_alpha_numeric(roi)

                sharpened_img = self.sharpen_image(img)
                sharpened_roi = sharpened_img[y:y+h, x:x+w]

                sharp_pred_number = self.easyReader.get_high_confidence_alpha_numeric(sharpened_roi)

                if pred_number == gt_numbers[i]:
                    true_preds += 1
                else:
                    if sharp_pred_number == gt_numbers[i]:
                        true_preds += 1

        print(f'Total labels detected: {str(detected_labels)}')
        print(f'Of which correctly predicted numbers: {str(true_preds)}')


    def read_with_binarization(self, test_file, numbers_file):
        paths, gt_numbers = self.get_paths_and_numbers(test_file, numbers_file)

        true_preds = 0
        detected_labels = 0     

        for i, path in enumerate(paths):
            img = cv.imread(path)
            objects = self.detector.detect_label(img)
            if len(objects) > 0:
                detected_labels += 1
                box, _ = self.detector.get_highest_confidence_object(img)
                box = [int(coord) for coord in box]

                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]

                roi = img[y:y+h, x:x+w]
                pred_number = self.easyReader.get_high_confidence_alpha_numeric(roi)

                _, binary_image = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
                binary_roi = binary_image[y:y+h, x:x+w]

                binary_pred_number = self.easyReader.get_high_confidence_alpha_numeric(binary_roi)

                if pred_number == gt_numbers[i]:
                        true_preds += 1
                else:
                    if binary_pred_number == gt_numbers[i]:
                        true_preds += 1

        print(f'Total labels detected: {str(detected_labels)}')
        print(f'Of which correctly predicted numbers: {str(true_preds)}')


    def read_with_contrast(self, test_file, numbers_file):
        paths, gt_numbers = self.get_paths_and_numbers(test_file, numbers_file)

        true_preds = 0
        detected_labels = 0     

        for i, path in enumerate(paths):
            img = cv.imread(path)
            objects = self.detector.detect_label(img)
            if len(objects) > 0:
                detected_labels += 1
                box, _ = self.detector.get_highest_confidence_object(img)
                box = [int(coord) for coord in box]

                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]

                roi = img[y:y+h, x:x+w]
                pred_number = self.easyReader.get_high_confidence_alpha_numeric(roi)

                contrast_image = cv.convertScaleAbs(img, alpha=2.0, beta=0)
                contrast_roi = contrast_image[y:y+h, x:x+w]

                contrast_pred_number = self.easyReader.get_high_confidence_alpha_numeric(contrast_roi)

                if pred_number == gt_numbers[i]:
                        true_preds += 1
                else:
                    if contrast_pred_number == gt_numbers[i]:
                        true_preds += 1

        print(f'Total labels detected: {str(detected_labels)}')
        print(f'Of which correctly predicted numbers: {str(true_preds)}')


