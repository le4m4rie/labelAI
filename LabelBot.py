import cv2 as cv
import numpy as np
from Detector import Detector
from EasyReader import EasyReader
import os
import random


class LabelBot:
    def __init__(self):
        self.detector = Detector('model/cascade020624.xml')
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


    def detect_and_read_workflow(self, img):
        """
        Retrieves highest confidence bounding box from image and cuts it out to predict item number.

        Parameters:
        img: OpenCV image object.

        Returns:
        pred_number: The predicted item number.
        """
        pred_box, _ = self.detector.get_highest_confidence_object(img)
        pred_box = [int(coord) for coord in pred_box]
        x = pred_box[0]
        y = pred_box[1]
        w = pred_box[2]
        h = pred_box[3]

        roi = img[y:y+h, x:x+w]
        pred_number = self.easyReader.get_high_confidence_alpha_numeric(roi)
        return pred_number
    

    def rotate_detect(self, path: str):
        """
        Rotates image until label is detected. In case of detection, returns confidence for testing purposes.
        """
        rotation_count = 0
        for rotation_count in range(4):
            img = cv.imread(path, cv.IMREAD_GRAYSCALE)
            #assert img is not None, "file could not be read"
            objects = self.detector.detect_label(img)
            if len(objects) > 0:
                _, prob = self.detector.get_highest_confidence_object(img)
                break
            else:
                prob = 0

            if rotation_count < 3:
                img = self.rotate_image(img)
                print('rotating' + path)

        return prob
    

    def rotate_detect_bbox(self, img):
        """
        Rotates image until label is detected. In case of detection, returns bounding box.
        """
        rotation_count = 0
        for rotation_count in range(4):
            #assert img is not None, "file could not be read"
            objects = self.detector.detect_label(img)
            if len(objects) > 0:
                box, _ = self.detector.get_highest_confidence_object(img)
                break
            else:
                box = 0

            if rotation_count < 3:
                img = self.rotate_image(img)

        return box


    def rotate_detect_read(self, path: str):
        """
        This function tries to detect a label and then read it. If it cannot be detected or read, it rotates the label and tries again.
        After the third 90 degree rotation, it stops.

        Parameters:
        path: Path to the image

        Returns:
        number_detected (bool): The number could be detected or not.
        """
        img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        assert img is not None, "file could not be read"

        for rotation_count in range(4):
            objects = self.detector.detect_label(img)
            if len(objects) > 0:
                pred_num = self.detect_and_read_workflow(img)
                if pred_num != 0:
                    print(pred_num)
                    number_detected = pred_num
                    break

            if rotation_count < 3:
                img = self.rotate_image(img)

        return number_detected


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
    

    def get_paths_and_numbers(self, file):
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

        with open(file, 'r') as file:
            for line in file:
                values = line.strip().split()
                img_path = values[0]
                paths.append(img_path)
                gt_number = values[6]
                gt_numbers.append(gt_number)
        
        return paths, gt_numbers


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


    def show_problem_images(self, file):
        """
        Shows images where detection = 0 to analyze problematic images.

        Parameters:
        file: File of image paths and ground truth numbers.

        Returns:
        none
        """
        paths, gt_numbers = self.get_paths_and_numbers(file)


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
    
    def add_median_filter(self, image):
        kernel_size = random.randint(3, 5)
        if kernel_size % 2 == 0:
            kernel_size += 1
        median_filtered = cv.medianBlur(image, kernel_size)
        return median_filtered
    

    def test_reading_preprocessing(self, file):
        """
        This function detects labels in the test file and then applies the different preprocessing methods before trying to read the item number.
        This should give an insight on which preprocessing methods can enhance the reading accuracy.

        Parameters:
        test_file: .txt file of positive test images.
        numbers_file: .txt file of ground truth numbers in positive test images.

        Returns:
        none
        """
        paths, gt_numbers = self.get_paths_and_numbers(file)

        detected_labels = 0
        nonreadable = 0
        none = 0  
        erosion = 0
        dilation = 0
        contrast = 0
        sharp = 0
        blur = 0
        noise = 0
        all = 0

        for i, path in enumerate(paths):
            img = cv.imread(path)
            objects = self.detector.detect_label(img)
            if len(objects) > 0:
                detected_labels += 1
                if str(gt_numbers[i]) == '0':
                    nonreadable += 1
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

                contrast_img = cv.convertScaleAbs(img, alpha=2.0, beta=0)
                contrast_roi = contrast_img[y:y+h, x:x+w]

                sharpened_img = self.sharpen_image(img)
                sharpened_roi = sharpened_img[y:y+h, x:x+w]

                blurred_img = self.add_median_filter(img)
                blurred_roi = blurred_img[y:y+h, x:x+w]

                denoised_img = cv.bilateralFilter(img, 9, 75, 75)
                denoised_roi = denoised_img[y:y+h, x:x+w]

                pred_number = self.easyReader.get_high_confidence_alpha_numeric(roi)
                thin_pred_number = self.easyReader.get_high_confidence_alpha_numeric(thin_roi)
                thick_pred_number = self.easyReader.get_high_confidence_alpha_numeric(thick_roi)
                contrast_pred_number = self.easyReader.get_high_confidence_alpha_numeric(contrast_roi)
                sharp_pred_number = self.easyReader.get_high_confidence_alpha_numeric(sharpened_roi)
                blurred_pred_number = self.easyReader.get_high_confidence_alpha_numeric(blurred_roi)
                desnoised_pred_number = self.easyReader.get_high_confidence_alpha_numeric(denoised_roi)

                if pred_number == gt_numbers[i]:
                    none += 1
                if thin_pred_number == gt_numbers[i]:
                    erosion += 1
                if thick_pred_number == gt_numbers[i]:
                    dilation += 1
                if contrast_pred_number == gt_numbers[i]:
                    contrast += 1
                if sharp_pred_number == gt_numbers[i]:
                    sharp += 1
                if blurred_pred_number == gt_numbers[i]:
                    blur += 1
                if desnoised_pred_number == gt_numbers[i]:
                    noise += 1


        print(f'Total labels detected: {str(detected_labels)}')
        print(f'Of which readable: {str(detected_labels - nonreadable)}')
        print(f'Correctly predicted no preprocessing: {str(none)}')
        print(f'Correctly predicted erosion: {str(erosion)}')
        print(f'Correctly predicted dilation: {str(dilation)}')
        print(f'Correctly predicted contrast: {str(contrast)}')
        print(f'Correctly predicted sharpened: {str(sharp)}')
        print(f'Correctly predicted blurred: {str(blur)}')
        print(f'Correctly predicted noise removal: {str(noise)}')


    def test_reading_all_preprocessing(self, file):
        """
        This function tests whether adding preprocessing methods until the number can be read can improve the number of correctly read item numbers.

        Parameters:
        test_file: .txt file of positive test images.
        numbers_file: .txt file of ground truth numbers in positive test images.

        Returns:
        none
        """
        paths, gt_numbers = self.get_paths_and_numbers(file)

        detected_labels = 0
        nonreadable = 0
        read = 0

        for i, path in enumerate(paths):
            img = cv.imread(path)
            objects = self.detector.detect_label(img)
            if len(objects) > 0:
                detected_labels += 1
                if str(gt_numbers[i]) == '0':
                    nonreadable += 1
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

                blurred_img = self.add_median_filter(img)
                blurred_roi = blurred_img[y:y+h, x:x+w]

                pred_number = self.easyReader.get_high_confidence_alpha_numeric(roi)
                thin_pred_number = self.easyReader.get_high_confidence_alpha_numeric(thin_roi)
                thick_pred_number = self.easyReader.get_high_confidence_alpha_numeric(thick_roi)
                sharp_pred_number = self.easyReader.get_high_confidence_alpha_numeric(sharpened_roi)
                blurred_pred_number = self.easyReader.get_high_confidence_alpha_numeric(blurred_roi)

                if pred_number == gt_numbers[i]:
                    read += 1
                elif thin_pred_number == gt_numbers[i]:
                    read += 1
                elif sharp_pred_number == gt_numbers[i]:
                    read += 1
                elif thick_pred_number == gt_numbers[i]:
                    read += 1
                elif blurred_pred_number == gt_numbers[i]:
                    read += 1
           
        print('Correctly read item numbers: ' + str(read))



#labelBot = LabelBot()
#labelBot.show_problem_images('training/test_old/numbers.txt')