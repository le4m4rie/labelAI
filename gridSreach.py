import cv2 as cv
import random
from EasyReader import EasyReader
from Detector import Detector


def grid_search_ocr_confidence(test_pos, numbers_file, confidences):
    """
    Performs a grid search to optimize the confidence of the easyOCR model as a decision parameter.
    
    Parameters:
    test_pos: File of positive test images.
    numbers_file: File of ground truth item numbers according to the test images.
    confidences: An array of different confidences to choose from.
    
    Returns:
    best_param: The best confidence.
    best_accuracy: The best accuracy.
    """
    detector = Detector('model/cascade180724.xml')
    easyReader = EasyReader()
    best_param = None
    best_accuracy = 0
    gt_nums = []
    paths  = []
    
    for confidence in confidences:

        accurate_preds = 0
        non_accurate_preds = 0
        detected_labels = 0

        with open(numbers_file, 'r') as ground_truth:
            for line in ground_truth:
                values = line.strip().split()
                num = values[0]
                gt_nums.append(num)

        with open(test_pos, 'r') as file:
            for line in file:
                values = line.strip().split()
                img_path = values[0]
                paths.append(img_path)

        for i, path in enumerate(paths):
            img = cv.imread(path) 
            objects = detector.detect_label(img)
            if len(objects) > 0:
                detected_labels += 1
                box, _ = detector.get_highest_confidence_object(img)
                box = [int(coord) for coord in box]

                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]

                roi = img[y:y+h, x:x+w]
                pred_num = easyReader.get_high_confidence_alpha_numeric(roi, confidence)


                if pred_num == gt_nums[i]:
                    accurate_preds += 1
                else:
                    non_accurate_preds += 1
                    

                accuracy = accurate_preds / detected_labels
                    
                if accuracy > best_accuracy:
                    best_param = confidence
                    best_accuracy = accuracy
    
    return best_param, best_accuracy

confidences = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
confidence, accuracy = grid_search_ocr_confidence('training/test/test_pos_single_instances.txt', 'training/test/numbers.txt', confidences)
print('best confidence: ' + str(confidence))
print('best accuracy: ' + str(accuracy))

