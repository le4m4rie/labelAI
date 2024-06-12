import cv2
import numpy as np
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
    detector = Detector()
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
            img = cv2.imread(path) 
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
                pred_num = easyReader.get_variable_confidence_alpha_numeric(roi, confidence)


                if pred_num == gt_nums[i]:
                    accurate_preds += 1
                else:
                    non_accurate_preds += 1
                    

                accuracy = accurate_preds / detected_labels
                    
                if accuracy > best_accuracy:
                    best_param = confidence
                    best_accuracy = accuracy
    
    return best_param, best_accuracy


def calculate_iou(bbox1, bbox2):
   """
   Function to calculate Intersection Over Union of two bounding boxes.

   Parameters:
   bbox1, bbox2: Arrays of box coordinates as x, y, width, height.

   Returns:
   iou (int)
   """
   x1 = bbox1[0]
   y1 = bbox1[1]
   width1 = bbox1[2]
   height1 = bbox1[3]
   x2 = x1 + width1
   y2 = y1 + height1

   x3 = bbox2[0]
   y3 = bbox2[1]
   width2 = bbox2[2]
   height2 = bbox2[3]
   x4 = x3 + width2
   y4 = y3 + height2
   
   intersection_width = max(0, min(x2, x4) - max(x1, x3))
   intersection_height = max(0, min(y2, y4) - max(y1, y3))
   intersection_area = intersection_width * intersection_height
   
   area_box1 = width1 * height1
   area_box2 = width2 * height2
   union_area = area_box1 + area_box2 - intersection_area
   
   iou = intersection_area / union_area
   
   return iou


def monte_carlo_test_set(test_pos):
    """
    Performs Monte Carlo search on whole test set to get best parameters for detectMultiScale function.

    Parameters:
    test_pos: Positive test file

    Returns:
    dict: A dictionary containing the best parameters and the corresponding performance metrics

    """
    best_params = None
    best_precision = 0
    best_recall = 0

    model = cv2.CascadeClassifier('model/cascade020624.xml')

    gt_box = []

    num_iterations = 3
    scale_factor_range = (1.05, 1.4)
    min_neighbors_range = (5, 20)
    min_size_width_range = (75, 150)
    min_size_height_range = (50, 100)
    max_size_width_range = (500, 800)
    max_size_height_range = (300, 450)

    with open(test_pos, 'r') as file:
        for line in file:
            values = line.strip().split()
            img_path = values[0]
            img = cv2.imread(img_path)
            gt_box.append(values[2])
            gt_box.append(values[3])
            gt_box.append(values[4])
            gt_box.append(values[5])
            gt_box = [int(coord) for coord in gt_box]

            for _ in range(num_iterations):
                scale_factor = random.uniform(*scale_factor_range)
                min_neighbors = random.uniform(*min_neighbors_range)
                min_width = random.randint(*min_size_width_range)
                min_height = random.randint(*min_size_height_range)
                max_width = random.randint(*max_size_width_range)
                max_height = random.randint(*max_size_height_range)
                min_size = (min_width, min_height)
                max_size = (max_width, max_height)

                detected_boxes = model.detectMultiScale(img, scaleFactor=scale_factor, minNeighbors=int(min_neighbors), minSize=min_size, maxSize=max_size)
                if len(detected_boxes) > 0:

                    true_positives = 0
                    false_positives = 0
                    false_negatives = 0

                    for det_box in detected_boxes:
                        det_box = [int(coord) for coord in det_box]
                        if calculate_iou(gt_box, det_box) > 0.5:
                            true_positives += 1
                        else:
                            false_negatives += 1

                                
                    false_positives = len(detected_boxes) - true_positives
                
                    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
                    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
                    
                    if precision > best_precision and recall > best_recall:
                        best_params = {
                            'scaleFactor': scale_factor,
                            'minNeighbors': int(min_neighbors),
                            'minSize': min_size,
                            'maxSize': max_size
                        }
                        best_precision = precision
                        best_recall = recall
        
    return {
        'best_params': best_params,
        'best_precision': best_precision,
        'best_recall': best_recall
        }




#result = monte_carlo_test_set('training/test/test_pos_single_instances.txt')

# Print the best parameters and performance metrics
#print(f"Best Parameters: {result['best_params']}")
#print(f"Best Precision: {result['best_precision']}")
#print(f"Best Recall: {result['best_recall']}")

confidences = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
confidence, accuracy = grid_search_ocr_confidence('training/test/fortestingnumbers.txt', 'training/test/test_single_instances_numbers.txt', confidences)
print('Best confidence: ' + confidence)
print('Best accuracy: ' + accuracy)
