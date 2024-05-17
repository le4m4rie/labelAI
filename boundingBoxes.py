import numpy as np
import cv2 as cv
from Detector import Detector
import re


def calculate_iou(bbox1, bbox2):
   """
   Function to calculate Intersection Over Union of two bounding boxes.

   Keyword arguments:
   bbox1, bbox2 -- arrays of box coordinates as x, y, width height.

   Return varialbes
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


def get_predictions(pos_file, output_file):
   """
   Function to get predicted bounding boxes from test dataset.

   Keyword arguments:
   pos_file -- .txt file of positive test images
   output_file -- predictions get written to new .txt file

   Return variables:
   none
   """
   detector = Detector()
   with open(pos_file, 'r') as file:
      with open(output_file, 'w') as output:
        for line in file:
            values = line.strip().split()
            img_path = values[0]
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            objects, confidence = detector.detect_labels_with_weights(img)
            if len(objects) == 0:
                output.write("0\n")  
            else:  
                #find box with highest confidence
                max_index = np.argmax(confidence)
                object_with_highest_confidence = objects[max_index]
                output.write(f"{object_with_highest_confidence}\n")


def get_FP(neg_file):
    """
    Function to get number of False Positive predictions.

    Keyword arguments:
    neg_file -- .txt file of negative images

    Return varialbes:
    false_positive_count (int) 
    """
    false_positive_count = 0
    total = 0
    detector = Detector()
    with open(neg_file, 'r') as file:
        for line in file:
            total += 1
            values = line.strip().split()
            img_path = values[0]
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            objects = detector.detect_label(img)
            if len(objects) > 0:
                false_positive_count += 1

    return int(false_positive_count), int(total)


def get_metrics(test_pos_file, predictions_file, threshold):
    """
    Function to calculate number of True Positive and False Negative predicitons.

    Keyword arguments:
    test_pos_file -- path to the positive .txt test file
    predictions_file -- path to model predicitions .txt file
    threshold -- to calculate IoU

    Return variables:
    true_positive_count -- TP (int)
    false_negative_count -- FN (int) 
    """
    total = 0
    with open(test_pos_file, 'r') as file:
        ground_truth = []
        for line in file:
            total += 1
            values = line.strip().split()
            box = values[2:]
            ground_truth.append(box)

    with open(predictions_file, 'r') as file:
        predictions = [re.split(r'\s+', line.strip()) for line in file]

    true_positive_count = 0
    false_negative_count = 0

    for gt_box, pred_box in zip(ground_truth, predictions):
        gt_box = [int(coord) for coord in gt_box]
        pred_box = [int(coord) for coord in pred_box]

        if len(pred_box) == 1:
            false_negative_count += 1
        else:
            iou_score = calculate_iou(gt_box, pred_box)

            if iou_score > threshold:
                true_positive_count += 1
            else:
                false_negative_count += 1
    
    return int(true_positive_count), int(false_negative_count), int(total)


def remove_brackets(predictions_file, new_file):
    """
    Function to remove unwanted brackets in predicitions file.

    Keyword arguments:
    predictions_file -- .txt file of model predictions
    new_file -- new .txt file to be written to

    Return variables:
    none
    """
    with open(predictions_file, 'r') as file:
        text = file.read()

    text_without_brackets = re.sub(r'\[|\]', '', text)

    with open(new_file, 'w') as file:
        file.write(text_without_brackets)


def get_all_metrics(test_neg, test_pos, preds, thresh):
    """
    Calculating FP, TP, FN.

    Keyword arguments:
    test_neg -- the negative test file
    test_pos -- the positive test file
    preds -- name of the txt file the predictions get saved to
    thresh -- threshod for iou

    Return variables:
    none
    """
    fp, totalfalse = get_FP(test_neg)
    fp_percentage = round(fp / totalfalse, 2)
    get_predictions(test_pos, preds)
    remove_brackets(preds, 'preds_no_brackets.txt')
    tp, fn, totaltrue = get_metrics(test_pos, 'preds_no_brackets.txt', thresh)
    tp_percentage = round(tp / totaltrue, 2)
    fn_percentage = round(fn / totaltrue, 2)
    print(' FP: ' + str(fp_percentage) + ' TP: ' + str(tp_percentage) + ' FN: ' + str(fn_percentage))


#get_all_metrics('training/test/test_neg.txt', 'training/test/test_pos_single_instances.txt', 'preds.txt', 0.4)
