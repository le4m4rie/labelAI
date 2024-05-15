import numpy as np
import matplotlib.pyplot as plt
from Detector import Detector
import cv2 as cv


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


def get_ytrue(pos, neg):
    """
    Function to count number of lines in neg and pos test files and create ground truth array.

    Keyword arguments:
    pos -- positive test file
    neg -- negative test file

    Return variables:
    y_true -- array of ground truths
    """
    y_true = []
    with open(pos, 'r') as file:
        lines = file.readlines()
        num_lines = len(lines)
        y_true = [1] * num_lines
    with open(neg, 'r') as file2:
        lines2 = file2.readlines()
        num_lines2 = len(lines2)
        y_true.extend([0] * num_lines2)
        
    return y_true


def get_probs(pos, neg):
    detector = Detector()
    probs = []
    with open(pos, 'r') as file:
        for line in file:
            values = line.strip().split()
            img_path = values[0]
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            objects = detector.detect_label(img)
            if len(objects) > 0:
                box, confidence = detector.get_highest_confidence_object(img)
                probs.append(confidence)
            else:
                probs.append(0)

    with open(neg, 'r') as file2:
        for line in file2:
            values = line.strip().split()
            img_path2 = values[0]
            img2 = cv.imread(img_path2, cv.IMREAD_GRAYSCALE)
            objects = detector.detect_label(img2)
            if len(objects) > 0:
                box2, confidence2 = detector.get_highest_confidence_object(img2)
                probs.append(confidence)
            else:
                probs.append(0)

    return probs


def get_ious(pos, neg):
    detector = Detector()
    ious = []
    with open(pos, 'r') as file:
        for line in file:
            values = line.strip().split()
            img_path = values[0]
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            objects = detector.detect_label(img)
            if len(objects) > 0:
                box, confidence = detector.get_highest_confidence_object(img)
            else:
                ious.append(0)
            ground_truth_box = values[:2]
            iou = calculate_iou(box, ground_truth_box)
            ious.append(iou)


    with open(neg, 'r') as file2:
        for line in file2:
            values = line.strip().split()
            img_path2 = values[0]
            img2 = cv.imread(img_path2, cv.IMREAD_GRAYSCALE)
            objects = detector.detect_label(img2)
            if len(objects) > 0:
                box2, confidence2 = detector.get_highest_confidence_object(img)
                ground_truth_box = values[:2]
                iou = calculate_iou(box2, ground_truth_box)
                ious.append(iou)
            else:
                ious.append(0)

    return ious


def show_roc_curve(y_test, probs):
    """
    Plots a ROC curve.

    Keyword arguments:
    ytest -- array of ground truth values for test data
    probs -- array of the predicted probabilities for each sample

    Return variables:
    none
    """
    fpr, tpr, thresholds = roc_curve(y_test, probs) 
    roc_auc = auc(fpr, tpr)

    plt.figure()  
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Label Detection')
    plt.legend()
    plt.show()


def show_precision_recall(true_labels, probs):
    thresholds = sorted(set(probs))

    precision_values = []
    recall_values = []

    for threshold in thresholds:
        binary_predictions = [1 if p >= threshold else 0 for p in probs]

        true_positives = sum(1 for true, pred in zip(true_labels, binary_predictions) if true == 1 and pred == 1)
        false_positives = sum(1 for true, pred in zip(true_labels, binary_predictions) if true == 0 and pred == 1)
        false_negatives = sum(1 for true, pred in zip(true_labels, binary_predictions) if true == 1 and pred == 0)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

        precision_values.append(precision)
        recall_values.append(recall)

    plt.figure(figsize=(8, 6))
    plt.plot(recall_values, precision_values, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.show()


#y_true = get_ytrue('training/test/test_pos_single_instances.txt', 'training/test/test_neg.txt')
#ious = get_probs('training/test/test_pos_single_instances.txt', 'training/test/test_neg.txt')
#show_precision_recall(y_true, ious)


        