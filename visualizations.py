import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve
from Detector import Detector
import cv2 as cv

###########################################################################
# This file contains anything that has to to with visualizing performance #
###########################################################################


def calculate_iou(bbox1, bbox2):
   """
   Function to calculate Intersection Over Union of two bounding boxes.

   Parameters:
   bbox1, bbox2: arrays of box coordinates as x, y, width height.

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


def get_ytrue(pos, neg):
    """
    Function to count number of lines in neg and pos test files and create ground truth array.

    Parameters:
    pos: positive test file
    neg: negative test file

    Returns:
    y_true: array of ground truths
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
    """
    Gets probabilities of model on a detection.

    Parameters:
    pos: positive test file
    neg: negative test file

    Returns:
    probs: array of probabilities
    """
    detector = Detector()
    probs = []
    with open(pos, 'r') as file:
        for line in file:
            values = line.strip().split()
            img_path = values[0]
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            objects = detector.detect_label(img)
            if len(objects) > 0:
                _, confidence = detector.get_highest_confidence_object(img)
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
                _, confidence2 = detector.get_highest_confidence_object(img2)
                probs.append(confidence2)
            else:
                probs.append(0)

    return probs


def plot_precision_recall(pos, neg):
    y_true = get_ytrue(pos, neg)
    y_scores = get_probs(pos, neg)
    precision, recall, thresholds = precision_recall_curve(
        y_true, y_scores)


    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()


def plot_roc(pos, neg):
    y_true = get_ytrue(pos, neg)
    y_pred = get_probs(pos, neg)

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')  
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('ROC Curve')
    plt.show()


def visualize_metrics(tpr, tpr2, tpr3):
    """
    Visualizes fixed FPR and TPR line for IoU range of 0.3 to 0.6.

    Parameters:
    tpr: numpy array of true positive rates for according IoU 

    Returns:
    none
    """
    ious = np.array([0.3, 0.4, 0.5, 0.6])
    plt.plot(ious, tpr, label='model with FP = 96 %')
    plt.plot(ious, tpr2, label='model with FP = 77 %')
    plt.plot(ious, tpr3, label='model with FP = 37 %')
    plt.xlabel('IoU')
    plt.ylabel('True Positive Rate')
    plt.title('TPR for different IoU thresholds')
    plt.legend()
    plt.show()

        