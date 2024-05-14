import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from Detector import Detector
import cv2 as cv


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
        lines = file.readLines()
        num_lines = len(lines)
        y_true = [1] * num_lines
    with open(neg, 'w') as file2:
        lines2 = file2.readLines()
        num_lines2 = len(lines2)
        y_true.extend([0] * num_lines2)
        
    return y_true


def get_probs(y_test):
    detector = Detector()
    probs = []
    with open(y_test, 'r') as file:
        for line in file:
            values = line.strip().split
            img_path = values[0]
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            box, confidence = detector.get_highest_confidence_object(img)
            probs.append(confidence)

    return probs


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



        