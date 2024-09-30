import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score
from Detector import Detector
from LabelBot import LabelBot
import cv2 as cv


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
    pos: Positive test file.
    neg: Negative test file.

    Returns:
    probs: Array of probabilities.
    """
    detector = Detector('model/cascade020624.xml')
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


def get_probs2(pos, neg):
    """
    Gets probabilities of model on a detection.

    Parameters:
    pos: positive test file
    neg: negative test file

    Returns:
    probs: array of probabilities
    """
    detector = Detector('model/cascade050824.xml')
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


def get_probs3(pos, neg):
    """
    Gets probabilities of model on a detection.

    Parameters:
    pos: Positive test file.
    neg: Negative test file.

    Returns:
    probs: Array of the models probabilities.
    """
    detector = Detector('model/cascade180724.xml')
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


def get_probs_rotation(pos, neg):
    """
    To test if rotating the image catches more labels.

    Parameters:
    pos: Positive test file.
    neg: Negative test file.

    Returns:
    probs: Array of models probabilities.
    """
    labelBot = LabelBot()
    probs = []
    with open(pos, 'r') as file:
        for line in file:
            values = line.strip().split()
            img_path = values[0]
            confidence = labelBot.rotate_detect(img_path)
            probs.append(confidence)

    with open(neg, 'r') as file2:
        print('POSITIVE DONE')
        for line in file2:
            values2 = line.strip().split()
            img_path2 = values2[0]
            confidence2 = labelBot.rotate_detect(img_path2)
            probs.append(confidence2)

    return probs


def plot_precision_recall(pos, neg):
    """
    Plots precision recall curve.

    Parameters:
    pos: Positive test file.
    neg: Negative test file.

    Returns:
    none
    """
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
    """
    Plots ROC curve.

    Parameters:
    pos: Positive test file.
    neg: Negative test file.

    Returns:
    none
    """
    y_true = get_ytrue(pos, neg)
    y_pred = get_probs(pos, neg)

    fpr, tpr, _ = roc_curve(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')  
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('ROC Curve')
    plt.show()


def plot_roc_of_3_models_diff_test_train(pos1, neg1, pos2, neg2):
    """
    Plots ROC curve.

    Parameters:
    pos: Positive test file.
    neg: Negative test file.

    Returns:
    none
    """
    y_true1 = get_ytrue(pos1, neg1)
    y_true2 = get_ytrue(pos2, neg2)
    y_pred1 = get_probs(pos1, neg1)
    y_pred2 = get_probs2(pos2, neg2)
    #y_pred3 = get_probs3(pos2, neg2)

    fpr1, tpr1, _ = roc_curve(y_true1, y_pred1)
    fpr2, tpr2, _ = roc_curve(y_true2, y_pred2)
    #fpr3, tpr3, _ = roc_curve(y_true2, y_pred3)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr1, tpr1, label='best before optimization')
    plt.plot(fpr2, tpr2, label='optimized')
    #plt.plot(fpr3, tpr3, label='1:2')
    plt.plot([0, 1], [0, 1], 'k--')  
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()


def plot_roc_of_2_models(pos, neg):
    """
    Plots ROC curve.

    Parameters:
    pos: Positive test file.
    neg: Negative test file.

    Returns:
    none
    """
    y_true = get_ytrue(pos, neg)
    y_pred1 = get_probs(pos, neg)
    y_pred2 = get_probs2(pos, neg)

    fpr1, tpr1, _ = roc_curve(y_true, y_pred1)
    fpr2, tpr2, _ = roc_curve(y_true, y_pred2)

    auc1 = roc_auc_score(y_true, y_pred1)
    auc2 = roc_auc_score(y_true, y_pred2)
    print(auc1, auc2)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr1, tpr1, label='default', linestyle='--')
    plt.plot(fpr2, tpr2, label='optimized')
    plt.plot([0, 1], [0, 1], 'k--')  
    plt.xlabel('False Positive Rate (1 - Specificity)', fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity)', fontweight='bold')
    plt.title('ROC curves: Default parameter model and optimized model', fontweight='bold')
    plt.legend()
    plt.show()


plot_roc_of_2_models('training/test_old/test_pos_single_instances.txt', 'training/test_old/test_neg.txt')


def plot_roc_of_3_models(pos, neg):
    """
    Plots ROC curve.

    Parameters:
    pos: Positive test file.
    neg: Negative test file.

    Returns:
    none
    """
    y_true = get_ytrue(pos, neg)
    y_pred1 = get_probs(pos, neg)
    y_pred2 = get_probs2(pos, neg)
    y_pred3 = get_probs3(pos, neg)

    fpr1, tpr1, _ = roc_curve(y_true, y_pred1)
    fpr2, tpr2, _ = roc_curve(y_true, y_pred2)
    fpr3, tpr3, _ = roc_curve(y_true, y_pred3)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr1, tpr1, label='First model')
    plt.plot(fpr2, tpr2, label='Second model')
    plt.plot(fpr3, tpr3, label='Third model')
    plt.plot([0, 1], [0, 1], 'k--')  
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()


#plot_roc_of_3_models_diff_test_train('training/test_old/test_pos_single_instances.txt', 'training/test_old/test_neg.txt', 'training/test/test_pos_single_instances.txt', 'training/test/test_neg.txt')


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


def visualize_tp_fp_increasing_data():
    """
    To visualize the TP and FP Rates for increasing TOTAL training samples.

    Parameters:
    None

    Returns:
    None
    """
    tprs = np.array([0.53, 0.63, 0.67, 0.86, 0.89])
    fprs = np.array([0.80, 0.65, 0.60, 0.28, 0.11])
    data_total = np.array([2300, 3200, 4800, 6400, 9000])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(data_total, tprs, label='True Positive rate', color='#0066cc', linestyle='--')
    ax.scatter(data_total, tprs, color='#0066cc', s=50)
    ax.plot(data_total, fprs, label='False Positive rate', color='#99ccff')
    ax.scatter(data_total, fprs, color='#99ccff', s=50)

    #for x in data_total:
        #ax.axvline(x, color='gray', linestyle='--', linewidth=1)

    ax.set_xlabel('Total number of training samples', fontweight='bold')
    ax.set_ylabel('Rate of TP/FP', fontweight='bold')
    ax.legend()
    plt.title('Trends in TP and FP rates with increasing training data', fontweight='bold')
    plt.show()


def visualize_tp_fp_increasing_stages():
    """
    To visualize the TP and FP Rates for increasing training stages.

    Parameters:
    None

    Returns:
    None
    """
    tprs = np.array([0.83, 0.86, 0.89, 0.89])
    fprs = np.array([0.45, 0.31, 0.20, 0.11])
    stages = np.array([12, 13, 14, 15])

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.fill_between(stages, 0.83, tprs, color='#0066cc', alpha=0.5)

# Fill the area under the FP rate line
    ax.fill_between([12, 13, 14, 15], 0.11, fprs, color='#99ccff', alpha=0.5)

    ax.plot(stages, tprs, label='True Positive rate', color='#0066cc', linestyle='--')
    ax.scatter(stages, tprs, color='#0066cc', s=50)
    ax.plot(stages, fprs, label='False Positive rate', color='#99ccff')
    ax.scatter(stages, fprs, color='#99ccff', s=50)

    ax.set_xticks(stages)
    ax.set_xticklabels(stages)

    ax.set_xlabel('Training stages', fontweight='bold')
    ax.set_ylabel('Rate of TP/FP', fontweight='bold')
    ax.legend()
    plt.title('Trends in TP and FP rates for increasing training stages', fontweight='bold')
    plt.show()

#visualize_tp_fp_increasing_stages()