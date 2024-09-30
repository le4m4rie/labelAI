from bayes_opt import BayesianOptimization, UtilityFunction
import os
import shutil
import subprocess
import re
from Detector import Detector
import cv2 as cv
import numpy as np
import math


def calculate_iou(bbox1, bbox2):
   """
   Function to calculate Intersection Over Union of two bounding boxes.

   Parameters:
   bbox1, bbox2: Arrays of box coordinates as x, y, width height.

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


def get_predictions(model_path, pos_file, output_file, scaleFactor, minNeighbours):
   """
   Function to get predicted bounding boxes from test dataset.

   Parameters:
   pos_file: .txt file of positive test images.
   output_file: Predictions get written to new .txt file.

   Returns:
   none
   """
   detector = Detector(model_path)
   with open(pos_file, 'r') as file:
      with open(output_file, 'w') as output:
        for line in file:
            values = line.strip().split()
            img_path = values[0]
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            objects, confidence = detector.detect_labels_weights_variable_multiscale(img, scaleFactor, minNeighbours)
            if len(objects) == 0:
                output.write("0\n")  
            else:  
                max_index = np.argmax(confidence)
                object_with_highest_confidence = objects[max_index]
                output.write(f"{object_with_highest_confidence}\n")


def get_FP(model_path, neg_file):
    """
    Function to get number of False Positive predictions from negative test images.

    Parameters:
    neg_file: .txt file of negative images.

    Returns:
    false_positive_count (int) 
    """
    false_positive_count = 0
    total = 0
    detector = Detector(model_path)
    with open(neg_file, 'r') as file:
        for line in file:
            total += 1
            values = line.strip().split()
            img_path = values[0]
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            objects = detector.detect_label(img)
            count = len(objects)
            false_positive_count += count

    return int(false_positive_count), int(total)


def get_TP_FP_FN(test_pos_file, predictions_file, threshold):
    """
    Function to calculate number of True Positive, False Posistive and False Negative predicitons in the positive test set.

    Parameters:
    test_pos_file: Path to the positive .txt test file.
    predictions_file: Path to model predicitions .txt file.
    threshold: To calculate IoU.

    Returns:
    true_positive_count: TP (int)
    false_negative_count: FN (int) 
    """

    total = 0
    true_positive_count = 0
    false_negative_count = 0
    false_positive_count = 0

    with open(test_pos_file, 'r') as file:
        ground_truth = []
        for line in file:
            total += 1
            values = line.strip().split()
            box = values[2:]
            ground_truth.append(box)

    with open(predictions_file, 'r') as file:
        predictions = [re.split(r'\s+', line.strip()) for line in file]

    for gt_box, pred_box in zip(ground_truth, predictions):
        gt_box = [int(coord) for coord in gt_box]
        pred_box = [int(coord) for coord in pred_box]

        if len(pred_box) == 1:
            false_negative_count += 1
        else:
            iou_score = calculate_iou(gt_box, pred_box)

            if iou_score >= threshold:
                true_positive_count += 1
            else:
                false_positive_count += 1
    
    return int(true_positive_count), int(false_negative_count), int(false_positive_count), int(total)


def remove_brackets(predictions_file, new_file):
    """
    Removing unwanted brackets in predicitions file.

    Parameters:
    predictions_file: .txt file of model predictions.
    new_file: new .txt file to be written to.

    Returns:
    none
    """
    with open(predictions_file, 'r') as file:
        text = file.read()

    text_without_brackets = re.sub(r'\[|\]', '', text)

    with open(new_file, 'w') as file:
        file.write(text_without_brackets)


def get_all_metrics(model_path, test_pos, test_neg, preds, thresh, scaleFactor, minNeighbours):
    """
    Calculating FP, TP, FN.

    Parameters:
    test_neg: The negative test file.
    test_pos: The positive test file.
    preds: Name of the .txt file the predictions get saved to.
    thresh: Threshold for IoU.

    Returns:
    none
    """
    get_predictions(model_path, test_pos, preds, scaleFactor, minNeighbours)
    remove_brackets(preds, 'preds_no_brackets.txt')
    fp1, _ = get_FP(model_path, test_neg)
    tp, fn, fp2, _ = get_TP_FP_FN(test_pos, 'preds_no_brackets.txt', thresh)
    fp = fp1 + fp2
    return fp, tp, fn


def training_workflow(numPos, numNeg, acceptanceRatioBreakValue, maxFalseAlarmRate):
    """
    The cascade training workflow for the bayes optimization function.

    Parameters:
    w: Width of detection window.
    h: Height of detection window.

    Returns:
    destination_file: Path to the trained model.
    """
    numPos = math.floor(numPos)
    numNeg = math.floor(numNeg)

    #remove old vec file
    vectors_path = r'C:\Users\q659840\Desktop\opencv\build\x64\vc15\bin\pos.vec'
    if os.path.exists(vectors_path):
        os.remove(vectors_path)

    directory = r'C:\Users\q659840\Desktop\opencv\build\x64\vc15\bin'
    vectors_command = f"opencv_createsamples.exe -info transformed_pos.txt -w 50 -h 30 -num 3000 -vec pos.vec"

    os.chdir(directory)

    #delete old cascade files
    cascade_dir = os.path.join(directory, 'cascade')
    for filename in os.listdir(cascade_dir):
        file_path = os.path.join(cascade_dir, filename)
        os.remove(file_path)

    subprocess.run(vectors_command, shell=True)

    command = f"opencv_traincascade.exe -data cascade/ -vec pos.vec -bg transformed_neg.txt -w 50 -h 30 -numPos {numPos} -numNeg {numNeg} -numStages 100 -acceptanceRatioBreakValue {acceptanceRatioBreakValue} -minHitRate 0.995 -maxFalseAlarmRate {maxFalseAlarmRate}"

    subprocess.run(command, shell=True)

    source_file = os.path.join(directory, 'cascade', 'cascade.xml')
    destination_file = r'C:\Users\q659840\Desktop\labelAI-main\cascade.xml'

    if os.path.exists(destination_file):
        os.remove(destination_file)

    shutil.copy(source_file, destination_file)
    
    return destination_file


def internal_method(numPos, numNeg, maxFalseAlarmRate, acceptanceRatioBreakValue, scaleFactor, minNeighbours):
    """
    This is the objective function for bayesian optimization to get the best training and architecture parameters for the cascade model.

    Parameters:
    numPos: Positive training samples.
    numNeg: Negative training samples.
    maxFalseAlarmRate: FP.
    acceptanceRatioBreakValue: Stop parameter.
    scaleFactor: Factor at which input image is re-scaled.
    minNeighbours: Minimum detections in same place necessary.

    Returns:
    F1: The F1-score of the current model.
    """
    current_model_path = training_workflow(numPos, numNeg, acceptanceRatioBreakValue, maxFalseAlarmRate)

    data_dir = r'C:\Users\q659840\Desktop\labelAI-main'
    os.chdir(data_dir)

    FP, TP, FN = get_all_metrics(current_model_path, 'training/test/test_pos_single_instances.txt', 'training/test/test_neg.txt', 'preds.txt', 0.5, scaleFactor, minNeighbours)


    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    if precision + recall == 0:
        F1 = 0
    else:
        F1 = 2 * (precision * recall) / (precision + recall)

    # Return the performance metric
    return F1


# Define the search space
pbounds = {
    'numPos': [400, 800],
    'numNeg': [400, 800],
    'maxFalseAlarmRate': (0.1, 0.3),
    'acceptanceRatioBreakValue': (0.0001, 0.00001),
    'scaleFactor': (1.01, 1.1),
    'minNeighbours': (1, 20)
}

# Create the Bayesian Optimization object
optimizer = BayesianOptimization(
    f=internal_method,
    pbounds=pbounds,
    verbose=2,
    random_state=7
)

# Set the Gaussian Process parameters
optimizer.set_gp_params(
    kernel=None, 
    alpha=1e-6,   
    n_restarts_optimizer=5
)

# Create an instance of UtilityFunction
utility = UtilityFunction(
    kind='ucb',  
    kappa=2.5,  
    xi=0.0
)

# Perform the optimization
optimizer.maximize(
    init_points=3,
    n_iter=10,
    acquisition_function=utility
)

# Get the optimal parameters
optimal_params = optimizer.max['params']
print(optimal_params)
