import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from Detector import Detector
from LabelBot import LabelBot
import re
from PIL import Image


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


def nms_vs_highest_confidence(pos):
    """
    Tests if it is necessary to use NMS by comparing the resulting box of NMS with the highest confidence box of the detector.

    Parameters:
    pos: The positive test file.

    Returns:
    none
    """
    detector = Detector('model/cascade090824.xml')
    num_same_boxes = 0
    objects_total = 0
    with open(pos, 'r') as file:
        for line in file:
            values = line.strip().split()
            img_path = values[0]
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            objects, confidences = detector.detect_labels_with_weights(img)
            if len(objects) == 0:
                objects_total += 1
                indices = cv.dnn.NMSBoxes(objects, confidences, 0.0, 0.5)
                idx = indices[0]
                nms_box = objects[idx]
                highest_confidence_box = detector.get_highest_confidence_object(img)

                nms_box = [int(coord) for coord in nms_box]
                highest_confidence_box = [int(coord) for coord in highest_confidence_box] 

                for i in nms_box, highest_confidence_box:
                    if nms_box[i] == highest_confidence_box[i]:
                        num_same_boxes += 1 

    print('Ratio of same boxes: ' + str(num_same_boxes))


def get_predictions(pos_file, output_file):
   """
   Function to get predicted bounding boxes from positive test dataset.

   Parameters:
   pos_file: .txt file of positive test images.
   output_file: Predictions get written to new .txt file.

   Returns:
   none
   """
   detector = Detector('model/cascade020624.xml')
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
                max_index = np.argmax(confidence)
                object_with_highest_confidence = objects[max_index]
                output.write(f"{object_with_highest_confidence}\n")


def get_predictions_with_rotation(pos_file, output_file):
   """
   Function to get predicted bounding boxes from positive test dataset with the added rotation sequence.

   Parameters:
   pos_file: .txt file of positive test images.
   output_file: Predictions get written to new .txt file.

   Returns:
   none
   """
   detector = LabelBot()
   with open(pos_file, 'r') as file:
      with open(output_file, 'w') as output:
        for line in file:
            values = line.strip().split()
            img_path = values[0]
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            box = detector.rotate_detect_bbox(img)
            output.write(f"{box}\n")


def get_FP(neg_file):
    """
    Function to get number of False Positive predictions from negative test images.

    Parameters:
    neg_file: .txt file of negative images.

    Returns:
    false_positive_count: Number of false positive detections.
    """
    false_positive_count = 0
    total = 0
    detector = Detector('model/cascade020624.xml')
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


def get_FP_with_rotation(neg_file):
    """
    Function to get number of False Positive predictions from negative test images.

    Parameters:
    neg_file: .txt file of negative images.

    Returns:
    false_positive_count: Number of false positive detections.
    """
    false_positive_count = 0
    total = 0
    detector = LabelBot()
    with open(neg_file, 'r') as file:
        for line in file:
            total += 1
            values = line.strip().split()
            img_path = values[0]
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            box = detector.rotate_detect_bbox(img)
            box = np.array(box)
            if box.shape == ():
                false_positive_count += 1

    return int(false_positive_count), int(total)


def get_TP_FP_FN(test_pos_file, predictions_file, threshold):
    """
    Function to calculate number of True Positive and False Negative predicitons.

    Parameters:
    test_pos_file: Path to the positive .txt test file.
    predictions_file: Path to model predicitions .txt file.
    threshold: To calculate IoU.

    Returns:
    true_positive_count: Number of true positive detections.
    false_negative_count: Number of false negative detections. 
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
    Function to remove unwanted brackets in predicitions file.

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


def get_all_metrics(test_neg, test_pos, preds, thresh):
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
    total_neg = 400
    total_pos = 193
    fp_neg, _ = get_FP_with_rotation(test_neg)
    get_predictions_with_rotation(test_pos, preds)
    remove_brackets(preds, 'preds_no_brackets.txt')
    tp, fn, fp, _ = get_TP_FP_FN(test_pos, 'preds_no_brackets.txt', thresh)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print(' TPR: ' + str(tp/total_pos) + ' FPR:' + str((fp_neg+fp)/(total_neg+total_pos)) + ' Precision: ' + str(precision) + ' Recall: ' + str(recall))
    return fp, tp, fn


#get_all_metrics('training/test_old/test_neg.txt', 'training/test_old/test_pos_single_instances.txt', 'preds.txt', 0.5)


def show_gt_and_pred(pos):
    """
    Visualizing ground truth and predicted bounding boxes for the test data set.

    Parameters:
    pos: Test file.

    Returns:
    None    
    """
    detector = Detector()
    with open(pos, 'r') as file:
        for line in file:
            values = line.strip().split()
            img_path = values[0]
            gt_box = values[2:]
    
            img = cv.imread(img_path)

            pred_boxes = detector.detect_label(img)
            if len(pred_boxes) > 0:
                pred_box, conf = detector.get_highest_confidence_object(img)

                gt_box = [int(coord) for coord in gt_box]
                pred_box = [int(coord) for coord in pred_box]

                x = int(pred_box[0])
                y = int(pred_box[1])
                xmax = x + int(pred_box[2])
                ymax = y + int(pred_box[3])
                cv.rectangle(img, (x, y), (xmax, ymax), (128, 0, 128), 2)

                xgt = int(gt_box[0])
                ygt = int(gt_box[1])
                xmaxgt = xgt + int(gt_box[2])
                ymaxgt = ygt + int(gt_box[3])
                cv.rectangle(img, (xgt, ygt), (xmaxgt, ymaxgt), (0, 255, 0), 2)

                conf = round(conf, 2)
                text = str(conf)

                text_x = xmax 
                text_y = ymax

                cv.putText(img, text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 128), 2)

                cv.imshow('Predicted in purple and ground truth in green', img)
                cv.waitKey(0)
                cv.destroyAllWindows
            else:
                gt_box = [int(coord) for coord in gt_box]

                xgt = int(gt_box[0])
                ygt = int(gt_box[1])
                xmaxgt = xgt + int(gt_box[2])
                ymaxgt = ygt + int(gt_box[3])
                cv.rectangle(img, (xgt, ygt), (xmaxgt, ymaxgt), (0, 255, 0), 2)

                cv.imshow('Predicted in purple and ground truth in green', img)
                cv.waitKey(0)
                cv.destroyAllWindows


def get_max_min_size(test_file):
    """
    Retrieves the biggest and smallest box from the test data.

    Parameters:
    test_file: The test file.

    Returns:
    none
    """

    boxes = []

    with open(test_file, 'r') as file:
        for line in file:
            values = line.strip().split()
            box = values[2:]
            box = [int(coord) for coord in box]

            width = box[2]
            height = box[3]

            boxes.append((width, height))

        min_box = min(boxes, key=lambda box: box[0] * box[1])
    
        max_box = max(boxes, key=lambda box: box[0] * box[1])
    

    print('Min: ' + str(min_box))
    print('Max:' + str(max_box))






