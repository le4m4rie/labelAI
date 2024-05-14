import numpy as np
import cv2 as cv
from Detector import Detector
import re


# TESTING #############################################################
def test_model(pos_test_file, neg_test_file):
    """
    Function to test the cascade model based on precision and recall.

    Keyword arguments: 
    pos_test_file: .txt file contatining positive images
    neg_test_file: .txt file containing negative images

    Return variables:
    None
    """
    get_predictions(pos_test_file, 'preds.txt')
    remove_brackets('preds.txt', 'preds_no_brackets.txt')
    fp = get_FP(neg_test_file)
    tp, fn = get_TP_FN(pos_test_file, 'preds_no_brackets.txt', 0.5)
    precision, recall = calculate_precision_recall(tp, fp, fn)
    print('Precision: ' + precision + ' and Recall: ' + recall)


def calculate_precision_recall(tp, fp, fn):
    """
    Calculate precision and recall.

    Keyword arguments:
    tp -- true positive count
    fp -- false positive count
    fn -- false negative count

    Return variables:
    recall, precision
    """
    precision = tp / (tp + fp)

    recall = tp / (tp + fn)

    return precision, recall


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

    detector = Detector()
    with open(neg_file, 'r') as file:
        for line in file:
            values = line.strip().split()
            img_path = values[0]
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            objects = detector.detect_label(img)
            if len(objects) > 0:
                false_positive_count += 1

    return false_positive_count


def get_TP_FN(test_pos_file, predictions_file, threshold):
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
    with open(test_pos_file, 'r') as file:
        ground_truth = []
        for line in file:
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

        iou_score = calculate_iou(gt_box, pred_box)

        if iou_score > threshold:
            true_positive_count += 1
        else:
            false_negative_count += 1
    
    return true_positive_count, false_negative_count


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


def non_max_suppression(boxes, scores, threshold):
    """
    Perform non-maximum suppression to remove overlapping bounding boxes and return the box with the highest score.

    Keyword arguments:
    boxes -- List of bounding box coordinates in the format (x, y, w, h).
    scores -- List of confidence scores corresponding to each bounding box.
    threshold -- Threshold value for deciding whether to keep overlapping boxes.

    Return values:
    best_box_index (int) -- Index of the bounding box with the highest confidence score after NMS.
    """

    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    best_box_index = sorted_indices[0]  
    max_score = scores[best_box_index]

    while True:
        max_index = sorted_indices.pop(0)


        box = boxes[max_index]
        overlapping_indices = []
        for idx in sorted_indices:
            iou = calculate_iou(box, boxes[idx])

            if iou > threshold:
                overlapping_indices.append(idx)


        for idx in overlapping_indices:
            sorted_indices.remove(idx)


        if not sorted_indices:
            break


        next_box_index = sorted_indices[0]
        next_box_score = scores[next_box_index]
        if next_box_score > max_score:
            best_box_index = next_box_index
            max_score = next_box_score

    return best_box_index

# END OF TESTING ##################################################################

# OTHER ###########################################################################

def nms_before_and_after(img_path):
    """
    Show bounding boxes before and after NMS.

    Keyword arguments:
    img_path -- path to image

    Return variables:
    none
    """
    detector = Detector()
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    detector.show_objects_detected(img_path)
    boxes, scores = detector.detect_labels_with_weights(img)

    box_idx = non_max_suppression(boxes, scores, 0.5)
    final_box = boxes[box_idx]
    x = int(final_box[0])
    y = int(final_box[1])
    xmax = x + int(final_box[2])
    ymax = y + int(final_box[3])
    cv.rectangle(img, (x, y), (xmax, ymax), (0, 255, 255), 2)
    cv.imshow('final box', img)
    cv.waitKey(0)


def print_boxes_and_weights(img_path):
    """
    Printing bounding boxes and according weights.
    """
    detector = Detector()
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    boxes, weights = detector.detect_labels_with_weights(img)
    print('BOXES: \n' + str(boxes) + '\n' + ' WEIGHTS: \n' + str(weights))


def compare_nms_and_highest_confidence(img_path):
    """
    Checks if result from NMS is the same bounding box as the highest confidence bounding box.

    Keyword arguments:
    img_path -- path to image

    Return variables:
    none
    """
    detector = Detector()
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    boxes, confidence = detector.detect_labels_with_weights(img)
    nms_idx = non_max_suppression(boxes, confidence, 0.5)
    max_idx = np.argmax(confidence)
    if nms_idx == max_idx:
        print('Same bounding box')
    else:
        print('Not same bounding box')


# END OF OTHER ###################################################################