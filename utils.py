import os
import numpy as np
import PIL.Image as Image
import torch
import warnings
import config as cfg




def compute_classification_metrics(dataset, num_classes=19):
    """
    Compute accuracy per sample, overall accuracy, per-class F1,
    macro F1, and weighted F1.

    Returns:
    - overall_accuracy: float
    - macro_f1: float
    - weighted_f1: float
    - f1_per_class: list of floats
    - accuracy_per_sample: list of floats
    """
    y_true_all = []
    y_pred_all = []
    class_support = np.zeros(num_classes, dtype=int)
    accuracy_per_sample = []

    for data_test in dataset:
        if not hasattr(data_test, "pred") or not hasattr(data_test, "y"):
            warnings.warn(f"Skipping sample : missing `pred` or `y` attribute.")
            continue

        y_true = data_test.y.numpy()
        y_pred = data_test.pred.numpy()

        correct = np.sum(y_true == y_pred)
        accuracy = correct / len(y_true)
        accuracy_per_sample.append(accuracy)

        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)

        for cls in range(num_classes):
            class_support[cls] += np.sum(y_true == cls)

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    # Overall Accuracy
    overall_accuracy = np.mean(y_true_all == y_pred_all)

    # Per-class F1 computation
    f1_scores = []
    for cls in range(num_classes):
        tp = np.sum((y_pred_all == cls) & (y_true_all == cls))
        fp = np.sum((y_pred_all == cls) & (y_true_all != cls))
        fn = np.sum((y_pred_all != cls) & (y_true_all == cls))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)

    f1_scores = np.array(f1_scores)

    # Macro and Weighted F1
    macro_f1 = np.mean(f1_scores)
    total_support = np.sum(class_support)
    weighted_f1 = np.sum(f1_scores * class_support) / total_support if total_support > 0 else 0

    return overall_accuracy, macro_f1, weighted_f1, f1_scores, accuracy_per_sample

def distance(pos1,pos2):
    """
    Computes Euclidean distance between two 2D points.

    Args:
        pos1 (tuple or list): Coordinates (x1, y1).
        pos2 (tuple or list): Coordinates (x2, y2).

    Returns:
        float: Euclidean distance between pos1 and pos2.
    """
    return np.sqrt((pos1[0]-pos2[0])**2+(pos1[1]-pos2[1])**2)

def rearrange_labels_ROI_hippocampus(arr):
    """
    Reassigns and filters labels in a hippocampus ROI mask.
    Args:
        arr (ndarray): Input label array.

    Returns:
        ndarray: New array with relabeled and filtered classes.
    """
    new_labels = arr.copy()
    new_labels[arr == 1] = 0
    new_labels[arr == 2] = 1
    new_labels[arr == 102] = 2
    mask=np.isin(new_labels,[1,2])
    new_labels[~mask]=0
    return new_labels


def rearrange_labels_ROI_image(arr):
    """
    Reassigns labels in a generic ROI image according to a custom mapping.
    The final output is reduced modulo 91, possibly for downstream processing.

    Args:
        arr (ndarray): Input label array.

    Returns:
        ndarray: Remapped and filtered label array.
    """
    new_labels=arr.copy()
    new_labels[arr == 112] = 103

    new_labels[arr == 11] = 8
    new_labels[arr == 111] = 108

    new_labels[arr == 13] = 9
    new_labels[arr == 113] = 109


    mask=np.isin(new_labels,[1,2,3,4,5,6,7,8,9,101,102,103,104,105,106,107,108,109])
    new_labels[~mask]=0
    return new_labels%91

def read_image(img_url):
    return np.asarray(Image.open(img_url))




