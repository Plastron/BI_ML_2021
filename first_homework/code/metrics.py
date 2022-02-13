import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """
    res_list = []
    for i in range(len(y_pred)):
        if (y_pred[i] == "0") and (y_true[i] == "0"):
            res_list.append("tn")
        elif (y_pred[i] == "1") and (y_true[i] == "1"):
            res_list.append("tp")
        elif (y_pred[i] == "1") and (y_true[i] == "0"):
            res_list.append("fp")
        elif (y_pred[i] == "0") and (y_true[i] == "1"):
            res_list.append("fn")

    true_positive = res_list.count("tp")
    true_negative = res_list.count("tn")
    false_positive = res_list.count("fp")
    false_negative = res_list.count("fn")

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = true_positive / (true_positive + ((false_positive + false_negative) / 2))
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)

    final_dict = {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}

    return final_dict

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score


def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    correct = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            correct += 1

    final_dict = {"accuracy": correct / len(y_pred)}

    return final_dict


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """
    upper_sum = 0
    lower_sum = 0
    mean_y_true = y_true.mean()

    for i in range(len(y_true)):
        upper_sum += (y_true[i] - y_pred[i])**2
        lower_sum += (y_true[i] - mean_y_true)**2

    r2 = 1 - (upper_sum / lower_sum)

    return r2


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """
    sq_sum = 0
    for i in range(len(y_true)):
        sq_sum += (y_true[i] - y_pred[i])**2

    mse = sq_sum / len(y_true)

    return mse


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """
    abs_sum = 0
    for i in range(len(y_true)):
        abs_sum += abs(y_true[i] - y_pred[i])

    mae = abs_sum / len(y_true)

    return mae
