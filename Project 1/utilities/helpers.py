"""Some helper functions for project 1."""

import csv
import numpy as np
import os


def load_csv_data(data_path, sub_sample=False):
    """
    This function loads the data and returns the respectinve numpy arrays.
    Remember to put the 3 files in the same folder and to not change the names of the files.

    Args:
        data_path (str): datafolder path
        sub_sample (bool, optional): If True the data will be subsempled. Default to False.

    Returns:
        x_train (np.array): training data
        x_test (np.array): test data
        y_train (np.array): labels for training data in format (-1,1)
        train_ids (np.array): ids of training data
        test_ids (np.array): ids of test data
    """
    print("test")
    ###################################################################
    with open(os.path.join(data_path, "x_train.csv")) as f:
        headers_train = f.readline().strip().split(",")
    ###################################################################

    y_train = np.genfromtxt(
        os.path.join(data_path, "y_train.csv"),
        delimiter=",",
        skip_header=1,
        dtype=int,
        usecols=1,
    )
    x_train = np.genfromtxt(
        os.path.join(data_path, "x_train.csv"), delimiter=",", skip_header=1
    )
    x_test = np.genfromtxt(
        os.path.join(data_path, "x_test.csv"), delimiter=",", skip_header=1
    )

    train_ids = x_train[:, 0].astype(dtype=int)
    test_ids = x_test[:, 0].astype(dtype=int)
    x_train = x_train[:, 1:]
    x_test = x_test[:, 1:]

    # sub-sample
    if sub_sample:
        y_train = y_train[::50]
        x_train = x_train[::50]
        train_ids = train_ids[::50]

    return x_train, x_test, y_train, train_ids, test_ids, headers_train


def create_csv_submission(ids, y_pred, name):
    """
    This function creates a csv file named 'name' in the format required for a submission in Kaggle or AIcrowd.
    The file will contain two columns the first with 'ids' and the second with 'y_pred'.
    y_pred must be a list or np.array of 1 and -1 otherwise the function will raise a ValueError.

    Args:
        ids (list,np.array): indices
        y_pred (list,np.array): predictions on data correspondent to indices
        name (str): name of the file to be created
    """
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


def submission(x_test, ids_test, limit, w_opt, name, model_labels):
    if model_labels == {0, 1}:
        y_pred = x_test @ w_opt
        y_pred[y_pred <= limit] = 0
        y_pred[y_pred > limit] = 1
        y_pred = np.where(y_pred == 0, -1, y_pred)
    elif model_labels == {-1, 1}:
        y_pred = np.sign(x_test @ w_opt)

    create_csv_submission(ids_test, y_pred, name)

    return y_pred


def evaluate_performance(x, y, w, model_labels, limit):
    if model_labels == {-1, 1}:
        # Calculate the predictions
        y_pred = np.sign(x @ w)

        # Check if all values in y_pred are either -1 or 1
        if np.all((y_pred == -1) | (y_pred == 1)):
            pass
        else:
            raise ValueError("The array contains values other than -1 or 1.")

        # Calculate accuracy
        accuracy = np.mean(y == y_pred)

        # Compute confusion matrix components
        TP = np.sum((y == 1) & (y_pred == 1))
        FP = np.sum((y == -1) & (y_pred == 1))
        FN = np.sum((y == 1) & (y_pred == -1))

    elif model_labels == {0, 1}:
        # Calculate the predictions
        y_pred = x @ w

        # Apply threshold to convert predictions to binary classes
        y_pred[y_pred <= limit] = 0
        y_pred[y_pred > limit] = 1

        # Check if all values in y_pred are either -1 or 1
        if np.all((y_pred == 0) | (y_pred == 1)):
            pass
        else:
            raise ValueError("The array contains values other than 0 or 1.")

        # Calculate accuracy
        accuracy = np.mean(y == y_pred)

        # Compute confusion matrix components
        TP = np.sum((y == 1) & (y_pred == 1))
        FP = np.sum((y == 0) & (y_pred == 1))
        FN = np.sum((y == 1) & (y_pred == 0))
    else:
        raise ValueError("model_labels should be either {-1, 1} or {0, 1}")

    # Calculate Precision and Recall
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # Calculate F1 Score
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return y_pred, accuracy, precision, recall, f1_score
