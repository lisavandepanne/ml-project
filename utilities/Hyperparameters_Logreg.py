### This file is used to test different hyperparameters, and then selecting the best one


import numpy as np
import matplotlib.pyplot as plt
from implementations import *


def evaluate_performance(x, y, w, limit):
    """
    Evaluates the performance of a binary classifier by calculating various metrics,
    including accuracy, precision, recall, and F1 score.

    Parameters:
    x (np.ndarray): Input feature matrix (shape: [n_samples, n_features]).
    y (np.ndarray): True binary labels (shape: [n_samples]).
    w (np.ndarray): Weight vector for the model (shape: [n_features]).
    limit (float): Threshold to convert predictions to binary classes (0 or 1).

    Returns:
    tuple: A tuple containing:
        - y_pred (np.ndarray): Predicted binary labels (0 or 1) based on the threshold.
        - accuracy (float): Ratio of correct predictions to total predictions.
        - precision (float): Ratio of true positives to all predicted positives.
        - recall (float): Ratio of true positives to all actual positives.
        - f1_score (float): Harmonic mean of precision and recall, representing model balance.

    Raises:
    ValueError: If `y_pred` contains values other than 0 or 1 after applying the threshold.
    """

    # Calculate the predictions
    y_pred = x @ w

    # Apply threshold to convert predictions to binary classes
    y_pred[y_pred <= limit] = 0
    y_pred[y_pred > limit] = 1

    # Check if all values in y_pred are either 0 or 1
    if np.all((y_pred == 0) | (y_pred == 1)):
        None
        # print("All values are either 0 or 1.")
    else:
        raise ValueError("The array contains values other than 0 or 1.")

    # Calculate accuracy
    accuracy = np.mean(y == y_pred)
    # print("Accuracy:", accuracy)

    # Compute confusion matrix components
    TP = np.sum((y == 1) & (y_pred == 1))
    FP = np.sum((y == 0) & (y_pred == 1))
    FN = np.sum((y == 1) & (y_pred == 0))

    # Calculate Precision and Recall
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # Calculate F1 Score
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    # Print metrics
    # print("Precision:", precision)
    # print("Recall (True positive rate):", recall)
    # print("F1 Score:", f1_score)

    return y_pred, accuracy, precision, recall, f1_score


def build_k_indices(y, k_fold, seed=1):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)  # nombre d'éléments qui seront dans chaque fold
    np.random.seed(seed)
    indices = np.random.permutation(num_row)  # permute les indices
    k_indices = [
        indices[k * interval : (k + 1) * interval] for k in range(k_fold)
    ]  # sépare les indices
    return np.array(k_indices)


def cross_validation(y, x, k_indices, k, lambda_, initial_w, max_iters, gamma):

    # ***************************************************
    # x_tr = np.array([])
    # y_tr = np.array([])
    x_tr = np.empty((0, x.shape[1]))
    y_tr = np.empty((0,))

    for i in range(np.shape(k_indices)[0]):
        if i == k:
            x_te = x[k_indices[i]]
            y_te = y[k_indices[i]]
        else:
            x_tr = np.concatenate((x_tr, x[k_indices[i]]))
            y_tr = np.concatenate((y_tr, y[k_indices[i]]))
    # ***************************************************

    # ***************************************************
    w, loss_tr, list_loss_tr = reg_logistic_regression_hyper(
        y_tr, x_tr, lambda_, initial_w, max_iters, gamma
    )
    loss_te = calculate_loss(y_te, x_te, w)
    # ***************************************************

    return w, loss_tr, list_loss_tr, loss_te


def cross_validation_demo(x, y, k_fold, initial_w, lambda_, max_iters, gamma, limit):
    """cross validation over regularisation parameter lambda.

    Args:
        degree: integer, degree of the polynomial expansion
        k_fold: integer, the number of folds
        lambdas: shape = (p, ) where p is the number of values of lambda to test
    Returns:
        best_lambda : scalar, value of the best lambda
        best_rmse : scalar, the associated root mean squared error for the best lambda
    """
    f1 = 0
    loss_train = 0
    loss_test = 0
    seed = 1
    k_fold = k_fold
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    # ***************************************************
    for i in range(k_fold):
        # print(f"Currently at fold {i}")
        w, loss_tr, list_loss_tr, loss_te = cross_validation(
            y, x, k_indices, i, lambda_, initial_w, max_iters, gamma
        )
        y_pred_tr, accuracy, precision, recall, f1_score = evaluate_performance(
            x, y, w, limit
        )
        f1 += f1_score
        loss_train += loss_tr
        loss_test += loss_te
    av_loss_tr = loss_train / k_fold
    av_loss_te = loss_test / k_fold
    av_f1_score = f1 / k_fold
    # ***************************************************
    return av_f1_score, av_loss_tr, av_loss_te


def Hyperparameters(
    x_tr, y_tr, k_fold, max_iters, regularization_term, learning_rate, limits, initial_w
):

    results = []
    for lambda_ in regularization_term:
        for gamma in learning_rate:
            for limit in limits:
                av_f1_score, av_loss_tr, av_loss_te = cross_validation_demo(
                    x_tr, y_tr, k_fold, initial_w, lambda_, max_iters, gamma, limit
                )
                results.append(
                    {
                        "regularization_term": lambda_,
                        "learning_rate": gamma,
                        "limit": limit,
                        "av_f1_score": av_f1_score,
                    }
                )

    return results


def Get_best_results(results):
    # Initialize variables to track the best F1-score and corresponding parameters
    best_f1_score = 0
    best_params = None

    # Loop through the results to find the best F1-score on the validation set
    for result in results:
        f1_score_val = result["av_f1_score"]

        # Update if a new best F1-score is found
        if f1_score_val > best_f1_score:
            best_f1_score = f1_score_val
            best_params = {
                "regularization_term": result["regularization_term"],
                "learning_rate": result["learning_rate"],
                "limit": result["limit"],
            }

    # Output the best F1-score and corresponding parameters
    print("Best F1-Score:", best_f1_score)
    print("Best Parameters and Metrics:", best_params)
    return best_f1_score, best_params


# same function as in implementations but without the prints
def reg_logistic_regression_hyper(y, tx, lambda_, initial_w, max_iters, gamma):
    """Perform optimisation steps in logistic regression, with penalisation term (regularisation)

    Args:
        y:  shape=(N, ) labels are 0 or 1
        tx: shape=(N, D)
        lambda_: float, the regularization term
        inital_w:  shape=(D, )
        max_iters: int
        gamma: float, the learning rate

    Returns:
        loss: scalar number
        w: shape=(D, ) optimal weights
    """

    # init parameters
    threshold = 1e-8
    prev_loss = float("inf")
    w = initial_w
    list_loss = np.zeros((max_iters))

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        w, loss = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        list_loss[iter] = loss

        # converge criterion
        if np.abs(loss - prev_loss) < threshold:
            break
        # Update previous loss
        prev_loss = loss

    loss = calculate_loss(y, tx, w)
    return w, loss, list_loss
