import numpy as np
from implementations import *
from helpers import *


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


def cross_validation(y, x, k_indices, k, lambda_, a, penalty_factor, max_iters, gamma):

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
    w_opt, loss_tr = sgd_for_svm(
        y_tr, x_tr, max_iters, gamma, lambda_, a, penalty_factor
    )
    loss_te = calculate_primal_objective(y_te, x_te, w_opt, lambda_, penalty_factor)
    y_pred, accuracy, precision, recall, f1_score_te = evaluate_performance(
        x_te, y_te, w_opt, model_labels={-1, 1}, limit=0
    )
    # ***************************************************

    return f1_score_te, loss_tr, loss_te


def cross_validation_demo(x, y, k_fold, lambda_, a, penalty_factor, max_iters, gamma):
    """cross validation over regularisation parameter lambda.

    Args:
        degree: integer, degree of the polynomial expansion
        k_fold: integer, the number of folds
        lambdas: shape = (p, ) where p is the number of values of lambda to test
    Returns:
        best_lambda : scalar, value of the best lambda
        best_rmse : scalar, the associated root mean squared error for the best lambda
    """

    # To find the best weights :
    best_weights = None
    best_loss_val = float("inf")
    f1 = 0
    loss_test = 0
    loss_train = 0
    seed = 1
    k_fold = k_fold
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    # ***************************************************
    for k in range(k_fold):
        # print(f"Currently at fold {i}")
        f1_score, loss_tr, loss_te = cross_validation(
            y, x, k_indices, k, lambda_, a, penalty_factor, max_iters, gamma
        )
        loss_train += loss_tr
        loss_test += loss_te
        f1 += f1_score

    av_loss_tr = loss_train / k_fold
    av_loss_te = loss_test / k_fold
    av_f1_score = f1 / k_fold
    # ***************************************************
    return av_f1_score, av_loss_tr, av_loss_te


def Hyperparameter(
    x_tr, y_tr, penalty_factor_list, gamma_list, lambda_list, a_list, k_fold
):
    results = []
    for penalty_factor in penalty_factor_list:
        for gamma in gamma_list:
            max_iters = int(10 / gamma)
            for lambda_ in lambda_list:
                for a in a_list:
                    av_f1_score, av_loss_tr, av_loss_te = cross_validation_demo(
                        x=x_tr,
                        y=y_tr,
                        k_fold=k_fold,
                        lambda_=lambda_,
                        a=a,
                        penalty_factor=penalty_factor,
                        max_iters=max_iters,
                        gamma=gamma,
                    )
                    # Store results in dictionary
                    result = {
                        "penalty_factor": penalty_factor,
                        "gamma": gamma,
                        "lambda": lambda_,
                        "a": a,
                        "f1-score": av_f1_score,
                    }
                    results.append(result)
    return results


def Get_best_results(results):
    # Initialize variables to track the best F1-score and corresponding parameters
    best_f1_score = 0
    best_params = None

    # Loop through the results to find the best F1-score on the validation set
    for result in results:
        f1_score_val = result["f1-score"]

        # Update if a new best F1-score is found
        if f1_score_val > best_f1_score:
            best_f1_score = f1_score_val
            best_params = {
                "penalty_factor": result["penalty_factor"],
                "gamma": result["gamma"],
                "lambda": result["lambda"],
                "a": result["a"],
            }

    # Output the best F1-score and corresponding parameters
    print("Best F1-Score:", best_f1_score)
    print("Best Parameters and Metrics:", best_params)
    return best_f1_score, best_params
