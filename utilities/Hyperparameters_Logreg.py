import numpy as np
import matplotlib.pyplot as plt


def evaluate_performance(x, y, w, limit):

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
    print("Precision:", precision)
    print("Recall (True positive rate):", recall)
    print("F1 Score:", f1_score)

    return y_pred, accuracy, precision, recall, f1_score


def build_k_indices(y, k_fold, seed=1):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
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


def cross_validation_demo(x, y, k_fold, lambda_, initial_w, max_iters, gamma):
    """cross validation over regularisation parameter lambda.

    Args:
        degree: integer, degree of the polynomial expansion
        k_fold: integer, the number of folds
        lambdas: shape = (p, ) where p is the number of values of lambda to test
    Returns:
        best_lambda : scalar, value of the best lambda
        best_rmse : scalar, the associated root mean squared error for the best lambda
    """

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
        loss_tr += loss_tr
        loss_te += loss_te
    av_loss_tr = loss_tr / k_fold
    av_loss_te = loss_te / k_fold
    # ***************************************************
    return w, av_loss_tr, av_loss_te


def hyperparameters(
    x_train,
    y_train,
    headers_train,
    ratio_missing,
    regularization_term,
    learning_rate,
    limits,
    initial_w,
    max_iters,
    zero=False,
    mean=False,
    mode=False,
    k_fold=None,
):
    results = []

    for ratio in ratio_missing:
        print(f"Ratio of missing data : {ratio}")
        x_tr, x_val, y_tr, y_val, data_filled, remaining_headers = data_preprocess(
            x_train,
            y_train,
            headers_train,
            ratio_miss=ratio,
            ratio_train=0.8,
            standardization=True,
            zero=zero,
            mean=mean,
            mode=mode,
            hyper=True,
        )
        for lambda_ in regularization_term:
            for gamma in learning_rate:
                if k_fold:
                    w_opt, loss_opt, list_loss = cross_validation_demo(
                        x_tr, y_tr, k_fold, lambda_, initial_w, max_iters, gamma
                    )
                else:
                    w_opt, loss_opt, list_loss = reg_logistic_regression_hyper(
                        y_tr, x_tr, lambda_, initial_w, max_iters, gamma
                    )
                for limit in limits:
                    y_pred_tr, accuracy_tr, precision_tr, recall_tr, f1_score_tr = (
                        evaluate_performance(x_tr, y_tr, w_opt, limit)
                    )
                    (
                        y_pred_val,
                        accuracy_val,
                        precision_val,
                        recall_val,
                        f1_score_val,
                    ) = evaluate_performance(x_val, y_val, w_opt, limit)

                    results.append(
                        {
                            "ratio_missing": ratio,
                            "regularization_term": lambda_,
                            "learning_rate": gamma,
                            "limit": limit,
                            "loss_opt": loss_opt,
                            "list_loss": list_loss,
                            "w_opt": w_opt,
                            "accuracy_tr": accuracy_tr,
                            "precision_tr": precision_tr,
                            "recall_tr": recall_tr,
                            "f1_score_tr": f1_score_tr,
                            "accuracy_val": accuracy_val,
                            "precision_val": precision_val,
                            "recall_val": recall_val,
                            "f1_score_val": f1_score_val,
                        }
                    )

    return results


def analyze_results(results):

    # Find the configuration with the lowest loss_opt
    lowest_loss_entry = min(results, key=lambda x: x["loss_opt"])

    # Extract parameters corresponding to the lowest loss
    target_gamma = lowest_loss_entry["learning_rate"]
    target_lambda = lowest_loss_entry["regularization_term"]
    target_ratio = lowest_loss_entry["ratio_missing"]

    print(f"Lowest loss_opt found with parameters:")
    print(
        f"Gamma: {target_gamma}, Lambda: {target_lambda}, Ratio: {target_ratio}, Loss: {lowest_loss_entry['loss_opt']}"
    )

    # Filter the results for the identified parameters
    filtered_results = [
        res
        for res in results
        if res["learning_rate"] == target_gamma
        and res["regularization_term"] == target_lambda
        and res["ratio_missing"] == target_ratio
    ]

    filtered_results = sorted(filtered_results, key=lambda x: x["limit"])

    # Extract the limit values and corresponding training metrics
    limits = [res["limit"] for res in filtered_results]
    accuracy_tr_values = [res["accuracy_tr"] for res in filtered_results]
    precision_tr_values = [res["precision_tr"] for res in filtered_results]
    recall_tr_values = [res["recall_tr"] for res in filtered_results]
    f1_score_tr_values = [res["f1_score_tr"] for res in filtered_results]

    # Plot each metric as a function of 'limit'
    plt.figure(figsize=(10, 6))
    plt.plot(limits, accuracy_tr_values, marker="o", label="Accuracy")
    plt.plot(limits, precision_tr_values, marker="o", label="Precision")
    plt.plot(limits, recall_tr_values, marker="o", label="Recall")
    plt.plot(limits, f1_score_tr_values, marker="o", label="F1 Score")

    plt.xlabel("Limit")
    plt.ylabel("Metric Value")
    plt.title(
        f"Variation of Training Metrics with Limit\n(gamma={target_gamma}, lambda={target_lambda}, ratio={target_ratio})"
    )
    plt.legend()
    plt.grid(True)
    plt.show()


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
