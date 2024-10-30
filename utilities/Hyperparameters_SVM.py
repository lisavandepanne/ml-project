import numpy as np
import matplotlib.pyplot as plt


def evaluate_performance(x, y, w):

    # Calculate the predictions
    y_pred = np.sign(x @ w)

    # Check if all values in y_pred are either -1 or 1
    if np.all((y_pred == -1) | (y_pred == 1)):
        None
        # print("All values are either -1 or 1.")
    else:
        raise ValueError("The array contains values other than -1 or 1.")

    # Calculate accuracy
    accuracy = np.mean(y == y_pred)
    print("Accuracy:", accuracy)

    # Compute confusion matrix components
    TP = np.sum((y == 1) & (y_pred == 1))
    FP = np.sum((y == -1) & (y_pred == 1))
    FN = np.sum((y == 1) & (y_pred == -1))

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
):
    results = []

    for ratio in ratio_missing:
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
        )
        for lambda_ in regularization_term:
            for gamma in learning_rate:
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
