import numpy as np


def compute_MSE(y, tx, w):
    """Calculate the loss using MSE.

    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        w: shape=(D,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """

    e = y - tx @ w
    return e @ e / (2 * y.size)


def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        w: shape=(D, ). The vector of model parameters.

    Returns:
        An array of shape (D, ) (same shape as w), containing the gradient of the loss at w.
    """

    e = y - tx @ w
    return -tx.T @ e / y.size


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar (mse)
    """

    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient(y, tx, w)
        loss = compute_MSE(y, tx, w)

        w = w - gamma * grad

        print(
            "GD iter. {bi}/{ti}: loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss)
        )
    loss = compute_MSE(y, tx, w)
    return w, loss


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from a data sample batch of size B, where B < N, and their corresponding labels.

    Args:
        y: numpy array of shape=(B, )
        tx: numpy array of shape=(B,D)
        w: numpy array of shape=(D, ). The vector of model parameters.

    Returns:
        A numpy array of shape (D, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """

    e = y - tx @ w
    return -tx.T @ e / y.size


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.

    Example:

     Number of batches = 9

     Batch size = 7                              Remainder = 3
     v     v                                         v v
    |-------|-------|-------|-------|-------|-------|---|
        0       7       14      21      28      35   max batches = 6

    If shuffle is False, the returned batches are the ones started from the indexes:
    0, 7, 14, 21, 28, 35, 0, 7, 14

    If shuffle is True, the returned batches start in:
    7, 28, 14, 35, 14, 0, 21, 28, 7

    To prevent the remainder datapoints from ever being taken into account, each of the shuffled indexes is added a random amount
    8, 28, 16, 38, 14, 0, 22, 28, 9

    This way batches might overlap, but the returned batches are slightly more representative.

    Disclaimer: To keep this function simple, individual datapoints are not shuffled. For a more random result consider using a batch_size of 1.

    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """

    data_size = len(y)  # NUmber of data points.
    batch_size = min(data_size, batch_size)  # Limit the possible size of the batch.
    max_batches = int(
        data_size / batch_size
    )  # The maximum amount of non-overlapping batches that can be extracted from the data.
    remainder = (
        data_size - max_batches * batch_size
    )  # Points that would be excluded if no overlap is allowed.

    if shuffle:
        # Generate an array of indexes indicating the start of each batch
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            # Add an random offset to the start of each batch to eventually consider the remainder points
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        # If no shuffle is done, the array of indexes is circular.
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start in idxs:
        start_index = start  # The first data point of the batch
        end_index = (
            start_index + batch_size
        )  # The first data point of the following batch
        yield y[start_index:end_index], tx[start_index:end_index]


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar (mse)
    """

    w = initial_w

    batch_size = 1

    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, shuffle=True):

            grad = compute_stoch_gradient(minibatch_y, minibatch_tx, w)
            loss = compute_MSE(y, tx, w)

            w = w - gamma * grad

            print(
                "SGD iter. {bi}/{ti}: loss={l}".format(
                    bi=n_iter, ti=max_iters - 1, l=loss
                )
            )
    loss = compute_MSE(y, tx, w)
    return w, loss


def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar (mse)
    """

    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    loss = compute_MSE(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar, the regularization term

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar (mse)
    """

    N = y.shape[0]
    D = tx.shape[1]
    reg_matrix = 2 * N * np.identity(D) * lambda_
    w = np.linalg.solve(tx.T @ tx + reg_matrix, tx.T @ y)
    loss = compute_MSE(y, tx, w)
    return w, loss


def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array
    """

    return 1 / (1 + np.exp(-t))


def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, ) labels are 0 or 1
        tx: shape=(N, D)
        w:  shape=(D, )

    Returns:
        a non-negative loss
    """

    N = y.size
    assert N == tx.shape[0]
    assert tx.shape[1] == w.size

    y_hat = tx @ w
    log_term = np.log1p(np.exp(y_hat))
    loss = (-y * y_hat + log_term).sum() / N
    return loss


def calculate_gradient(y, tx, w):
    """compute the gradient of loss.

    Args:
        y:  shape=(N, ) labels are 0 or 1
        tx: shape=(N, D)
        w:  shape=(D, )

    Returns:
        a vector of shape (D, )
    """

    y_hat = tx @ w
    return tx.T @ (sigmoid(y_hat) - y) / y.size


def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression. Return the loss and the updated w.

    Args:
        y:  shape=(N, ) labels are 0 or 1
        tx: shape=(N, D)
        w:  shape=(D, )
        gamma: float, the learning rate

    Returns:
        loss: scalar number
        w: shape=(D, ) updated weights
    """

    loss = calculate_loss(y, tx, w)
    w = w - gamma * calculate_gradient(y, tx, w)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Perform optimisation steps in logistic regression.

    Args:
        y:  shape=(N, ) labels are 0 or 1
        tx: shape=(N, D)
        inital_w:  shape=(D, )
        max_iters: int
        gamma: float

    Returns:
        loss: scalar number
        w: shape=(D, )
    """

    # init parameters
    threshold = 1e-8
    prev_loss = float("inf")
    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        w, loss = learning_by_gradient_descent(y, tx, w, gamma)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        if np.abs(loss - prev_loss) < threshold:
            break
        # Update previous loss
        prev_loss = loss

    loss = calculate_loss(y, tx, w)
    print("loss={l}".format(l=loss))
    return w, loss


def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss and gradient.

    Args:
        y:  shape=(N, ) labels are 0 or 1
        tx: shape=(N, D)
        w:  shape=(D, )
        lambda_: scalar, the regularization term

    Returns:
        loss: scalar number
        gradient: shape=(D, )
    """

    loss = calculate_loss(y, tx, w)
    gradient = calculate_gradient(y, tx, w) + 2 * lambda_ * w

    return loss, gradient


def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.

    Args:
        y:  shape=(N, ) labels are 0 or 1
        tx: shape=(N, D)
        w:  shape=(D, )
        gamma: scalar, the learning rate
        lambda_: scalar, the regularization term

    Returns:
        loss: scalar number
        w: shape=(D, ) updated weights
    """

    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    w = w - gamma * gradient
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
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
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        if np.abs(loss - prev_loss) < threshold:
            break
        # Update previous loss
        prev_loss = loss

    loss = calculate_loss(y, tx, w)
    print("loss={l}".format(l=loss))
    return w, loss
