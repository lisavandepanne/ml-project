import numpy as np
import matplotlib.pyplot as plt
from utilities.helpers import *
from utilities.Data_preprocessing_global import *
from utilities.Hyperparameters_SVM import *
from implementations import *

# Importing the data
# You have to change the path for it to work
data_path = r"C:\Users\natha\Documents\EPFL\Cours_MA1\ML\ML_course\projects\project1 - withGit\data\dataset"
x_train, x_test, y_train, train_ids, test_ids, headers_train = load_csv_data(
    data_path, sub_sample=False
)

# Pre-processing
x_tr, x_val, y_tr, y_val, x_train_full, x_test_formatted, remaining_headers = (
    data_preprocess(
        x_train,
        y_train,
        x_test,
        headers_train,
        model_labels={-1, 1},
        ratio_miss=0.1,
        ratio_train=1,
        standardization=True,
    )
)

# Training
penalty_factor = 10
gamma = 0.01
lambda_ = 0.01
a = 0.5
max_iter = int(10 / gamma)

w_opt, loss_tr = sgd_for_svm(y_tr, x_tr, max_iter, gamma, lambda_, a, penalty_factor)

y_pred, accuracy, precision, recall, f1_score = evaluate_performance(
    x_tr, y_tr, w_opt, model_labels={-1, 1}, limit=0
)
print(accuracy)
print(f1_score)

# Submission
best_limit = 0  # It doesn't matter here, if we say that the model_labels are {-1, 1}
best_w = w_opt
name = "SVM_submission_file.csv"
model_labels = {-1, 1}

y_pred_test = submission(
    x_test_formatted, test_ids, best_limit, best_w, name, model_labels
)
