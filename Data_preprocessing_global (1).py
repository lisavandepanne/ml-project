import numpy as np


### This file compiles all the function used to preprocess the data


def find_missing_values(data, headers=None):
    """
    Finds the count of missing (NaN) values in each column of the dataset.

    Parameters:
    data (np.ndarray): The input NumPy array containing the data.
    headers (list of str, optional): List of column names in the dataset. If provided, column names will be used in the output.
                                      If not provided, column indices will be used.

    Returns:
    dict: A dictionary where keys are column names (if headers provided) or column indices, and values are the count of missing
    values (NaNs) in the respective columns.
    """
    num_rows, num_cols = data.shape
    missing_count = np.zeros(num_cols, dtype=int)
    columns = np.linspace(0, num_cols, num_cols + 1)

    for col in range(num_cols):
        # Count the missing values
        missing_count[col] = np.sum(np.isnan(data[:, col]))
    if headers:
        # Returning only the columns with missing values
        missing_info = {
            headers[col]: missing_count[col]
            for col in range(num_cols)
            if missing_count[col] > 0
        }
    else:
        missing_info = {
            columns[col]: missing_count[col]
            for col in range(num_cols)
            if missing_count[col] > 0
        }
    return missing_info


### TEST


# We fix a threshold of 10% inside the function remove_high_missing_columns. We choose 10% because we assume that if there are
# more missing values than that, the feature becomes unreliable for prediction.
# !!! This threshold must not be modified, because my data modification function that considers each feature individually was not
# specifically designed to account for columns with a higher number of missing values that are not in columns_to_keep. !!!
def remove_high_missing_columns(
    data,
    test_points,
    ratio=10,
    headers=None,
    headers_to_keep=None,
    headers_to_remove=None,
):
    """
    Removes columns with high missing values from the dataset, with options to automatically keep or remove specific columns.

    Parameters:
    data (np.ndarray): The input NumPy array containing the data.
    headers (list of str): List of all column names in the dataset.
    headers_to_keep (list of str): List of column names to keep regardless of missing values.
    headers_to_remove (list of str): List of column names to remove regardless of missing values.

    Returns:
    np.ndarray: A filtered NumPy array with the remaining columns.
    list of str: The list of remaining headers.
    """
    assert data.shape[1] == len(headers)

    num_rows, num_cols = data.shape
    threshold = num_rows / ratio

    # Find the missing values count for each column
    missing_count = find_missing_values(data, headers)

    # Determine the indices of columns to automatically keep and remove
    # indices_to_keep = [headers.index(col) for col in headers_to_keep if col in headers] if headers_to_keep else []
    # indices_to_remove = [headers.index(col) for col in headers_to_remove if col in headers] if headers_to_remove else []

    # Create the mask for columns that meet the criteria
    if headers_to_keep and headers_to_remove:
        columns_to_keep = [
            col
            for col in range(num_cols)
            if (
                (headers[col] in headers_to_keep)
                or (
                    headers[col] not in headers_to_remove
                    and missing_count.get(headers[col], 0) <= threshold
                )
            )
        ]
        remaining_headers = [headers[col] for col in columns_to_keep]
    else:
        # Identify columns to keep (those with missing values below the threshold)
        columns_to_keep = [
            col for col, count in missing_count.items() if count <= threshold
        ]

    # If headers are provided, adjust the list of headers to match the remaining columns
    # if headers:
    #    remaining_headers = [headers[col] for col in columns_to_keep]
    # else:
    #    remaining_headers= None

    # Filter the data and headers to only include the columns that meet the criteria
    filtered_data = data[:, columns_to_keep]
    filtered_test = test_points[:, columns_to_keep]

    return filtered_data, remaining_headers, filtered_test


def remove_high_missing_columns2(data, test_points, ratio=2, headers=None):
    """
    Removes columns from the dataset with a high proportion of missing values based on a specified threshold.

    Parameters:
    data (np.ndarray): The input NumPy array containing the data.
    headers (list of str, optional): List of all column names in the dataset.

    Returns:
    np.ndarray: A filtered NumPy array with columns containing excess missing values removed.
    list of str: The list of remaining headers after column removal.
    """
    # Ensure the headers, if provided, match the number of columns
    if headers:
        assert data.shape[1] == len(
            headers
        ), "The number of headers must match the number of columns in the data."

    num_rows, num_cols = data.shape
    threshold = num_rows / ratio

    # Find the missing values count for each column
    missing_count = find_missing_values(data, headers)

    # Identify columns to keep based on missing value threshold
    columns_to_keep = [
        col
        for col in range(num_cols)
        if missing_count.get(headers[col] if headers else col, 0) <= threshold
    ]

    # Adjust headers if provided
    remaining_headers = [headers[col] for col in columns_to_keep] if headers else None

    # Filter the data to include only columns that meet the criteria
    filtered_data = data[:, columns_to_keep]
    filtered_test_points = test_points[:, columns_to_keep]

    return filtered_data, remaining_headers, filtered_test_points


def replace_nan(data, zero=None, mean=None, mode=None):
    """
    Replaces NaN values in each column of the dataset with the mean of the non-NaN values in that column.

    Parameters:
    data (np.ndarray): The input NumPy array containing the data with potential NaN values.

    Returns:
    np.ndarray: A new array with NaN values replaced by the column means.
    """
    # Copy the data to avoid modifying the original array
    data_filled = data.copy()

    if mean:
        # Iterate through each column
        for col in range(data.shape[1]):
            # Calculate the mean of non-NaN values in the column
            col_mean = np.nanmean(data[:, col])

            # Replace NaNs in the column with the mean
            nan_indices = np.isnan(data[:, col])
            data_filled[nan_indices, col] = col_mean

    if zero:
        data_filled = np.nan_to_num(data, nan=0.0)

    # Replace the more frequent value (good with categorical data)
    if mode:
        for col in range(data.shape[1]):
            nan_indices = np.isnan(data[:, col])
            column = data_filled[:, col]
            if len(nan_indices) > 0:
                unique_values, counts = np.unique(column, return_counts=True)
                mode_value = np.argmax(counts)
                data_filled[nan_indices, col] = mode_value
            else:
                None

    return data_filled


def replace_nan_in_binary(data, binary_features):
    data_replaced = data.copy()  # Create a copy to avoid modifying the original data

    for feature_index in binary_features:
        # Calculate the mode, ignoring NaN values
        non_nan_values = data_replaced[
            ~np.isnan(data_replaced[:, feature_index]), feature_index
        ]
        unique_values, counts = np.unique(non_nan_values, return_counts=True)
        feature_mode = unique_values[np.argmax(counts)]

        # Replace NaNs with the mode
        nan_mask = np.isnan(data_replaced[:, feature_index])
        data_replaced[nan_mask, feature_index] = feature_mode

    return data_replaced


def replace_nan_in_ordinal(data, ordinal_features):
    data_replaced = data.copy()  # Create a copy to avoid modifying the original data

    for feature_index in ordinal_features:
        # Replace NaNs with 0
        nan_mask = np.isnan(data_replaced[:, feature_index])
        data_replaced[nan_mask, feature_index] = 0

    return data_replaced


def replace_zero_with_minus_one_in_binary(data, binary_features):
    data_modified = data.copy()  # Create a copy to avoid modifying the original data

    for feature_index in binary_features:
        # Replace 0 with -1 in the specified binary feature
        zero_mask = data_modified[:, feature_index] == 0
        data_modified[zero_mask, feature_index] = -1

    return data_modified


# A function to check if the column has specific values:
def unique_values(data, column_name, headers):
    """
    Returns a sorted list of unique values from a specified column, excluding NaN values.

    Parameters:
    data (np.ndarray): The input NumPy array containing the data.
    column_name (str): The name of the column to analyze.
    headers (list of str): List of all column names in the dataset.

    Returns:
    list: A sorted list of unique values from the specified column, excluding NaNs.
    """
    if column_name not in headers:
        raise ValueError(f"Column '{column_name}' not found in headers.")

    # Get the index of the specified column
    col_index = headers.index(column_name)

    # Extract the column values and filter out NaN values
    column = data[:, col_index]
    non_nan_values = column[~np.isnan(column)]

    # Get the unique values and sort them in ascending order
    unique_values = sorted(set(non_nan_values))

    return unique_values


# After analysing every feature, a sample of the rules for data transformation are given below and transcribed in the function replace_specific_values:

# DIABETE3 replace 2 by 1, replace 3, 4, 7, 9 by 0
# SMOKDAY2 replace 2 by 1, replace 3, 7, 9 by 0
# USENOW3 replace 2 by 1, replace 3, 7, 9 by 0
# ALCDAY5 replace 777, 888, 999 by 0
# AVEDRNK replace 77, 99 by 0
# DRNK3GE5 replace 77, 88, 99 by 0
# PREDIAB1 replace 2 by 1, replace 3, 7, 9 by 0
# BLDSUGAR if starts by 1, two last*365. If starts by 2, two last*52. If starts by 3, two last*12, if starts by 4, two last. If > 499, 0.
# DOCTDIAB if >76, replace by 0
# FC60_ replace 99900 by average of rest
# Some sets have 1,2,3,4,5,6,7,9 should replace the 9
# MARITAL if not 1, replace by 0
# PHYSHLTH replace 88, 77, 99 by 0
# POORHLTH replace 88, 77, 99 by 0
# PERSDOC2 replace 2 by 1, replace 3, 7, 9 by 0
# SEX replace 2 by 0
# EDUCA replace 9 by 0
# INCOME2 replace 77 and 99 by 4
# WEIGHT2 if starts by 9, replace by 162
# LASTSMK2 replace 77, 99 by Nan
# MAXDRNKS replace 77, 99 by 0
# FRUITJU1 if >= 300 replace by 0, if < 300 replace by 1
# FRUIT1 if >= 300 replace by 0, if < 300 replace by 1
# FVBEANS if >= 300 replace by 0, if < 300 replace by 1
# FVORANG if >= 300 replace by 0, if < 300 replace by 1
# VEGETAB1 if >= 300 replace by 0, if < 300 replace by 1
# EXERHMM1 replace by 1, 2, 3, 4 if 1<30<100<200
# JOINPAIN replace 77 and 99 by 0
# _ASTHMS1 replace 2 by 1, replace 3, 9 by 0
# _PRACE1 one hot encoding
# _MRACE1 one hot encoding


def replace_specific_values(old_data, headers):
    """
    Iterates through each column in the dataset and performs an action if the unique values of the column match the target values.

    Parameters:
    old_data (np.ndarray): The input NumPy array containing the data.
    headers (list of str): List of all column names in the dataset.

    Returns:
    data (np.ndarray): The output NumPy array with the modified values.
    """
    data = np.copy(old_data)

    for column_name in headers:
        # Get unique values from the column
        unique_vals = unique_values(data, column_name, headers)

        if unique_vals == [1, 2, 9]:
            # Find the column index
            col_idx = headers.index(column_name)
            column = data[:, col_idx]

            column = np.where(column == 2, 0, column)
            column = np.where(column == 9, np.nan, column)

            # Update the data with modified column
            data[:, col_idx] = column

        elif unique_vals == [1, 2, 7, 9]:
            col_idx = headers.index(column_name)
            column = data[:, col_idx]

            column = np.where(column == 2, 0, column)
            column = np.where(column == 7, np.nan, column)
            column = np.where(column == 9, np.nan, column)

            data[:, col_idx] = column

        elif unique_vals == [1, 2, 3, 9]:
            col_idx = headers.index(column_name)
            column = data[:, col_idx]

            column = np.where(column == 2, 1, column)
            column = np.where(column == 3, 0, column)
            column = np.where(column == 9, np.nan, column)

            data[:, col_idx] = column

        elif unique_vals == [1, 2, 3, 7, 9]:
            col_idx = headers.index(column_name)
            column = data[:, col_idx]

            column = np.where(column == 2, 0, column)
            column = np.where(column == 3, 0, column)
            column = np.where(column == 7, np.nan, column)
            column = np.where(column == 9, np.nan, column)

            data[:, col_idx] = column

        elif unique_vals == [1, 2, 3, 4, 9] and column_name != "_PACAT1":
            col_idx = headers.index(column_name)
            column = data[:, col_idx]

            column = np.where(column == 2, 1, column)
            column = np.where(column == 3, 1, column)
            column = np.where(column == 4, 0, column)
            column = np.where(column == 9, np.nan, column)

            data[:, col_idx] = column

        elif unique_vals == [1, 2, 3, 4, 7, 9]:
            col_idx = headers.index(column_name)
            column = data[:, col_idx]

            column = np.where(column == 2, 0, column)
            column = np.where(column == 3, 0, column)
            column = np.where(column == 4, 0, column)
            column = np.where(column == 7, np.nan, column)
            column = np.where(column == 9, np.nan, column)

            data[:, col_idx] = column

        elif unique_vals == [1, 2, 3, 4, 5, 6, 7, 9]:
            col_idx = headers.index(column_name)
            column = data[:, col_idx]

            column = np.where(column == 9, np.nan, column)

            data[:, col_idx] = column

        elif unique_vals == [1, 2, 3]:
            col_idx = headers.index(column_name)
            column = data[:, col_idx]

            column = np.where(column == 3, np.nan, column)

            data[:, col_idx] = column

        elif column_name == "SEX":
            col_idx = headers.index(column_name)
            column = data[:, col_idx]

            column = np.where(column == 2, 0, column)

            data[:, col_idx] = column

        elif column_name in ["ALCDAY5", "EXERHMM1", "STRENGTH"]:
            col_idx = headers.index(column_name)
            column = data[:, col_idx]

            column = np.where(column == 888, 0, column)
            column = np.where(column == 777, np.nan, column)
            column = np.where(column == 999, np.nan, column)

            data[:, col_idx] = column

        elif column_name in ["AVEDRNK2", "INCOME2", "LASTSMK2", "MAXDRNKS", "JOINPAIN"]:
            col_idx = headers.index(column_name)
            column = data[:, col_idx]

            column = np.where(column == 77, np.nan, column)
            column = np.where(column == 99, np.nan, column)

            data[:, col_idx] = column

        elif column_name in [
            "DRNK3GE5",
            "PHYSHLTH",
            "MENTHLTH",
            "POORHLTH",
            "DOCTDIAB",
        ]:
            col_idx = headers.index(column_name)
            column = data[:, col_idx]

            column = np.where(column == 88, 0, column)
            column = np.where(column == 77, np.nan, column)
            column = np.where(column == 99, np.nan, column)

            data[:, col_idx] = column

        elif column_name == "_PACAT1":
            col_idx = headers.index(column_name)
            column = data[:, col_idx]

            column = np.where(column == 9, np.nan, column)

            data[:, col_idx] = column

        elif column_name in ["FC60_", "_DRNKWEK", "MAXVO2_", "STRFREQ_"]:
            col_idx = headers.index(column_name)
            column = data[:, col_idx]

            column = np.where(column == 88, 0, column)
            column = np.where(column == 77, np.nan, column)
            column = np.where(column == 99900, np.nan, column)
            column = np.where(column == 999, np.nan, column)

            data[:, col_idx] = column

        elif column_name in [
            "EDUCA",
            "_CHLDCNT",
            "_EDUCAG",
            "SMOKER3",
            "_INCOMG",
            "PAMISS1_",
        ]:
            col_idx = headers.index(column_name)
            column = data[:, col_idx]

            column = np.where(column == 9, np.nan, column)

            data[:, col_idx] = column

        elif column_name in ["_FRUITEX", "_VEGETEX"]:
            col_idx = headers.index(column_name)
            column = data[:, col_idx]

            column = np.where(column == 2, np.nan, column)

            data[:, col_idx] = column

        elif column_name == "_AGEG5YR":
            col_idx = headers.index(column_name)
            column = data[:, col_idx]

            column = np.where(column == 14, np.nan, column)

            data[:, col_idx] = column

        elif column_name == "DROCDY3_":
            col_idx = headers.index(column_name)
            column = data[:, col_idx]

            column = np.where(column == 900, np.nan, column)

            data[:, col_idx] = column

        elif column_name == "MARITAL":
            col_idx = headers.index(column_name)
            column = data[:, col_idx]

            column = np.where(column != 1, 0, column)

            data[:, col_idx] = column

        elif column_name in ["WEIGHT2", "HEIGHT3"]:
            col_idx = headers.index(column_name)
            column = data[:, col_idx]

            column = np.where(column == 777, np.nan, column)
            column = np.where(column == 7777, np.nan, column)
            column = np.where((column >= 9000) & (column // 1000 == 9), np.nan, column)

            data[:, col_idx] = column

        elif column_name in [
            "FRUITJU1",
            "FRUIT1",
            "FVBEANS",
            "FVGREEN",
            "FVORANG",
            "VEGETAB1",
        ]:
            col_idx = headers.index(column_name)
            column = data[:, col_idx]

            column = np.where(column >= 300, 1, 0)

            data[:, col_idx] = column

        elif column_name == "BLDSUGAR":
            col_idx = headers.index(column_name)
            column = data[:, col_idx]

            column = np.where(column == 888, 0, column)
            column = np.where(column == 777, np.nan, column)
            column = np.where(column == 999, np.nan, column)

            column_str = column.astype(str)
            # Remove decimal points by casting to float first, then to int for `zfill`
            column_str = np.char.replace(column_str, ".0", "")
            mask_1 = np.char.startswith(column_str, "1")
            column[mask_1] = np.char.zfill(column_str[mask_1], 3).astype(int) % 100
            mask_2 = np.char.startswith(column_str, "2")
            column[mask_2] = (
                np.char.zfill(column_str[mask_2], 3).astype(int) % 100
            ) / 7
            mask_3 = np.char.startswith(column_str, "3")
            column[mask_3] = (
                np.char.zfill(column_str[mask_3], 3).astype(int) % 100
            ) / 30.4
            mask_4 = np.char.startswith(column_str, "4")
            column[mask_4] = (
                np.char.zfill(column_str[mask_4], 3).astype(int) % 100
            ) / 365.25

            data[:, col_idx] = column

        elif column_name == "EXERHMM1":
            col_idx = headers.index(column_name)
            column = data[:, col_idx]

            column = np.where(column == 888, 0, column)
            column = np.where(column == 777, np.nan, column)
            column = np.where(column == 999, np.nan, column)

            column_str = column.astype(str)
            greater_than_100_mask = column.astype(float) >= 100
            column_str_gt_100 = column_str[greater_than_100_mask]

            column[greater_than_100_mask] = (
                np.char.zfill(column_str_gt_100, 3).astype(int) // 100 * 60
                + np.char.zfill(column_str_gt_100, 3).astype(int) % 100
            )

            data[:, col_idx] = column

    return data


# To check that we didn't neglect any modification, we write a function that looks for the columns that haven't been accessed during the data modification.
def get_unmodified_headers(old_data, new_data, headers):
    """
    Compares the old dataset with the new dataset and returns the headers
    that have not been modified.

    Parameters:
    old_data (np.ndarray): The original NumPy array containing the old data.
    new_data (np.ndarray): The modified NumPy array containing the new data.
    headers (list of str): List of all column names in the dataset.

    Returns:
    list: A list of headers that have not been modified.
    """
    # Check the types of old_data and new_data
    if not isinstance(old_data, np.ndarray):
        raise ValueError("old_data must be a numpy ndarray")
    if not isinstance(new_data, np.ndarray):
        raise ValueError("new_data must be a numpy ndarray")

    if old_data.shape[1] != new_data.shape[1]:
        raise ValueError("old_data and new_data must have the same number of columns")

    unmodified_headers = []

    for i, column_name in enumerate(headers):
        # Check if the column in the old data is the same as in the new data
        if np.array_equal(old_data[:, i], new_data[:, i]):
            unmodified_headers.append(column_name)

    return unmodified_headers


# We can notice that there are some specific features that have still not been adressed: those relating to "Race". For them, the scalar vales
# have no real order, so what we should do instead is "One-hot encoding" in order to better reflect the categorical nature of those features:
def one_hot_encode(data, headers, headers_to_encode):
    """
    One-hot encodes specified columns in the dataset without modifying the original data.

    Parameters:
    data (np.ndarray): The input NumPy array containing the data.
    headers (list of str): List of all column names in the dataset.
    headers_to_encode (list of str): List of column names to one-hot encode.

    Returns:
    np.ndarray: The new dataset with one-hot encoded columns.
    list of str: The updated headers list with new column names.
    """
    # Create a copy of the data to avoid modifying the original dataset
    data_copy = np.copy(data)

    # List to store new columns and headers
    new_columns = []
    new_headers = []

    if data_copy.shape[1] != len(headers):
        raise ValueError(
            "The number of headers does not match the number of columns in the data."
        )

    # Iterate through all columns
    for column_name in headers:
        if column_name in headers_to_encode:
            # Get the index of the column
            col_idx = headers.index(column_name)

            # Get the sorted unique values of the column
            unique_vals = unique_values(data_copy, column_name, headers)

            # Create a one-hot column for each unique value
            for val in unique_vals:
                one_hot_column = np.where(data_copy[:, col_idx] == val, 1, 0)
                new_columns.append(one_hot_column)
                new_headers.append(f"{column_name}_{int(val)}")
        else:
            # If not one-hot encoding, add the original column
            col_idx = headers.index(column_name)
            new_columns.append(data_copy[:, col_idx])
            new_headers.append(column_name)

    # Stack all the new columns horizontally
    data_modified = np.column_stack(new_columns)

    return data_modified, new_headers


# The last step of this "data modification" step was to get a subsample of the modified training set, and
# to visually inspect if there was any problem/detail we forgot:
def save_csv_with_headers(data, headers, file_name="data_with_headers.csv"):
    """
    Saves a dataset as a CSV file with headers as the first row, overwriting if the file already exists.

    Parameters:
    data (np.ndarray): The dataset to save, of shape (num_samples, num_features).
    headers (list of str): List of headers for the columns.
    file_name (str): The name of the file to save. Default is "data_with_headers.csv".

    Returns:
    str: The path to the saved file.
    """
    # Check that the number of headers matches the number of columns in the data
    if len(headers) != data.shape[1]:
        raise ValueError("Number of headers must match the number of columns in data.")

    # Define header as a single comma-separated string
    header_str = ",".join(headers)

    # Open file in write mode and save data with headers
    with open(file_name, "w") as f:
        np.savetxt(f, data, delimiter=",", header=header_str, comments="", fmt="%s")

    return file_name


#### Standardization : only standardize the data that is non binary


def identify_feature_types(X):
    continuous_indices = []
    binary_indices = []

    for i in range(X.shape[1]):
        unique_values = np.unique(X[:, i])
        if np.issubdtype(X[:, i].dtype, np.number):
            if len(unique_values) > 2:
                continuous_indices.append(i)  # Continuous feature
            elif len(unique_values) == 2:
                binary_indices.append(i)  # Binary feature

    return continuous_indices, binary_indices


def standardize_std(X, test, continuous_indices):
    # Calculate the mean and standard deviation for specified continuous features, ignoring NaNs
    mean = np.nanmean(X[:, continuous_indices], axis=0)
    std_dev = np.nanstd(X[:, continuous_indices], axis=0)

    # Standardize only the continuous features, preserving NaNs
    X_standardized = X.copy()
    test_standardized = test.copy()

    # Standardize X
    X_standardized[:, continuous_indices] = (X[:, continuous_indices] - mean) / std_dev
    X_standardized[:, continuous_indices][np.isnan(X[:, continuous_indices])] = np.nan

    # Standardize test
    test_standardized[:, continuous_indices] = (
        test[:, continuous_indices] - mean
    ) / std_dev
    test_standardized[:, continuous_indices][
        np.isnan(test[:, continuous_indices])
    ] = np.nan

    return X_standardized, test_standardized


### Splitting the data


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing. If ratio times the number of samples is not round
    you can use np.floor. Also check the documentation for np.random.permutation,
    it could be useful.

    Args:
        x: numpy array of shape (N,), N is the number of samples.
        y: numpy array of shape (N,).
        ratio: scalar in [0,1]
        seed: integer.

    Returns:
        x_tr: numpy array containing the train data.
        x_te: numpy array containing the test data.
        y_tr: numpy array containing the train labels.
        y_te: numpy array containing the test labels.
    """

    # set seed
    np.random.seed(seed)

    indices = np.random.permutation(x.shape[0])
    x_shuffled = x[indices]
    y_shuffled = y[indices]

    first_index_te = int(np.floor(x.shape[0] * ratio))
    x_tr = x_shuffled[:first_index_te]
    x_te = x_shuffled[first_index_te:]
    y_tr = y_shuffled[:first_index_te]
    y_te = y_shuffled[first_index_te:]
    return x_tr, x_te, y_tr, y_te


def data_preprocess(
    x_train,
    y_train,
    x_test,
    headers_train,
    model_labels,
    ratio_miss=10,
    ratio_train=0.8,
    standardization=False,
):
    """
    Preprocesses the training data by removing the column that we judged unintersting, and with too high missing data,
    one hot encoding some column, and replacing the NaN with the possibility of using one of three methods.

    Parameters:
    -----------
    x_train : np.ndarray
        The training data array containing raw data with potential missing values and irrelevant columns.

    headers_train : list
        The list of column headers corresponding to the data in `x_train`.

    ratio : int, optional (default=10)
        Threshold ratio to determine which columns to remove based on the proportion of missing values.

    zero : bool, optional (default=None)
        If True, replaces NaN values with zero in columns with missing data.

    mean : bool, optional (default=None)
        If True, replaces NaN values with the mean of non-NaN values in each column.

    mode : bool, optional (default=None)
        If True, replaces NaN values with the mode (most frequent value) of non-NaN values in each column.

    Returns:
    --------
    data_filled : np.ndarray
        The processed data with NaN values handled and columns removed or encoded as specified.

    remaining_headers : list
        The list of headers corresponding to the columns in the processed data.
    """
    if "Id" in headers_train:
        headers_train.remove("Id")
    print(
        f"See the different shapes : x_train {x_train.shape}, x_test {x_test.shape}, y_train {y_train.shape}, headers_train: {len(headers_train)}"
    )

    columns_to_keep = [
        "AGE",
        "SEX",
        "_RACE",
        "_HISPANC",
        "GENHLTH",
        "PHYSHLTH",
        "MENTHLTH",
        "POORHLTH",
        "BPHIGH4",
        "BLOODCHO",
        "CHOLCHK",
        "TOLDHI2",
        "CVDINFR4",
        "CVDCRHD4",
        "CVDSTRK3",
        "ASTHMA3",
        "DIABETE3",
        "DIABAGE2",
        "DIABEDU",
        "BLDSUGAR",
        "INSULIN",
        "PDIABTST",
        "PREDIAB1",
        "DOCTDIAB",
        "SMOKE100",
        "SMOKDAY2",
        "_SMOKER3",
        "STOPSMK2",
        "ALCDAY5",
        "AVEDRNK2",
        "DRNK3GE5",
        "EXERANY2",
        "_FRUTSUM",
        "METVL11",
        "FC60_",
        "MINAC11",
        "WEIGHT2",
        "_BMI5",
        "HAVARTH3",
        "BPMEDS",
        "EXERHMM1",
    ]

    columns_to_remove = [
        "Id",
        "FMONTH",
        "IDATE",
        "IMONTH",
        "IDAY",
        "IYEAR",
        "DISPCODE",
        "SEQNO",
        "_STATE",
        "_PSU",
        "_STSTR",
        "HHADULT",
        "CPDEMO1",
        "EMPLOY1",
        "CHILDREN",
        "INTERNET",
        "SEATBELT",
        "IMFVPLAC",
        "QSTVER",
        "MSCODE",
        "_LLCPWT",
        "_STRWT",
        "_RAWRAKE",
        "_WT2RAKE",
        "CLLCPWT",
        "DUALCOR",
        "WTKG3",
    ]

    sliced_x_train, sliced_features, sliced_x_test = remove_high_missing_columns(
        x_train, x_test, 10, headers_train, columns_to_keep, columns_to_remove
    )

    modified_data = replace_specific_values(sliced_x_train, sliced_features)
    modified_test_points = replace_specific_values(sliced_x_test, sliced_features)

    headers_to_encode = ["_PRACE1", "_MRACE1", "_RACE", "_RACEGR3", "_RACE_G1"]
    encoded_data, encoded_headers = one_hot_encode(
        modified_data, sliced_features, headers_to_encode
    )
    encoded_test_points, encoded_headers_bis = one_hot_encode(
        modified_test_points, sliced_features, headers_to_encode
    )

    filtered_data, remaining_headers, filtered_test_points = (
        remove_high_missing_columns2(
            encoded_data, encoded_test_points, ratio=ratio_miss, headers=encoded_headers
        )
    )

    # Standardization
    if standardization:
        ordinal_indices, binary_indices = identify_feature_types(filtered_data)
        std_data, std_test_points = standardize_std(
            filtered_data, filtered_test_points, ordinal_indices
        )

    # Imputation of the missing binary values by the Mode
    data_filled = replace_nan_in_binary(std_data, binary_indices)
    test_points_filled = replace_nan_in_binary(std_test_points, binary_indices)

    # Imputation of the missing ordinal values by 0 (they have been standardize with mean 0)
    data_filled = replace_nan_in_ordinal(data_filled, ordinal_indices)
    test_points_filled = replace_nan_in_ordinal(test_points_filled, ordinal_indices)

    ### Check missing values in training set:
    train_columns_with_missing_data = find_missing_values(data_filled)
    train_has_nan = np.isnan(data_filled).any()

    print(
        f"After preprocessing (train) : column with missing values {train_columns_with_missing_data}, are there NaN ? {train_has_nan}"
    )

    ### Check missing values in test set:
    test_columns_with_missing_data = find_missing_values(test_points_filled)
    test_has_nan = np.isnan(test_points_filled).any()

    print(
        f"After preprocessing (test) : column with missing values {test_columns_with_missing_data}, are there NaN ? {test_has_nan}"
    )

    # Modifications in X and y according to if our ML model is meant for labels {-1,1} or {0,1}
    if model_labels == {-1, 1}:
        assert np.all(
            (y_train == -1) | (y_train == 1)
        ), "The array contains values other than -1 or 1."
        data_filled = replace_zero_with_minus_one_in_binary(data_filled, binary_indices)
    elif model_labels == {0, 1}:
        y_train[y_train == -1] = 0
        assert np.all(
            (y_train == 0) | (y_train == 1)
        ), "The array contains values other than 0 or 1."
    else:
        raise ValueError("model_labels should be either {-1, 1} or {0, 1}")

    # if you want to see what the data looks like
    save_csv_with_headers(data_filled[:100, :], remaining_headers)

    # Splitting the data between training and validation
    x_tr, x_val, y_tr, y_val = split_data(data_filled, y_train, ratio_train, seed=1)
    print(
        f"See the different shapes : x_tr {x_tr.shape}, x_val {x_val.shape}, y_tr {y_tr.shape}, y_te{y_val.shape}, x_test_formatted{test_points_filled.shape}"
    )

    return x_tr, x_val, y_tr, y_val, data_filled, test_points_filled, remaining_headers
