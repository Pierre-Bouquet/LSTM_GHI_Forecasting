import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def max_error(y_ref, y_test):
    """
    Calculate the maximum error between two arrays.

    Parameters:
        y_ref (array-like): Reference array.
        y_test (array-like): Test array.

    Returns:
        float: Maximum error between the two arrays.
    """
    y_test = np.array(y_test)
    y_ref = np.array(y_ref)
    y_mean = np.mean(y_ref)
    return np.max(np.divide(np.abs(y_test-y_ref), y_mean))


def median_error(y_ref, y_test):
    """
    Calculate the median error between two arrays.

    Parameters:
        y_ref (array-like): Reference array.
        y_test (array-like): Test array.

    Returns:
        float: Median error between the two arrays.
    """
    y_test = np.array(y_test)
    y_ref = np.array(y_ref)
    y_mean = np.mean(y_ref)
    return np.median(np.divide(np.abs(y_test-y_ref), y_mean))


def error(y_ref, y_test):
    """
    Calculate the error between two arrays.

    Parameters:
        y_ref (array-like): Reference array.
        y_test (array-like): Test array.

    Returns:
        array: Error between the two arrays.
    """
    y_test = np.array(y_test)
    y_ref = np.array(y_ref)
    y_mean = np.mean(y_ref)
    return np.divide(np.abs(y_test-y_ref), y_mean)


def MBE(y_ref, y_test):
    """
    Calculate the Mean Bias Error (MBE) between two arrays.

    Parameters:
        y_ref (array-like): Reference array.
        y_test (array-like): Test array.

    Returns:
        float: Mean Bias Error (MBE).
    """
    mbe = np.sum(y_test-y_ref)/len(y_ref)
    return mbe


def invTransform(scaler, data, colName, colNames):
    """
    Inverse transform a column in a DataFrame using a scaler.

    Parameters:
        scaler (object): Scaler object with inverse_transform method.
        data (array-like): Data to transform.
        colName (str): Name of the column to transform.
        colNames (list): List of column names in the DataFrame.

    Returns:
        array: Inverse transformed column data.
    """
    dummy = pd.DataFrame(np.zeros((len(data), len(colNames))), columns=colNames)
    dummy[colName] = data
    dummy = pd.DataFrame(scaler.inverse_transform(dummy), columns=colNames)
    return dummy[colName].values


def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate the Mean Absolute Percentage Error (MAPE) between two arrays.

    Parameters:
        y_true (array-like): True values.
        y_pred (array-like): Predicted values.

    Returns:
        float: Mean Absolute Percentage Error (MAPE).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Check for zero division (when y_true is zero, MAPE is undefined)
    mask = y_true != 0
    y_true_nonzero = y_true[mask]
    y_pred_nonzero = y_pred[mask]
    # Calculate the percentage error for each data point
    percentage_errors = np.abs((y_true_nonzero - y_pred_nonzero) / y_true_nonzero)
    # Calculate the mean absolute percentage error
    mape = np.mean(percentage_errors) * 100.0

    return mape

# Define directories and constants
data_dir = './GHI_dataset/cleaned_sampled_data/'
reportPersistence_dir = './reports/persistence_model/'
samplingFrequencies_eng = ["15_minutes", "30_minutes", "45_minutes", "1_hour", "2_hours",
                            "4_hours", "6_hours", "12_hours", "24_hours", "48_hours",
                            "72_hours", "4_days", "5_days", "6_days", "7_days"]

# Making folder for reports if it doesn't exist
if not os.path.exists(reportPersistence_dir):
    os.makedirs(reportPersistence_dir)

# Initialize DataFrame to store results
persistenceResult_df = pd.DataFrame(columns=['samplingFrequency', 'accuracy', 'MAPE', 'RMSE',
                                             'nRMSE', 'MAE', 'nMAE', 'R2', 'MBE', 'maxNormError'])

# Loop through different sampling frequencies
for k, samplingFrequency in enumerate(samplingFrequencies_eng):

    # Importing the data
    file_path = os.path.join(data_dir, f'GHI_sampled_{samplingFrequency}.csv')
    if os.path.exists(file_path):
        data_df = pd.read_csv(file_path, index_col=0)

        # Split the data into train, validation, and test sets
        y = data_df['GHI'].values
        y_train, y_test = train_test_split(y, test_size=0.2, shuffle=False)
        y_validation, y_test = train_test_split(y_test, test_size=0.5, shuffle=False)

        # Testing the same 10% of the dataset we're testing the LSTM on
        pred = np.roll(y_test, shift=1)[1:]
        test = y_test[1:]

        # Persistence Performance evaluation
        persistenceResult_df.loc[k, 'samplingFrequency'] = samplingFrequency
        persistenceResult_df.loc[k, 'accuracy'] = np.mean(np.abs(pred - test))  # Accuracy
        persistenceResult_df.loc[k, 'MAPE'] = mean_absolute_percentage_error(test, pred)  # MAPE
        persistenceResult_df.loc[k, 'RMSE'] = np.sqrt(mean_squared_error(test, pred))  # RMSE
        persistenceResult_df.loc[k, 'nRMSE'] = np.sqrt(mean_squared_error(test, pred)) / np.mean(test)  # nRMSE
        persistenceResult_df.loc[k, 'MAE'] = mean_absolute_error(test, pred)  # MAE
        persistenceResult_df.loc[k, 'nMAE'] = mean_absolute_error(test, pred) / np.mean(test)  # nMAE
        persistenceResult_df.loc[k, 'R2'] = r2_score(test, pred)  # R2
        persistenceResult_df.loc[k, 'MBE'] = MBE(test, pred)  # MBE
        persistenceResult_df.loc[k, 'maxNormError'] = max_error(test, pred)  # Maximum normalized error

    else:
        print(f"File not found: {file_path}")

csv_file_path = os.path.join(reportPersistence_dir, 'persistence_results.csv')
persistenceResult_df.to_csv(csv_file_path, index=False)

print(f"Persistence performance report exported to {csv_file_path}")