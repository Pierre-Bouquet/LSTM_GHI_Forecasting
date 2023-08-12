import os
import pandas as pd
import numpy as np 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error, r2_score
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping

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

def calculate_95_CI(data, mean):
    """
    Calculate the 95% confidence interval as a percentage of the mean.

    Parameters:
        data (array-like): 1-dimensional array representing the distribution.
        mean (float): The mean value of the distribution.

    Returns:
        float: The 95% confidence interval as a percentage of the mean.
    """
    # Calculate the lower and upper bounds of the 95% confidence interval
    lower_bound = np.percentile(data, 2.5)
    upper_bound = np.percentile(data, 97.5)
    
    # Calculate the width of the confidence interval as a percentage of the mean
    ci_width_percent = (upper_bound - lower_bound) / mean
    
    return ci_width_percent

def summary_report(lstmRunResult_df):
    """
    Generate a summary report of LSTM run results.

    Parameters:
        lstmRunResult_df (DataFrame): DataFrame containing the LSTM run results.

    Returns:
        list: A list containing various summary statistics for the LSTM run results.
    """
    nbEpoch_mean = lstmRunResult_df['nbEpoch'].mean()

    # Calculate the mean and confidence intervals for MAPE
    MAPE_mean = lstmRunResult_df['MAPE'].mean() 
    MAPE_min = lstmRunResult_df['MAPE'].min()
    MAPE_med = lstmRunResult_df['MAPE'].median()
    MAPE_max = lstmRunResult_df['MAPE'].max()
    MAPE_std = lstmRunResult_df['MAPE'].std()
    MAPE_CI = calculate_95_CI(lstmRunResult_df['MAPE'], MAPE_mean)                          

    # Calculate the mean and confidence intervals for RMSE
    RMSE_mean = lstmRunResult_df['RMSE'].mean() 
    RMSE_min = lstmRunResult_df['RMSE'].min()
    RMSE_med = lstmRunResult_df['RMSE'].median()
    RMSE_max = lstmRunResult_df['RMSE'].max()
    RMSE_std = lstmRunResult_df['RMSE'].std()
    RMSE_CI = calculate_95_CI(lstmRunResult_df['RMSE'], RMSE_mean)
    nRMSE_mean = lstmRunResult_df['nRMSE'].mean()

    # Calculate the mean and confidence intervals for MAE
    MAE_mean = lstmRunResult_df['MAE'].mean() 
    MAE_min = lstmRunResult_df['MAE'].min()
    MAE_med = lstmRunResult_df['MAE'].median()
    MAE_max = lstmRunResult_df['MAE'].max()
    MAE_std = lstmRunResult_df['MAE'].std()
    MAE_CI = calculate_95_CI(lstmRunResult_df['MAE'], MAE_mean)
    nMAE_mean = lstmRunResult_df['nMAE'].mean()

    # Calculate the mean and confidence intervals for R2
    R2_mean = lstmRunResult_df['R2'].mean() 
    R2_min = lstmRunResult_df['R2'].min()
    R2_med = lstmRunResult_df['R2'].median()
    R2_max = lstmRunResult_df['R2'].max()
    R2_std = lstmRunResult_df['R2'].std()
    R2_CI = calculate_95_CI(lstmRunResult_df['R2'], R2_mean)

    # Calculate the mean and confidence intervals for MBE
    MBE_mean = lstmRunResult_df['MBE'].mean() 
    MBE_min = lstmRunResult_df['MBE'].min()
    MBE_med = lstmRunResult_df['MBE'].median()
    MBE_max = lstmRunResult_df['MBE'].max()
    MBE_std = lstmRunResult_df['MBE'].std()
    MBE_CI = calculate_95_CI(lstmRunResult_df['MBE'], MBE_mean)

    # Return a list of summary statistics for the LSTM run results
    return [samplingFrequency, polynomialAug, featureColumns_eng_list[k], nPrevSteps, nbIter, nbEpoch_mean, 
            MAPE_min, MAPE_med, MAPE_max, MAPE_std, MAPE_CI, MAPE_mean,
            RMSE_min, RMSE_med, RMSE_max, RMSE_std, RMSE_CI, RMSE_mean, nRMSE_mean,
            MAE_min, MAE_med, MAE_max, MAE_std, MAE_CI, MAE_mean, nMAE_mean,
            R2_min, R2_med, R2_max, R2_std, R2_CI, R2_mean,
            MBE_min, MBE_med, MBE_max, MBE_std, MBE_CI, MBE_mean]


nbIter = 10
nbEpoch = 50

nPrevSteps_list = [5, 10, 15, 20, 25]
polynomialAugm_list = [1, 2, 3, 4, 5]

samplingFrequencies_eng = ["7_days", "6_days", "5_days", "4_days", "72_hours", 
                           "48_hours", "24_hours", "12_hours", "6_hours", "4_hours",
                           "2_hours", "1_hour", "45_minutes", "30_minutes", "15_minutes"]

basicFeatures = ['GHI', 'GHIcs', 'k']
seasonalFeatures = ['month', 'day', 'hour']
d1Features = ['GHI_d1', 'GHIcs_d1', 'k_d1']
d2Features = ['GHI_d2', 'GHIcs_d2', 'k_d2']
d3Features = ['GHI_d3', 'GHIcs_d3', 'k_d3']

featureColumns_list = [basicFeatures,
                       basicFeatures + d1Features,
                       basicFeatures + d1Features + d2Features,
                       basicFeatures + d1Features + d2Features + d3Features,
                       basicFeatures + seasonalFeatures,
                       basicFeatures + seasonalFeatures + d1Features,
                       basicFeatures + seasonalFeatures + d1Features + d2Features,
                       basicFeatures + seasonalFeatures + d1Features + d2Features + d3Features]

featureColumns_eng_list = ['GHI', 'GHI_d1', 'GHI_d1_d2', 'GHI_d1_d2_d3', 
                           'GHI_season', 'GHI_season_d1', 'GHI_season_d1_d2', 'GHI_season_d1_d2_d3']

# Set data directory
data_dir = './GHI_dataset/cleaned_sampled_data/'

# Set report directory
reportLSTM_dir = './reports/LSTM_models/'

# Loop through different sampling frequencies
for samplingFrequency in samplingFrequencies_eng:

    # Create dataframe to store summary results
    lstmSummaryResult_df = pd.DataFrame(columns = [
    'samplingFrequency', 'polynomialAugmentation', 'features', 'history', 
    'nbIter', 'nbEpoch',  
    'MAPE_min', 'MAPE_med', 'MAPE_max', 'MAPE_std', 'MAPE_CI', 'MAPE_mean',  
    'RMSE_min', 'RMSE_med', 'RMSE_max', 'RMSE_std', 'RMSE_CI', 'RMSE_mean', 'nRMSE',
    'MAE_min', 'MAE_med', 'MAE_max', 'MAE_std', 'MAE_CI', 'MAE_mean', 'nMAE',
    'R2_min', 'R2_med', 'R2_max', 'R2_std', 'R2_CI', 'R2_mean',
    'MBE_min', 'MBE_med', 'MBE_max', 'MBE_std', 'MBE_CI', 'MBE_mean'
    ])
    
    # Make report directory for this sampling frequency
    report_dir = os.path.join(reportLSTM_dir, samplingFrequency)
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    
    # Import preprocessed data for this sampling frequency 
    file_path = os.path.join(data_dir, f'GHI_sampled_{samplingFrequency}.csv')
    if os.path.exists(file_path):
        data_df = pd.read_csv(file_path, index_col=0)
    else:
        print(f"File not found: {file_path}")
    
    # Set batch size based on RAM
    batch_size = min(256, int(data_df.shape[0]))
    
    # Loop through different feature sets
    for k, featureColumns in enumerate(featureColumns_list):

        # Loop through polynomial augmentation orders
        for polynomialAug in polynomialAugm_list:

            # Loop through history lengths
            for nPrevSteps in nPrevSteps_list:
      
                # Extract features
                nFeatures = len(featureColumns)
                X = data_df.loc[:, featureColumns]
                
                # Polynomial augmentation
                poly = PolynomialFeatures(polynomialAug)
                X_poly = poly.fit_transform(X)
                colNames = poly.get_feature_names_out(X.columns)
                
                # Min-max scaling
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaler.fit(X_poly)
                scaled = scaler.transform(X_poly)
                scaled = pd.DataFrame(scaled, columns = colNames, index = data_df.index)
                scaled = scaled.drop("1", axis=1)
                nFeatureColumns = scaled.shape[1]
                
                # Shift features to create lagged inputs
                train_test_df = pd.DataFrame()
                for i in range(0, nPrevSteps+1, 1):
                    title = scaled.columns + 't(-' + str(i) + ')'
                    temp = scaled.shift(periods = i)
                    temp.columns = title
                    train_test_df = pd.concat([train_test_df, temp], axis=1)
                
                train_test_df = train_test_df.dropna()
        
                # Extract X and y        
                title_0 = scaled.columns + 't(-' + str(0) + ')'
                X = train_test_df.drop(title_0, axis = 1).values
                y = train_test_df['GHIt(-0)'].values
                
                # Train/validation/test split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
                X_validation, X_test, y_validation, y_test = train_test_split(X_test, y_test, test_size=0.5, shuffle=False)
                
                # Reshape X for LSTM 
                X_train_r = X_train.reshape((X_train.shape[0], nPrevSteps, nFeatureColumns))
                X_validation_r = X_validation.reshape((X_validation.shape[0], nPrevSteps, nFeatureColumns))
                X_test_r = X_test.reshape((X_test.shape[0], nPrevSteps, nFeatureColumns))
                
                # Run LSTM nbIter times to compute statistics
                lstmRunResult_df = pd.DataFrame(columns = ['samplingFrequency', 'polynomialAugmentation', 'features', 'history', 'iter',
                                                           'nbEpoch', 'MAPE', 'RMSE', 'nRMSE', 'MAE', 'nMAE','R2', 'MBE'])
                
                for i in range(nbIter):
  
                    # LSTM model
                    es = EarlyStopping(monitor='val_loss', mode='min', verbose = 0, patience=10, restore_best_weights=True)
                    opt = keras.optimizers.Adam(learning_rate=0.001)
                    model = Sequential()
                    model.add(LSTM(300, return_sequences=True, input_shape=(X_train_r.shape[1], X_train_r.shape[2])))
                    model.add(LSTM(300, return_sequences=True, input_shape=(X_train_r.shape[1], X_train_r.shape[2]))) 
                    model.add(LSTM(300, input_shape=(X_train_r.shape[1], X_train_r.shape[2])))
                    model.add(Dense(1))
                    model.compile(loss='mse', optimizer=opt)
                    
                    # Fit model
                    history = model.fit(X_train_r, y_train, epochs=nbEpoch, validation_data=(X_validation_r, y_validation), 
                                        batch_size = batch_size, verbose = 0, callbacks = [es], shuffle=False)
                    
                    # Make predictions
                    y_pred = model.predict(X_test_r)
          
                    # Inverse transform predictions
                    pred = invTransform(scaler, y_pred, 'GHI', colNames)
                    test = invTransform(scaler, y_test, 'GHI', colNames)
          
                    # Evaluate model
                    lstmRunResult_df.loc[i, 'samplingFrequency'] = samplingFrequency
                    lstmRunResult_df.loc[i, 'polynomialAugmentation'] = polynomialAug
                    lstmRunResult_df.loc[i, 'features'] = featureColumns_eng_list[k]  
                    lstmRunResult_df.loc[i, 'history'] = nPrevSteps
                    lstmRunResult_df.loc[i, 'iter'] = i
                    lstmRunResult_df.loc[i, 'nbEpoch'] = es.stopped_epoch
                    lstmRunResult_df.loc[i, 'MAPE'] = mean_absolute_percentage_error(test, pred)
                    lstmRunResult_df.loc[i, 'RMSE'] = np.sqrt(mean_squared_error(test, pred))
                    lstmRunResult_df.loc[i, 'nRMSE'] = np.sqrt(mean_squared_error(test, pred)) / np.mean(test)
                    lstmRunResult_df.loc[i, 'MAE'] = mean_absolute_error(test, pred)
                    lstmRunResult_df.loc[i, 'nMAE'] = mean_absolute_error(test, pred) / np.mean(test) 
                    lstmRunResult_df.loc[i, 'R2'] = r2_score(test, pred)
                    lstmRunResult_df.loc[i, 'MBE'] = MBE(test, pred)

                # Save results for this run
                run_fileName = 'run_file_' + samplingFrequency + '_' + featureColumns_eng_list[k] + '_P' + str(polynomialAug) + '_H' + str(nPrevSteps) + '.csv'
                csv_run_file_path = os.path.join(report_dir, run_fileName)
                lstmRunResult_df.to_csv(csv_run_file_path, index=False)
                
                # Calculate summary statistics for lstmRunResult_df
                lstmSummaryResult_array = summary_report(lstmRunResult_df)
                # Convert lstmSummaryResult_array to a DataFrame
                lstmSummaryResult_row = pd.DataFrame([lstmSummaryResult_array], columns=lstmSummaryResult_df.columns)
                # Append the new row to lstmSummaryResult_df using pd.concat
                lstmSummaryResult_df = pd.concat([lstmSummaryResult_df, lstmSummaryResult_row], ignore_index=True)
                  
                # Save summary results
                summary_fileName = 'summary_file_' + samplingFrequency + '.csv'
                csv_summary_file_path = os.path.join(report_dir, summary_fileName)
                lstmSummaryResult_df.to_csv(csv_summary_file_path, index=False)