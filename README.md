# LSTM_GHI_Forecasting
AI-Based Forecasting for Optimized Solar Energy Management and Smart Grid Efficiency

## Citation
This repository contains the data and code used to generate the results in the paper:
_AI-Based Forecasting for Optimized Solar Energy Management and Smart Grid Efficiency_ [1]

## Authors
By: Pierre Bouquet, Ilya Jackson, Mostafa Nick, Amin Kaboli

## Description
The LSTM_GHI_Forecasting repository contains the code and data for forecasting Global Horizontal Irradiance (GHI) using LSTM models. The raw GHI data was provided by the DESL laboratory at the Swiss Federal Institute of Technology (EPFL).

## Folder Structure
- `.\GHI_dataset\` - Contains all the data
    - `raw_data` - Raw GHI measurements from 2016 to 2021 with a sampling frequency of 10s in .csv format.
    - `cleaned_sampled_data` - Cleaned and sampled GHI measurements from 2016 to 2021 in .csv format. The data is obtained after running `GHI_preprocessing.ipynb`.

- `.\code\` - Contains all the code
    - `preprocessing.ipynb` - Preprocessing pipeline for cleaning and sampling the raw GHI data.
    - `persistence_model.ipynb` - Implementation of the persistence model.
    - `LSTM_feature_grid_search.py` - Feature grid search algorithm detailed in the paper.

- `.\reports` - Contains the performance from the models
    - `persistence_model` - Performance report of the persistence model.
    - `LSTM_model`
        - `run_file_[...]` - Detailed performance of the LSTM model with a set of parameters.
        - `summary_file_HORIZON` - Summary of the performance of the LSTM model for each horizon.

## Dataset Description

### GHI Raw Data
GHI data in .csv measured on DESL laboratory roof from 2016 to 2021 with a sampling frequency of 10s. Contains Nans and outliers.

### GHI Cleaned and Sampled Data
Obtained from running preprocessing.ipynb on GHI raw data.
Output .csv with the following pre-processing performed:
 - Removed NaNs.
 - Removed outliers.
 - Added GHIcs.
 - Removed night measurements.
 - Added Clear Sky index.
 - Added backward finite difference.
 - Added seasonality.
 - Downsampled from 15 minutes to 7 days.

## References

[1] _Under review_