# LSTM_GHI_Forecasting
AI-Based Forecasting for Optimized Solar Energy Management and Smart Grid Efficiency

## Citation
This repository contains the data and code used to generate the results in the paper:
_AI-Based Forecasting for Optimized Solar Energy Management and Smart Grid Efficiency_ [1]

By Pierre Bouquet, Ilya Jackson, Mostafa Nick, Amin Kaboli

The raw data is provided by the DESL laboratory at the Swiss Federal Institute of Technology (EPFL).
The code is provided by Pierre Bouquet.

## Description
The LSTM_GHI_Forecasting repository contains the code and data for forecasting Global Horizontal Irradiance (GHI) using LSTM models.

## Folder Structure
- `.\GHI_dataset\` - Contains all the data
    - `raw_data` - Raw GHI measurements from 2016 to 2021 with a sampling frequency of 10s in .csv format.
    - `cleaned_sampled_data` - Cleaned and sampled GHI measurements from 2016 to 2021 in .csv format. The data is obtained after running `GHI_preprocessing.ipynb`.

- `.\code\` - Contains all the code
    - `preprocessing.ipynb` - Preprocessing pipeline for cleaning and sampling the raw GHI data.
    - `persistence_model.ipynb` - Implementation of the persistence model.
    - `LSTM_feature_grid_search.py` - Feature grid search algorithm detailed in the paper.

- `.\reports\` - Contains the performance reports from the models
    - `persistence_model` - Performance report of the persistence model.
    - `\LSTM_model\HORIZON_\` - Performance reports of the LSTM models for each forecasting horizon.
        - `run_file_HORIZON_FEATURES_POLYNOMIAL_HISTORY.csv` - Detailed performance of the LSTM model for each set of parameters.
        - `summary_file_HORIZON` - Summary of the performance of the LSTM model for each horizon.

- `.\requirements` - Contains mac OS environments associated to each piece of code.

## Dataset Description

### GHI Raw Data
GHI data in .csv measured on DESL laboratory roof from 2016 to 2021 with a sampling frequency of 10s. Contains Nans and outliers.

### GHI Cleaned and Sampled Data
Obtained from running preprocessing.ipynb on GHI raw data.
Output .csv with the following pre-processing performed:
1. Removed NaNs.
2. Removed outliers.
3. Added GHIcs.
4. Removed night measurements.
5. Added Clear Sky index.
6. Added backward finite difference.
7. Added seasonality.
8. Downsampled from 15 minutes to 7 days.

## References

[1] Bouquet, P., Jackson, I., Nick, M., & Amin Kaboli. (2023). AI-based forecasting for optimised solar energy management and smart grid efficiency. *International Journal of Production Research*, 1–22. [https://doi.org/10.1080/00207543.2023.2269565](https://doi.org/10.1080/00207543.2023.2269565) 