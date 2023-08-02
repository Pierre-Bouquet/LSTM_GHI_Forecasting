# LSTM_GHI_Forecasting
The data and code in this folder generate all the results present in the paper:

_AI-Based Forecasting for Optimized Solar Energy Management and Smart Grid Efficiency_ [1]

By Pierre Bouquet, Ilya Jackson, Mostafa Nick, Amin Kaboli

The raw data was provided by the DESL laboratory at the Swiss Federal Institute of Technology (EPFL).
The code was provided by Pierre Bouquet.

# Description of the folders and files

- `.\GHI_dataset\` - Contains all the data

    * `raw_data` - Raw GHI measurements from 2016 to 2021 with a sampling frequency of 10s in .csv format.
    * `cleaned_sampled_data` - Cleaned and sampled GHI measurements from 2016 to 2021 in .csv format. The data is obtained after running `GHI_preprocessing.ipynb`.

- `.\code\` - Contains all the code

    * `preprocessing.ipynb` - It is the preprocessing pipeline, the code cleans the raw GHI data to the cleaned and samples it. It is built as a notebook to allow tweeking in the pipeline.
    * `persistence_model.ipynb` - It is the persistence model.
    * `LSTM_feature_grid_search.py` - It the feature grid search algorithm detailed in the paper.

- `.\reports` - Contains the performance from the models

    * `persistence_model` - It contains the performance report of the persistence model.
    * `LSTM_model`
        - `run_file_[...]` -  It contains the n iterations of the LSTM model with a set of parameters.
        - `summary_file_HORIZON` - It contains a summary of the performance of the LSTM model for each set of parameters.


# Dataset description

## GHI Raw Data

GHI data in .csv measured on DESL laboratory roof from 2016 to 2021 with a sampling frequency of 10s. 
Contains Nans and outliers.

## GHI Cleaned and Sampled Data

* Obtained from running preprocessing.ipynb on GHI raw data.

Output .csv  with the following pre-processing performed:
 - Removed NaNs.
 - Removed outliers.
 - Added GHIcs.
 - Removed night measurements.
 - Added Clear Sky index.
 - Added backward finite difference.
 - Added seasonality.
 - Downsampled from 15 minutes to 7 days.