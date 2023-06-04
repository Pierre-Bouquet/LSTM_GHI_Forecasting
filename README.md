# LSTM_GHI_Forecasting
AI-Based Forecasting for Optimized Solar Energy Management and Smart Grid Efficiency

## GHI dataset

### GHI Raw Data

GHI data in .csv measured on DESL laboratory roof from 2016 to 2021 with a sampling frequency of 10s. Contains Nans and outliers

### GHI Clean Sampled

Preprocessing jupyter notebook and GHI data in .csv.

Pre-processing performed:
 - Removed NaNs.
 - Removed outliers.
 - Added GHIcs.
 - Removed night measurements.
 - Added Clear Sky index.
 - Added backward finite difference.
 - Added seasonality.
 - Downsampled from 15 minutes to 7 days.

