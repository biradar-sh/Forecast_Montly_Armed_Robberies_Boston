
 ### Monthly Armed Robberies in Boston — Time Series Forecasting with ARIMA


## 📌 Project Overview

This is a time series forecasting project that analyzes monthly armed robbery incidents in Boston. Using the ARIMA model and walk-forward validation, this project aims to predict short-term future robbery counts, helping city officials or analysts anticipate high-crime periods and make informed decisions.

---

## 📈 Key Features

- ✅ Real-world dataset on monthly armed robberies  
- ✅ Box-Cox transformation to stabilize variance  
- ✅ Rolling (walk-forward) validation for realistic performance testing  
- ✅ ARIMA(0,1,2) model tuned for non-stationary behavior  
- ✅ Custom prediction evaluation with RMSE and visual comparison  
- ✅ Handles data transformation, forecasting, inverse transformation, and plotting in a modular loop  
- ✅ Visual plots for actual vs predicted crime counts

---

### How It Works
Data Load & Split:
Reads robberies.csv, splits the last 12 months as a validation set.

Transformation:
Applies Box-Cox transformation to make the data more ARIMA-friendly.

Modeling:
Trains an ARIMA(0,1,2) model on rolling windows of historical data.

Forecasting:
Makes 1-step-ahead predictions, then updates history with true values.

Evaluation:
Computes RMSE and visualizes predicted vs actual crime levels.

### 📚 Techniques Used
ARIMA modeling (statsmodels)

Walk-forward validation

Box-Cox transformation and inverse

Time series split & holdout validation

RMSE for accuracy

Matplotlib for visualization
