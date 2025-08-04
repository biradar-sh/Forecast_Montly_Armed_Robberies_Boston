# Re-run the README generation code after the kernel reset

readme_full_content = """
 
### Monthly Armed Robberies in Boston — Time Series Forecasting with ARIMA

![Boston skyline](https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Boston_skyline_from_Logan_Airport.jpg/1280px-Boston_skyline_from_Logan_Airport.jpg)

---

## 📌 Project Overview

**Forecast Under Fire** is a time series forecasting project that analyzes monthly armed robbery incidents in Boston. Using the ARIMA model and walk-forward validation, this project aims to predict short-term future robbery counts, helping city officials or analysts anticipate high-crime periods and make informed decisions.

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

## 📂 Project Structure


---

## 🚀 Getting Started

### 1. Clone the repository

git clone https://github.com/yourusername/forecast-under-fire.git
cd forecast-under-fire

### 2. Install dependencies

pip install -r requirements.txt

### 3. How It Works
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
