# Amazon Stock Price Prediction with ARIMA, SARIMAX, and GARCH

## Overview

This project investigates whether next-day **Amazon (AMZN)** stock returns can be predicted using classical time-series methods and market-based exogenous variables. The analysis combines:

- **Univariate return forecasting** with **ARIMA**
- **Return forecasting with external signals** using **SARIMAX**
- **Volatility forecasting** with **GARCH(1,1)**

The workflow is organized into three notebooks covering data collection and exploratory analysis, model training, and out-of-sample evaluation.

Rather than modeling raw stock prices directly, the project focuses on **daily returns**, which are more suitable for statistical time-series modeling because they are much closer to stationarity.

---

## Project Objectives

The main goals of the project are to:

1. Build a clean and reproducible **AMZN forecasting pipeline** from raw market data.
2. Compare whether **past AMZN returns alone** or **lagged market variables** improve next-day forecasts.
3. Assess whether external signals such as the **S&P 500** and the **VIX** provide incremental predictive value.
4. Evaluate not only return prediction, but also **volatility dynamics** and **price reconstruction at horizon t+1**.
5. Highlight the gap between:
   - the difficulty of predicting **daily returns**, and
   - the relative ease of producing visually accurate **one-step-ahead price forecasts**.

---

## Data

### Sources
The project uses daily financial data downloaded with **Yahoo Finance (`yfinance`)** for:

- **AMZN** — Amazon stock
- **^GSPC** — S&P 500 index
- **^VIX** — CBOE Volatility Index

### Time Period
- **Start date:** 2005-01-01  
- **End date:** 2025-01-01

### Engineered Variables
The notebooks create and use the following features:

- `Price`: adjusted close price for AMZN
- `Return`: daily AMZN return
- `SP500_Return`: daily S&P 500 return
- `VIX_Return`: daily change in VIX
- `SP500_Return_lag1`: lagged S&P 500 return
- `VIX_Return_lag1`: lagged VIX change
- calendar variables such as weekday and month

The final cleaned dataset contains **5,031 observations** and **14 columns** after preprocessing.

---

## Notebook Structure

### 1. `01_data_download_and_eda.ipynb`
This notebook covers data acquisition, cleaning, exploratory analysis, and feature engineering.

Main steps:

- Download AMZN, S&P 500, and VIX data from Yahoo Finance
- Standardize the adjusted close as the working **price** variable
- Compute daily returns
- Explore:
  - price evolution
  - return distributions
  - rolling volatility
  - trading volume behavior
  - AMZN vs S&P 500 rolling correlation
  - VIX regime behavior
  - drawdowns
- Run **ADF stationarity tests**
- Analyze **ACF/PACF** for returns
- Save the final dataset to:
  - `../data/raw/amzn_sp500_vix_clean.csv`

### 2. `02_model_training.ipynb`
This notebook trains the forecasting models and generates rolling one-step-ahead predictions.

Main steps:

- Load the cleaned dataset
- Apply a **chronological train-test split** at **2017-01-01**
- Training set: **2005-01-05 to 2016-12-30** (**3,019 observations**)
- Test set: **2017-01-03 to 2024-12-31** (**2,012 observations**)
- Estimate:
  - **ARIMA** benchmark
  - **SARIMAX** with lagged S&P 500 return
  - **SARIMAX** with lagged VIX change
  - **SARIMAX** with both exogenous variables
  - **GARCH(1,1)** for volatility
- Produce **rolling expanding-window t+1 forecasts**
- Save forecasts to:
  - `../data/forecasts.csv`

### 3. `03_evaluation_and_plots.ipynb`
This notebook evaluates model performance out of sample and visualizes the predictions.

Main steps:

- Load forecasts and test data
- Compare models using:
  - **RMSE**
  - **MAE**
  - **Directional Accuracy**
- Evaluate GARCH against realized volatility
- Visualize forecasted vs actual returns
- Analyze residuals
- Track **rolling RMSE** over time
- Compare performance across different market regimes
- Zoom in on crisis periods
- Reconstruct **one-step-ahead prices** from predicted returns
- Summarize limitations and future improvements

---

## Methodology

## 1. Why returns instead of prices?
Raw stock prices are typically **non-stationary**, while daily returns are much closer to stationary behavior. Since ARIMA-family models assume a stable mean structure, modeling returns is more appropriate than modeling price levels directly.

This is also why the selected models use **d = 0**: the return series was already treated as stationary after transformation.

## 2. Baseline model: ARIMA
The univariate benchmark was selected with `auto_arima` and resulted in:

- **ARIMA(0,0,0)**

This means the AMZN return series behaves approximately like a **constant plus white noise** process, which is itself an important result: past AMZN returns contain very little linear information for predicting next-day AMZN returns.

## 3. Exogenous models: SARIMAX
Three exogenous specifications were tested:

- SARIMAX with **lagged S&P 500 returns**
- SARIMAX with **lagged VIX changes**
- SARIMAX with **both variables**

For all three cases, `auto_arima` selected the same order:

- **SARIMAX(2,0,0)**

This suggests that once exogenous information is included, a parsimonious AR(2) structure provides the best in-sample fit.

## 4. Volatility model: GARCH
A **GARCH(1,1)** model was estimated separately to forecast conditional volatility.

This model addresses a different question from ARIMA/SARIMAX:

- **ARIMA / SARIMAX** model the **conditional mean** of returns
- **GARCH** models the **conditional variance** of returns

The project finds strong volatility persistence, with **alpha + beta = 0.9111**, which is consistent with volatility clustering in financial markets.

## 5. Forecasting design
All predictive models are evaluated using **rolling one-step-ahead forecasts with an expanding window**, which is more realistic than fitting once and forecasting the full test set in a single batch.

---

## Key Results

### Return Forecasting Performance (2017–2024 test set)

| Model | RMSE | MAE | Directional Accuracy (%) |
|---|---:|---:|---:|
| ARIMA | 0.020811 | 0.014581 | 53.51 |
| SARIMAX (SP500) | 0.020794 | 0.014627 | 50.97 |
| SARIMAX (VIX) | 0.020824 | 0.014666 | 48.98 |
| SARIMAX (Both) | 0.020794 | 0.014645 | 50.27 |
| Naive (Random Walk) | 0.029799 | 0.020965 | 51.52 |

### Volatility Forecasting Performance

| Model | RMSE vs realized vol | MAE vs realized vol | Correlation with realized vol |
|---|---:|---:|---:|
| GARCH(1,1) | 0.5977 | 0.5112 | 0.8526 |

### Price Reconstruction at t+1
The one-step-ahead price reconstruction is visually strong, with the top models reaching approximately:

- **Price RMSE:** 2.61 USD
- **Price MAE:** 1.77 USD

This does **not** mean returns are easy to predict. It reflects the fact that the next-day price forecast is anchored on the **known observed price at time t**:

```math
\hat{P}_{t+1} = P_t (1 + \hat{r}_{t+1})
```

Even when predicted returns are close to zero, price forecasts can still look very accurate at horizon t+1.

### Main Findings
- Next-day AMZN returns are extremely difficult to predict.
- The best univariate specification selected by auto_arima is ARIMA(0,0,0), indicating very weak linear predictability in returns.
- Adding exogenous variables improves RMSE only marginally.
- SARIMAX-SP500 and SARIMAX-BOTH slightly outperform ARIMA on RMSE, but the gains are economically very small.
- ARIMA remains the strongest parsimonious benchmark overall, with the lowest MAE and the highest directional accuracy.
- VIX alone is a weaker signal for mean return forecasting than the S&P 500.
- Volatility is much more predictable than returns, as shown by the strong GARCH fit to realized volatility.

### Interpretation

A central takeaway of the project is that return prediction and price prediction should not be interpreted in the same way.

The notebooks show that:
- return forecasts are weak and often close to zero,
- yet one-step-ahead price forecasts can still appear excellent.

This is not a contradiction. It is a consequence of short-horizon price reconstruction being anchored on the known price level at the previous date.

The project therefore supports a classic empirical finance conclusion: **Returns are hard to forecast, but volatility is more predictable than the direction of returns.**

### Limitations

The current framework has several important limitations:

1 - Weak signal in daily returns :  
Daily stock returns contain very little stable linear information.

2 - Linear modeling assumptions :  
ARIMA and SARIMAX may miss nonlinear and regime-dependent effects.

3 - Limited feature set :  
Only a small number of exogenous variables were included.

4 - Regime instability :  
Relationships may change across normal periods, crises, and tightening cycles.

5 - Short-horizon focus :  
The project emphasizes t+1 forecasting, which is statistically useful but limited for broader investment forecasting applications.

### Potential Extensions

Several directions could improve the forecasting framework:

Add richer exogenous variables:
- VIX level and changes
- bond yields / policy rates
- Nasdaq or sector ETF returns
- earnings-event indicators
- sentiment and news-based signals

Test regime-aware models:
- Markov-switching models
- threshold models

Extend volatility modeling:
- EGARCH
- GJR-GARCH
- GARCH-in-Mean

Explore nonlinear methods:
- XGBoost
- Random Forest
- LSTM / GRU
- Transformer-based sequence models

Evaluate longer horizons:
- 5-day forecasts
- 20-day forecasts
- direct multi-horizon price prediction

### Project Structure
```
project/
├── data/
│   ├── raw/
│   │   └── amzn_sp500_vix_clean.csv
│   └── forecasts.csv
├── notebooks/
│   ├── 01_data_download_and_eda.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation_and_plots.ipynb
└── README.md
```

Depending on your local repository layout, the notebooks expect the cleaned dataset under ../data/raw/ and save forecasts to ../data/.

### Installation
pip install pandas numpy matplotlib seaborn yfinance statsmodels scikit-learn pmdarima arch

### How to Run the Project

Run the notebooks in the following order:

- 01_data_download_and_eda.ipynb
Downloads data, performs EDA, engineers features, and saves the cleaned dataset.

- 02_model_training.ipynb
Loads the cleaned file, trains ARIMA/SARIMAX/GARCH models, and saves rolling forecasts.

- 03_evaluation_and_plots.ipynb
Loads the saved forecasts and produces evaluation tables, regime analysis, and plots.

### Technologies Used
- Python
- Pandas / NumPy
- Matplotlib / Seaborn
- yfinance
- statsmodels
- pmdarima
- arch
- scikit-learn
- Jupyter Notebook

### Conclusion

This project provides a clear and well-structured benchmark study for AMZN next-day forecasting using classical time-series methods.

The empirical conclusions are strong and realistic:
- predicting next-day returns is very difficult,
- adding simple exogenous variables helps only slightly,
- and volatility is far more predictable than returns.

As a result, the project is particularly valuable not because it “solves” stock prediction, but because it demonstrates, with a rigorous workflow, the difference between:
- modeling returns,
- reconstructing prices, and
- forecasting risk dynamics.
