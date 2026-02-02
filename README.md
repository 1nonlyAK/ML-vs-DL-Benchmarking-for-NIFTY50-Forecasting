# ML vs DL Benchmarking for NIFTY 50 Forecasting

## Overview
This project presents a large-scale benchmarking study comparing **Machine Learning (ML)** and **Deep Learning (DL)** models for **financial time-series forecasting** using historical **NIFTY 50 stock price data**.

Instead of assuming deep learning superiority, the project evaluates multiple models across different **time windows** and **price targets** to determine which approaches generalize best for structured financial data.

---

## Objectives
- Benchmark ML and DL models for stock price forecasting  
- Analyze the impact of **time window sizes** on prediction accuracy  
- Compare predictability across **Open, High, Low, Close (OHLC)** prices  
- Identify models that generalize best under **time-aware validation**

---

## Dataset
- **Market:** NIFTY 50  
- **Features:** Open, High, Low, Close  
- **Task:** Univariate time-series regression  
- **Input Windows:** 30 to 250 days  

---

## Methodology

### Time-Series Framing
- Converted price sequences into supervised learning datasets using a **sliding window approach**
- Evaluated **8 temporal horizons** to capture short- and mid-term dependencies

### Models Evaluated

#### Machine Learning Models
- Linear Regression, Ridge, Lasso  
- Random Forest, Gradient Boosting  
- Support Vector Regression (SVR)  
- K-Nearest Neighbors (KNN)  
- XGBoost, LightGBM  

#### Deep Learning Models
- Simple RNN  
- LSTM  
- GRU  
- Bidirectional LSTM  

---

## Evaluation Strategy
- Applied **time-aware train–test splitting** to prevent data leakage  
- Evaluated models using **MAE** and **RMSE**  
- Ranked models based on **test error performance**

---

## Key Findings
- Classical ML models consistently **outperformed deep learning models**
- **30–120 day windows** dominated top-performing configurations
- The **High price** was the most predictable target among OHLC features
- Deep learning models did not appear among the top-ranked configurations

These results highlight the importance of **empirical benchmarking over model complexity** in financial forecasting tasks.

---

## Results & Analysis
- Compared **train vs test MAE** for top-performing models  
- Analyzed frequency of optimal **time windows**, **target variables**, and **model types**  
- Visualized trends using Matplotlib for clear model comparison

---

## Reproducibility
- All trained models and metrics are saved using **Joblib**
- Experiment results are exported as CSV files
- Models can be reloaded for inference on unseen sequences

---

## Future Work
- Hyperparameter tuning for top-performing ML models  
- Finer-grained time window analysis (5–150 days)  
- Ensemble modeling using best configurations  
- Extension to multivariate financial indicators

---

## Technologies Used
- Python  
- NumPy, Pandas  
- Scikit-learn  
- TensorFlow / Keras  
- XGBoost, LightGBM  
- Matplotlib  

---

## Project Structure
