# Market Regime Detection & Signal Stability Suite

**Quantitative Research | Financial Time-Series | Machine Learning**

## Project Overview

Financial markets are inherently non-stationary; trading strategies that perform well in "bull" regimes often fail during periods of high volatility or sideways movement. This project implements a **Stacked Machine Learning Architecture** designed to detect shifts in market regimes and provide stable signals for algorithmic trading.

By utilising unsupervised learning outputs (HMM/GMM) as feature inputs for advanced classifiers, this system identifies the underlying "state" of equity markets to optimise P&L rather than simple statistical accuracy.

## Key Quantitative Features

### 1. Regime Clustering (HMM & GMM)
* Implementation of **Hidden Markov Models (HMM)** and **Gaussian Mixture Models (GMM)**.
* Classifies market states based on latent volatility clusters and return distributions.
* Filters out "noise" to identify true structural shifts in market behaviour.

### 2. Custom Loss Function: Directional Big Error (DBE)
Standard loss functions like MSE or MAE are insufficient for financial forecasting because they treat all errors equally. Using an engineered custom loss function, **DBE**, which penalises "costly" errors more heavily than neutral ones.

* **Directional Penalty:** High penalty for predicting the wrong direction (e.g., predicting Up when the market goes Down).
* **Magnitude Weighting:** Scales the penalty based on the size of the move, prioritising the avoidance of large drawdowns.
* **Goal:** Optimizes the model for Profit & Loss (P&L) rather than geometric fit.

### 3. Stacked Classifier Architecture
* **Level 1:** Unsupervised clustering (Regime Detection).
* **Level 2:** Supervised forecasting using Regime probabilities as features.
* **Result:** A meta-model that adapts its predictions based on the current market environment.

## Technical Architecture

The system follows a **Medallion Data Architecture** to ensure data integrity from ingestion to inference:

1.  **Bronze Layer:** Ingestion of raw OHLCV (Open, High, Low, Close, Volume) data from financial APIs.
2.  **Silver Layer:** Feature engineering, including rolling volatility, RSI, MACD, and momentum indicators.
3.  **Gold Layer:** Inference-ready dataset containing regime labels, stacked features, and clean target variables.

## Tech Stack

* **Core Logic:** Python 3.10+
* **Modelling:** Scikit-learn, PyTorch (for CNN implementation), Statsmodels.
* **Data Manipulation:** Pandas, NumPy.
* **Backtesting:** Custom vector-based backtesting engine.

## Installation & Usage

```bash
# Clone the repository
git clone [https://github.com/DrWho1369/market-regime-detection.git](https://github.com/DrWho1369/market-regime-detection.git)

# Install dependencies
pip install -r requirements.txt

# Run the regime detection pipeline
python src/main_pipeline.py --mode train --tickers SPY,QQQ

## Future Improvements
* Integration of Transformer-based architecture for longer-term dependency mapping.
* Deployment of the inference engine via a REST API for live trading signal generation.
