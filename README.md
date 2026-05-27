# Forecasting China's Electric Vehicle Sales Using Optimized SARIMA

> A complete end-to-end time series forecasting pipeline for monthly EV sales in China (Jan 2018 – Apr 2024), featuring nested rolling cross-validation, multi-model benchmarking, and a 12-month forward forecast.

---

## Table of Contents
1. [Overview](#overview)
2. [Key Results](#key-results)
3. [Repository Structure](#repository-structure)
4. [Dataset](#dataset)
5. [Methodology](#methodology)
6. [Model Performance](#model-performance)
7. [Forecast (May 2024 – Apr 2025)](#forecast-may-2024--apr-2025)
8. [Figures](#figures)
9. [Installation](#installation)
10. [Quick Start](#quick-start)
11. [References](#references)

---

## Overview

China is the world's largest electric vehicle (EV) market, accounting for over half of global EV sales annually. Accurate monthly sales forecasting is critical for production planning, battery procurement, charging infrastructure deployment, and policy evaluation.

This project constructs a rigorous SARIMA forecasting workflow applied to a 76-month monthly EV sales panel. The key methodological contribution is a **two-stage model selection procedure** — BIC-guided auto-ARIMA followed by **nested rolling cross-validation** — which produces a more robust and generalisable model than single-split evaluation. Five baseline models (non-seasonal ARIMA, Holt-Winters, Prophet, LSTM, and a log-space SARIMA) are benchmarked on the same hold-out window.

---

## Key Results

| Metric | Value |
|---|---|
| **Final model** | $\text{SARIMA}(1,1,0) \times (1,1,0)_{12}$ |
| **Test MAPE** | 17.42 % |
| **Test RMSE** | 108,617 units |
| **Ljung-Box** $Q(12)$ $p$-value | 0.336 (white-noise residuals ✓) |
| **Dec 2024 peak forecast** | ~810,000 units |
| **Feb 2025 trough forecast** | ~446,000 units |

- The model **outperforms non-seasonal ARIMA and LSTM** and is **competitive with Holt-Winters** across both RMSE and MAPE.
- An **additive specification on the original scale** substantially outperforms log-space (multiplicative) alternatives, due to the post-subsidy structural break in 2023.
- Strong **Q4 seasonality** (year-end peak) and **Q1 trough** (Lunar New Year) are captured reliably.

---

## Repository Structure

```
Forecasting-China-EV-Sales/
│
├── Data/
│   ├── Raw/
│   │   └── China Automobile Sales Data.csv     # 38,806 vehicle-model-month records
│   └── Processed/
│       └── ev_sales_monthly.csv                # 76 monthly observations (Jan 2018 – Apr 2024)
│
├── Notebooks/
│   ├── 01_Data_Prep_and_EDA.ipynb              # Data cleaning, aggregation, EDA, decomposition
│   └── 02_Time_Series_Modeling.ipynb           # SARIMA selection, diagnostics, forecast, baselines
│
├── Src/
│   ├── 01_data_prep_and_eda.py                 # Script version of notebook 01
│   └── 02_time_series_modeling.py              # Script version of notebook 02
│
├── figures/                                    # 11 publication-quality PNG charts
│
├── Report/
│   ├── Report_Project.tex                      # Full LaTeX report
│   └── Hoang_Phuc_report.pdf                   # Compiled PDF
│
├── requirements.txt
└── README.md
```

---

## Dataset

| Property | Detail |
|---|---|
| **Source** | [China Automobile Monthly Sales Data (2018–2024.4)](https://www.kaggle.com/datasets/felixzhao/china-automobile-monthly-sales-data-2018-2024-4) — Felixzhao on Kaggle |
| **Compiled from** | China Passenger Car Association (CPCA) official registration records |
| **Raw records** | 38,806 vehicle-model-month observations |
| **Fields** | `year_month`, `units_sold`, `brand`, `model`, `is_ev`, `body_type`, `brand_country` |
| **Processed series** | 76 monthly EV sales totals, Jan 2018 – Apr 2024 |
| **Range** | 8,988 units (Feb 2020, COVID lockdown) → 755,943 units (Dec 2023) |

**Preprocessing steps:** deduplication (33 rows removed), type conversion, EV filtering (`is_ev == "EV"`), monthly aggregation, monthly-start frequency alignment.

---

## Methodology

### 1. Exploratory Data Analysis
- Time series visualisation revealing three structural phases: slow growth (2018–2020), explosive acceleration (2021–2022), post-subsidy stabilisation (2023–2024).
- Additive seasonal decomposition (12-month period): stable seasonal amplitude ±75,000 units; Jan 2023 post-subsidy outlier confirmed at 3σ.
- Additive vs. multiplicative decomposition comparison.

### 2. Stationarity Testing
- **ADF** and **KPSS** dual testing on raw and differenced series.
- Selected differencing: $d = 1$, $D = 1$, $s = 12$ (confirmed stationary by both tests simultaneously).

### 3. Two-Stage Model Selection

**Stage 1 — Auto-ARIMA (BIC):** Stepwise grid search over $p, q \le 2$, $P, Q \le 1$ using `pmdarima`. BIC selected as criterion to penalise over-parameterisation (64 training observations).

**Stage 2 — Nested Rolling Cross-Validation:** Expanding-window CV (step = 1, horizon = 1) refitting auto-ARIMA at each fold. Specification with lowest mean RMSE across all folds is selected as the final model.

### 4. Diagnostics & Benchmarking
- Residual ACF, histogram, Q-Q plot, and Ljung-Box $Q(12)$ test.
- Robustness check: additive (original scale) vs. multiplicative (log-space) SARIMA.
- Benchmarks: Non-seasonal ARIMA, Holt-Winters (ETS), Prophet, LSTM (32 units, lookback = 24).

### 5. Forward Forecast
- Model re-estimated on all 76 observations; 12-month forecast (May 2024 – Apr 2025) with 95% prediction intervals.

---

## Model Performance

### Test Set Comparison (May 2023 – Apr 2024)

| Model | RMSE (units) | MAPE (%) |
|---|---|---|
| **SARIMA$(1,1,0)\times(1,1,0)_{12}$ (Nested CV)** | 108,617 | **17.42** |
| Holt-Winters (ETS, additive) | **97,746** | 17.25 |
| Prophet | 119,070 | 18.45 |
| LSTM (lookback = 24) | 138,427 | 22.99 |
| ARIMA (non-seasonal) | 142,781 | 21.29 |

### Additive vs. Multiplicative and Log vs. No-log (Robustness Check)

| Specification | Test RMSE | Test MAPE (%) |
|---|---|---|
| SARIMA — no-log (Nested CV)  | 108,617 | **17.42** |
| SARIMA — no-log (Auto-ARIMA) | 140,031 | 25.28 |
| SARIMA — log-space (Auto-ARIMA) | 146,423 | 27.14 |
| SARIMA — log-space (Nested CV) | 146,423 | 27.14 |
| ETS — Additive | 97,746 | 17.25 |
| ETS — Multiplicative | 147,854 | 26.57 |

> The log-space model learns the pre-2023 high-growth trajectory and over-extrapolates beyond the structural break, compounding errors upon back-transformation.

---

## Forecast (May 2024 – Apr 2025)

| Month | Forecast | Lower 95% | Upper 95% |
|---|---|---|---|
| May 2024 | 535,606 | 401,629 | 669,584 |
| Jun 2024 | 646,477 | 483,176 | 809,778 |
| Jul 2024 | 620,896 | 426,384 | 815,407 |
| Aug 2024 | 687,121 | 467,514 | 906,729 |
| Sep 2024 | 706,742 | 464,147 | 949,336 |
| Oct 2024 | 707,084 | 443,633 | 970,534 |
| Nov 2024 | 761,747 | 478,936 | 1,044,557 |
| **Dec 2024** | **810,054** | 509,137 | 1,110,970 |
| Jan 2025 | 503,186 | 185,190 | 821,182 |
| **Feb 2025** | **446,472** | 112,268 | 780,675 |
| Mar 2025 | 614,824 | 265,163 | 964,485 |
| Apr 2025 | 576,971 | 212,508 | 941,434 |

The December 2024 peak reflects year-end promotions and fleet procurement; the February 2025 trough reflects the Lunar New Year retail suspension.

---

## Figures

| File | Description |
|---|---|
| `01_units_sold_distribution.png` | Right-skewed distribution of raw vehicle model sales |
| `02_ev_sales_monthly.png` | Aggregated monthly EV sales time series (Jan 2018 – Apr 2024) |
| `03_decomposition_compare.png` | Additive vs. multiplicative seasonal decomposition comparison |
| `04_seasonal_decompose.png` | Full additive decomposition: observed, trend, seasonal, residuals |
| `05_acf_pacf_raw.png` | ACF/PACF of the raw (non-stationary) series |
| `06_acf_pacf_diff.png` | ACF/PACF after $(d=1, D=1, s=12)$ differencing |
| `07_residual_diagnostics.png` | SARIMA residual diagnostics: time plot, histogram, ACF, Q-Q |
| `08_test_forecast_nested_sarima.png` | Test set forecast vs. actuals with 95% CI |
| `09_test_forecast_additiive_vs_multiplicative_ets.png` | Test set accuracy comparison between SARIMA and ETS models (robustness check) |
| `10_forecast_comparison.png` | All baseline model forecasts overlaid on test actuals |
| `11_future_forecast.png` | 12-month forward forecast with 95% confidence bands |

---

## Installation

**Requirements:** Python 3.9+

```bash
# 1. Clone the repository
git clone https://github.com/banana17v-netizen/Forecasting-China-EV-Sales.git
cd Forecasting-China-EV-Sales

# 2. (Recommended) Create a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

**Dependencies** (`requirements.txt`):

```
numpy
pandas
matplotlib
seaborn
statsmodels
pmdarima
scikit-learn
prophet
tensorflow
```

> **Note:** `prophet` and `tensorflow` are required only for baseline model comparison. If not installed, those steps will be skipped automatically.

---

## Quick Start

### Option A — Run scripts directly

```bash
# Step 1: Data preparation and EDA
python Src/01_data_prep_and_eda.py

# Step 2: Modeling, diagnostics, and forecasting
python Src/02_time_series_modeling.py
```

### Option B — Interactive notebooks

Open and run the notebooks in order:

```
Notebooks/01_Data_Prep_and_EDA.ipynb
Notebooks/02_Time_Series_Modeling.ipynb
```

> If `Data/Processed/ev_sales_monthly.csv` already exists, the scripts and notebooks will load it directly; otherwise the raw CSV is processed automatically.

All random seeds are fixed at `42` for reproducibility.

---

## References

| Citation | Details |
|---|---|
| Ge et al. (2024) | *Forecasting and Impact Analysis of the Development Trends of China's New Energy Electric Vehicles Based on Time Series Causal Analysis.* Highlights in Science, Engineering and Technology, 118, pp. 187–196. |
| Yang (2024) | *A Study on Forecasting Sales of New Energy Vehicles in China Based on Time Series Analysis.* Highlights in Science, Engineering and Technology, 98, pp. 182–192. |
| Liang (2025) | *SARIMA Model for Sales Forecast: Evidence from New Energy Vehicles.* Proc. ICEMGD 2025 Symposium. |
| Chen et al. (2025) | *Research on the Forecasting of Sales Volume of New Energy Vehicles Based on SARIMA Model.* Journal of Education, Humanities and Social Sciences, 55, pp. 76–82. |
| Lu (2024) | *UK Battery Electric Vehicle Sales Prediction Using ARIMA and SARIMA Models.* Proc. 8th Int. Conf. Economic Management and Green Development. |
| Çetin & Taşdemir (2024) | *Forecasting Electric Vehicle Sales Using Optimized SARIMA Model: A Two-Year Predictive Analysis.* Veri Bilimi Dergisi, 7(2), pp. 41–51. |
| Felixzhao (2024) | *China Automobile Monthly Sales Data (2018–2024.4).* Kaggle dataset. |

---

## Author

**Ngo Hoang Phuc** — National Economics University