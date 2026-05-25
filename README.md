# Forecasting China EV Sales

This repository contains code and notebooks for forecasting electric vehicle (NEV) sales in China using SARIMA and several baseline models.

## Overview
The project performs data cleaning and aggregation, exploratory data analysis (EDA), model selection using nested cross‑validation, comparison with baseline models (Holt‑Winters, Prophet, LSTM), and a 12‑month forecast.

## Project Structure
- Data/
	- Raw/China Automobile Sales Data.csv
	- Processed/ev_sales_monthly.csv
- Notebooks/
	- 01_Data_Prep_and_EDA.ipynb
	- 02_Time_Series_Modeling.ipynb
- Src/
	- 01_data_prep_and_eda.py
	- 02_time_series_modeling.py
- figures/ (plots and charts)
- requirements.txt

## Installation
1. (Optional) Create a virtual environment:

	 ```bash
	 python -m venv .venv
	 # Windows
	 .venv\Scripts\activate
	 # macOS / Linux
	 source .venv/bin/activate
	 ```

2. Install dependencies:

	 ```bash
	 pip install -r requirements.txt
	 ```

## Quick Start
- Run the processing and modeling scripts (these mirror the notebooks):

	```bash
	python Src/01_data_prep_and_eda.py
	python Src/02_time_series_modeling.py
	```

- Or open and run the notebooks interactively:

	- Notebooks/01_Data_Prep_and_EDA.ipynb
	- Notebooks/02_Time_Series_Modeling.ipynb

## Data
- Raw data: `Data/Raw/China Automobile Sales Data.csv`
- Processed monthly EV sales: `Data/Processed/ev_sales_monthly.csv`
- If the processed file exists, scripts and notebooks will use it; otherwise the raw file will be converted.

## Notes
- Some packages (Prophet, TensorFlow) are optional; if not installed, the corresponding steps will be skipped.
- Main outcome: an optimized SARIMA model selected via nested CV and a 12‑month forecast.