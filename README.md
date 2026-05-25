# Forecasting China EV Sales

Forecasting China electric vehicle sales using SARIMA and baseline models.

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
- figures/

## How To Run
1. Install dependencies:
	 - `pip install -r requirements.txt`
2. Run the scripts (optional, mirrors the notebooks):
	 - `python Src/01_data_prep_and_eda.py`
	 - `python Src/02_time_series_modeling.py`

## Notes
- The first script creates `Data/Processed/ev_sales_monthly.csv`.
- The second script runs SARIMA modeling, baselines, and a 12-month forecast.
- Prophet and TensorFlow are optional; if missing, the baseline or LSTM step is skipped.
