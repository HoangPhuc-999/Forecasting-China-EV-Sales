"""Data preparation and EDA for China EV sales."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller


plt.style.use("seaborn-v0_8")

# Base directory (project root) computed from this script's location
BASE_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = BASE_DIR / "figures"


def save_fig(name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / name
    plt.savefig(out_path, dpi=200, bbox_inches="tight")


def adf_pvalue(series: pd.Series) -> float:
    result = adfuller(series.dropna(), autolag="AIC")
    return float(result[1])


def main() -> None:
    # Load raw data
    raw_path = BASE_DIR / "Data" / "Raw" / "China Automobile Sales Data.csv"
    df_raw = pd.read_csv(raw_path)

    # Basic inspection
    _ = df_raw.head()
    _ = df_raw.info()
    _ = df_raw.isna().sum()

    # Remove fully duplicated rows
    df_raw = df_raw.drop_duplicates()

    # Clean units_sold
    df_raw["units_sold"] = (
        df_raw["units_sold"].astype(str).str.replace(",", "", regex=False)
    )
    df_raw["units_sold"] = pd.to_numeric(df_raw["units_sold"], errors="coerce")

    print("Missing units_sold after convert:", df_raw["units_sold"].isna().sum())
    print("Negative units_sold:", (df_raw["units_sold"] < 0).sum())

    _ = df_raw["units_sold"].describe()
    _ = df_raw["is_ev"].value_counts()

    plt.figure(figsize=(10, 5))
    sns.histplot(df_raw["units_sold"], bins=100, kde=True, color="teal")
    plt.title("Distribution of Monthly Units Sold (Raw Data)", fontweight="bold")
    plt.xlabel("Units Sold")
    plt.ylabel("Frequency (Number of Models)")
    plt.tight_layout()
    save_fig("01_units_sold_distribution.png")
    plt.show()

    # Aggregate monthly data
    df = df_raw.copy()
    df["year_month"] = pd.to_datetime(df["year_month"], errors="coerce")
    df = df.dropna(subset=["year_month"])

    total_sales = df.groupby("year_month")["units_sold"].sum()
    ev_sales = df[df["is_ev"] == "EV"].groupby("year_month")["units_sold"].sum()

    monthly = pd.concat([ev_sales, total_sales], axis=1)
    monthly.columns = ["EV_Sales", "Total_Sales"]
    monthly = monthly.asfreq("MS")
    monthly["EV_Sales"] = monthly["EV_Sales"].fillna(0)
    monthly["Total_Sales"] = monthly["Total_Sales"].fillna(0)

    # Compare additive vs. multiplicative decomposition on the positive series
    series_pos = monthly["EV_Sales"].replace(0, np.nan).dropna()
    add_decomp = seasonal_decompose(series_pos, model="additive", period=12)
    mul_decomp = seasonal_decompose(series_pos, model="multiplicative", period=12)

    comparison_table = pd.DataFrame(
        {
            "additive_resid_std": [add_decomp.resid.dropna().std()],
            "multiplicative_resid_std": [mul_decomp.resid.dropna().std()],
            "additive_resid_mean": [add_decomp.resid.dropna().mean()],
            "multiplicative_resid_mean": [mul_decomp.resid.dropna().mean()],
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True)
    axes[0, 0].plot(add_decomp.trend, color="#1f77b4")
    axes[0, 0].set_title("Additive Trend")
    axes[0, 1].plot(mul_decomp.trend, color="#ff7f0e")
    axes[0, 1].set_title("Multiplicative Trend")
    axes[1, 0].plot(add_decomp.resid, color="#1f77b4")
    axes[1, 0].set_title("Additive Residuals")
    axes[1, 1].plot(mul_decomp.resid, color="#ff7f0e")
    axes[1, 1].set_title("Multiplicative Residuals")

    plt.tight_layout()
    save_fig("02_additive_vs_multiplicative_decomposition.png")
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly.index, monthly["EV_Sales"], color="#1f77b4")
    ax.set_title("EV Sales (Monthly)")
    ax.set_ylabel("Units Sold")
    ax.set_xlabel("Year-Month")
    plt.tight_layout()
    save_fig("02_ev_sales_monthly.png")
    plt.show()

    # Decomposition
    series = monthly["EV_Sales"].dropna()
    decomp = seasonal_decompose(series, model="additive", period=12)

    fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(series.index, series.values, "-", linewidth=1)
    axes[0].set_title("Observed")

    axes[1].plot(decomp.trend.index, decomp.trend.values, "-", linewidth=1)
    axes[1].set_title("Trend")

    axes[2].plot(decomp.seasonal.index, decomp.seasonal.values, "-", linewidth=1)
    axes[2].set_title("Seasonal")

    axes[3].plot(decomp.resid.index, decomp.resid.values, "-", linewidth=1)
    axes[3].set_title("Residuals")

    plt.tight_layout()
    save_fig("03_seasonal_decompose.png")
    plt.show()

    # Simple outlier check on residuals
    print("Check Outliers in Residuals:")
    residuals = decomp.resid.dropna()
    mean_resid = residuals.mean()
    std_resid = residuals.std()
    lower_bound = mean_resid - 3 * std_resid
    upper_bound = mean_resid + 3 * std_resid
    time_series_outliers = residuals[
        (residuals < lower_bound) | (residuals > upper_bound)
    ]

    if len(time_series_outliers) == 0:
        print(
            "-> Result: NO macroeconomic shocks were detected. "
            "The residuals fluctuated stably within the 3-sigma threshold"
        )
        print("-> Data is clean and ready for SARIMA modeling")
    else:
        print(
            f"-> Result: {len(time_series_outliers)} months "
            "with abnormal fluctuations detected:"
        )
        print(time_series_outliers)

    # Log vs no-log ADF check
    series_no_log = monthly["EV_Sales"].astype(float)
    series_log = np.log1p(series_no_log)

    adf_table = pd.DataFrame(
        [
            {
                "transform": "no_log",
                "level_pvalue": adf_pvalue(series_no_log),
                "diff1_pvalue": adf_pvalue(series_no_log.diff(1)),
            },
            {
                "transform": "log1p",
                "level_pvalue": adf_pvalue(series_log),
                "diff1_pvalue": adf_pvalue(series_log.diff(1)),
            },
        ]
    ).sort_values("transform").reset_index(drop=True)

    print(adf_table)

    # Save processed monthly series
    out_path = BASE_DIR / "Data" / "Processed" / "ev_sales_monthly.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(out_path)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
