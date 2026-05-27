"""Time series modeling and forecasting for China EV sales."""

import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pmdarima import auto_arima
from pmdarima.model_selection import RollingForecastCV, cross_val_score
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss


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


def mape_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.where(y_true == 0, np.nan, y_true)
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100)


def compute_metrics(actual: pd.Series, pred: pd.Series) -> tuple[float, float]:
    rmse = float(np.sqrt(mean_squared_error(actual, pred)))
    mape = float(np.mean(np.abs((actual - pred) / actual.replace(0, np.nan))) * 100)
    return rmse, mape


def main() -> None:
    data_path = BASE_DIR / "Data" / "Processed" / "ev_sales_monthly.csv"

    if data_path.exists():
        monthly = pd.read_csv(
            data_path, parse_dates=["year_month"], index_col="year_month"
        )
    else:
        raw_path = BASE_DIR / "Data" / "Raw" / "China Automobile Sales Data.csv"
        raw = pd.read_csv(raw_path)
        raw["year_month"] = pd.to_datetime(raw["year_month"], errors="coerce")
        raw = raw.dropna(subset=["year_month"])
        raw["units_sold"] = raw["units_sold"].astype(str).str.replace(",", "", regex=False)
        raw["units_sold"] = pd.to_numeric(raw["units_sold"], errors="coerce")
        ev_sales = (
            raw[raw["is_ev"] == "EV"].groupby("year_month")["units_sold"].sum()
        )
        monthly = ev_sales.to_frame("EV_Sales").sort_index()

    monthly = monthly.asfreq("MS")
    monthly["EV_Sales"] = monthly["EV_Sales"].fillna(0)

    y = monthly["EV_Sales"].dropna()

    # Stationarity checks
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(y, ax=axes[0], lags=24)
    plot_pacf(y, ax=axes[1], lags=24, method="ywm")
    axes[0].set_title("ACF")
    axes[1].set_title("PACF")
    plt.tight_layout()
    save_fig("01_acf_pacf_raw.png")
    plt.show()

    # Differencing grid with ADF
    seasonal_period = 12
    combo_results: list[dict[str, float]] = []
    for d_try in [0, 1, 2]:
        base = y.diff(d_try) if d_try > 0 else y.copy()
        for d_seasonal in [0, 1]:
            series = base.diff(seasonal_period) if d_seasonal > 0 else base
            series = series.dropna()
            if len(series) < 12:
                continue
            combo_results.append(
                {
                    "d": d_try,
                    "D": d_seasonal,
                    "pvalue": adf_pvalue(series),
                    "nobs": len(series),
                }
            )

    combo_table = (
        pd.DataFrame(combo_results)
        .sort_values(["pvalue", "d", "D"])
        .reset_index(drop=True)
    )
    print(combo_table)

    adf_rows: list[dict[str, object]] = []
    series_specs = [
        ("Raw data", 0, 0),
        ("Differenced data", 1, 1),
    ]

    for label, d_val, D_val in series_specs:
        srs = y.copy()
        if d_val > 0:
            srs = srs.diff(d_val)
        if D_val > 0:
            srs = srs.diff(seasonal_period)
        srs = srs.dropna()

        res = adfuller(srs, autolag="AIC")
        crit = res[4]

        adf_rows.append(
            {
                "series": label,
                "d": d_val,
                "D": D_val,
                "adf_statistic": float(res[0]),
                "p_value": float(res[1]),
                "critical_value_1%": float(crit["1%"]),
                "critical_value_5%": float(crit["5%"]),
                "critical_value_10%": float(crit["10%"]),
            }
        )

    adf_compare_table = pd.DataFrame(adf_rows).round(4)
    print(adf_compare_table)

    kpss_rows: list[dict[str, object]] = []
    for label, d_val, D_val in series_specs:
        srs = y.copy()
        if d_val > 0:
            srs = srs.diff(d_val)
        if D_val > 0:
            srs = srs.diff(seasonal_period)
        srs = srs.dropna()

        stat, pval, lags, crit = kpss(srs, regression="c", nlags="auto")
        kpss_rows.append(
            {
                "series": label,
                "d": d_val,
                "D": D_val,
                "kpss_statistic": float(stat),
                "p_value": float(pval),
                "critical_value_10%": float(crit["10%"]),
                "critical_value_5%": float(crit["5%"]),
                "critical_value_2.5%": float(crit["2.5%"]),
                "critical_value_1%": float(crit["1%"]),
                "conclusion_5%": "stationary" if pval > 0.05 else "non-stationary",
            }
        )

    kpss_table = pd.DataFrame(kpss_rows).round(4)
    print(kpss_table)

    # Detailed ADF table with test statistic and critical values (1%, 5%, 10%)
    adf_rows: list[dict[str, object]] = []
    for _, r in combo_table.iterrows():
        d_val = int(r["d"])
        D_val = int(r["D"])
        base = y.diff(d_val) if d_val > 0 else y.copy()
        series = base.diff(seasonal_period) if D_val > 0 else base
        series = series.dropna()
        if len(series) < 12:
            continue
        res = adfuller(series, autolag="AIC")
        test_stat = float(res[0])
        pval = float(res[1])
        usedlag = int(res[2])
        nobs_adf = int(res[3])
        crit = res[4]
        adf_rows.append(
            {
                "d": d_val,
                "D": D_val,
                "usedlag": usedlag,
                "nobs": nobs_adf,
                "test_stat": test_stat,
                "pvalue": pval,
                "cv_1%": float(crit.get("1%")),
                "cv_5%": float(crit.get("5%")),
                "cv_10%": float(crit.get("10%")),
            }
        )

    if adf_rows:
        adf_table = (
            pd.DataFrame(adf_rows)
            .sort_values(["pvalue", "d", "D"]) 
            .reset_index(drop=True)
        )
        print("\nADF detailed table (test statistic, p-value, critical values):")
        print(adf_table.round(4))

    # ACF/PACF after chosen differencing
    d_choice = 1
    D_choice = 1
    y_diff = y.diff(d_choice) if d_choice > 0 else y.copy()
    if D_choice > 0:
        y_diff = y_diff.diff(seasonal_period)
    y_diff = y_diff.dropna()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(y_diff, ax=axes[0], lags=24)
    plot_pacf(y_diff, ax=axes[1], lags=24, method="ywm")
    axes[0].set_title(f"ACF (d={d_choice}, D={D_choice})")
    axes[1].set_title(f"PACF (d={d_choice}, D={D_choice})")
    plt.tight_layout()
    save_fig("02_acf_pacf_diff.png")
    plt.show()

    # Train/test split
    train = y.iloc[:64]
    test = y.iloc[64:]
    s = 12
    d_used = 1
    d_seasonal_used = 1

    # Fit the best auto_arima candidate and show the ranked top fits
    fit_results = auto_arima(
        train,
        seasonal=True,
        m=s,
        d=d_used,
        D=d_seasonal_used,
        max_p=2,
        max_q=2,
        max_P=1,
        max_Q=1,
        information_criterion="bic",
        stepwise=True,
        trace=True,
        error_action="ignore",
        suppress_warnings=True,
        method="lbfgs",
        maxiter=2000,
        return_valid_fits=True,
    )

    if isinstance(fit_results, tuple):
        valid_fits = list(fit_results)
        model = valid_fits[0]
    else:
        model = fit_results
        valid_fits = [model]

    summary_rows = []
    for fitted in valid_fits:
        summary_rows.append(
            {
                "s": s,
                "order": fitted.order,
                "seasonal_order": fitted.seasonal_order,
                "aic": fitted.aic(),
                "bic": fitted.bic(),
            }
        )

    summary_table = (
        pd.DataFrame(summary_rows)
        .sort_values(["bic", "aic"])
        .head(3)
        .reset_index(drop=True)
    )
    summary_table.insert(0, "rank", np.arange(1, len(summary_table) + 1))
    print(summary_table)

    # Rolling CV
    cv = RollingForecastCV(
        h=1,
        step=1,
        initial=max(24, int(len(train) * 0.7)),
    )

    rmse_scores = cross_val_score(
        model,
        train,
        cv=cv,
        scoring="mean_squared_error",
    )
    rmse_cv = float(np.sqrt(rmse_scores).mean())

    mape_scores = cross_val_score(
        model,
        train,
        cv=cv,
        scoring=mape_score,
    )
    mape_cv = float(np.mean(mape_scores))

    cv_table = (
        pd.DataFrame([{"s": s, "rmse_cv": rmse_cv, "mape_cv": mape_cv}])
        .sort_values("rmse_cv")
        .reset_index(drop=True)
    )
    print(cv_table)

    # Nested CV for SARIMA order
    cv_nested = RollingForecastCV(
        h=1,
        step=1,
        initial=max(24, int(len(train) * 0.7)),
    )

    nested_rows: list[dict[str, object]] = []
    for fold, (train_idx, test_idx) in enumerate(cv_nested.split(train), start=1):
        y_train = train.iloc[train_idx]
        y_test = train.iloc[test_idx]

        fold_model = auto_arima(
            y_train,
            seasonal=True,
            m=s,
            d=d_used,
            D=d_seasonal_used,
            max_p=2,
            max_q=2,
            max_P=1,
            max_Q=1,
            information_criterion="bic",
            stepwise=True,
            trace=False,
            error_action="ignore",
            suppress_warnings=True,
            method="lbfgs",
            maxiter=2000,
        )

        y_pred = fold_model.predict(n_periods=len(y_test))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mape = mape_score(y_test, y_pred)

        nested_rows.append(
            {
                "fold": fold,
                "order": fold_model.order,
                "seasonal_order": fold_model.seasonal_order,
                "rmse": rmse,
                "mape": mape,
            }
        )

    nested_table = pd.DataFrame(nested_rows)
    nested_summary = (
        nested_table.groupby(["order", "seasonal_order"], as_index=False)
        .agg(
            mean_rmse=("rmse", "mean"),
            mean_mape=("mape", "mean"),
            n_folds=("fold", "count"),
        )
        .sort_values(["mean_rmse", "mean_mape"])
        .reset_index(drop=True)
    )

    best_row = nested_summary.iloc[0]
    best_order = best_row["order"]
    best_seasonal_order = best_row["seasonal_order"]

    print(nested_summary)

    # Residual diagnostics
    sarima_nested_model = SARIMAX(
        train,
        order=best_order,
        seasonal_order=best_seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False, maxiter=2000)

    resid = sarima_nested_model.resid

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes[0, 0].plot(resid.index, resid.values, color="#1f77b4")
    axes[0, 0].set_title("Residuals Over Time")

    axes[0, 1].hist(resid.values, bins=30, color="#2ca02c", alpha=0.7)
    axes[0, 1].set_title("Residual Histogram")

    plot_acf(resid.dropna(), ax=axes[1, 0], lags=24)
    axes[1, 0].set_title("Residual ACF")

    qqplot(resid.dropna(), line="s", ax=axes[1, 1])
    axes[1, 1].set_title("Residual Q-Q Plot")

    plt.suptitle(
        f"Residual Diagnostics - SARIMA{best_order}x{best_seasonal_order}",
        y=1.02,
    )
    plt.tight_layout()
    save_fig("03_residual_diagnostics.png")
    plt.show()

    lb = acorr_ljungbox(resid, lags=[12], return_df=True)
    print("Ljung-Box p-value (lag 12):", lb["lb_pvalue"].iloc[0])

    # Test forecast
    n_test = len(test)
    actual_train_sales = monthly["EV_Sales"].loc[train.index]
    actual_test_sales = monthly["EV_Sales"].loc[test.index]

    pred_train_sales = sarima_nested_model.get_prediction(
        start=train.index[0],
        end=train.index[-1],
    ).predicted_mean
    rmse, mape = compute_metrics(actual_train_sales, pred_train_sales)
    metrics = [
        {
            "s": s,
            "split": "train",
            "rmse": rmse,
            "mape": mape,
            "lb_pvalue": np.nan,
        }
    ]

    forecast_res = sarima_nested_model.get_forecast(steps=n_test)
    pred_test_sales_nested = forecast_res.predicted_mean
    pred_test_sales_nested.index = test.index
    ci_test = forecast_res.conf_int(alpha=0.05)
    ci_test.index = test.index
    lower_95 = ci_test.iloc[:, 0]
    upper_95 = ci_test.iloc[:, 1]
    rmse, mape = compute_metrics(actual_test_sales, pred_test_sales_nested)
    metrics.append(
        {
            "s": s,
            "split": "test",
            "rmse": rmse,
            "mape": mape,
            "lb_pvalue": lb["lb_pvalue"].iloc[0] if "lb" in globals() else np.nan,
        }
    )

    metrics_table = (
        pd.DataFrame(metrics)
        .sort_values(["split", "rmse"])
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        test.index,
        actual_test_sales,
        label="Actual (Test)",
        color="#1f77b4",
        linewidth=2.5,
        zorder=10,
    )
    ax.plot(
        test.index,
        pred_test_sales_nested,
        label="SARIMA (nested CV)",
        color="#E63946",
        linewidth=2.0,
    )

    ax.fill_between(
        test.index,
        lower_95,
        upper_95,
        color="#E63946",
        alpha=0.2,
        label="95% CI",
    )

    from matplotlib.ticker import StrMethodFormatter

    ax.set_title("Test Set Forecast - Nested SARIMA", fontsize=13)
    ax.set_ylabel("Units Sold")
    ax.set_xlabel("Month")
    ax.ticklabel_format(style="plain", axis="y")
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.legend(loc="upper left")

    plt.tight_layout()
    save_fig("04_test_forecast_nested_sarima.png")
    plt.show()

    print(metrics_table)

    # Log vs no-log comparison
    actual_test_sales = monthly["EV_Sales"].loc[test.index]

    # Log-space ADF and nested-CV SARIMA for the multiplicative-style model
    y_log = np.log1p(monthly["EV_Sales"].astype(float))

    adf_log_rows = [
        {"series": "level", "pvalue": adf_pvalue(y_log)},
        {"series": "diff1", "pvalue": adf_pvalue(y_log.diff(1))},
        {"series": "seasonal_diff12", "pvalue": adf_pvalue(y_log.diff(12))},
        {
            "series": "diff1_seasonal_diff12",
            "pvalue": adf_pvalue(y_log.diff(1).diff(12)),
        },
    ]
    adf_log_table = pd.DataFrame(adf_log_rows)

    combo_log_rows = []
    for d_try in [0, 1, 2]:
        base = y_log.diff(d_try) if d_try > 0 else y_log.copy()
        for D_try in [0, 1]:
            series = base.diff(seasonal_period) if D_try > 0 else base
            series = series.dropna()
            if len(series) < 12:
                continue
            combo_log_rows.append(
                {
                    "d": d_try,
                    "D": D_try,
                    "pvalue": adf_pvalue(series),
                    "nobs": len(series),
                }
            )

    combo_log_table = (
        pd.DataFrame(combo_log_rows)
        .sort_values(["pvalue", "d", "D"])
        .reset_index(drop=True)
    )

    best_combo_log = combo_log_table.iloc[0]
    d_used_log = int(best_combo_log["d"])
    D_used_log = int(best_combo_log["D"])

    train_log = y_log.loc[train.index]
    test_log = y_log.loc[test.index]

    cv_nested_log = RollingForecastCV(
        h=1,
        step=1,
        initial=max(24, int(len(train_log) * 0.7)),
    )

    nested_rows_log: list[dict[str, object]] = []
    for fold, (train_idx, test_idx) in enumerate(cv_nested_log.split(train_log), start=1):
        y_train = train_log.iloc[train_idx]
        y_test = train_log.iloc[test_idx]

        fold_model = auto_arima(
            y_train,
            seasonal=True,
            m=s,
            d=d_used_log,
            D=D_used_log,
            max_p=2,
            max_q=2,
            max_P=1,
            max_Q=1,
            information_criterion="bic",
            stepwise=True,
            trace=False,
            error_action="ignore",
            suppress_warnings=True,
            method="lbfgs",
            maxiter=2000,
        )

        y_pred = fold_model.predict(n_periods=len(y_test))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mape = float(np.mean(np.abs((y_test - y_pred) / y_test.replace(0, np.nan))) * 100)

        nested_rows_log.append(
            {
                "fold": fold,
                "order": fold_model.order,
                "seasonal_order": fold_model.seasonal_order,
                "rmse": rmse,
                "mape": mape,
            }
        )

    nested_table_log = pd.DataFrame(nested_rows_log)
    nested_summary_log = (
        nested_table_log.groupby(["order", "seasonal_order"], as_index=False)
        .agg(mean_rmse=("rmse", "mean"), mean_mape=("mape", "mean"), n_folds=("fold", "count"))
        .sort_values(["mean_rmse", "mean_mape"])
        .reset_index(drop=True)
    )

    best_row_log = nested_summary_log.iloc[0]
    best_order_log = best_row_log["order"]
    best_seasonal_order_log = best_row_log["seasonal_order"]

    log_model_nested = SARIMAX(
        train_log,
        order=best_order_log,
        seasonal_order=best_seasonal_order_log,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False, maxiter=2000)

    pred_test_log_nested = log_model_nested.get_forecast(steps=len(test)).predicted_mean
    pred_test_log_nested.index = test.index
    pred_test_log_sales_nested = np.expm1(pred_test_log_nested)

    actual_test_sales = monthly["EV_Sales"].loc[test.index]
    rmse_log_nested = float(
        np.sqrt(mean_squared_error(actual_test_sales, pred_test_log_sales_nested))
    )
    mape_log_nested = float(
        np.mean(
            np.abs((actual_test_sales - pred_test_log_sales_nested) / actual_test_sales.replace(0, np.nan))
        )
        * 100
    )

    metrics_table_log = pd.DataFrame(
        [
            {
                "transform": "log1p",
                "d": d_used_log,
                "D": D_used_log,
                "order": best_order_log,
                "seasonal_order": best_seasonal_order_log,
                "rmse_test": rmse_log_nested,
                "mape_test": mape_log_nested,
            }
        ]
    )

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        test.index,
        actual_test_sales,
        label="Actual (Test)",
        color="#1f77b4",
        linewidth=2.5,
        zorder=10,
    )

    ax.plot(
        test.index,
        pred_test_log_sales_nested,
        label="SARIMA log1p (nested CV)",
        color="#E63946",
        linewidth=2.0,
    )

    from matplotlib.ticker import StrMethodFormatter

    ax.set_title("Test Set Forecast - Log-Space SARIMA (Nested CV)", fontsize=13)
    ax.set_ylabel("Units Sold")
    ax.set_xlabel("Month")
    ax.ticklabel_format(style="plain", axis="y")
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.show()

    print("ADF-guided differencing for log-space model:")
    print(adf_log_table)
    print("Joint differencing candidates:")
    print(combo_log_table)
    print(f"Selected d={d_used_log}, D={D_used_log}")

    _ = metrics_table_log

    pred_test_sales_auto = pd.Series(model.predict(n_periods=n_test), index=test.index)
    rmse_raw_auto, mape_raw_auto = compute_metrics(actual_test_sales, pred_test_sales_auto)

    best_model_raw = SARIMAX(
        train,
        order=best_order,
        seasonal_order=best_seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False, maxiter=2000)

    pred_test_sales_nested_raw = best_model_raw.get_forecast(steps=n_test).predicted_mean
    pred_test_sales_nested_raw.index = test.index
    rmse_raw_nested, mape_raw_nested = compute_metrics(
        actual_test_sales, pred_test_sales_nested_raw
    )

    y_log = np.log1p(monthly["EV_Sales"])
    train_log = y_log.loc[train.index]

    log_arima = auto_arima(
        train_log,
        seasonal=True,
        m=s,
        d=None,
        D=None,
        max_d=2,
        max_D=1,
        max_p=2,
        max_q=2,
        max_P=1,
        max_Q=1,
        information_criterion="bic",
        stepwise=True,
        trace=False,
        error_action="ignore",
        suppress_warnings=True,
        method="lbfgs",
        maxiter=2000,
    )

    log_model_auto = SARIMAX(
        train_log,
        order=log_arima.order,
        seasonal_order=log_arima.seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False, maxiter=2000)

    pred_test_log_auto = log_model_auto.get_forecast(steps=n_test).predicted_mean
    pred_test_log_auto.index = test.index
    pred_test_log_sales_auto = np.expm1(pred_test_log_auto)
    rmse_log_auto, mape_log_auto = compute_metrics(
        actual_test_sales, pred_test_log_sales_auto
    )

    d_used_log = log_arima.order[1]
    d_seasonal_used_log = log_arima.seasonal_order[1]

    cv_nested_log = RollingForecastCV(
        h=1,
        step=1,
        initial=max(24, int(len(train_log) * 0.7)),
    )

    nested_rows_log: list[dict[str, object]] = []
    for fold, (train_idx, test_idx) in enumerate(cv_nested_log.split(train_log), start=1):
        y_train = train_log.iloc[train_idx]
        y_test = train_log.iloc[test_idx]

        fold_model = auto_arima(
            y_train,
            seasonal=True,
            m=s,
            d=d_used_log,
            D=d_seasonal_used_log,
            max_p=2,
            max_q=2,
            max_P=1,
            max_Q=1,
            information_criterion="bic",
            stepwise=True,
            trace=False,
            error_action="ignore",
            suppress_warnings=True,
            method="lbfgs",
            maxiter=2000,
        )

        y_pred = fold_model.predict(n_periods=len(y_test))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mape = mape_score(y_test, y_pred)

        nested_rows_log.append(
            {
                "fold": fold,
                "order": fold_model.order,
                "seasonal_order": fold_model.seasonal_order,
                "rmse": rmse,
                "mape": mape,
            }
        )

    nested_table_log = pd.DataFrame(nested_rows_log)
    nested_summary_log = (
        nested_table_log.groupby(["order", "seasonal_order"], as_index=False)
        .agg(
            mean_rmse=("rmse", "mean"),
            mean_mape=("mape", "mean"),
            n_folds=("fold", "count"),
        )
        .sort_values(["mean_rmse", "mean_mape"])
        .reset_index(drop=True)
    )

    best_row_log = nested_summary_log.iloc[0]
    best_order_log = best_row_log["order"]
    best_seasonal_order_log = best_row_log["seasonal_order"]

    log_model_nested = SARIMAX(
        train_log,
        order=best_order_log,
        seasonal_order=best_seasonal_order_log,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False, maxiter=2000)

    pred_test_log_nested = log_model_nested.get_forecast(steps=n_test).predicted_mean
    pred_test_log_nested.index = test.index
    pred_test_log_sales_nested = np.expm1(pred_test_log_nested)
    rmse_log_nested, mape_log_nested = compute_metrics(
        actual_test_sales, pred_test_log_sales_nested
    )

    log_compare = pd.DataFrame(
        [
            {
                "transform": "no_log",
                "model": "auto_arima",
                "order": model.order,
                "seasonal_order": model.seasonal_order,
                "rmse": rmse_raw_auto,
                "mape": mape_raw_auto,
            },
            {
                "transform": "no_log",
                "model": "nested_cv",
                "order": best_order,
                "seasonal_order": best_seasonal_order,
                "rmse": rmse_raw_nested,
                "mape": mape_raw_nested,
            },
            {
                "transform": "log1p",
                "model": "auto_arima",
                "order": log_arima.order,
                "seasonal_order": log_arima.seasonal_order,
                "rmse": rmse_log_auto,
                "mape": mape_log_auto,
            },
            {
                "transform": "log1p",
                "model": "nested_cv",
                "order": best_order_log,
                "seasonal_order": best_seasonal_order_log,
                "rmse": rmse_log_nested,
                "mape": mape_log_nested,
            },
        ]
    ).sort_values(["rmse", "mape"]).reset_index(drop=True)

    print(log_compare)

    # Additive vs multiplicative exponential smoothing
    def compute_metrics_hw(actual: pd.Series, pred: pd.Series) -> tuple[float, float]:
        rmse = float(np.sqrt(mean_squared_error(actual, pred)))
        mape = float(np.mean(np.abs((actual - pred) / actual.replace(0, np.nan))) * 100)
        return rmse, mape

    if (monthly["EV_Sales"] <= 0).any():
        raise ValueError(
            "Multiplicative Holt-Winters requires strictly positive values, but EV_Sales contains zero or negative values."
        )

    train_hw = train.astype(float)
    test_hw = test.astype(float)

    hw_add = ExponentialSmoothing(
        train_hw,
        trend="add",
        seasonal="add",
        seasonal_periods=12,
    ).fit(optimized=True)

    pred_add = pd.Series(hw_add.forecast(len(test_hw)), index=test_hw.index)
    rmse_add, mape_add = compute_metrics_hw(test_hw, pred_add)

    hw_mul = ExponentialSmoothing(
        train_hw,
        trend="add",
        seasonal="mul",
        seasonal_periods=12,
    ).fit(optimized=True)

    pred_mul = pd.Series(hw_mul.forecast(len(test_hw)), index=test_hw.index)
    rmse_mul, mape_mul = compute_metrics_hw(test_hw, pred_mul)

    comparison_hw = pd.DataFrame(
        [
            {
                "seasonality": "additive",
                "rmse": rmse_add,
                "mape": mape_add,
                "model": "ETS(add,add)",
            },
            {
                "seasonality": "multiplicative",
                "rmse": rmse_mul,
                "mape": mape_mul,
                "model": "ETS(add,mul)",
            },
        ]
    ).sort_values(["rmse", "mape"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_hw.index, train_hw, label="Train", color="#1f77b4", linewidth=2.0)
    ax.plot(test_hw.index, test_hw, label="Actual (Test)", color="#111111", linewidth=2.5)
    ax.plot(pred_add.index, pred_add, label="ETS additive", color="#E63946", linewidth=2.0)
    ax.plot(pred_mul.index, pred_mul, label="ETS multiplicative", color="#2A9D8F", linewidth=2.0)

    from matplotlib.ticker import StrMethodFormatter

    ax.set_title("Test Set Forecast - Additive vs Multiplicative ETS", fontsize=13)
    ax.set_ylabel("Units Sold")
    ax.set_xlabel("Month")
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.show()

    print(comparison_hw)

    # Baselines
    baseline_results: list[dict[str, object]] = []

    arima_model = auto_arima(
        train,
        seasonal=False,
        d=d_used,
        max_p=3,
        max_q=3,
        information_criterion="bic",
        stepwise=True,
        error_action="ignore",
        suppress_warnings=True,
        method="lbfgs",
        maxiter=2000,
    )
    pred_test_arima = pd.Series(arima_model.predict(n_periods=n_test), index=test.index)
    rmse, mape = compute_metrics(actual_test_sales, pred_test_arima)
    baseline_results.append({"model": "ARIMA", "rmse": rmse, "mape": mape})

    hw_model = ExponentialSmoothing(
        train,
        trend="add",
        seasonal="add",
        seasonal_periods=12,
    ).fit(optimized=True)
    pred_test_hw = pd.Series(hw_model.forecast(n_test), index=test.index)
    rmse, mape = compute_metrics(actual_test_sales, pred_test_hw)
    baseline_results.append({"model": "Holt-Winters", "rmse": rmse, "mape": mape})

    try:
        from prophet import Prophet

        train_df = train.reset_index()
        train_df.columns = ["ds", "y"]
        prophet_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
        )
        prophet_model.fit(train_df)
        future_df = pd.DataFrame({"ds": test.index})
        forecast = prophet_model.predict(future_df)
        pred_test_prophet = pd.Series(forecast["yhat"].values, index=test.index)
        rmse, mape = compute_metrics(actual_test_sales, pred_test_prophet)
        baseline_results.append({"model": "Prophet", "rmse": rmse, "mape": mape})
    except Exception as exc:
        baseline_results.append({"model": "Prophet", "rmse": np.nan, "mape": np.nan})
        print("Prophet not available or failed:", exc)
        pred_test_prophet = None

    baseline_table = (
        pd.DataFrame(baseline_results)
        .sort_values(["rmse", "mape"])
        .reset_index(drop=True)
    )
    print(baseline_table)

    # LSTM baseline (optional)
    os.environ["PYTHONHASHSEED"] = "42"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
        from sklearn.preprocessing import MinMaxScaler
    except Exception as exc:
        print("TensorFlow not available or failed to import:", exc)
        best_lstm_pred_series = None
    else:
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        try:
            tf.keras.utils.set_random_seed(seed)
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass

        series = monthly["EV_Sales"].astype(float).values
        n_train = len(train)

        scaler = MinMaxScaler()
        scaler.fit(series[:n_train].reshape(-1, 1))
        series_scaled = scaler.transform(series.reshape(-1, 1)).flatten()

        lookbacks = [6, 12, 24]
        dropout = 0.2

        def make_supervised(values: np.ndarray, lb: int) -> tuple[np.ndarray, np.ndarray]:
            X, y_vals = [], []
            for i in range(lb, len(values)):
                X.append(values[i - lb : i])
                y_vals.append(values[i])
            return np.array(X), np.array(y_vals)

        def train_eval_lstm(lb: int) -> dict[str, object] | None:
            X_all, y_all = make_supervised(series_scaled, lb)
            split_idx = max(0, n_train - lb)
            X_train, y_train = X_all[:split_idx], y_all[:split_idx]
            X_test, y_test = X_all[split_idx:], y_all[split_idx:]

            if len(X_train) == 0 or len(X_test) == 0:
                return None

            X_train = X_train[..., np.newaxis]
            X_test = X_test[..., np.newaxis]

            lstm_model = models.Sequential(
                [
                    layers.Input(shape=(lb, 1)),
                    layers.LSTM(32),
                    layers.Dropout(dropout),
                    layers.Dense(1),
                ]
            )
            lstm_model.compile(optimizer="adam", loss="mse")

            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=15,
                restore_best_weights=True,
            )

            lstm_model.fit(
                X_train,
                y_train,
                validation_split=0.2,
                epochs=200,
                batch_size=8,
                verbose=0,
                callbacks=[early_stop],
            )

            pred_scaled = lstm_model.predict(X_test, verbose=0).flatten()
            pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
            actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

            pred_series = pd.Series(pred, index=test.index)
            actual_series = pd.Series(actual, index=test.index)

            lstm_rmse, lstm_mape = compute_metrics(actual_series, pred_series)
            return {
                "model": f"LSTM (lookback={lb}, dropout={dropout})",
                "lookback": lb,
                "rmse": lstm_rmse,
                "mape": lstm_mape,
                "pred_series": pred_series,
            }

        lstm_candidates = []
        for lb in lookbacks:
            result = train_eval_lstm(lb)
            if result is None:
                print(f"Not enough data for lookback={lb}.")
            else:
                lstm_candidates.append(result)

        if lstm_candidates:
            lstm_table = (
                pd.DataFrame(lstm_candidates)
                .drop(columns=["pred_series"])
                .sort_values(["rmse", "mape"])
                .reset_index(drop=True)
            )
            print(lstm_table)

            best_lb = int(lstm_table.iloc[0]["lookback"])
            best_lstm_pred_series = next(
                item["pred_series"]
                for item in lstm_candidates
                if item["lookback"] == best_lb
            )
        else:
            print("No LSTM results to display.")
            best_lstm_pred_series = None

    # Forecast comparison plot
    plot_candidates: dict[str, pd.Series] = {
        "SARIMA (nested CV)": pred_test_sales_nested,
    }
    plot_candidates["ARIMA"] = pred_test_arima
    plot_candidates["Holt-Winters"] = pred_test_hw
    if isinstance(pred_test_prophet, pd.Series):
        plot_candidates["Prophet"] = pred_test_prophet
    if isinstance(best_lstm_pred_series, pd.Series):
        plot_candidates["LSTM (best)"] = best_lstm_pred_series

    fig, ax = plt.subplots(figsize=(11, 5))

    color_map = {
        "SARIMA (nested CV)": "#E63946",
        "ARIMA": "#457B9D",
        "Holt-Winters": "#F4A261",
        "Prophet": "#2A9D8F",
        "LSTM (best)": "#8338EC",
    }

    style_map = {
        "SARIMA (nested CV)": "-",
        "ARIMA": "--",
        "Holt-Winters": "-.",
        "Prophet": ":",
        "LSTM (best)": (0, (3, 1, 1, 1)),
    }

    ax.plot(
        test.index,
        actual_test_sales,
        label="Actual (Test)",
        color="#1f77b4",
        linewidth=2.5,
        zorder=10,
    )

    for label, series in plot_candidates.items():
        ax.plot(
            test.index,
            series,
            label=label,
            color=color_map.get(label, "#999999"),
            linestyle=style_map.get(label, "-"),
            linewidth=1.8,
            alpha=0.85,
        )

    ax.set_title("Test Set Forecast Comparison", fontsize=13, fontweight="bold")
    ax.set_ylabel("Units Sold")
    ax.set_xlabel("Month")
    ax.legend(loc="upper left")

    plt.tight_layout()
    save_fig("05_forecast_comparison.png")
    plt.show()

    comparison_rows = []
    for label, series in plot_candidates.items():
        rmse, mape = compute_metrics(actual_test_sales, series)
        comparison_rows.append(
            {
                "model": label,
                "rmse": rmse,
                "mape": mape,
            }
        )

    comparison_table = (
        pd.DataFrame(comparison_rows)
        .sort_values(["rmse", "mape"])
        .reset_index(drop=True)
    )
    print(comparison_table)

    # Final refit and forecast
    forecast_horizon = 12

    full_model = SARIMAX(
        y,
        order=best_order,
        seasonal_order=best_seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False, maxiter=2000)

    forecast_frame = full_model.get_forecast(steps=forecast_horizon).summary_frame()

    future_index = pd.date_range(
        start=monthly.index[-1] + pd.offsets.MonthBegin(1),
        periods=forecast_horizon,
        freq="MS",
    )

    future_sales = pd.Series(forecast_frame["mean"].values, index=future_index)
    future_lower = pd.Series(forecast_frame["mean_ci_lower"].values, index=future_index)
    future_upper = pd.Series(forecast_frame["mean_ci_upper"].values, index=future_index)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(monthly.index, monthly["EV_Sales"], label="History", color="#1f77b4")
    ax.plot(
        future_index, future_sales, label="Future Forecast (s=12)", color="#2ca02c"
    )

    ax.fill_between(
        future_index,
        future_lower,
        future_upper,
        color="#2ca02c",
        alpha=0.2,
        label="95% CI (Future)",
    )

    ax.set_title("EV Sales Forecast (s=12)")
    ax.set_ylabel("Units Sold")
    ax.set_xlabel("Month")
    ax.legend()
    plt.tight_layout()
    save_fig("06_future_forecast.png")
    plt.show()

    forecast_table = pd.DataFrame(
        {
            "month": future_index,
            "forecast": future_sales.values,
            "lower_95": future_lower.values,
            "upper_95": future_upper.values,
        }
    )
    print(forecast_table)


if __name__ == "__main__":
    main()
