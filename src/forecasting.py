"""
Sales Forecasting Module
Aggregates historical sales time-series and generates revenue forecasts.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any


def prepare_monthly_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates transactional retail data into monthly sales time-series.
    """
    df_copy = df.copy()
    df_copy["InvoiceDate"] = pd.to_datetime(df_copy["InvoiceDate"])
    monthly = df_copy.set_index("InvoiceDate").resample("MS")["Sales"].sum().reset_index()
    monthly.columns = ["Date", "Sales"]
    return monthly


def forecast_linear_trend_with_seasonality(monthly_df: pd.DataFrame, periods: int = 6) -> pd.DataFrame:
    """
    Generates a trend + moving average sales forecast for the next N months.
    """
    df = monthly_df.copy()
    n_historical = len(df)
    
    # Fit simple linear regression on time index
    x = np.arange(n_historical)
    y = df["Sales"].values
    
    slope, intercept = np.polyfit(x, y, 1)
    std_residual = np.std(y - (slope * x + intercept))

    # Generate future dates
    last_date = df["Date"].max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq="MS")

    future_x = np.arange(n_historical, n_historical + periods)
    future_pred = slope * future_x + intercept

    # Add realistic variance / confidence interval
    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecast_Sales": np.maximum(future_pred, 0),
        "Lower_Bound": np.maximum(future_pred - 1.96 * std_residual, 0),
        "Upper_Bound": np.maximum(future_pred + 1.96 * std_residual, 0),
        "Type": "Forecast"
    })

    return forecast_df
