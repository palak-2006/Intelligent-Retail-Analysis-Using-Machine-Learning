"""
Preprocessing Module
Handles data loading, cleaning, validation, and RFM feature engineering.
"""
import pandas as pd
import numpy as np
import datetime as dt
from typing import Tuple, Optional


def load_and_clean_retail_data(filepath: str) -> pd.DataFrame:
    """
    Loads and cleans the online retail transactions dataset.
    Removes cancellations, missing customer IDs, and non-positive values.
    """
    if filepath.endswith(".xlsx"):
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)

    # Clean Customer ID
    df = df.dropna(subset=["Customer ID"]).copy()
    df["Customer ID"] = df["Customer ID"].astype(int)

    # Drop duplicates
    df = df.drop_duplicates()

    # Filter cancellations (Invoices starting with 'C')
    df = df[~df["Invoice"].astype(str).str.startswith("C")]

    # Filter positive quantity and price
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]

    # Feature engineering
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Month"] = df["InvoiceDate"].dt.to_period("M")
    df["Sales"] = df["Quantity"] * df["Price"]

    return df


def calculate_rfm(df: pd.DataFrame, snapshot_date: Optional[dt.datetime] = None) -> pd.DataFrame:
    """
    Calculates Recency, Frequency, and Monetary metrics for each customer.
    """
    if snapshot_date is None:
        snapshot_date = df["InvoiceDate"].max() + dt.timedelta(days=1)

    rfm = df.groupby("Customer ID").agg({
        "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
        "Invoice": "nunique",
        "Sales": "sum"
    })

    rfm.columns = ["Recency", "Frequency", "Monetary"]
    return rfm


def clip_rfm_outliers(rfm: pd.DataFrame, lower_pct: float = 0.01, upper_pct: float = 0.99) -> pd.DataFrame:
    """
    Clips extreme outliers in RFM features for robust model training.
    """
    rfm_clipped = rfm.copy()
    for col in ["Recency", "Frequency", "Monetary"]:
        lower = rfm[col].quantile(lower_pct)
        upper = rfm[col].quantile(upper_pct)
        rfm_clipped[col] = rfm_clipped[col].clip(lower, upper)
    return rfm_clipped
