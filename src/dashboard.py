"""
Dashboard Module
Contains dashboard page layout renderers and UI components for Streamlit.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from src.utils import format_currency


def render_kpi_metrics(df_clean: pd.DataFrame):
    """Renders top 4 business KPI cards."""
    total_rev = df_clean["Sales"].sum()
    total_orders = df_clean["Invoice"].nunique()
    total_customers = df_clean["Customer ID"].nunique()
    avg_order_value = total_rev / total_orders if total_orders > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Total Revenue", format_currency(total_rev))
    col2.metric("📦 Total Orders", f"{total_orders:,}")
    col3.metric("👤 Unique Customers", f"{total_customers:,}")
    col4.metric("🏷️ Avg Order Value", f"${avg_order_value:.2f}")
