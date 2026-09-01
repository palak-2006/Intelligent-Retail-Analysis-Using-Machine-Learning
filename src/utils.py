"""
Utility Functions Module
Provides formatting, chart helpers, and project-wide configuration.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any


def format_currency(amount: float) -> str:
    """Formats float numbers into currency strings."""
    if amount >= 1_000_000:
        return f"${amount / 1_000_000:.2f}M"
    elif amount >= 1_000:
        return f"${amount / 1_000:.2f}K"
    else:
        return f"${amount:,.2f}"


def create_rfm_3d_scatter(rfm_df: pd.DataFrame) -> go.Figure:
    """
    Creates an interactive 3D scatter plot of Recency, Frequency, and Monetary clusters.
    """
    fig = px.scatter_3d(
        rfm_df,
        x="Recency",
        y="Frequency",
        z="Monetary",
        color="Customer Segment",
        hover_data=["Customer Segment", "Recency", "Frequency", "Monetary"],
        title="3D Customer Segmentation Space",
        opacity=0.7,
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=30),
        scene=dict(
            xaxis_title="Recency (Days)",
            yaxis_title="Frequency (Orders)",
            zaxis_title="Monetary ($ Spend)"
        )
    )
    return fig


def create_revenue_trend_plot(historical_df: pd.DataFrame, forecast_df: pd.DataFrame = None) -> go.Figure:
    """
    Creates a unified Plotly chart showing historical revenue and projected forecast.
    """
    fig = go.Figure()

    # Historical Line
    fig.add_trace(go.Scatter(
        x=historical_df["Date"],
        y=historical_df["Sales"],
        mode="lines+markers",
        name="Historical Sales",
        line=dict(color="#1f77b4", width=3),
        marker=dict(size=6)
    ))

    # Forecast Line
    if forecast_df is not None and not forecast_df.empty:
        fig.add_trace(go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["Forecast_Sales"],
            mode="lines+markers",
            name="Forecasted Sales",
            line=dict(color="#2ca02c", width=3, dash="dash"),
            marker=dict(size=6)
        ))

        # Confidence Interval Ribbon
        fig.add_trace(go.Scatter(
            x=list(forecast_df["Date"]) + list(forecast_df["Date"][::-1]),
            y=list(forecast_df["Upper_Bound"]) + list(forecast_df["Lower_Bound"][::-1]),
            fill="toself",
            fillcolor="rgba(44, 160, 44, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="95% Confidence Interval"
        ))

    fig.update_layout(
        title="Monthly Revenue Trend & Demand Forecast",
        xaxis_title="Timeline",
        yaxis_title="Revenue ($)",
        template="plotly_white",
        hovermode="x unified"
    )
    return fig
