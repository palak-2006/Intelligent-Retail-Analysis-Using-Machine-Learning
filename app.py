"""
RetailIQ - Intelligent Retail Analytics Platform
Interactive Streamlit Application
"""
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Import custom src modules
from src.preprocessing import load_and_clean_retail_data, calculate_rfm
from src.segmentation import CustomerSegmentationModel, DEFAULT_SEGMENT_NAMES, DEFAULT_SEGMENT_ACTIONS
from src.recommendation import ProductRecommender
from src.forecasting import prepare_monthly_timeseries, forecast_linear_trend_with_seasonality
from src.utils import format_currency, create_rfm_3d_scatter, create_revenue_trend_plot

# Page Configuration
st.set_page_config(
    page_title="RetailIQ - AI Retail Analytics",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_clean_data():
    csv_path = "data/online_retail_clean.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        return df
    return None


@st.cache_data
def load_segments_data():
    csv_path = "data/customer_segments.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None


@st.cache_resource
def load_models():
    model_path = "models/customer_segmentation_kmeans.pkl"
    scaler_path = "models/rfm_scaler.pkl"
    rules_path = "data/market_basket_rules.csv"

    seg_model = CustomerSegmentationModel()
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            seg_model.load(model_path, scaler_path)
        except Exception:
            pass

    recommender = ProductRecommender(rules_path)
    return seg_model, recommender


# Load Data & Models
df_clean = load_clean_data()
segments_df = load_segments_data()
seg_model, recommender = load_models()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/isometric/100/shopping-cart-loaded.png", width=70)
st.sidebar.title("RetailIQ Analytics")
st.sidebar.markdown("Machine Learning-Powered Retail Intelligence Platform")

page = st.sidebar.radio(
    "Navigate to:",
    ["📊 Executive Dashboard", "👥 Customer Segmentation", "🛒 Smart Recommender", "📈 Sales Forecasting"]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Use the sidebar to switch between analytics engines.")


# ==============================================================================
# PAGE 1: EXECUTIVE DASHBOARD
# ==============================================================================
if page == "📊 Executive Dashboard":
    st.markdown('<div class="main-header">📊 Executive Retail Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time business performance metrics, sales velocity, and geographical distribution.</div>', unsafe_allow_html=True)

    if df_clean is not None:
        # Top KPI Metrics
        total_rev = df_clean["Sales"].sum()
        total_orders = df_clean["Invoice"].nunique()
        total_customers = df_clean["Customer ID"].nunique()
        avg_order_value = total_rev / total_orders if total_orders > 0 else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("💰 Total Revenue", format_currency(total_rev))
        col2.metric("📦 Total Orders", f"{total_orders:,}")
        col3.metric("👤 Unique Customers", f"{total_customers:,}")
        col4.metric("🏷️ Avg Order Value", f"${avg_order_value:.2f}")

        st.markdown("---")

        # Charts Section
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("📅 Monthly Sales Trend")
            monthly_sales = df_clean.set_index("InvoiceDate").resample("MS")["Sales"].sum().reset_index()
            fig_month = px.line(
                monthly_sales,
                x="InvoiceDate",
                y="Sales",
                markers=True,
                title="Revenue Over Time",
                labels={"InvoiceDate": "Month", "Sales": "Revenue ($)"}
            )
            fig_month.update_traces(line_color="#2563EB", line_width=3)
            st.plotly_chart(fig_month, use_container_width=True)

        with c2:
            st.subheader("🌍 Top 10 Countries by Revenue")
            country_sales = df_clean.groupby("Country")["Sales"].sum().sort_values(ascending=False).head(10).reset_index()
            fig_country = px.bar(
                country_sales,
                x="Sales",
                y="Country",
                orientation="h",
                title="Sales by Country",
                labels={"Sales": "Revenue ($)", "Country": "Country"},
                color="Sales",
                color_continuous_scale="Blues"
            )
            fig_country.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_country, use_container_width=True)

        st.subheader("🏆 Top 10 Best-Selling Products")
        top_prod = df_clean.groupby("Description").agg({"Sales": "sum", "Quantity": "sum"}).sort_values(by="Sales", ascending=False).head(10).reset_index()
        top_prod["Sales"] = top_prod["Sales"].apply(lambda x: f"${x:,.2f}")
        top_prod["Quantity"] = top_prod["Quantity"].apply(lambda x: f"{x:,}")
        st.dataframe(top_prod, use_container_width=True)
    else:
        st.warning("Clean dataset not found. Please run the preprocessing pipeline first.")


# ==============================================================================
# PAGE 2: CUSTOMER SEGMENTATION
# ==============================================================================
elif page == "👥 Customer Segmentation":
    st.markdown('<div class="main-header">👥 RFM Customer Segmentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-powered customer grouping using Recency, Frequency, and Monetary behavior.</div>', unsafe_allow_html=True)

    if segments_df is not None:
        t1, t2 = st.tabs(["🔍 Segment Insights & Distribution", "🎯 Real-Time Customer Predictor"])

        with t1:
            c1, c2 = st.columns([1, 1])

            with c1:
                st.subheader("Segment Distribution")
                seg_counts = segments_df["Customer Segment"].value_counts().reset_index()
                seg_counts.columns = ["Customer Segment", "Count"]
                fig_pie = px.pie(
                    seg_counts,
                    names="Customer Segment",
                    values="Count",
                    color="Customer Segment",
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with c2:
                st.subheader("Average RFM Profile per Segment")
                seg_profile = segments_df.groupby("Customer Segment").agg({
                    "Recency": "mean",
                    "Frequency": "mean",
                    "Monetary": "mean",
                    "Customer ID": "count"
                }).rename(columns={"Customer ID": "Customer Count"}).round(2).reset_index()
                st.dataframe(seg_profile, use_container_width=True)

            st.subheader("3D Interactive RFM Space")
            st.plotly_chart(create_rfm_3d_scatter(segments_df), use_container_width=True)

        with t2:
            st.subheader("🎯 Test Customer Segment Prediction")
            st.write("Enter custom purchase metrics to classify a customer into an actionable segment:")

            col1, col2, col3 = st.columns(3)
            rec = col1.number_input("Recency (Days since last purchase)", min_value=1, max_value=365, value=15)
            freq = col2.number_input("Frequency (Total unique orders)", min_value=1, max_value=100, value=12)
            mon = col3.number_input("Monetary (Total lifetime spend $)", min_value=1.0, max_value=50000.0, value=3500.0)

            if st.button("Classify Customer", type="primary"):
                if seg_model.scaler is not None:
                    cluster_id, seg_name, strategy = seg_model.predict_single(rec, freq, mon)
                    st.success(f"**Assigned Segment:** {seg_name} (Cluster #{cluster_id})")
                    st.info(f"**Recommended Marketing Action:** {strategy}")
                else:
                    st.error("Segmentation model weights not found. Please train/save models first.")
    else:
        st.warning("Customer segments file not found. Please run the segmentation notebook first.")


# ==============================================================================
# PAGE 3: SMART PRODUCT RECOMMENDER
# ==============================================================================
elif page == "🛒 Smart Recommender":
    st.markdown('<div class="main-header">🛒 Smart Cross-Sell Recommender</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Discover product affinities and automated bundle suggestions using Market Basket Analysis.</div>', unsafe_allow_html=True)

    all_products = recommender.get_all_products()
    if all_products:
        selected_product = st.selectbox(
            "Select a Product to view complementary recommendations:",
            options=all_products,
            index=0
        )

        top_k = st.slider("Number of recommendations to show:", min_value=1, max_value=10, value=5)

        if st.button("Generate Recommendations", type="primary"):
            recs = recommender.recommend(selected_product, top_n=top_k)

            if recs:
                st.subheader(f"Recommended Products for: *{selected_product}*")
                for idx, item in enumerate(recs, start=1):
                    with st.container():
                        st.markdown(f"""
                        **{idx}. {item['recommended_item']}**  
                        * **Confidence:** `{item['confidence']}%` (Likelihood of being bought together)  
                        * **Lift:** `{item['lift']}x` (Strength of association over random chance)  
                        * **Support:** `{item['support']}%` of total transactions  
                        ---
                        """)
            else:
                st.warning(f"No direct association rules found for '{selected_product}'.")
    else:
        st.warning("Association rules not found. Please run the market basket analysis notebook first.")


# ==============================================================================
# PAGE 4: SALES FORECASTING
# ==============================================================================
elif page == "📈 Sales Forecasting":
    st.markdown('<div class="main-header">📈 Sales Demand Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-driven predictive demand modeling and revenue trajectory forecasting.</div>', unsafe_allow_html=True)

    if df_clean is not None:
        monthly_df = prepare_monthly_timeseries(df_clean)
        
        forecast_horizon = st.slider("Select Forecast Horizon (Months):", min_value=3, max_value=12, value=6)
        
        forecast_df = forecast_linear_trend_with_seasonality(monthly_df, periods=forecast_horizon)
        
        # Display Combined Plot
        fig_forecast = create_revenue_trend_plot(monthly_df, forecast_df)
        st.plotly_chart(fig_forecast, use_container_width=True)

        st.subheader("📋 Forecast Summary Table")
        forecast_table = forecast_df.copy()
        forecast_table["Date"] = forecast_table["Date"].dt.strftime("%B %Y")
        forecast_table["Forecast_Sales"] = forecast_table["Forecast_Sales"].apply(lambda x: f"${x:,.2f}")
        forecast_table["Lower_Bound"] = forecast_table["Lower_Bound"].apply(lambda x: f"${x:,.2f}")
        forecast_table["Upper_Bound"] = forecast_table["Upper_Bound"].apply(lambda x: f"${x:,.2f}")
        st.dataframe(forecast_table, use_container_width=True)
    else:
        st.warning("Clean dataset not found for forecasting.")
