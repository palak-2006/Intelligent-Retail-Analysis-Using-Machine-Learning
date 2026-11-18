# RetailIQ - Intelligent Retail Analysis Platform Using Machine Learning

An end-to-end, production-ready retail analytics platform powered by Machine Learning and statistical modeling. RetailIQ connects modern, responsive frontends with a high-performance Flask REST API to provide executive KPIs, sales forecasting, customer RFM segmentation, customer lifetime value (CLV) predictive intelligence, market basket product recommendations, and individual customer profile intelligence.

---

## 🌟 Key Features

### 1. 🔐 Authentication & Session Security
- User registration and login backed by SQLite (`data/users.db`) with SHA-256 password hashing.
- Token-based session management using JSON Web Tokens (PyJWT).
- Seamless client-side auto-redirects on login/logout with bearer token persistence in `localStorage`.

### 2. 📊 Executive KPI Dashboard
- **Real-Time KPIs**: Total Revenue, Total Orders, Active Customers, and Average Order Value (AOV).
- **Sales Trend Line**: Historical monthly revenue trends with interactive tooltips.
- **Top 5 Best-Selling Products**: Revenue-ranked product breakdown.
- **Recent Transactions Feed**: High-value invoice log with customer IDs, items, and timestamps.

### 3. 📈 Sales & Demand Forecasting Analytics
- **Historical Sales Breakdown**: Dynamic aggregation by month, week, or day.
- **30-Day Sales Forecasting**: Predictive modeling with confidence bands for inventory and demand planning.
- **Category & Geographic Mix**: Revenue contribution by category and country distribution.

### 4. 👥 Customer RFM Segmentation & Interactive Directory
- **RFM Analysis + K-Means Clustering**: Classifies retail customers into actionable cohorts (*Champions*, *Loyal Customers*, *Potential Loyalists*, *At Risk*, *Lost*).
- **Interactive Segment Matrix**: Filterable summary cards with cluster-level spending, recency, and size.
- **Live Search & Customer Directory**: Search customers by ID, filter by cohort, and click to view their full profile.

### 5. 💎 Customer Lifetime Value (CLV) Prediction & Value Tier Classification
- **Continuous Spend Regression**: Predicts expected 90-day future spend ($) per customer using Random Forest Regressor ($R^2 = 0.7256$, $\text{MAE} = \$493.31$, $\text{RMSE} = \$1009.11$).
- **Multi-Class CLV Tier Classification**: Classifies customers into strategic cohorts (*High CLV*, *Medium CLV*, *Low CLV*) with comprehensive evaluation:
  - **Accuracy**: $55.47\%$
  - **Weighted F1-Score**: $0.5447$ | **Macro F1-Score**: $0.5339$
  - **Confusion Matrix** with full breakdown of true vs predicted tier assignments.
- **Binary Top-Spender Identification**: Detects prospective VIPs ($\ge \$500$ future spend) with **$72.93\%$ Accuracy**, **$0.6065$ F1-Score**, and **$0.8025$ ROC-AUC**.
- **Real-Time Scoring API**: Fast on-the-fly inference endpoint (`/api/clv/predict`) for any customer profile.

### 6. 🔍 Customer 360° Profile & Next Best Action
- **Customer Metrics**: Historical Spend (LTV), ML Predicted CLV (Next 90 Days), CLV Tier Badge, Order Count, Recency (days), and Average Basket Size.
- **Spend Trajectory**: Historical timeline of monthly spend per customer.
- **Category Affinity Breakdown**: Product category preference distribution.
- **Market Basket Cross-Sell Recommender**: Powered by FP-Growth association rules to recommend the next best product based on past purchases.
- **Invoice Audit**: Detailed table of historical invoices and purchased items.

### 7. 🛡️ Enterprise Risk Analysis & Vulnerability Matrix
- **Multidimensional Risk Evaluation**: Rigorous empirical assessment covering 8 strategic retail risk dimensions:
  - **Regional Risk (Critical)**: Geographic over-concentration in the United Kingdom (**83.90% of global sales, $7.38M** across 37 active markets).
  - **Customer Risk (High)**: Whale dependency (**Top 10 accounts command 16.50% / $1.45M** and Top 1% command **28.58% / $2.51M**) and customer dormancy (**1,002 at-risk accounts** with **$504.4k** historical spend at risk).
  - **Demand & Seasonality Risk (High)**: Monthly sales coefficient of variation of **28.7%** with an extreme **+107.4% Q4 holiday demand surge** ($493k Jan-Feb trough vs $1.16M peak).
  - **Inventory Risk (Medium)**: Long-tail dead stock drag with **998 SKUs (22.46% of catalog)** generating <$100 annual revenue.
  - **Product Risk (High)**: Extreme Pareto concentration where **22.3% of SKUs (991 items)** generate **80%** of total revenue, creating high stockout sensitivity.
  - **Sales Risk (Medium)**: One-time buyer acquisition leakage (**32.91% of customers** never place a repeat order).
  - **Operational Risk (Medium)**: Order volume surges of 2.1x in peak months across 37 export nations.
- **Interactive Risk Assessment Matrix**: Clickable 8-risk probability vs impact matrix grid with pop-up containment protocols.
- **Dynamic Visualizations**: 8-dimension vulnerability radar chart, seasonal demand volatility envelope (±1 Std Dev bands), and regional revenue bar chart.
- **Mitigation Roadmap**: Phased action plan (Immediate 0-30d, Medium-Term 30-90d, Long-Term 90-180d) and academic dataset measurement boundary disclosure.

---

## 🧠 Customer Lifetime Value (CLV) Theory & Mathematical Formulations

### What is CLV?
**Customer Lifetime Value (CLV/LTV)** measures the total economic value (revenue or margin) a business can anticipate from a customer over the duration of their relationship.

### Core Formulations:
1. **Historical CLV**:
   $$\text{CLV}_{\text{hist}} = \sum_{j=1}^{N_i} \text{Order Value}_{i,j} = \text{AOV}_i \times \text{Purchase Frequency}_i$$

2. **Traditional / Deterministic CLV**:
   $$\text{CLV} = \text{Average Order Value} \times \text{Purchase Frequency} \times \text{Customer Lifespan}$$
   With margin $m$, retention rate $r$, and discount rate $d$:
   $$\text{CLV} = m \times \left(\frac{r}{1 + d - r}\right)$$

3. **Machine Learning Supervised CLV Framework**:
   - **Observation Window**: Features computed over historical transactions (Recency, Frequency, Monetary, Tenure, Average Order Value, Unique SKUs, Average Basket Size).
   - **Future Holdout Window**: Ground truth future monetary spend measured in the forward 90 days.
   - **Dual Model**:
     - *Regression*: Continuous spend prediction evaluated via MAE, RMSE, and $R^2$.
     - *Classification*: Value tier categorization evaluated via **Accuracy, F1-Score, Precision, Recall, Confusion Matrix, and ROC-AUC**.

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.9+ or Python 3.10+
- `pip` package manager

### 1. Installation
```bash
git clone https://github.com/palak-2006/Intelligent-Retail-Analysis-Using-Machine-Learning.git
cd Intelligent-Retail-Analysis-Using-Machine-Learning-main
pip install -r requirements.txt
```

### 2. Launch the Platform
Start both the backend server and frontend launcher with a single command:
```bash
python run.py
```
This will:
1. Initialize the SQLite database and seed demo data if not already present.
2. Load and verify preprocessed retail datasets, association rules, and CLV predictive models.
3. Start the Flask server on `http://localhost:5000`.
4. Automatically launch your default web browser to the Login page.

### 3. Default Login Credentials
- **Email**: `admin@retailiq.com`
- **Password**: `RetailIQ@2024`

---

## 🏗️ Architecture & Dual-Mode Compatibility

1. **Full-Stack Mode (Recommended)**:
   Run `python run.py`. The Flask server serves both the REST API endpoints and routes all frontend pages cleanly:
   - `/login` &rarr; Login Page
   - `/signup` &rarr; Sign Up Page
   - `/dashboard` &rarr; Executive KPI Dashboard
   - `/sales` &rarr; Sales Analytics & Forecasting
   - `/segmentation` &rarr; Customer RFM Segmentation & Directory
   - `/customer-details?id=14911` &rarr; Customer 360° Profile Deep-Dive (including ML CLV)
   - `/clv` &rarr; Customer Lifetime Value (CLV) Engine & Value Tier Intelligence
   - `/risk` &rarr; Enterprise Risk Intelligence Engine & Vulnerability Matrix

2. **Standalone Frontend Mode (`file://`)**:
   Open any `.html` file inside the `frontend/` directory directly in your browser. All frontend scripts automatically detect `file://` mode and route API calls to `http://localhost:5000` while preserving relative file navigation.

---

## 📡 REST API Reference

| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| `POST` | `/api/auth/register` | Register a new user | No |
| `POST` | `/api/auth/login` | Authenticate and obtain JWT token | No |
| `GET` | `/api/auth/me` | Fetch authenticated user profile | Yes (Bearer) |
| `GET` | `/api/dashboard/kpis` | Executive summary metrics (Revenue, Orders, Customers, AOV) | Yes (Bearer) |
| `GET` | `/api/dashboard/charts` | Monthly sales trend and top 5 products | Yes (Bearer) |
| `GET` | `/api/dashboard/recent-transactions` | Last 10 invoices | Yes (Bearer) |
| `GET` | `/api/sales/monthly-actuals` | Monthly sales aggregations, revenue by country, top products | Yes (Bearer) |
| `GET` | `/api/sales/forecast` | 30-day forward sales forecast with confidence intervals | Yes (Bearer) |
| `GET` | `/api/sales/forecast-metrics` | Time-series backtest metrics (MAE, RMSE, MAPE, R²) | Yes (Bearer) |
| `GET` | `/api/segmentation/overview` | RFM cluster distribution, cluster metrics, and personas | Yes (Bearer) |
| `POST` | `/api/segmentation/predict` | Predict customer segment using supervised KNN classifier | Yes (Bearer) |
| `GET` | `/api/customers/list` | Sample customer list across all segments | Yes (Bearer) |
| `GET` | `/api/customers/search?q={query}` | Search customer directory by ID or cohort | Yes (Bearer) |
| `GET` | `/api/customers/<int:customer_id>` | Detailed 360° customer metrics, ML CLV, trajectory, & recommendations | Yes (Bearer) |
| `GET` | `/api/clv/overview` | CLV executive totals, tier breakdown, accuracy, F1, & confusion matrix | Yes (Bearer) |
| `GET` | `/api/clv/predictions` | Paginated customer CLV predictions with tier filters | Yes (Bearer) |
| `POST` | `/api/clv/predict` | Real-time ML prediction for continuous spend & CLV tier | Yes (Bearer) |
| `GET` | `/api/risk/overview` | Executive risk scores, overall vulnerability index, & summary metrics | Yes (Bearer) |
| `GET` | `/api/risk/matrix` | Full 8-dimension Risk Assessment Matrix with likelihood & mitigations | Yes (Bearer) |
| `GET` | `/api/risk/regional` | Geographic concentration analysis (UK vs 36 international corridors) | Yes (Bearer) |
| `GET` | `/api/risk/demand` | Monthly demand volatility series with ±1 Std Dev risk bands | Yes (Bearer) |
| `GET` | `/api/risk/products` | High-velocity stockout risk SKUs vs slow-moving dead-stock tail | Yes (Bearer) |
| `GET` | `/api/risk/all` | Complete consolidated risk intelligence payload for dashboard hydration | Yes (Bearer) |
| `GET` | `/api/recommendations/products` | All candidate products for market basket recommendation | Yes (Bearer) |
| `POST` | `/api/recommendations/recommend` | Next-best product recommendation via Market Basket rules | Yes (Bearer) |
| `GET` | `/api/health` | Service health status and readiness checks | No |

---

## 📁 Project Structure

```
├── api.py                   # Central Flask REST API with auth, RFM, CLV & ML endpoints
├── run.py                   # One-click launcher (starts server + opens browser)
├── requirements.txt         # Python package dependencies
├── data/                    # Cleaned retail data, customer segments, CLV predictions, MBA rules
│   ├── online_retail_clean.csv
│   ├── customer_segments.csv
│   ├── clv_predictions.csv
│   ├── churn_predictions.csv
│   └── market_basket_rules.csv
├── models/                  # Serialized ML models & scalers
│   ├── clv_model.pkl        # CLV Random Forest Regressor
│   ├── clv_classifier.pkl   # CLV Value Tier Random Forest Classifier
│   ├── clv_features.pkl     # Feature column names
│   ├── clv_scaler.pkl       # Feature StandardScaler
│   ├── clv_metrics.pkl      # Accuracy, F1, Confusion Matrix, MAE, RMSE, R2
│   ├── churn_model.pkl      # Customer Churn Classifier
│   └── customer_segmentation_kmeans.pkl
├── notebooks/               # Step-by-step Jupyter analysis notebooks
│   ├── 01_online_retail_eda.ipynb
│   ├── 02_store_sales_eda.ipynb
│   ├── 03_superstore_eda.ipynb
│   ├── 04_market_basket_analysis.ipynb
│   ├── 05_sales_prediction.ipynb
│   ├── 06_churn_prediction.ipynb
│   ├── 07_clv_prediction.ipynb        # Comprehensive CLV prediction & tier classification
│   └── 08_risk_analysis.ipynb         # Empirical Enterprise Risk Analysis & Assessment
├── src/                               # Modular backend algorithms
│   ├── preprocessing.py               # Data cleaning and feature engineering
│   ├── segmentation.py                # RFM modeling and K-Means clustering
│   ├── forecasting.py                 # Time-series trend and forecasting
│   ├── clv.py                         # Customer Lifetime Value models, metrics & inference
│   ├── risk.py                        # Multi-dimensional Enterprise Risk Analysis Engine
│   └── recommendation.py              # FP-Growth Market Basket association rules
└── frontend/                          # Interactive UI prototypes
    ├── retailiq_login_silk_white_bg/
    ├── retailiq_sign_up_purple_glow_centered/
    ├── retailiq_dashboard_theme_toggle_purple_glow/
    ├── retailiq_sales_analytics_theme_toggle_pie_chart/
    ├── retailiq_customer_segmentation_theme_toggle_purple_glow/
    ├── retailiq_clv_analytics_theme_toggle_purple_glow/
    ├── retailiq_risk_analytics_theme_toggle_purple_glow/
    └── retailiq_customer_details_theme_toggle_purple_glow/
```

---

## 🛡️ License & Acknowledgements
Developed as part of the Intelligent Retail Analytics project using real-world retail transaction data.
