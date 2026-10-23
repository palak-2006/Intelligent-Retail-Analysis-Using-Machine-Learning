# RetailIQ - Intelligent Retail Analysis Platform Using Machine Learning

An end-to-end, production-ready retail analytics platform powered by Machine Learning and statistical modeling. RetailIQ connects modern, responsive frontends with a high-performance Flask REST API to provide executive KPIs, sales forecasting, customer RFM segmentation, market basket product recommendations, and individual customer profile intelligence.

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

### 5. 🔍 Customer 360° Profile & Next Best Action
- **Customer Metrics**: Exact Lifetime Value (LTV), Order Count, Recency (days), and Average Basket Size.
- **Spend Trajectory**: Historical timeline of monthly spend per customer.
- **Category Affinity Breakdown**: Product category preference distribution.
- **Market Basket Cross-Sell Recommender**: Powered by FP-Growth association rules to recommend the next best product based on past purchases.
- **Invoice Audit**: Detailed table of historical invoices and purchased items.

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.9+ or Python 3.10+
- `pip` package manager

### 1. Installation
Clone this repository and install required dependencies:
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
2. Load and verify preprocessed retail datasets and association rules.
3. Start the Flask server on `http://localhost:5000`.
4. Automatically launch your default web browser to the Login page.

### 3. Default Login Credentials
You can sign up for a new account directly on the signup screen, or log in with the pre-seeded administrator account:
- **Email**: `admin@retailiq.com`
- **Password**: `RetailIQ@2024`

---

## 🏗️ Architecture & Dual-Mode Compatibility

The platform is designed to be fully decoupled and supports two modes of operation:

1. **Full-Stack Mode (Recommended)**:
   Run `python run.py`. The Flask server serves both the REST API endpoints and routes all frontend pages cleanly:
   - `/login` &rarr; Login Page
   - `/signup` &rarr; Sign Up Page
   - `/dashboard` &rarr; Executive KPI Dashboard
   - `/sales` &rarr; Sales Analytics & Forecasting
   - `/segmentation` &rarr; Customer RFM Segmentation & Directory
   - `/customer-details?id=17850` &rarr; Customer 360° Profile Deep-Dive

2. **Standalone Frontend Mode (`file://`)**:
   You can open any `.html` file inside the `frontend/` directory directly in your browser. All frontend scripts automatically detect `file://` mode and route API calls to `http://localhost:5000` while preserving relative file navigation.

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
| `GET` | `/api/sales/overview` | Monthly sales aggregations, revenue by country, top products | Yes (Bearer) |
| `GET` | `/api/sales/forecast` | 30-day forward sales forecast with confidence intervals | Yes (Bearer) |
| `GET` | `/api/customers/segmentation` | RFM cluster distribution, cluster metrics, and sample profiles | Yes (Bearer) |
| `GET` | `/api/customers/list` | Sample customer list across all segments | Yes (Bearer) |
| `GET` | `/api/customers/search?q={query}&segment={segment}` | Search customer directory | Yes (Bearer) |
| `GET` | `/api/customers/<int:customer_id>` | Detailed 360° customer metrics, affinity, trajectory, & recommendations | Yes (Bearer) |
| `POST` | `/api/recommendations/product` | Next-best product recommendation via Market Basket rules | Yes (Bearer) |

---

## 📁 Project Structure

```
├── api.py                   # Central Flask REST API with auth & ML endpoints
├── run.py                   # One-click launcher (starts server + opens browser)
├── requirements.txt         # Python package dependencies
├── data/                    # Cleaned retail data, customer segments, MBA rules, SQLite DB
├── src/                     # Core analytics, segmentation, and forecasting modules
│   ├── preprocessing.py     # Data cleaning and feature engineering
│   ├── segmentation.py      # RFM modeling and K-Means clustering
│   ├── forecasting.py       # Time-series trend and forecasting
│   └── recommendation.py    # FP-Growth Market Basket association rules
└── frontend/                # Interactive UI prototypes
    ├── retailiq_login_silk_white_bg/
    ├── retailiq_sign_up_purple_glow_centered/
    ├── retailiq_dashboard_theme_toggle_purple_glow/
    ├── retailiq_sales_analytics_theme_toggle_pie_chart/
    ├── retailiq_customer_segmentation_theme_toggle_purple_glow/
    └── retailiq_customer_details_theme_toggle_purple_glow/
```

---

## 🛡️ License & Acknowledgements
Developed as part of the Intelligent Retail Analytics project using real-world retail transaction data.
