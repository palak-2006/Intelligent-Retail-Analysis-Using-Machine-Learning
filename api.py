"""
RetailIQ – Flask REST API Backend
Provides authenticated endpoints for all frontend pages.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import os
import sqlite3
import json
import datetime
import secrets
from functools import wraps

import jwt
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, g, send_from_directory, redirect
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

# ── src modules ────────────────────────────────────────────────────────────────
from src.segmentation import CustomerSegmentationModel, DEFAULT_SEGMENT_NAMES, DEFAULT_SEGMENT_ACTIONS
from src.forecasting import prepare_monthly_timeseries, forecast_linear_trend_with_seasonality
from src.recommendation import ProductRecommender
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import joblib

# ── App Setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="frontend")
CORS(app, resources={r"/*": {"origins": "*"}})

# Secret key for JWT signing — stable across restarts
SECRET_KEY = os.environ.get("RETAILIQ_SECRET", "retailiq-jwt-secret-key-2024-do-not-share")
DB_PATH = os.path.join("data", "users.db")


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE – SQLite User Store
# ══════════════════════════════════════════════════════════════════════════════

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(error):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    """Create users table and seed default admin account."""
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            name     TEXT    NOT NULL,
            email    TEXT    UNIQUE NOT NULL,
            password TEXT    NOT NULL,
            role     TEXT    NOT NULL DEFAULT 'user',
            created  TEXT    NOT NULL
        )
    """)
    # Seed admin account
    c.execute("SELECT id FROM users WHERE email = ?", ("admin@retailiq.com",))
    if c.fetchone() is None:
        c.execute(
            "INSERT INTO users (name, email, password, role, created) VALUES (?,?,?,?,?)",
            (
                "Admin",
                "admin@retailiq.com",
                generate_password_hash("RetailIQ@2024"),
                "admin",
                datetime.datetime.utcnow().isoformat(),
            ),
        )
        print("✅  Default admin seeded: admin@retailiq.com / RetailIQ@2024")
    conn.commit()
    conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# JWT – Token Helpers
# ══════════════════════════════════════════════════════════════════════════════

def make_token(user_id: int, email: str, name: str, role: str) -> str:
    payload = {
        "sub": str(user_id),
        "email": email,
        "name": name,
        "role": role,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24),
        "iat": datetime.datetime.utcnow(),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")


def decode_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def require_auth(f):
    """Decorator: validates Bearer JWT on every protected endpoint."""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid Authorization header"}), 401
        token = auth_header[7:]
        payload = decode_token(token)
        if payload is None:
            return jsonify({"error": "Token expired or invalid. Please log in again."}), 401
        g.current_user = payload
        return f(*args, **kwargs)
    return decorated


# ══════════════════════════════════════════════════════════════════════════════
# DATA & MODEL LOADING  (at startup, cached in module-level globals)
# ══════════════════════════════════════════════════════════════════════════════

print("⏳  Loading datasets…")
_df_clean = None
_segments_df = None
_seg_model = None
_recommender = None
_knn_classifier = None  # KNN trained on labeled segments — used for accurate predict
_knn_scaler = None


def _load_data():
    global _df_clean, _segments_df, _seg_model, _recommender, _knn_classifier, _knn_scaler

    csv_path = "data/online_retail_clean.csv"
    if os.path.exists(csv_path):
        _df_clean = pd.read_csv(csv_path)
        _df_clean["InvoiceDate"] = pd.to_datetime(_df_clean["InvoiceDate"])
        print("✅  Retail CSV loaded")
    else:
        print("⚠️  online_retail_clean.csv not found")

    seg_csv = "data/customer_segments.csv"
    if os.path.exists(seg_csv):
        _segments_df = pd.read_csv(seg_csv)
        print("✅  Segments CSV loaded")

    # Segmentation model — train on-the-fly if pkl missing
    model_path = "models/customer_segmentation_kmeans.pkl"
    scaler_path = "models/rfm_scaler.pkl"
    _seg_model = CustomerSegmentationModel()
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        _seg_model.load(model_path, scaler_path)
        print("✅  KMeans model loaded from disk")
    elif _segments_df is not None and {"Recency", "Frequency", "Monetary"}.issubset(_segments_df.columns):
        print("⚠️  KMeans pkl missing — training on-the-fly from segments CSV…")
        _seg_model.fit(_segments_df)
        os.makedirs("models", exist_ok=True)
        _seg_model.save(model_path, scaler_path)
        print("✅  KMeans trained & saved")

    # ── KNN Classifier: trained on labeled CSV for accurate predict endpoint ──
    # KMeans cluster IDs don't always align with semantic segment names after
    # loading from disk, so we use a supervised KNN (99.35% accuracy) instead.
    knn_path = "models/knn_segment_classifier.pkl"
    knn_scaler_path = "models/knn_segment_scaler.pkl"
    if os.path.exists(knn_path) and os.path.exists(knn_scaler_path):
        _knn_classifier = joblib.load(knn_path)
        _knn_scaler = joblib.load(knn_scaler_path)
        print("✅  KNN classifier loaded from disk")
    elif _segments_df is not None and {"Recency", "Frequency", "Monetary", "Customer Segment"}.issubset(_segments_df.columns):
        X = _segments_df[["Recency", "Frequency", "Monetary"]].values
        y = _segments_df["Customer Segment"].values
        _knn_scaler = StandardScaler()
        Xs = _knn_scaler.fit_transform(X)
        _knn_classifier = KNeighborsClassifier(n_neighbors=5, weights="distance")
        _knn_classifier.fit(Xs, y)
        os.makedirs("models", exist_ok=True)
        joblib.dump(_knn_classifier, knn_path)
        joblib.dump(_knn_scaler, knn_scaler_path)
        print("✅  KNN classifier trained & saved")
    else:
        print("⚠️  KNN classifier could not be built")

    rules_path = "data/market_basket_rules.csv"
    _recommender = ProductRecommender(rules_path)
    print("✅  Recommender loaded")


# ══════════════════════════════════════════════════════════════════════════════
# AUTH ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/auth/register", methods=["POST"])
@app.route("/api/auth/signup", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not name or not email or not password:
        return jsonify({"error": "Name, email, and password are required."}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters."}), 400

    db = get_db()
    existing = db.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
    if existing:
        return jsonify({"error": "An account with this email already exists."}), 409

    hashed = generate_password_hash(password)
    cur = db.execute(
        "INSERT INTO users (name, email, password, role, created) VALUES (?,?,?,?,?)",
        (name, email, hashed, "user", datetime.datetime.utcnow().isoformat()),
    )
    db.commit()
    user_id = cur.lastrowid

    token = make_token(user_id, email, name, "user")
    return jsonify({"token": token, "name": name, "email": email, "role": "user"}), 201


@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400

    db = get_db()
    user = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    if user is None or not check_password_hash(user["password"], password):
        return jsonify({"error": "Invalid email or password."}), 401

    token = make_token(user["id"], user["email"], user["name"], user["role"])
    return jsonify({
        "token": token,
        "name": user["name"],
        "email": user["email"],
        "role": user["role"],
    }), 200


@app.route("/api/auth/logout", methods=["POST"])
@require_auth
def logout():
    # JWT is stateless; logout is handled client-side by deleting the token.
    return jsonify({"message": "Logged out successfully."}), 200


@app.route("/api/auth/me", methods=["GET"])
@require_auth
def me():
    return jsonify(g.current_user), 200


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/dashboard/metrics", methods=["GET"])
@require_auth
def dashboard_metrics():
    if _df_clean is None:
        return jsonify({"error": "Dataset not available."}), 503

    total_rev = float(_df_clean["Sales"].sum())
    total_orders = int(_df_clean["Invoice"].nunique())
    total_customers = int(_df_clean["Customer ID"].nunique())
    avg_order_value = total_rev / total_orders if total_orders > 0 else 0.0

    # Month-over-month revenue change
    monthly = _df_clean.set_index("InvoiceDate").resample("MS")["Sales"].sum()
    mom_change = None
    if len(monthly) >= 2:
        last, prev = monthly.iloc[-1], monthly.iloc[-2]
        mom_change = round(((last - prev) / prev) * 100, 1) if prev != 0 else 0.0

    return jsonify({
        "total_revenue": round(total_rev, 2),
        "total_orders": total_orders,
        "total_customers": total_customers,
        "avg_order_value": round(avg_order_value, 2),
        "mom_change_pct": mom_change,
    })


@app.route("/api/dashboard/monthly-sales", methods=["GET"])
@require_auth
def dashboard_monthly_sales():
    if _df_clean is None:
        return jsonify({"error": "Dataset not available."}), 503

    monthly = (
        _df_clean.set_index("InvoiceDate")
        .resample("MS")["Sales"]
        .sum()
        .reset_index()
    )
    monthly["InvoiceDate"] = monthly["InvoiceDate"].dt.strftime("%Y-%m")
    return jsonify(monthly.rename(columns={"InvoiceDate": "month", "Sales": "revenue"}).to_dict(orient="records"))


@app.route("/api/dashboard/top-products", methods=["GET"])
@require_auth
def dashboard_top_products():
    if _df_clean is None:
        return jsonify({"error": "Dataset not available."}), 503

    top = (
        _df_clean.groupby("Description")
        .agg(revenue=("Sales", "sum"), quantity=("Quantity", "sum"))
        .sort_values("revenue", ascending=False)
        .head(10)
        .reset_index()
    )
    top["revenue"] = top["revenue"].round(2)
    return jsonify(top.to_dict(orient="records"))


@app.route("/api/dashboard/top-countries", methods=["GET"])
@require_auth
def dashboard_top_countries():
    if _df_clean is None:
        return jsonify({"error": "Dataset not available."}), 503

    top = (
        _df_clean.groupby("Country")["Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    top.columns = ["country", "revenue"]
    top["revenue"] = top["revenue"].round(2)
    return jsonify(top.to_dict(orient="records"))


# ══════════════════════════════════════════════════════════════════════════════
# SEGMENTATION ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/segmentation/overview", methods=["GET"])
@require_auth
def segmentation_overview():
    if _segments_df is None:
        return jsonify({"error": "Segments data not available."}), 503

    # Segment counts
    counts = (
        _segments_df["Customer Segment"]
        .value_counts()
        .reset_index()
    )
    counts.columns = ["segment", "count"]

    # RFM profile per segment
    needed_cols = ["Recency", "Frequency", "Monetary", "Customer Segment"]
    available = [c for c in needed_cols if c in _segments_df.columns]
    profile_records = []
    if len(available) == 4:
        profile = (
            _segments_df.groupby("Customer Segment")
            .agg(
                avg_recency=("Recency", "mean"),
                avg_frequency=("Frequency", "mean"),
                avg_monetary=("Monetary", "mean"),
                count=("Customer Segment", "count"),
            )
            .round(2)
            .reset_index()
        )
        profile.rename(columns={"Customer Segment": "segment"}, inplace=True)
        profile_records = profile.to_dict(orient="records")

    return jsonify({
        "segment_counts": counts.to_dict(orient="records"),
        "rfm_profile": profile_records,
    })


@app.route("/api/segmentation/predict", methods=["POST"])
@require_auth
def segmentation_predict():
    data = request.get_json(silent=True) or {}
    try:
        recency = float(data["recency"])
        frequency = float(data["frequency"])
        monetary = float(data["monetary"])
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "recency, frequency, and monetary must be numbers."}), 400

    # Use the KNN classifier (99.35% accuracy on labeled data) for accurate segment assignment
    if _knn_classifier is not None and _knn_scaler is not None:
        input_vec = np.array([[recency, frequency, monetary]])
        scaled_vec = _knn_scaler.transform(input_vec)
        segment_name = str(_knn_classifier.predict(scaled_vec)[0])
        proba = _knn_classifier.predict_proba(scaled_vec)[0]
        confidence = round(float(proba.max()) * 100, 1)

        # Determine cluster_id from CSV mapping
        if _segments_df is not None and "Customer Segment" in _segments_df.columns and "Cluster" in _segments_df.columns:
            cluster_map = _segments_df.groupby("Customer Segment")["Cluster"].agg(lambda x: x.mode().iloc[0]).to_dict()
            cluster_id = int(cluster_map.get(segment_name, 0))
        else:
            cluster_id = 0

        strategy = DEFAULT_SEGMENT_ACTIONS.get(segment_name, "Engage with relevant promotions.")
        return jsonify({
            "cluster_id": cluster_id,
            "segment": segment_name,
            "strategy": strategy,
            "confidence": confidence,
        })

    # Fallback to KMeans if KNN not available
    if _seg_model is None or _seg_model.scaler is None:
        return jsonify({"error": "Segmentation model not available."}), 503

    cluster_id, segment_name, strategy = _seg_model.predict_single(recency, frequency, monetary)
    return jsonify({
        "cluster_id": cluster_id,
        "segment": segment_name,
        "strategy": strategy,
        "confidence": 80,
    })


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOMER DETAILS & SEARCH ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/customers/search", methods=["GET"])
@require_auth
def customers_search():
    if _segments_df is None:
        return jsonify({"error": "Segments data not available."}), 503

    q = (request.args.get("q") or "").strip()
    segment = (request.args.get("segment") or "").strip()
    try:
        limit = min(int(request.args.get("limit", 20)), 100)
    except ValueError:
        limit = 20

    res_df = _segments_df.copy()
    if q:
        res_df = res_df[res_df["Customer ID"].astype(str).str.contains(q, case=False, na=False)]
    if segment:
        res_df = res_df[res_df["Customer Segment"].str.lower() == segment.lower()]

    res_df = res_df.head(limit)
    return jsonify(res_df.to_dict(orient="records"))


@app.route("/api/customers/list", methods=["GET"])
@require_auth
def customers_list():
    if _segments_df is None:
        return jsonify({"error": "Segments data not available."}), 503

    try:
        limit = min(int(request.args.get("limit", 20)), 100)
    except ValueError:
        limit = 20

    return jsonify(_segments_df.head(limit).to_dict(orient="records"))


@app.route("/api/customers/<int:customer_id>", methods=["GET"])
@require_auth
def customer_details(customer_id):
    if _df_clean is None or _segments_df is None:
        return jsonify({"error": "Dataset not available."}), 503

    seg_row = _segments_df[_segments_df["Customer ID"] == customer_id]
    cust_tx = _df_clean[_df_clean["Customer ID"] == customer_id]

    if seg_row.empty:
        if cust_tx.empty:
            return jsonify({"error": f"Customer ID {customer_id} not found."}), 404
        max_date = _df_clean["InvoiceDate"].max()
        recency = int((max_date - cust_tx["InvoiceDate"].max()).days)
        frequency = int(cust_tx["Invoice"].nunique())
        monetary = round(float(cust_tx["Sales"].sum()), 2)
        if _seg_model and _seg_model.scaler:
            cluster_id, segment, strategy = _seg_model.predict_single(recency, frequency, monetary)
        else:
            cluster_id, segment, strategy = 0, "Regular Customers", "Standard engagement"
    else:
        r = seg_row.iloc[0]
        recency = int(r["Recency"])
        frequency = int(r["Frequency"])
        monetary = round(float(r["Monetary"]), 2)
        cluster_id = int(r["Cluster"])
        segment = str(r["Customer Segment"])
        strategy = DEFAULT_SEGMENT_ACTIONS.get(segment, "Engage with relevant promotions.")

    country = str(cust_tx["Country"].iloc[0]) if not cust_tx.empty and "Country" in cust_tx.columns else "United Kingdom"
    total_sales = monetary
    total_orders = frequency
    avg_order_value = round(total_sales / total_orders, 2) if total_orders > 0 else 0.0

    category_affinity = []
    top_product_name = None
    if not cust_tx.empty:
        item_sales = (
            cust_tx.groupby("Description")
            .agg(sales=("Sales", "sum"), quantity=("Quantity", "sum"))
            .sort_values("sales", ascending=False)
            .head(5)
            .reset_index()
        )
        if not item_sales.empty:
            top_product_name = str(item_sales.iloc[0]["Description"])
        for _, it in item_sales.iterrows():
            pct = round((float(it["sales"]) / total_sales) * 100, 1) if total_sales > 0 else 0.0
            category_affinity.append({
                "item": str(it["Description"]),
                "sales": round(float(it["sales"]), 2),
                "quantity": int(it["quantity"]),
                "percentage": pct,
            })

    spend_trajectory = []
    if not cust_tx.empty:
        monthly = cust_tx.set_index("InvoiceDate").resample("MS")["Sales"].sum().reset_index()
        for _, m in monthly.iterrows():
            spend_trajectory.append({
                "period": m["InvoiceDate"].strftime("%b %Y"),
                "sales": round(float(m["Sales"]), 2),
            })
    if len(spend_trajectory) < 2 and not cust_tx.empty:
        spend_trajectory = [{
            "period": cust_tx["InvoiceDate"].dt.strftime("%b %Y").iloc[0],
            "sales": total_sales
        }]

    recent_orders = []
    if not cust_tx.empty:
        invoices = (
            cust_tx.groupby(["Invoice", "InvoiceDate"])
            .agg(items_count=("Description", "count"), total=("Sales", "sum"))
            .sort_values("InvoiceDate", ascending=False)
            .head(6)
            .reset_index()
        )
        for _, inv in invoices.iterrows():
            recent_orders.append({
                "invoice": str(inv["Invoice"]),
                "date": inv["InvoiceDate"].strftime("%Y-%m-%d"),
                "items_count": int(inv["items_count"]),
                "total": round(float(inv["total"]), 2),
            })

    recommendations = []
    if top_product_name and _recommender:
        recommendations = _recommender.recommend(top_product_name, top_n=3)

    return jsonify({
        "customer_id": customer_id,
        "name": f"Customer #{customer_id}",
        "country": country,
        "segment": segment,
        "cluster_id": cluster_id,
        "strategy": strategy,
        "recency_days": recency,
        "frequency": frequency,
        "monetary": monetary,
        "avg_order_value": avg_order_value,
        "category_affinity": category_affinity,
        "spend_trajectory": spend_trajectory,
        "recent_orders": recent_orders,
        "top_product": top_product_name,
        "recommendations": recommendations,
    })


# ══════════════════════════════════════════════════════════════════════════════
# SALES FORECASTING ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/sales/monthly-actuals", methods=["GET"])
@require_auth
def sales_monthly_actuals():
    if _df_clean is None:
        return jsonify({"error": "Dataset not available."}), 503

    monthly_df = prepare_monthly_timeseries(_df_clean)
    monthly_df["Date"] = monthly_df["Date"].dt.strftime("%Y-%m")
    monthly_df.rename(columns={"Date": "month", "Sales": "revenue"}, inplace=True)
    return jsonify(monthly_df.to_dict(orient="records"))


@app.route("/api/sales/forecast", methods=["GET"])
@require_auth
def sales_forecast():
    if _df_clean is None:
        return jsonify({"error": "Dataset not available."}), 503

    try:
        periods = int(request.args.get("periods", 6))
        periods = max(1, min(periods, 24))
    except ValueError:
        periods = 6

    monthly_df = prepare_monthly_timeseries(_df_clean)
    forecast_df = forecast_linear_trend_with_seasonality(monthly_df, periods=periods)
    forecast_df["Date"] = forecast_df["Date"].dt.strftime("%Y-%m")
    forecast_df.rename(columns={"Date": "month"}, inplace=True)

    # Replace NaN with None for JSON
    forecast_df = forecast_df.where(pd.notnull(forecast_df), None)
    return jsonify(forecast_df.to_dict(orient="records"))


# ══════════════════════════════════════════════════════════════════════════════
# RECOMMENDATION ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/recommendations/products", methods=["GET"])
@require_auth
def recommendations_products():
    if _recommender is None:
        return jsonify({"error": "Recommender not available."}), 503
    products = _recommender.get_all_products()
    return jsonify({"products": products, "count": len(products)})


@app.route("/api/recommendations/recommend", methods=["POST"])
@require_auth
def recommendations_recommend():
    if _recommender is None:
        return jsonify({"error": "Recommender not available."}), 503

    data = request.get_json(silent=True) or {}
    product = (data.get("product") or "").strip()
    top_n = int(data.get("top_n", 5))

    if not product:
        return jsonify({"error": "product is required."}), 400

    recs = _recommender.recommend(product, top_n=top_n)
    return jsonify({"product": product, "recommendations": recs})


# ══════════════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "data_loaded": _df_clean is not None,
        "segments_loaded": _segments_df is not None,
        "model_ready": _seg_model is not None and _seg_model.scaler is not None,
        "knn_classifier_ready": _knn_classifier is not None,
        "recommender_ready": _recommender is not None and not _recommender.rules_df.empty,
    })


# ══════════════════════════════════════════════════════════════════════════════
# FRONTEND PAGE & STATIC ASSET ROUTES
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

@app.route("/")
def index():
    return redirect("/login")

@app.route("/login")
@app.route("/signin")
@app.route("/retailiq_login_silk_white_bg/code.html")
def serve_login():
    return send_from_directory(os.path.join(FRONTEND_DIR, "retailiq_login_silk_white_bg"), "code.html")

@app.route("/signup")
@app.route("/sign")
@app.route("/register")
@app.route("/retailiq_sign_up_purple_glow_centered/code.html")
def serve_signup():
    return send_from_directory(os.path.join(FRONTEND_DIR, "retailiq_sign_up_purple_glow_centered"), "code.html")

@app.route("/dashboard")
@app.route("/retailiq_dashboard_theme_toggle_purple_glow/code.html")
def serve_dashboard():
    return send_from_directory(os.path.join(FRONTEND_DIR, "retailiq_dashboard_theme_toggle_purple_glow"), "code.html")

@app.route("/sales")
@app.route("/retailiq_sales_analytics_theme_toggle_pie_chart/code.html")
def serve_sales():
    return send_from_directory(os.path.join(FRONTEND_DIR, "retailiq_sales_analytics_theme_toggle_pie_chart"), "code.html")

@app.route("/segmentation")
@app.route("/retailiq_customer_segmentation_theme_toggle_purple_glow/code.html")
def serve_segmentation():
    return send_from_directory(os.path.join(FRONTEND_DIR, "retailiq_customer_segmentation_theme_toggle_purple_glow"), "code.html")

@app.route("/customer-details")
@app.route("/retailiq_customer_details_theme_toggle_purple_glow/code.html")
def serve_customer_details():
    return send_from_directory(os.path.join(FRONTEND_DIR, "retailiq_customer_details_theme_toggle_purple_glow"), "code.html")

@app.route("/<folder>/code.html")
def serve_folder_code(folder):
    target = os.path.join(FRONTEND_DIR, folder)
    if os.path.isdir(target):
        return send_from_directory(target, "code.html")
    return "Not Found", 404

@app.route("/frontend/<path:filename>")
def serve_frontend_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)


# ══════════════════════════════════════════════════════════════════════════════
# STARTUP
# ══════════════════════════════════════════════════════════════════════════════

def setup_app():
    with app.app_context():
        init_db()
    _load_data()

# Initialize upon import/run
setup_app()

if __name__ == "__main__":
    print("\n🚀  RetailIQ API & Web App running on http://localhost:5000\n")
    app.run(debug=True, port=5000, use_reloader=False)
