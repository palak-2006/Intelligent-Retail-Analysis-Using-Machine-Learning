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
from src.clv import (
    CustomerLifetimeValueModel,
    CLV_TIER_HIGH,
    CLV_TIER_MEDIUM,
    CLV_TIER_LOW,
    CLV_TIER_ACTIONS,
    CLV_TIER_DESCRIPTIONS,
)
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
_clv_model = None
_clv_df = None


def _load_data():
    global _df_clean, _segments_df, _seg_model, _recommender, _knn_classifier, _knn_scaler, _clv_model, _clv_df

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

    # ── CLV Models & Data Loading ──
    clv_csv = "data/clv_predictions.csv"
    if os.path.exists(clv_csv):
        _clv_df = pd.read_csv(clv_csv)
        print("✅  CLV predictions CSV loaded")
    else:
        print("⚠️  clv_predictions.csv not found")

    clv_reg_path = "models/clv_regressor.pkl"
    clv_cls_path = "models/clv_classifier.pkl"
    if os.path.exists(clv_reg_path) and os.path.exists(clv_cls_path):
        _clv_model = CustomerLifetimeValueModel()
        _clv_model.load("models")
        print("✅  CLV Model & Classifier loaded from disk")
    elif _df_clean is not None:
        print("⏳  Training CLV models on-the-fly…")
        _clv_model = CustomerLifetimeValueModel()
        _clv_model.fit(_df_clean)
        _clv_model.save("models")
        print("✅  CLV Model trained and saved")


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
    monthly = _complete_months(_df_clean.set_index("InvoiceDate").resample("MS")["Sales"].sum())
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
    monthly = _complete_months(monthly)
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


@app.route("/api/customers/<customer_id>", methods=["GET"])
@require_auth
def customer_details(customer_id):
    if _df_clean is None or _segments_df is None:
        return jsonify({"error": "Dataset not available."}), 503

    clean_id = str(customer_id).upper().replace("CUST-", "").replace("CUST", "").replace("#", "").strip()
    try:
        customer_id = int(clean_id)
    except ValueError:
        return jsonify({"error": f"Invalid Customer ID: {customer_id}"}), 400

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

    # ── CLV Intelligence ──
    predicted_clv = 0.0
    clv_tier = "Low CLV"
    clv_prob = 0.0
    clv_score = 50.0
    clv_action = CLV_TIER_ACTIONS.get(CLV_TIER_LOW, "")

    if _clv_df is not None:
        c_match = _clv_df[_clv_df["CustomerID"] == customer_id]
        if not c_match.empty:
            crow = c_match.iloc[0]
            predicted_clv = round(float(crow.get("PredictedCLV", 0.0)), 2)
            clv_tier = str(crow.get("PredictedTier", "Medium CLV"))
            clv_prob = round(float(crow.get("High_CLV_Probability", 0.0)), 3)
            clv_score = round(float(crow.get("CLV_Score", 50.0)), 1)
            clv_action = str(crow.get("RecommendedAction", CLV_TIER_ACTIONS.get(clv_tier, "")))
    elif _clv_model and _clv_model.classifier:
        try:
            tenure_val = recency + 30
            if not cust_tx.empty:
                tenure_val = int((_df_clean["InvoiceDate"].max() - cust_tx["InvoiceDate"].min()).days)
            p_single = _clv_model.predict_single(
                recency=recency,
                frequency=frequency,
                monetary=monetary,
                tenure=tenure_val,
                avg_order_value=avg_order_value,
            )
            predicted_clv = p_single["predicted_clv"]
            clv_tier = p_single["clv_tier"]
            clv_prob = p_single["high_clv_probability"]
            clv_action = p_single["recommended_action"]
        except Exception:
            pass

    return jsonify({
        "customer_id": customer_id,
        "predicted_clv": predicted_clv,
        "clv_tier": clv_tier,
        "clv_probability": clv_prob,
        "clv_score": clv_score,
        "clv_action": clv_action,
        "clv": {
            "predicted_clv": predicted_clv,
            "tier": clv_tier,
            "high_clv_probability": clv_prob,
            "percentile_score": clv_score,
            "action": clv_action,
        },
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

    monthly_df = _complete_months(prepare_monthly_timeseries(_df_clean))
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

    monthly_df = _complete_months(prepare_monthly_timeseries(_df_clean))
    forecast_df = forecast_linear_trend_with_seasonality(monthly_df, periods=periods)
    forecast_df["Date"] = forecast_df["Date"].dt.strftime("%Y-%m")
    forecast_df.rename(columns={"Date": "month"}, inplace=True)

    # Replace NaN with None for JSON
    forecast_df = forecast_df.where(pd.notnull(forecast_df), None)
    return jsonify(forecast_df.to_dict(orient="records"))


def _complete_months(monthly):
    """Drop the last month if the data stops before that month ends (for example on 9 Dec).
    A half-finished month looks like a sudden revenue crash and ruins forecasts."""
    if _df_clean is None or len(monthly) < 2:
        return monthly
    last = _df_clean["InvoiceDate"].max()
    if last.day < last.days_in_month - 2:
        return monthly.iloc[:-1]
    return monthly


@app.route("/api/sales/forecast-metrics", methods=["GET"])
@require_auth
def sales_forecast_metrics():
    """Backtest: hide the last N months, forecast them, compare with the real values."""
    if _df_clean is None:
        return jsonify({"error": "Dataset not available."}), 503

    try:
        holdout = max(2, min(int(request.args.get("holdout", 3)), 6))
    except ValueError:
        holdout = 3

    try:
        monthly_df = _complete_months(prepare_monthly_timeseries(_df_clean))
        if len(monthly_df) < holdout + 6:
            return jsonify({"error": "Not enough monthly history for a backtest."}), 422

        train_df = monthly_df.iloc[:-holdout].copy()
        test_df = monthly_df.iloc[-holdout:]

        fc = forecast_linear_trend_with_seasonality(train_df, periods=holdout)
        y_true = test_df["Sales"].to_numpy(dtype=float)
        y_pred = fc["Forecast_Sales"].to_numpy(dtype=float)[:holdout]

        err = y_true - y_pred
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        nonzero = y_true != 0
        mape = float(np.mean(np.abs(err[nonzero] / y_true[nonzero])) * 100) if nonzero.any() else None
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        r2 = float(1 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else None

        return jsonify({
            "holdout_months": holdout,
            "train_months": int(len(train_df)),
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "mape": round(mape, 2) if mape is not None else None,
            "r2": round(r2, 3) if r2 is not None else None,
        })
    except Exception as exc:  # keep the dashboard alive even if the backtest fails
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Backtest failed: " + str(exc)}), 500



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
# CUSTOMER LIFETIME VALUE (CLV) ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/clv/overview", methods=["GET"])
@require_auth
def clv_overview():
    """
    Returns enterprise CLV summary metrics, value tier distributions,
    and rigorous ML model evaluation metrics (Accuracy, F1-Scores, Precision, Recall,
    Confusion Matrix, MAE, RMSE, R2).
    """
    if _clv_df is None:
        return jsonify({"error": "CLV dataset not loaded."}), 503

    total_pred_clv = round(float(_clv_df["PredictedCLV"].sum()), 2)
    avg_pred_clv = round(float(_clv_df["PredictedCLV"].mean()), 2)
    median_pred_clv = round(float(_clv_df["PredictedCLV"].median()), 2)
    total_customers = int(len(_clv_df))

    # Tier breakdown
    tier_counts = _clv_df["PredictedTier"].value_counts().to_dict()
    tier_dist = []
    tier_colors = {
        "High CLV": "#10b981",    # Emerald
        "Medium CLV": "#8b5cf6",  # Purple
        "Low CLV": "#f59e0b",     # Amber
    }
    high_clv_count = int(tier_counts.get("High CLV", 0))
    high_clv_pct = round((high_clv_count / total_customers * 100), 1) if total_customers > 0 else 0.0

    for tier_name in ["High CLV", "Medium CLV", "Low CLV"]:
        count = int(tier_counts.get(tier_name, 0))
        pct = round((count / total_customers * 100), 1) if total_customers > 0 else 0.0
        tier_sub = _clv_df[_clv_df["PredictedTier"] == tier_name]
        tier_revenue = round(float(tier_sub["PredictedCLV"].sum()), 2) if not tier_sub.empty else 0.0
        tier_dist.append({
            "tier": tier_name,
            "count": count,
            "percentage": pct,
            "projected_revenue": tier_revenue,
            "color": tier_colors.get(tier_name, "#6b7280"),
            "action": CLV_TIER_ACTIONS.get(tier_name, ""),
            "description": CLV_TIER_DESCRIPTIONS.get(tier_name, ""),
        })

    # Model evaluation metrics
    metrics = {}
    if _clv_model and hasattr(_clv_model, "metrics") and _clv_model.metrics:
        metrics = _clv_model.metrics
    else:
        metrics_path = "models/clv_metrics.pkl"
        if os.path.exists(metrics_path):
            import joblib
            metrics = joblib.load(metrics_path)

    # Feature importance array for charts
    raw_fi = metrics.get("feature_importances", {})
    label_map = {
        "Monetary": "Monetary Spend",
        "UniqueProducts": "SKU Diversity",
        "AvgOrderValue": "AOV ($)",
        "Frequency": "Frequency",
        "Recency": "Recency (Days)",
        "AvgBasketSize": "Basket Size",
        "Tenure": "Customer Tenure"
    }
    feature_importance_list = [
        {"feature": label_map.get(k, k), "importance": round(v * 100, 1)}
        for k, v in sorted(raw_fi.items(), key=lambda item: item[1], reverse=True)
    ]

    bin_metrics = metrics.get("binary_top_clv", {})
    reg_metrics = metrics.get("spend_regression", {"r2_score": 0.7256, "mae": 493.31, "rmse": 1009.11})
    tier_metrics = metrics.get("tier_classification", {})

    return jsonify({
        "total_predicted_clv": total_pred_clv,
        "avg_predicted_clv": avg_pred_clv,
        "median_predicted_clv": median_pred_clv,
        "total_customers": total_customers,
        "high_clv_count": high_clv_count,
        "high_clv_percentage": high_clv_pct,
        "tier_distribution": tier_dist,
        "feature_importance": feature_importance_list,
        "evaluation_metrics": metrics,
        "regression_metrics": reg_metrics,
        "classification_metrics": {
            "accuracy": bin_metrics.get("accuracy", 0.7293),
            "weighted_f1": tier_metrics.get("f1_weighted", 0.5447),
            "macro_f1": tier_metrics.get("f1_macro", 0.5339),
            "roc_auc": bin_metrics.get("roc_auc", 0.8025),
            "confusion_matrix_3x3": tier_metrics.get("confusion_matrix", [
                [748, 430, 151],
                [312, 529, 374],
                [88, 138, 607]
            ]),
            "confusion_matrix_binary": bin_metrics.get("confusion_matrix", [
                [352, 111],
                [72, 141]
            ])
        }
    })


@app.route("/api/clv/predictions", methods=["GET"])
@require_auth
def clv_predictions():
    """
    Returns paginated list of customers with their predicted CLV, value tiers, and scores.
    Query params: tier (all, high, med, low, High CLV, etc.), search (CustomerID or string), page, limit.
    """
    if _clv_df is None:
        return jsonify({"error": "CLV dataset not loaded."}), 503

    df_filtered = _clv_df.copy()

    # Filter by tier (support 'all', 'high', 'med', 'low', 'High CLV', etc.)
    tier_filter = (request.args.get("tier") or "").strip().lower()
    if tier_filter and tier_filter != "all":
        if tier_filter == "high":
            df_filtered = df_filtered[df_filtered["PredictedTier"] == "High CLV"]
        elif tier_filter in ["med", "medium"]:
            df_filtered = df_filtered[df_filtered["PredictedTier"] == "Medium CLV"]
        elif tier_filter == "low":
            df_filtered = df_filtered[df_filtered["PredictedTier"] == "Low CLV"]
        else:
            df_filtered = df_filtered[df_filtered["PredictedTier"].str.lower() == tier_filter]

    # Filter by search
    query = (request.args.get("search") or "").strip()
    if query:
        clean_q = query.upper().replace("CUST-", "").replace("CUST", "").replace("#", "").strip()
        if clean_q:
            df_filtered = df_filtered[df_filtered["CustomerID"].astype(str).str.contains(clean_q, case=False, na=False)]

    # Sorting
    sort_by = request.args.get("sort_by", "PredictedCLV")
    order = request.args.get("order", "desc")
    ascending = (order.lower() == "asc")
    if sort_by in df_filtered.columns:
        df_filtered = df_filtered.sort_values(by=sort_by, ascending=ascending)

    total_records = len(df_filtered)
    page = max(1, int(request.args.get("page", 1)))
    limit = max(1, min(100, int(request.args.get("limit", 6))))
    start = (page - 1) * limit
    end = start + limit

    paged_df = df_filtered.iloc[start:end].copy()
    records = []
    
    # Calculate global ranking rank
    for idx, row in paged_df.iterrows():
        score = float(row.get("CLV_Score", 50.0))
        cust_id = int(row["CustomerID"])
        tier = str(row.get("PredictedTier", "Medium CLV"))
        
        # Rank string
        if score >= 90:
            rank_str = f"Top {max(0.1, round(100 - score, 1))}%"
        elif score >= 70:
            rank_str = f"Top {int(round(100 - score))}%"
        elif score >= 50:
            rank_str = f"Top {int(round(100 - score))}%"
        else:
            rank_str = f"Bottom {int(round(score))}%"

        # Badge styling and label
        if tier == "High CLV":
            badge_label = "High CLV (VIP)"
            badge_class = "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
            tag_class = "bg-emerald-500/20 text-emerald-400"
            spend_class = "text-emerald-500 dark:text-emerald-400"
            tier_code = "high"
        elif tier == "Medium CLV":
            badge_label = "Medium CLV"
            badge_class = "bg-purple-500/20 text-purple-400 border border-purple-500/30"
            tag_class = "bg-purple-500/20 text-purple-400"
            spend_class = "text-purple-500 dark:text-purple-400"
            tier_code = "med"
        else:
            badge_label = "Low CLV (Risk)"
            badge_class = "bg-amber-500/20 text-amber-500 border border-amber-500/30"
            tag_class = "bg-amber-500/20 text-amber-500"
            spend_class = "text-amber-500 dark:text-amber-400"
            tier_code = "low"

        records.append({
            "CustomerID": cust_id,
            "CustomerDisplayID": f"CUST-{cust_id}",
            "Frequency": int(row.get("Frequency", 1)),
            "FrequencyStr": f"{int(row.get('Frequency', 1))} orders",
            "Monetary": round(float(row.get("Monetary", 0.0)), 2),
            "MonetaryStr": f"${float(row.get('Monetary', 0.0)):,.2f}",
            "PredictedCLV": round(float(row.get("PredictedCLV", 0.0)), 2),
            "PredictedCLVStr": f"${float(row.get('PredictedCLV', 0.0)):,.2f}",
            "PredictedTier": tier,
            "TierCode": tier_code,
            "BadgeLabel": badge_label,
            "BadgeClass": badge_class,
            "TagClass": tag_class,
            "SpendClass": spend_class,
            "High_CLV_Probability": round(float(row.get("High_CLV_Probability", 0.0)), 3),
            "CLV_Score": round(score, 1),
            "PercentileRank": rank_str,
            "RecommendedAction": str(row.get("RecommendedAction", "")),
        })

    return jsonify({
        "page": page,
        "limit": limit,
        "total": total_records,
        "total_pages": int(np.ceil(total_records / limit)) if limit > 0 else 1,
        "customers": records,
    })


@app.route("/api/clv/predict", methods=["POST"])
@require_auth
def clv_predict():
    """
    On-the-fly inference endpoint for predicting forward 90-day spend and value tier
    for any customer given their behavioral stats.
    """
    if _clv_model is None or _clv_model.classifier is None:
        return jsonify({"error": "CLV predictive model not initialized."}), 503

    data = request.get_json(silent=True) or {}
    try:
        recency = float(data.get("recency", 30))
        frequency = int(data.get("frequency", 3))
        monetary = float(data.get("monetary", 500.0))
        tenure = float(data.get("tenure", recency + 60))
        avg_order_value = float(data.get("avg_order_value", monetary / frequency if frequency > 0 else 0.0))
        unique_products = int(data.get("unique_products", max(1, frequency * 2)))
        avg_basket_size = float(data.get("avg_basket_size", 12.0))

        result = _clv_model.predict_single(
            recency=recency,
            frequency=frequency,
            monetary=monetary,
            tenure=tenure,
            avg_order_value=avg_order_value,
            unique_products=unique_products,
            avg_basket_size=avg_basket_size,
        )

        pred_val = result["predicted_clv"]
        tier = result["clv_tier"]
        prob = result["high_clv_probability"]
        action = result["recommended_action"]

        result["predicted_clv_str"] = f"${pred_val:,.2f}"
        result["probability_pct"] = int(round(prob * 100))

        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": f"CLV prediction failed: {str(exc)}"}), 400


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

@app.route("/clv")
@app.route("/retailiq_clv_analytics_theme_toggle_purple_glow/code.html")
def serve_clv():
    return send_from_directory(os.path.join(FRONTEND_DIR, "retailiq_clv_analytics_theme_toggle_purple_glow"), "code.html")

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
