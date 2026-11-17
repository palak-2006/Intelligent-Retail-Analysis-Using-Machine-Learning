"""
Customer Lifetime Value (CLV) Prediction Module
Handles CLV feature engineering, predictive regression (future spend),
and value-tier classification (High/Medium/Low CLV) with full evaluation metrics
(Accuracy, F1-Score, Precision, Recall, Confusion Matrix, MAE, RMSE, R²).
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

# Standard Business CLV Value Tiers
CLV_TIER_HIGH = "High CLV"
CLV_TIER_MEDIUM = "Medium CLV"
CLV_TIER_LOW = "Low CLV"

CLV_TIER_DESCRIPTIONS = {
    CLV_TIER_HIGH: "Top-tier high spender. Premium account with highest expected future revenue.",
    CLV_TIER_MEDIUM: "Solid recurring spender with consistent repeat orders and growth potential.",
    CLV_TIER_LOW: "Low or irregular spender. Low future monetary expectation without active intervention.",
}

CLV_TIER_ACTIONS = {
    CLV_TIER_HIGH: "Assign VIP concierge, exclusive early-access previews, and priority loyalty perks.",
    CLV_TIER_MEDIUM: "Nurture with cross-sell category recommendations, bundled offers, and threshold rewards.",
    CLV_TIER_LOW: "Automate low-cost retargeting emails, discount coupons, and seasonal re-engagement campaigns.",
}

DEFAULT_FEATURE_COLS = [
    "Recency",
    "Frequency",
    "Monetary",
    "Tenure",
    "AvgOrderValue",
    "UniqueProducts",
    "AvgBasketSize",
]


def extract_clv_features(
    df: pd.DataFrame,
    cutoff_date: Optional[pd.Timestamp] = None,
    holdout_months: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits retail transaction data into an observation window and a future holdout window.
    Engineers RFM + Tenure + Basket metrics from observation period and calculates
    ground-truth future spend from the holdout period.
    """
    clean_df = df.copy()

    # Normalize column names if needed
    col_map = {}
    if "Customer ID" in clean_df.columns:
        col_map["Customer ID"] = "CustomerID"
    if "Invoice" in clean_df.columns:
        col_map["Invoice"] = "InvoiceNo"
    if "Price" in clean_df.columns:
        col_map["Price"] = "UnitPrice"
    if col_map:
        clean_df = clean_df.rename(columns=col_map)

    clean_df["InvoiceDate"] = pd.to_datetime(clean_df["InvoiceDate"])
    clean_df = clean_df.dropna(subset=["CustomerID"])
    clean_df["CustomerID"] = clean_df["CustomerID"].astype(int)

    # Filter valid sales rows
    if "TotalPrice" not in clean_df.columns:
        if "Sales" in clean_df.columns:
            clean_df["TotalPrice"] = clean_df["Sales"]
        elif "Quantity" in clean_df.columns and "UnitPrice" in clean_df.columns:
            clean_df["TotalPrice"] = clean_df["Quantity"] * clean_df["UnitPrice"]

    clean_df = clean_df[(clean_df["Quantity"] > 0) & (clean_df["TotalPrice"] > 0)]

    # Determine cutoff date
    max_date = clean_df["InvoiceDate"].max()
    if cutoff_date is None:
        window_end = max_date + pd.Timedelta(days=1)
        cutoff_date = window_end - pd.DateOffset(months=holdout_months)

    obs = clean_df[clean_df["InvoiceDate"] < cutoff_date]
    future = clean_df[clean_df["InvoiceDate"] >= cutoff_date]

    # Aggregate observation features
    features = obs.groupby("CustomerID").agg(
        last_purchase=("InvoiceDate", "max"),
        first_purchase=("InvoiceDate", "min"),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
        Quantity=("Quantity", "sum"),
        UniqueProducts=("StockCode", "nunique"),
    ).reset_index()

    features["Recency"] = (cutoff_date - features["last_purchase"]).dt.days
    features["Tenure"] = (cutoff_date - features["first_purchase"]).dt.days
    features["AvgOrderValue"] = (features["Monetary"] / features["Frequency"]).round(2)
    features["AvgBasketSize"] = (features["Quantity"] / features["Frequency"]).round(2)
    features = features.drop(columns=["last_purchase", "first_purchase"])

    # Aggregate future spend (target)
    future_spend = (
        future.groupby("CustomerID")["TotalPrice"]
        .sum()
        .reset_index()
        .rename(columns={"TotalPrice": "FutureMonetary"})
    )

    combined = features.merge(future_spend, on="CustomerID", how="left")
    combined["FutureMonetary"] = combined["FutureMonetary"].fillna(0.0).round(2)

    # Value tier definition:
    # Low: <= $100 future spend
    # Medium: $100 < spend < $700
    # High: >= $700 future spend
    def assign_tier(val: float) -> str:
        if val <= 100.0:
            return CLV_TIER_LOW
        elif val < 700.0:
            return CLV_TIER_MEDIUM
        else:
            return CLV_TIER_HIGH

    combined["CLV_Tier"] = combined["FutureMonetary"].apply(assign_tier)
    combined["Is_High_CLV"] = (combined["FutureMonetary"] >= 500.0).astype(int)

    return combined, future


class CustomerLifetimeValueModel:
    """
    Dual-task CLV Model:
    1. Regressor: Predicts future customer spend ($ amount).
    2. Classifier: Predicts customer CLV Tier (High, Medium, Low) and Top-Tier likelihood.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.feature_cols = DEFAULT_FEATURE_COLS
        self.regressor: Optional[RandomForestRegressor] = None
        self.classifier: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.metrics: Dict[str, Any] = {}

    def fit(self, df: pd.DataFrame, holdout_months: int = 3) -> "CustomerLifetimeValueModel":
        """
        Extracts features, trains both Regressor and Classifier, and evaluates comprehensive metrics.
        """
        combined, _ = extract_clv_features(df, holdout_months=holdout_months)

        X = combined[self.feature_cols]
        y_reg = combined["FutureMonetary"]
        y_tier = combined["CLV_Tier"]
        y_bin = combined["Is_High_CLV"]

        # Stratified Train/Test split for robust evaluation
        (
            X_train,
            X_test,
            y_train_t,
            y_test_t,
            y_train_r,
            y_test_r,
            y_train_b,
            y_test_b,
        ) = train_test_split(
            X,
            y_tier,
            y_reg,
            y_bin,
            test_size=0.2,
            random_state=self.random_state,
            stratify=y_tier,
        )

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # ── 1. Train Classifier (Value Tiers) ──
        self.classifier = RandomForestClassifier(
            n_estimators=250,
            max_depth=8,
            min_samples_split=6,
            class_weight="balanced",
            random_state=self.random_state,
        )
        self.classifier.fit(X_train, y_train_t)

        y_pred_tier = self.classifier.predict(X_test)

        # Multi-class tier evaluation metrics
        acc_tier = float(accuracy_score(y_test_t, y_pred_tier))
        f1_macro = float(f1_score(y_test_t, y_pred_tier, average="macro"))
        f1_weighted = float(f1_score(y_test_t, y_pred_tier, average="weighted"))
        prec_weighted = float(precision_score(y_test_t, y_pred_tier, average="weighted"))
        rec_weighted = float(recall_score(y_test_t, y_pred_tier, average="weighted"))

        tier_order = [CLV_TIER_LOW, CLV_TIER_MEDIUM, CLV_TIER_HIGH]
        cm_tier = confusion_matrix(y_test_t, y_pred_tier, labels=tier_order)

        # ── Binary Top-CLV Classifier evaluation ──
        bin_clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            class_weight="balanced",
            random_state=self.random_state,
        )
        bin_clf.fit(X_train, y_train_b)
        y_pred_b = bin_clf.predict(X_test)
        y_proba_b = bin_clf.predict_proba(X_test)[:, 1]

        acc_bin = float(accuracy_score(y_test_b, y_pred_b))
        f1_bin = float(f1_score(y_test_b, y_pred_b))
        prec_bin = float(precision_score(y_test_b, y_pred_b))
        rec_bin = float(recall_score(y_test_b, y_pred_b))
        auc_bin = float(roc_auc_score(y_test_b, y_proba_b))
        cm_bin = confusion_matrix(y_test_b, y_pred_b)

        # ── 2. Train Regressor (Continuous Future Spend) ──
        self.regressor = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_split=5,
            random_state=self.random_state,
        )
        self.regressor.fit(X_train, y_train_r)

        y_pred_reg = self.regressor.predict(X_test)
        mae = float(mean_absolute_error(y_test_r, y_pred_reg))
        rmse = float(np.sqrt(mean_squared_error(y_test_r, y_pred_reg)))
        r2 = float(r2_score(y_test_r, y_pred_reg))

        # Store comprehensive metrics
        self.metrics = {
            "tier_classification": {
                "accuracy": round(acc_tier, 4),
                "f1_macro": round(f1_macro, 4),
                "f1_weighted": round(f1_weighted, 4),
                "precision_weighted": round(prec_weighted, 4),
                "recall_weighted": round(rec_weighted, 4),
                "confusion_matrix": cm_tier.tolist(),
                "labels": tier_order,
                "classification_report": classification_report(
                    y_test_t, y_pred_tier, output_dict=True
                ),
            },
            "binary_top_clv": {
                "accuracy": round(acc_bin, 4),
                "f1_score": round(f1_bin, 4),
                "precision": round(prec_bin, 4),
                "recall": round(rec_bin, 4),
                "roc_auc": round(auc_bin, 4),
                "confusion_matrix": cm_bin.tolist(),
            },
            "spend_regression": {
                "mae": round(mae, 2),
                "rmse": round(rmse, 2),
                "r2_score": round(r2, 4),
            },
            "feature_importances": {
                feat: round(float(imp), 4)
                for feat, imp in zip(self.feature_cols, self.classifier.feature_importances_)
            },
            "training_samples": len(X_train),
            "test_samples": len(X_test),
        }

        return self

    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates predictions for a dataframe containing customer features.
        Returns predicted spend, predicted tier, high-value probability, and action recommendation.
        """
        if self.regressor is None or self.classifier is None:
            raise ValueError("Models not fitted. Call fit() or load() before predict().")

        df = features_df.copy()
        X = df[self.feature_cols]

        pred_spend = np.maximum(0.0, self.regressor.predict(X))
        pred_tier = self.classifier.predict(X)
        proba = self.classifier.predict_proba(X)

        # High CLV probability index
        classes = list(self.classifier.classes_)
        high_idx = classes.index(CLV_TIER_HIGH) if CLV_TIER_HIGH in classes else -1
        high_prob = proba[:, high_idx] if high_idx >= 0 else np.zeros(len(df))

        results = df.copy()
        results["PredictedCLV"] = np.round(pred_spend, 2)
        results["CLV_Tier"] = pred_tier
        results["High_CLV_Probability"] = np.round(high_prob, 3)
        results["RecommendedAction"] = [CLV_TIER_ACTIONS.get(t, "") for t in pred_tier]
        return results

    def predict_single(
        self,
        recency: float,
        frequency: int,
        monetary: float,
        tenure: float,
        avg_order_value: Optional[float] = None,
        unique_products: Optional[int] = None,
        avg_basket_size: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Real-time inference for a single customer.
        """
        if self.regressor is None or self.classifier is None:
            raise ValueError("Model is not initialized.")

        if avg_order_value is None:
            avg_order_value = monetary / frequency if frequency > 0 else 0.0
        if unique_products is None:
            unique_products = max(1, int(frequency * 3))
        if avg_basket_size is None:
            avg_basket_size = 15.0

        row = pd.DataFrame([{
            "Recency": float(recency),
            "Frequency": int(frequency),
            "Monetary": float(monetary),
            "Tenure": float(tenure),
            "AvgOrderValue": float(avg_order_value),
            "UniqueProducts": int(unique_products),
            "AvgBasketSize": float(avg_basket_size),
        }])[self.feature_cols]

        pred_spend = float(np.maximum(0.0, self.regressor.predict(row)[0]))
        pred_tier = str(self.classifier.predict(row)[0])
        proba = self.classifier.predict_proba(row)[0]
        classes = list(self.classifier.classes_)

        tier_probs = {c: round(float(p), 3) for c, p in zip(classes, proba)}
        high_prob = tier_probs.get(CLV_TIER_HIGH, 0.0)

        return {
            "predicted_clv": round(pred_spend, 2),
            "clv_tier": pred_tier,
            "high_clv_probability": high_prob,
            "probabilities": tier_probs,
            "tier_description": CLV_TIER_DESCRIPTIONS.get(pred_tier, ""),
            "recommended_action": CLV_TIER_ACTIONS.get(pred_tier, ""),
        }

    def save(self, models_dir: str = "models") -> None:
        """Saves fitted models and metadata to disk."""
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(self.regressor, os.path.join(models_dir, "clv_regressor.pkl"))
        joblib.dump(self.classifier, os.path.join(models_dir, "clv_classifier.pkl"))
        joblib.dump(self.scaler, os.path.join(models_dir, "clv_scaler.pkl"))
        joblib.dump(self.feature_cols, os.path.join(models_dir, "clv_features.pkl"))
        joblib.dump(self.metrics, os.path.join(models_dir, "clv_metrics.pkl"))
        print(f"[OK] CLV models successfully saved to {models_dir}/")

    def load(self, models_dir: str = "models") -> "CustomerLifetimeValueModel":
        """Loads fitted models and metadata from disk."""
        self.regressor = joblib.load(os.path.join(models_dir, "clv_regressor.pkl"))
        self.classifier = joblib.load(os.path.join(models_dir, "clv_classifier.pkl"))
        self.scaler = joblib.load(os.path.join(models_dir, "clv_scaler.pkl"))
        self.feature_cols = joblib.load(os.path.join(models_dir, "clv_features.pkl"))
        metrics_file = os.path.join(models_dir, "clv_metrics.pkl")
        if os.path.exists(metrics_file):
            self.metrics = joblib.load(metrics_file)
        return self
