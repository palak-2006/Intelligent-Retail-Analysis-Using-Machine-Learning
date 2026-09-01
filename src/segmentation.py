"""
Customer Segmentation Module
Handles KMeans model training, customer segment prediction, and segment mapping.
"""
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Dict, Tuple, Optional

# Standard business segment personas based on RFM profile
DEFAULT_SEGMENT_NAMES = {
    0: "VIP Customers",
    1: "Regular Customers",
    2: "Loyal High-Value Customers",
    3: "At-Risk Customers"
}

DEFAULT_SEGMENT_ACTIONS = {
    "VIP Customers": "Reward with exclusive perks, VIP access, and premium concierge service.",
    "Loyal High-Value Customers": "Upsell premium products, offer loyalty bonus points, and early access.",
    "Regular Customers": "Engage with personalized recommendations and seasonal discount promotions.",
    "At-Risk Customers": "Re-engage with win-back discount campaigns, surveys, and special return offers."
}


class CustomerSegmentationModel:
    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler: Optional[StandardScaler] = None
        self.kmeans: Optional[KMeans] = None
        self.segment_map: Dict[int, str] = DEFAULT_SEGMENT_NAMES

    def fit(self, rfm_df: pd.DataFrame) -> "CustomerSegmentationModel":
        """
        Fits the scaler and KMeans model on 3 RFM features: Recency, Frequency, Monetary.
        """
        features = rfm_df[["Recency", "Frequency", "Monetary"]].values
        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(features)

        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        self.kmeans.fit(scaled_features)
        return self

    def predict(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts cluster and maps segment names for an RFM DataFrame.
        """
        if self.scaler is None or self.kmeans is None:
            raise ValueError("Model is not fitted yet. Fit the model or load saved weights.")

        features = rfm_df[["Recency", "Frequency", "Monetary"]].values
        scaled = self.scaler.transform(features)
        clusters = self.kmeans.predict(scaled)

        result_df = rfm_df.copy()
        result_df["Cluster"] = clusters
        result_df["Customer Segment"] = result_df["Cluster"].map(self.segment_map).fillna("Other")
        return result_df

    def predict_single(self, recency: float, frequency: float, monetary: float) -> Tuple[int, str, str]:
        """
        Predicts the segment for a single customer given their R, F, M values.
        """
        input_data = np.array([[recency, frequency, monetary]])
        scaled = self.scaler.transform(input_data)
        cluster = int(self.kmeans.predict(scaled)[0])
        segment_name = self.segment_map.get(cluster, "Standard Customer")
        action = DEFAULT_SEGMENT_ACTIONS.get(segment_name, "Engage with relevant promotions.")
        return cluster, segment_name, action

    def evaluate_silhouette(self, rfm_df: pd.DataFrame) -> float:
        """
        Calculates the silhouette score for the clusters.
        """
        features = rfm_df[["Recency", "Frequency", "Monetary"]].values
        scaled = self.scaler.transform(features)
        clusters = self.kmeans.predict(scaled)
        return float(silhouette_score(scaled, clusters))

    def save(self, model_path: str, scaler_path: str) -> None:
        """
        Saves the trained KMeans and Scaler models to disk.
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.kmeans, model_path)
        joblib.dump(self.scaler, scaler_path)

    def load(self, model_path: str, scaler_path: str) -> "CustomerSegmentationModel":
        """
        Loads serialized KMeans and Scaler models from disk.
        """
        self.kmeans = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.n_clusters = self.kmeans.n_clusters
        return self
