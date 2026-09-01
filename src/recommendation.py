"""
Product Recommendation Module
Uses Market Basket Association Rules (FP-Growth) to recommend complementary products.
"""
import os
import pandas as pd
from typing import List, Dict, Any, Optional


class ProductRecommender:
    def __init__(self, rules_path: Optional[str] = None):
        self.rules_df: pd.DataFrame = pd.DataFrame()
        if rules_path and os.path.exists(rules_path):
            self.load_rules(rules_path)

    def load_rules(self, rules_path: str) -> None:
        """
        Loads association rules CSV file.
        """
        self.rules_df = pd.read_csv(rules_path)

    def get_all_products(self) -> List[str]:
        """
        Returns a sorted list of unique products available in the rules.
        """
        if self.rules_df.empty:
            return []
        products = set()
        for item in self.rules_df["antecedents"].dropna():
            for p in str(item).split(", "):
                products.add(p.strip())
        return sorted(list(products))

    def recommend(self, product_name: str, top_n: int = 5, min_confidence: float = 0.3) -> List[Dict[str, Any]]:
        """
        Recommends top complementary products for a given item.
        """
        if self.rules_df.empty:
            return []

        # Find rules where antecedent contains product_name
        matches = self.rules_df[
            self.rules_df["antecedents"].str.contains(product_name, case=False, na=False, regex=False) &
            (self.rules_df["confidence"] >= min_confidence)
        ]

        if matches.empty:
            return []

        # Sort by Lift descending, then Confidence descending
        sorted_matches = (
            matches.sort_values(by=["lift", "confidence"], ascending=[False, False])
                   .drop_duplicates(subset=["consequents"])
                   .head(top_n)
        )

        recommendations = []
        for _, row in sorted_matches.iterrows():
            recommendations.append({
                "recommended_item": row["consequents"],
                "confidence": round(float(row["confidence"]) * 100, 1),
                "lift": round(float(row["lift"]), 2),
                "support": round(float(row["support"]) * 100, 2)
            })

        return recommendations
