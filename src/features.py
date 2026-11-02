"""Step 1b - Feature engineering + target creation.

Everything that is needed at *inference* time (the Streamlit app) lives here as
well, so training and serving share exactly the same code path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

AMENITY_LIST = ["Clubhouse", "Garden", "Gym", "Playground", "Pool"]
TRANSPORT_MAP = {"Low": 1 / 3, "Medium": 2 / 3, "High": 1.0}

NUMERIC_FEATURES = [
    "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt_INR", "Age_of_Property",
    "Floor_No", "Total_Floors", "Floor_Ratio", "Nearby_Schools", "Nearby_Hospitals",
    "Amenities_Count", "Parking_Flag", "Security_Flag", "Transport_Score",
    "School_Density_Score", "Infrastructure_Score", "Size_per_BHK", "Is_Metro",
] + [f"Has_{a}" for a in AMENITY_LIST]
CATEGORICAL_FEATURES = ["City", "Property_Type", "Furnished_Status", "Facing",
                        "Owner_Type", "Availability_Status"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# --------------------------------------------------------------------------
# Derived features
# --------------------------------------------------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features. Works on a full data set or a single row."""
    df = df.copy()
    if "Price_per_SqFt_INR" not in df:
        df["Price_per_SqFt_INR"] = (df["Price_in_Lakhs"] * 1e5 / df["Size_in_SqFt"]).round(2)
    if "Age_of_Property" not in df:
        df["Age_of_Property"] = C.CURRENT_YEAR - df["Year_Built"]

    amen = df["Amenities"].fillna("").astype(str)
    for a in AMENITY_LIST:
        df[f"Has_{a}"] = amen.str.contains(a, regex=False).astype(int)
    df["Amenities_Count"] = df[[f"Has_{a}" for a in AMENITY_LIST]].sum(axis=1)

    df["Parking_Flag"] = (df["Parking_Space"] == "Yes").astype(int)
    df["Security_Flag"] = (df["Security"] == "Yes").astype(int)
    df["Transport_Score"] = df["Public_Transport_Accessibility"].map(TRANSPORT_MAP).astype(float)

    df["Floor_Ratio"] = (df["Floor_No"] / df["Total_Floors"].clip(lower=1)).round(4)
    df["Size_per_BHK"] = (df["Size_in_SqFt"] / df["BHK"].clip(lower=1)).round(2)
    # Schools nearby, normalised to 0-1 (dataset scale is 1-10)
    df["School_Density_Score"] = (df["Nearby_Schools"] / 10).round(3)
    # Composite 0-10 score: schools, hospitals, transport, parking, security, amenities
    df["Infrastructure_Score"] = (10 * (
        df["Nearby_Schools"] / 10 + df["Nearby_Hospitals"] / 10 + df["Transport_Score"]
        + df["Parking_Flag"] + df["Security_Flag"] + df["Amenities_Count"] / 5
    ) / 6).round(3)
    df["Is_Metro"] = df["City"].isin(C.METRO_CITIES).astype(int)
    return df


# --------------------------------------------------------------------------
# Reference statistics (medians used by the business rules)
# --------------------------------------------------------------------------
def build_reference(df: pd.DataFrame) -> dict:
    ppsf = df.groupby(["City", "Property_Type"])["Price_per_SqFt_INR"].median()
    return {
        "city_type_median_ppsf": {f"{c}|{t}": float(v) for (c, t), v in ppsf.items()},
        "city_median_price": df.groupby("City")["Price_in_Lakhs"].median().to_dict(),
        "city_median_ppsf": df.groupby("City")["Price_per_SqFt_INR"].median().to_dict(),
        "global_median_ppsf": float(df["Price_per_SqFt_INR"].median()),
        "global_median_price": float(df["Price_in_Lakhs"].median()),
        "infra_median": float(df["Infrastructure_Score"].median()),
    }


# --------------------------------------------------------------------------
# Target 1: Good Investment (multi-factor score, PDF: "Price vs Median",
# "Price per Sq Ft" and "Multi-factor Score")
# --------------------------------------------------------------------------
CRITERIA_LABELS = {
    "C1_Price_per_SqFt_below_median": "Price/sq ft ≤ median of same city & property type",
    "C2_Price_below_city_median": "Total price ≤ city median price",
    "C3_BHK_3_plus": "3 or more BHK",
    "C4_Ready_to_move": "Ready to move",
    "C5_Infra_above_median": "Infrastructure score ≥ market median",
}


def investment_criteria(df: pd.DataFrame, ref: dict) -> pd.DataFrame:
    key = df["City"].astype(str) + "|" + df["Property_Type"].astype(str)
    med_ppsf = key.map(ref["city_type_median_ppsf"]).fillna(ref["global_median_ppsf"])
    med_price = df["City"].map(ref["city_median_price"]).fillna(ref["global_median_price"])
    out = pd.DataFrame(index=df.index)
    out["C1_Price_per_SqFt_below_median"] = (df["Price_per_SqFt_INR"] <= med_ppsf).astype(int)
    out["C2_Price_below_city_median"] = (df["Price_in_Lakhs"] <= med_price).astype(int)
    out["C3_BHK_3_plus"] = (df["BHK"] >= 3).astype(int)
    out["C4_Ready_to_move"] = (df["Availability_Status"] == "Ready_to_Move").astype(int)
    out["C5_Infra_above_median"] = (df["Infrastructure_Score"] >= ref["infra_median"]).astype(int)
    return out


def add_good_investment(df: pd.DataFrame, ref: dict) -> pd.DataFrame:
    df = df.copy()
    crit = investment_criteria(df, ref)
    df["Investment_Score"] = crit.sum(axis=1)
    df[C.TARGET_CLF] = (df["Investment_Score"] >= C.GOOD_INVESTMENT_SCORE_THRESHOLD).astype(int)
    return df


# --------------------------------------------------------------------------
# Target 2: Future price after 5 years (PDF: by location/type + feature-based)
#     Future = Current * (1 + r) ** t
# --------------------------------------------------------------------------
def annual_growth_rate(df: pd.DataFrame) -> pd.Series:
    r = (
        C.BASE_GROWTH
        + df["Property_Type"].map(C.TYPE_GROWTH_ADJ).fillna(0)
        + df["Public_Transport_Accessibility"].map(C.TRANSPORT_ADJ).fillna(0)
        + df["Availability_Status"].map(C.CONSTRUCTION_ADJ).fillna(0)
        + np.where(df["City"].isin(C.METRO_CITIES), C.CITY_TIER_ADJ["metro"], C.CITY_TIER_ADJ["other"])
    )
    return pd.Series(r, index=df.index)


def add_future_price(df: pd.DataFrame, years: int = C.HORIZON_YEARS) -> pd.DataFrame:
    df = df.copy()
    df["Annual_Growth_Rate"] = annual_growth_rate(df)
    df[C.TARGET_REG] = (df["Price_in_Lakhs"] * (1 + df["Annual_Growth_Rate"]) ** years).round(2)
    return df
