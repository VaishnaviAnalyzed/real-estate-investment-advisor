"""Step 1a - Data preprocessing: load, clean, impute, flag outliers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

CAT_COLS = [
    "State", "City", "Locality", "Property_Type", "Furnished_Status",
    "Public_Transport_Accessibility", "Parking_Space", "Security", "Amenities",
    "Facing", "Owner_Type", "Availability_Status",
]
NUM_COLS = [
    "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt", "Year_Built",
    "Floor_No", "Total_Floors", "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals",
]


def load_raw(path=C.RAW_CSV) -> pd.DataFrame:
    return pd.read_csv(path)


def iqr_bounds(s: pd.Series, k: float = 1.5) -> tuple[float, float]:
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def clean(df: pd.DataFrame, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Return (clean_df, report). The report documents every change made."""
    df = df.copy()
    report: dict = {"rows_in": len(df)}

    # 1. Duplicates (ignore the surrogate ID)
    before = len(df)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "ID"])
    report["duplicates_removed"] = before - len(df)

    # 2. Standardise text columns
    for c in CAT_COLS:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip()
    df["Property_Type"] = df["Property_Type"].str.title()

    # 3. Missing values: median for numeric, mode for categorical
    report["missing_before"] = int(df.isna().sum().sum())
    for c in NUM_COLS:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    for c in CAT_COLS:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].mode().iloc[0])
    report["missing_after"] = int(df.isna().sum().sum())

    # 4. Logical consistency fixes
    #    a) a flat cannot be on a floor higher than the building's top floor
    bad_floor = df["Floor_No"] > df["Total_Floors"]
    report["floor_gt_total_fixed"] = int(bad_floor.sum())
    df["Floor_No_Was_Invalid"] = bad_floor.astype(int)
    df.loc[bad_floor, "Floor_No"] = df.loc[bad_floor, "Total_Floors"]
    #    b) Age must agree with Year_Built
    df["Age_of_Property"] = C.CURRENT_YEAR - df["Year_Built"]
    #    c) Price per sq ft in real rupees (source column is in lakhs / sq ft)
    df["Price_per_SqFt_INR"] = (df["Price_in_Lakhs"] * 1e5 / df["Size_in_SqFt"]).round(2)

    # 5. Outlier detection (IQR) - flagged, not removed, so valid luxury /
    #    compact listings stay in the data set.
    for col in ["Price_in_Lakhs", "Size_in_SqFt", "Price_per_SqFt_INR"]:
        lo, hi = iqr_bounds(df[col])
        flag = ((df[col] < lo) | (df[col] > hi)).astype(int)
        df[f"{col}_Outlier"] = flag
        report[f"outliers_{col}"] = int(flag.sum())

    # 6. dtypes
    for c in CAT_COLS:
        df[c] = df[c].astype(str)
    report["rows_out"] = len(df)
    if verbose:
        for k, v in report.items():
            print(f"  {k:32s}: {v}")
    return df.reset_index(drop=True), report
