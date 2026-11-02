"""Run: python -m src.build_dataset

Raw CSV -> cleaned + feature-engineered CSV with both target variables, plus
the reference statistics (medians) the app needs for explanations.
"""
import json

from . import config as C
from .features import add_features, add_future_price, add_good_investment, build_reference
from .preprocessing import clean, load_raw


def main():
    print("Loading raw data ...")
    raw = load_raw()
    print(f"  raw shape: {raw.shape}")
    print("Cleaning ...")
    df, report = clean(raw)
    print("Engineering features ...")
    df = add_features(df)
    ref = build_reference(df)
    df = add_good_investment(df, ref)
    df = add_future_price(df)

    C.CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.CLEAN_CSV, index=False)
    C.MODELS_DIR.mkdir(exist_ok=True)
    (C.MODELS_DIR / "reference_stats.json").write_text(json.dumps(ref, indent=2))
    (C.MODELS_DIR / "cleaning_report.json").write_text(json.dumps(report, indent=2))

    print(f"\nSaved {C.CLEAN_CSV}  shape={df.shape}")
    print("Good_Investment balance:\n", df[C.TARGET_CLF].value_counts(normalize=True).round(3).to_string())
    print("Investment_Score distribution:\n", df["Investment_Score"].value_counts().sort_index().to_string())
    print(df[[C.TARGET_REG, "Annual_Growth_Rate"]].describe().round(3).to_string())


if __name__ == "__main__":
    main()
