"""Central configuration for the Real Estate Investment Advisor project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "india_housing_prices.csv"
CLEAN_CSV = ROOT / "data" / "processed" / "cleaned_housing.csv"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
MLFLOW_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
MLFLOW_ARTIFACTS = (ROOT / "mlruns").as_posix()

CURRENT_YEAR = 2025
HORIZON_YEARS = 5
RANDOM_STATE = 42

# ---- Future-price growth model (annual appreciation rates) -------------
# The dataset has no future price, so the regression target is derived from
# domain rules (PDF options: "Fixed Rate", "By Location/Type", "Feature-based").
BASE_GROWTH = 0.08  # PDF example: r = 8 %
TYPE_GROWTH_ADJ = {"Apartment": 0.000, "Independent House": 0.004, "Villa": 0.008}
TRANSPORT_ADJ = {"High": 0.010, "Medium": 0.004, "Low": 0.000}
CONSTRUCTION_ADJ = {"Under_Construction": 0.008, "Ready_to_Move": 0.000}
# City tiers: metro cities appreciate faster than tier-3 cities.
METRO_CITIES = {"Mumbai", "New Delhi", "Bangalore", "Chennai", "Hyderabad", "Kolkata", "Pune",
                "Gurgaon", "Noida", "Ahmedabad"}
CITY_TIER_ADJ = {"metro": 0.012, "other": 0.000}

# ---- Good-investment label ---------------------------------------------
GOOD_INVESTMENT_SCORE_THRESHOLD = 3  # of 5 criteria (see features.py)

TARGET_CLF = "Good_Investment"
TARGET_REG = "Future_Price_5Y"
