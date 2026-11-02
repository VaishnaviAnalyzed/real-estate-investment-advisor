"""Run: pytest -q   (needs data/processed + models built first)"""
import pandas as pd
from src import config as C
from src.features import (MODEL_FEATURES, add_features, add_future_price, add_good_investment,
                          build_reference, investment_criteria)


def _row(**kw):
    base = dict(State="Maharashtra", City="Mumbai", Property_Type="Apartment", BHK=3, Size_in_SqFt=1500,
                Price_in_Lakhs=200.0, Year_Built=2010, Furnished_Status="Semi-furnished", Floor_No=5,
                Total_Floors=10, Nearby_Schools=6, Nearby_Hospitals=5, Public_Transport_Accessibility="High",
                Parking_Space="Yes", Security="Yes", Amenities="Gym, Pool", Facing="East",
                Owner_Type="Owner", Availability_Status="Ready_to_Move")
    base.update(kw)
    return pd.DataFrame([base])


def test_feature_engineering_columns():
    f = add_features(_row())
    assert set(MODEL_FEATURES).issubset(f.columns)
    assert f.Amenities_Count.iloc[0] == 2
    assert 0 <= f.Infrastructure_Score.iloc[0] <= 10


def test_future_price_formula():
    f = add_future_price(add_features(_row()))
    r = f.Annual_Growth_Rate.iloc[0]
    assert abs(f[C.TARGET_REG].iloc[0] - 200 * (1 + r) ** 5) < 0.01
    assert r >= C.BASE_GROWTH


def test_label_matches_score_threshold():
    df = pd.read_csv(C.CLEAN_CSV, nrows=5000)
    ref = build_reference(df)
    out = add_good_investment(df, ref)
    assert ((out.Investment_Score >= C.GOOD_INVESTMENT_SCORE_THRESHOLD).astype(int) == out[C.TARGET_CLF]).all()
    assert investment_criteria(out, ref).shape[1] == 5


def test_models_predict():
    import joblib
    clf = joblib.load(C.MODELS_DIR / "classifier.joblib")
    reg = joblib.load(C.MODELS_DIR / "regressor.joblib")
    X = add_features(_row())[MODEL_FEATURES]
    assert 0 <= clf.predict_proba(X)[0, 1] <= 1
    assert reg.predict(X)[0] > 200  # positive appreciation
