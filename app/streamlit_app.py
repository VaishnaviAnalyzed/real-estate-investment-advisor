"""Step 5 - Streamlit app.   Run:  streamlit run app/streamlit_app.py"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config as C
from src.eda import EDA_QUESTIONS
from src.features import (CRITERIA_LABELS, MODEL_FEATURES, add_features, annual_growth_rate,
                          investment_criteria)

st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏙️", layout="wide")


# ------------------------------------------------------------------ loaders
@st.cache_data(show_spinner="Loading data ...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(C.CLEAN_CSV)


@st.cache_resource(show_spinner="Loading models ...")
def load_models():
    return joblib.load(C.MODELS_DIR / "classifier.joblib"), joblib.load(C.MODELS_DIR / "regressor.joblib")


@st.cache_data
def load_json(name):
    return json.loads((C.MODELS_DIR / name).read_text())


def need_artifacts():
    missing = [p for p in [C.CLEAN_CSV, C.MODELS_DIR / "classifier.joblib", C.MODELS_DIR / "regressor.joblib"]
               if not p.exists()]
    if missing:
        st.error("Project artifacts missing. Run `python -m src.build_dataset` then `python -m src.train` first.")
        st.stop()


def lakh(v):
    return f"₹{v:,.1f} L" if v < 100 else f"₹{v / 100:,.2f} Cr"


# --------------------------------------------------------------------- pages
def page_overview(df):
    st.title("🏙️ Real Estate Investment Advisor")
    st.caption("Predicting property profitability & future value - classification + 5-year price regression")
    c = st.columns(5)
    c[0].metric("Listings", f"{len(df):,}")
    c[1].metric("Cities / States", f"{df.City.nunique()} / {df.State.nunique()}")
    c[2].metric("Avg price", lakh(df.Price_in_Lakhs.mean()))
    c[3].metric("Good-investment share", f"{df[C.TARGET_CLF].mean():.1%}")
    c[4].metric("Avg projected 5-yr price", lakh(df[C.TARGET_REG].mean()))

    st.subheader("Location heatmap")
    metric = st.radio("Metric", ["Price_in_Lakhs", "Price_per_SqFt_INR", C.TARGET_CLF, "Annual_Growth_Rate"],
                      horizontal=True, format_func=lambda x: {C.TARGET_CLF: "Good-investment rate",
                                                              "Annual_Growth_Rate": "Modelled annual growth",
                                                              "Price_in_Lakhs": "Avg price (₹L)",
                                                              "Price_per_SqFt_INR": "Avg ₹ / sq ft"}[x])
    pv = df.pivot_table(index="City", columns="Property_Type", values=metric, aggfunc="mean")
    st.plotly_chart(px.imshow(pv, aspect="auto", color_continuous_scale="YlGnBu", text_auto=".3g",
                              height=900), width="stretch")

    st.subheader("Growth outlook by city")
    g = df.groupby("City").agg(price=("Price_in_Lakhs", "mean"), future=(C.TARGET_REG, "mean")).reset_index()
    g["Return_%"] = (g.future / g.price - 1) * 100
    fig = px.bar(g.sort_values("Return_%"), x="Return_%", y="City", orientation="h", height=800,
                 title="Expected 5-year appreciation (%)")
    st.plotly_chart(fig, width="stretch")


def advisor_form(df):
    cities = sorted(df.City.unique())
    state_of = df.drop_duplicates("City").set_index("City")["State"].to_dict()
    with st.form("property_form"):
        a, b, c = st.columns(3)
        city = a.selectbox("City", cities, index=cities.index("Mumbai") if "Mumbai" in cities else 0)
        ptype = b.selectbox("Property type", sorted(df.Property_Type.unique()))
        status = c.selectbox("Availability", sorted(df.Availability_Status.unique()))
        a, b, c = st.columns(3)
        bhk = a.slider("BHK", 1, 5, 3)
        size = b.number_input("Size (sq ft)", 300, 10000, 1800, step=50)
        price = c.number_input("Asking price (₹ lakhs)", 5.0, 5000.0, 250.0, step=5.0)
        a, b, c = st.columns(3)
        year = a.number_input("Year built", 1980, C.CURRENT_YEAR, 2012)
        total_fl = b.number_input("Total floors", 1, 60, 12)
        floor = c.number_input("Floor number", 0, 60, 5)
        a, b, c = st.columns(3)
        furn = a.selectbox("Furnishing", sorted(df.Furnished_Status.unique()))
        facing = b.selectbox("Facing", sorted(df.Facing.unique()))
        owner = c.selectbox("Owner type", sorted(df.Owner_Type.unique()))
        a, b, c = st.columns(3)
        schools = a.slider("Nearby schools (0-10)", 0, 10, 6)
        hospitals = b.slider("Nearby hospitals (0-10)", 0, 10, 5)
        transport = c.select_slider("Public transport access", ["Low", "Medium", "High"], "Medium")
        a, b, c = st.columns(3)
        parking = a.checkbox("Parking space", True)
        security = b.checkbox("Security (gated / CCTV / guard)", True)
        amen = c.multiselect("Amenities", ["Clubhouse", "Garden", "Gym", "Playground", "Pool"], ["Gym", "Pool"])
        go_btn = st.form_submit_button("🔮 Analyse investment", type="primary")
    row = pd.DataFrame([{
        "State": state_of[city], "City": city, "Property_Type": ptype, "BHK": bhk, "Size_in_SqFt": size,
        "Price_in_Lakhs": price, "Year_Built": year, "Furnished_Status": furn,
        "Floor_No": min(floor, total_fl), "Total_Floors": total_fl, "Nearby_Schools": schools,
        "Nearby_Hospitals": hospitals, "Public_Transport_Accessibility": transport,
        "Parking_Space": "Yes" if parking else "No", "Security": "Yes" if security else "No",
        "Amenities": ", ".join(amen), "Facing": facing, "Owner_Type": owner, "Availability_Status": status,
    }])
    return go_btn, row


def page_advisor(df):
    st.title("🔮 Investment Advisor")
    clf, reg = load_models()
    ref = load_json("reference_stats.json")
    metrics = load_json("metrics.json")
    go_btn, row = advisor_form(df)
    if go_btn:
        st.session_state["row"] = row
    if "row" not in st.session_state:
        st.info("Fill in the property details and press **Analyse investment**.")
        return
    row = add_features(st.session_state["row"])
    X = row[MODEL_FEATURES]
    prob = float(clf.predict_proba(X)[0, 1])
    future = float(reg.predict(X)[0])
    price = float(row.Price_in_Lakhs.iloc[0])
    gain = future - price
    cagr = (future / price) ** (1 / C.HORIZON_YEARS) - 1
    good = prob >= 0.5
    conf = prob if good else 1 - prob

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### Is this a Good Investment?")
        (st.success if good else st.warning)(f"{'✅ YES - Good investment' if good else '⚠️ NO - Not a good investment'}")
        st.progress(conf, text=f"Model confidence: {conf:.1%}")
        st.caption(f"P(good investment) = {prob:.1%}")
    with c2:
        st.markdown("#### Estimated price after 5 years")
        st.metric("Projected price", lakh(future), f"{gain:+,.1f} L ({gain / price:+.1%})")
        st.caption(f"Implied growth ≈ {cagr:.2%} per year")
    with c3:
        st.markdown("#### Investment score")
        crit = investment_criteria(row, ref).iloc[0]
        st.metric("Rules passed", f"{int(crit.sum())} / 5", help=f"≥{C.GOOD_INVESTMENT_SCORE_THRESHOLD} = good")
        for k, v in crit.items():
            st.write(("✅ " if v else "❌ ") + CRITERIA_LABELS[k])

    left, right = st.columns(2)
    with left:
        yrs = np.arange(0, C.HORIZON_YEARS + 1)
        proj = price * (future / price) ** (yrs / C.HORIZON_YEARS)
        fig = go.Figure(go.Scatter(x=yrs, y=proj, mode="lines+markers", fill="tozeroy",
                                   line=dict(color="#2a9d8f"), name="Projected price"))
        fig.update_layout(title="Projected price path (₹ lakhs)", xaxis_title="Years from today",
                          height=360, template="plotly_white", margin=dict(t=50, b=10))
        st.plotly_chart(fig, width="stretch")
    with right:
        fi = pd.DataFrame(metrics["classification"][metrics["best_classifier"]["name"]]["feature_importance"]).head(12)
        fig = px.bar(fi.iloc[::-1], x="importance", y="feature", orientation="h",
                     title=f"What drives the 'Good Investment' call ({metrics['best_classifier']['name']})")
        fig.update_layout(height=360, template="plotly_white", margin=dict(t=50, b=10))
        st.plotly_chart(fig, width="stretch")

    st.subheader("Comparable listings")
    r0 = row.iloc[0]
    comps = df[(df.City == r0.City) & (df.Property_Type == r0.Property_Type) & (df.BHK == r0.BHK)
               & df.Size_in_SqFt.between(r0.Size_in_SqFt * 0.85, r0.Size_in_SqFt * 1.15)]
    if len(comps):
        a, b, c = st.columns(3)
        a.metric("Comparable listings", f"{len(comps):,}")
        b.metric("Median comparable price", lakh(comps.Price_in_Lakhs.median()),
                 f"{(price / comps.Price_in_Lakhs.median() - 1):+.1%} vs yours", delta_color="inverse")
        c.metric("Median ₹/sq ft", f"₹{comps.Price_per_SqFt_INR.median():,.0f}",
                 f"yours ₹{r0.Price_per_SqFt_INR:,.0f}", delta_color="off")
        st.dataframe(comps.nsmallest(8, "Price_in_Lakhs")[["State", "City", "Locality", "Property_Type", "BHK",
                     "Size_in_SqFt", "Price_in_Lakhs", C.TARGET_REG, C.TARGET_CLF]], hide_index=True,
                     width="stretch")
    else:
        st.caption("No comparable listings found for this combination.")
    st.caption("⚠️ Educational tool. The 5-year price follows rule-based growth assumptions (see Documentation); "
               "it is not financial advice.")


def page_explorer(df):
    st.title("🔎 Property Explorer")
    with st.sidebar:
        st.subheader("Filters")
        states = st.multiselect("State", sorted(df.State.unique()))
        pool = df[df.State.isin(states)] if states else df
        cities = st.multiselect("City", sorted(pool.City.unique()))
        types = st.multiselect("Property type", sorted(df.Property_Type.unique()))
        bhk = st.multiselect("BHK", sorted(df.BHK.unique()))
        area = st.slider("Area (sq ft)", 500, 5000, (500, 5000), step=50)
        prange = st.slider("Price (₹ lakhs)", 10, 500, (10, 500), step=5)
        avail = st.multiselect("Availability", sorted(df.Availability_Status.unique()))
        only_good = st.checkbox("Only 'Good Investment' listings")
    m = df.Size_in_SqFt.between(*area) & df.Price_in_Lakhs.between(*prange)
    for col, sel in [("State", states), ("City", cities), ("Property_Type", types), ("BHK", bhk),
                     ("Availability_Status", avail)]:
        if sel:
            m &= df[col].isin(sel)
    if only_good:
        m &= df[C.TARGET_CLF] == 1
    f = df[m]
    a, b, c, d = st.columns(4)
    a.metric("Matching listings", f"{len(f):,}")
    if len(f) == 0:
        st.warning("No listings match the filters.")
        return
    b.metric("Avg price", lakh(f.Price_in_Lakhs.mean()))
    c.metric("Good-investment share", f"{f[C.TARGET_CLF].mean():.1%}")
    d.metric("Avg projected 5-yr price", lakh(f[C.TARGET_REG].mean()))
    sort = st.selectbox("Rank by", ["Investment_Score", "Annual_Growth_Rate", "Price_in_Lakhs", "Price_per_SqFt_INR"])
    asc = sort in ("Price_in_Lakhs", "Price_per_SqFt_INR")
    top = f.sort_values([sort, "Annual_Growth_Rate"], ascending=[asc, False]).head(200)
    st.dataframe(top[["ID", "State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs",
                      "Price_per_SqFt_INR", "Availability_Status", "Investment_Score", C.TARGET_CLF,
                      C.TARGET_REG]], hide_index=True, width="stretch", height=380)
    st.download_button("⬇️ Download all matches (CSV)", f.to_csv(index=False).encode(), "filtered_properties.csv")
    x1, x2 = st.columns(2)
    pv = f.pivot_table(index="City", columns="BHK", values="Price_in_Lakhs", aggfunc="mean")
    x1.plotly_chart(px.imshow(pv, aspect="auto", color_continuous_scale="YlOrRd", title="Avg price by city & BHK (₹L)",
                              height=max(350, 22 * len(pv))), width="stretch")
    x2.plotly_chart(px.histogram(f, x="Price_in_Lakhs", color=f[C.TARGET_CLF].map({1: "Good", 0: "Not good"}),
                                 nbins=40, barmode="overlay", title="Price distribution of matches"),
                    width="stretch")


def page_insights(df):
    st.title("📊 Market Insights (EDA)")
    group = st.selectbox("Topic", list(dict.fromkeys(g for g, _, _ in EDA_QUESTIONS)))
    items = [(t, fn) for g, t, fn in EDA_QUESTIONS if g == group]
    for tab, (title, fn) in zip(st.tabs([t.split(" ")[0] for t, _ in items]), items):
        with tab:
            st.subheader(title)
            fig, insight = fn(df)
            st.plotly_chart(fig, width="stretch")
            st.info(insight)


def page_models():
    st.title("🧠 Model Performance")
    m = load_json("metrics.json")
    st.caption(f"Trained on {m['n_rows']:,} listings, 80/20 train-test split. Tracked in MLflow "
               f"(`mlflow ui --backend-store-uri {C.MLFLOW_URI}`).")
    st.warning("Both targets are **rule-derived** (the data set has no true future price or investment label), so high "
               "scores show the models recover the business rules - not that they can foresee real markets.")
    t1, t2 = st.tabs(["Classification - Good Investment", "Regression - Price in 5 years"])
    with t1:
        st.success(f"Champion: **{m['best_classifier']['name']}** (registry v{m['best_classifier']['registry_version']})")
        st.dataframe(pd.DataFrame(m["classification"]).T[["accuracy", "precision", "recall", "f1", "roc_auc"]].astype(float).round(4),
                     width="stretch")
        best = m["best_classifier"]["name"]
        a, b = st.columns(2)
        cm = np.array(m["classification"][best]["confusion_matrix"])
        a.plotly_chart(px.imshow(cm, text_auto=",", x=["Pred: No", "Pred: Yes"], y=["Actual: No", "Actual: Yes"],
                                 color_continuous_scale="Blues", title=f"Confusion matrix - {best}"), width="stretch")
        fig = go.Figure()
        for n, v in m["classification"].items():
            fig.add_scatter(x=v["roc"]["fpr"], y=v["roc"]["tpr"], name=f"{n} (AUC {v['roc_auc']:.3f})")
        fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="grey"), name="chance")
        fig.update_layout(title="ROC curves", xaxis_title="FPR", yaxis_title="TPR")
        b.plotly_chart(fig, width="stretch")
        fi = pd.DataFrame(m["classification"][best]["feature_importance"]).head(15)
        st.plotly_chart(px.bar(fi.iloc[::-1], x="importance", y="feature", orientation="h",
                               title="Feature importance"), width="stretch")
    with t2:
        st.success(f"Champion: **{m['best_regressor']['name']}** (registry v{m['best_regressor']['registry_version']})")
        st.dataframe(pd.DataFrame(m["regression"]).T[["rmse", "mae", "r2"]].astype(float).round(4), width="stretch")
        best = m["best_regressor"]["name"]
        fi = pd.DataFrame(m["regression"][best]["feature_importance"]).head(15)
        st.plotly_chart(px.bar(fi.iloc[::-1], x="importance", y="feature", orientation="h",
                               title="Feature importance (regression)"), width="stretch")


def page_about():
    st.title("📘 About & Methodology")
    st.markdown("""
**Pipeline** - raw CSV → cleaning (duplicates, missing values, logical fixes, IQR outlier flags) → feature
engineering (infrastructure & school-density scores, amenity flags, price per sq ft ...) → two targets → models
compared in MLflow → Streamlit.

**Good Investment** (classification) = at least 3 of 5 rules pass: price/sq ft ≤ city-type median · price ≤ city
median · BHK ≥ 3 · ready to move · infrastructure score ≥ market median.

**Future price** (regression) = `Price × (1 + r)^5` with `r = 8 % + property-type + transport + construction-status +
metro-city adjustments` (rates in `src/config.py`).

**Known data limits** - no crime-rate, RERA or sale-date columns exist; Security is used as a safety proxy. Prices
in the data set are nearly uniform and independent of location/size, so market-level conclusions should not be
over-interpreted.
""")


# ---------------------------------------------------------------------- main
need_artifacts()
data = load_data()
PAGES = {"🏠 Overview": lambda: page_overview(data), "🔮 Investment Advisor": lambda: page_advisor(data),
         "🔎 Property Explorer": lambda: page_explorer(data), "📊 Market Insights": lambda: page_insights(data),
         "🧠 Model Performance": page_models, "📘 About": page_about}
choice = st.sidebar.radio("Navigate", list(PAGES))
PAGES[choice]()
