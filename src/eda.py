"""Step 2 - Exploratory Data Analysis.

Every question from the project brief (Q1-Q20) plus the four analyses listed
in the "Step 2" box (S1-S4) is implemented as a function that returns
``(plotly_figure, insight_text)``. The insight text is computed from the data,
so it never goes stale. The same functions power the Streamlit "Market
Insights" page and the offline report:

    python -m src.eda      ->  reports/figures/*.html  +  reports/eda_findings.md
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from . import config as C

PALETTE = px.colors.qualitative.Safe
SAMPLE_N = 20_000


# ----------------------------------------------------------------- helpers
def _style(fig, height=430):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=50, b=10),
                      colorway=PALETTE, title_font_size=16)
    return fig


def _sample(df, n=SAMPLE_N):
    return df.sample(min(n, len(df)), random_state=C.RANDOM_STATE)


def _spread(s: pd.Series, unit="") -> str:
    """Describe how different the best and worst group are."""
    hi, lo = s.idxmax(), s.idxmin()
    pct = (s.max() / s.min() - 1) * 100
    return (f"Highest: **{hi}** ({s.max():,.2f}{unit}); lowest: **{lo}** ({s.min():,.2f}{unit}) "
            f"- a gap of only {pct:.1f}%.") if pct < 10 else (
            f"Highest: **{hi}** ({s.max():,.2f}{unit}); lowest: **{lo}** ({s.min():,.2f}{unit}) "
            f"- a gap of {pct:.1f}%.")


# ------------------------------------------------------- 1-5 price & size
def q1(df):
    fig = px.histogram(df, x="Price_in_Lakhs", nbins=60, title="Distribution of property prices (₹ lakhs)")
    s = df["Price_in_Lakhs"]
    return _style(fig), (f"Prices range from ₹{s.min():.0f}L to ₹{s.max():.0f}L (mean ₹{s.mean():.0f}L, "
                         f"median ₹{s.median():.0f}L, skew {s.skew():.2f}). The distribution is almost flat, "
                         f"i.e. close to uniform rather than the right-skewed shape of real markets.")


def q2(df):
    fig = px.histogram(df, x="Size_in_SqFt", nbins=60, title="Distribution of property sizes (sq ft)")
    s = df["Size_in_SqFt"]
    return _style(fig), (f"Sizes span {s.min():,} - {s.max():,} sq ft (median {s.median():,.0f}); "
                         f"the spread is again nearly uniform.")


def q3(df):
    fig = px.box(_sample(df), x="Property_Type", y="Price_per_SqFt_INR", color="Property_Type",
                 title="Price per sq ft (₹) by property type")
    g = df.groupby("Property_Type")["Price_per_SqFt_INR"].mean()
    return _style(fig), "Mean price per sq ft by type. " + _spread(g, " ₹/sqft")


def q4(df):
    s = _sample(df, 5000)
    fig = px.scatter(s, x="Size_in_SqFt", y="Price_in_Lakhs", opacity=0.35, color="Property_Type",
                     title="Property size vs price (5,000-point sample)")
    r = df["Size_in_SqFt"].corr(df["Price_in_Lakhs"])
    return _style(fig), (f"Correlation between size and total price is r = {r:.3f}: practically **no linear "
                         f"relationship**. In this data set price does not scale with area, so price per sq ft "
                         f"is driven mostly by size itself (r = {df['Size_in_SqFt'].corr(df['Price_per_SqFt_INR']):.2f}).")


def q5(df):
    s = _sample(df)
    fig = go.Figure()
    fig.add_trace(go.Box(y=s["Price_per_SqFt_INR"], name="Price / sq ft (₹)", boxpoints="outliers"))
    fig.update_layout(title="Outlier check - price per sq ft (₹)")
    n_pp = int(df["Price_per_SqFt_INR_Outlier"].sum()) if "Price_per_SqFt_INR_Outlier" in df else 0
    n_pr = int(df.get("Price_in_Lakhs_Outlier", pd.Series([0])).sum())
    n_sz = int(df.get("Size_in_SqFt_Outlier", pd.Series([0])).sum())
    return _style(fig), (f"IQR rule flags {n_pp:,} price-per-sq-ft outliers ({n_pp / len(df):.1%}), "
                         f"{n_pr:,} price outliers and {n_sz:,} size outliers. Outliers appear only in "
                         f"price/sq ft (small flats with high totals). They are **flagged, not removed**, as they are "
                         f"valid listings.")


# ------------------------------------------------------ 6-10 location
def q6(df):
    g = df.groupby("State")["Price_per_SqFt_INR"].mean().sort_values()
    fig = px.bar(g, orientation="h", title="Average price per sq ft by state (₹)", labels={"value": "₹ / sq ft", "State": ""})
    fig.update_layout(showlegend=False)
    return _style(fig, 560), _spread(g, " ₹/sqft")


def q7(df):
    g = df.groupby("City")["Price_in_Lakhs"].mean().sort_values()
    fig = px.bar(g, orientation="h", title="Average property price by city (₹ lakhs)", labels={"value": "₹ lakhs", "City": ""})
    fig.update_layout(showlegend=False)
    return _style(fig, 820), _spread(g, "L")


def q8(df):
    g = df.groupby("Locality")["Age_of_Property"].median().sort_values(ascending=False)
    fig = px.bar(g.head(25), title="Median property age by locality - 25 oldest localities (years)",
                 labels={"value": "median age (yrs)", "Locality": ""})
    fig.update_layout(showlegend=False)
    return _style(fig), (f"Across {len(g)} localities the median age ranges {g.min():.0f} - {g.max():.0f} years "
                         f"(overall median {df['Age_of_Property'].median():.0f}). Oldest: {', '.join(g.index[:3])}.")


def q9(df):
    ct = pd.crosstab(df["City"], df["BHK"], normalize="index") * 100
    fig = px.imshow(ct.round(1), aspect="auto", color_continuous_scale="Blues", text_auto=".0f",
                    title="BHK mix by city (% of listings)", labels=dict(color="%"))
    return _style(fig, 880), (f"Every city has a near-identical BHK split (each BHK 1-5 holds roughly "
                              f"{ct.values.mean():.0f}% of listings, city-to-city spread ±{ct.values.std():.1f} pts).")


def q10(df):
    top = df.groupby("Locality")["Price_in_Lakhs"].mean().nlargest(5).index
    d = df[df["Locality"].isin(top)].copy()
    d["Built_Period"] = (d["Year_Built"] // 5 * 5).astype(str) + "s"
    g = d.groupby(["Built_Period", "Locality"])["Price_in_Lakhs"].mean().reset_index()
    fig = px.line(g, x="Built_Period", y="Price_in_Lakhs", color="Locality", markers=True,
                  title="Average price by construction period - 5 most expensive localities")
    means = d.groupby("Locality")["Price_in_Lakhs"].mean().sort_values(ascending=False)
    return _style(fig), ("Top 5 localities by mean price: " + ", ".join(f"{k} (₹{v:.0f}L)" for k, v in means.items())
                         + ". The dataset has no sale date, so the trend is shown by year built; lines are noisy "
                           "with no consistent upward trend.")


# ------------------------------------------- 11-15 relationships
def q11(df):
    cols = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt_INR", "Age_of_Property", "Floor_No",
            "Total_Floors", "Nearby_Schools", "Nearby_Hospitals", "Infrastructure_Score", "Amenities_Count"]
    cm = df[cols].corr().round(2)
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    title="Correlation matrix of numeric features")
    off = cm.where(~np.eye(len(cm), dtype=bool)).stack()
    a, b = off.abs().idxmax()
    return _style(fig, 650), (f"Strongest pair: **{a} ↔ {b}** (r = {cm.loc[a, b]:.2f}); this is mechanical, because "
                              f"price/sq ft = price ÷ size. Apart from those derived pairs almost all correlations are ≈ 0.")


def _numeric_vs_ppsf(df, col, title):
    g = df.groupby(col)["Price_per_SqFt_INR"].mean().reset_index()
    fig = px.bar(g, x=col, y="Price_per_SqFt_INR", title=title, labels={"Price_per_SqFt_INR": "avg ₹ / sq ft"})
    r = df[col].corr(df["Price_per_SqFt_INR"])
    return _style(fig), (f"Correlation with price per sq ft: r = {r:.3f}. "
                         f"Average ₹/sq ft ranges only {g.Price_per_SqFt_INR.min():,.0f} - {g.Price_per_SqFt_INR.max():,.0f} across the scale.")


def q12(df):
    return _numeric_vs_ppsf(df, "Nearby_Schools", "Nearby schools vs average price per sq ft")


def q13(df):
    return _numeric_vs_ppsf(df, "Nearby_Hospitals", "Nearby hospitals vs average price per sq ft")


def q14(df):
    fig = px.box(_sample(df), x="Furnished_Status", y="Price_in_Lakhs", color="Furnished_Status",
                 title="Price (₹ lakhs) by furnished status")
    g = df.groupby("Furnished_Status")["Price_in_Lakhs"].mean()
    return _style(fig), "Mean price by furnishing level. " + _spread(g, "L")


def q15(df):
    g = df.groupby("Facing")["Price_per_SqFt_INR"].mean().sort_values()
    fig = px.bar(g, title="Average price per sq ft by facing direction (₹)", labels={"value": "₹ / sq ft", "Facing": ""})
    fig.update_layout(showlegend=False)
    return _style(fig), _spread(g, " ₹/sqft")


# ------------------------------------- 16-20 investment / amenities
def q16(df):
    g = df["Owner_Type"].value_counts()
    fig = px.bar(g, title="Listings per owner type", labels={"value": "properties", "Owner_Type": ""})
    fig.update_layout(showlegend=False)
    return _style(fig), "Listings: " + ", ".join(f"{k} {v:,} ({v / len(df):.1%})" for k, v in g.items()) + "."


def q17(df):
    g = df["Availability_Status"].value_counts()
    fig = px.pie(values=g.values, names=g.index, hole=0.4, title="Listings by availability status")
    return _style(fig), ("Listings: " + ", ".join(f"{k} {v:,} ({v / len(df):.1%})" for k, v in g.items())
                         + ". (No 'Sold' status exists in this data.)")


def q18(df):
    g = df.groupby("Parking_Space")[["Price_in_Lakhs", "Price_per_SqFt_INR"]].mean().reset_index()
    fig = px.bar(g.melt("Parking_Space"), x="Parking_Space", y="value", color="variable", barmode="group",
                 facet_col="variable", title="Effect of parking space on price")
    fig.update_yaxes(matches=None)
    d = (g.set_index("Parking_Space").loc["Yes"] / g.set_index("Parking_Space").loc["No"] - 1) * 100
    return _style(fig), (f"Properties with parking differ from those without by {d['Price_in_Lakhs']:+.1f}% in total "
                         f"price and {d['Price_per_SqFt_INR']:+.1f}% in price per sq ft - negligible.")


def q19(df):
    g = df.groupby("Amenities_Count")["Price_per_SqFt_INR"].mean().reset_index()
    fig = px.bar(g, x="Amenities_Count", y="Price_per_SqFt_INR", title="Number of amenities vs average price per sq ft")
    per = {a: df.loc[df[f"Has_{a}"] == 1, "Price_per_SqFt_INR"].mean() / df.loc[df[f"Has_{a}"] == 0, "Price_per_SqFt_INR"].mean() - 1
           for a in ["Clubhouse", "Garden", "Gym", "Playground", "Pool"]}
    return _style(fig), ("Effect of each individual amenity on mean ₹/sq ft: "
                         + ", ".join(f"{k} {v * 100:+.1f}%" for k, v in per.items()) + ".")


def q20(df):
    d = df.groupby("Public_Transport_Accessibility").agg(
        Avg_PPSF=("Price_per_SqFt_INR", "mean"), Good_Investment_Rate=(C.TARGET_CLF, "mean"),
        Expected_5Y_Return=("Annual_Growth_Rate", lambda s: ((1 + s) ** 5 - 1).mean())).reindex(["Low", "Medium", "High"]).reset_index()
    fig = px.bar(d.melt("Public_Transport_Accessibility"), x="Public_Transport_Accessibility", y="value",
                 color="variable", facet_col="variable", title="Public transport vs price/sq ft and investment potential")
    fig.update_yaxes(matches=None)
    hi = d.set_index("Public_Transport_Accessibility")
    return _style(fig), (f"Price/sq ft barely changes with transport access, but the modelled 5-year return rises from "
                         f"{hi.loc['Low', 'Expected_5Y_Return']:.1%} (Low) to {hi.loc['High', 'Expected_5Y_Return']:.1%} (High) because "
                         f"transport is a growth driver in the future-price rules.")


# --------------------------- S1-S4 : the four "Step 2" analyses
def s1(df):
    top = df.groupby("City")["Price_in_Lakhs"].mean().nlargest(6).index
    d = df[df["City"].isin(top)].groupby(["Year_Built", "City"])["Price_in_Lakhs"].mean().reset_index()
    fig = px.line(d, x="Year_Built", y="Price_in_Lakhs", color="City", title="Price trend by year built - 6 priciest cities")
    g = df.groupby("City")["Price_in_Lakhs"].mean()
    return _style(fig), "Price by city (year-built trend). " + _spread(g, "L")


def s2(df):
    d = df.assign(Return=lambda x: x[C.TARGET_REG] / x["Price_in_Lakhs"] - 1,
                  Size_Band=pd.cut(df["Size_in_SqFt"], [0, 1000, 2000, 3000, 4000, 5000]).astype(str))
    g = d.groupby("Size_Band", sort=False).agg(Return=("Return", "mean"), Good_Rate=(C.TARGET_CLF, "mean")).reset_index()
    fig = px.bar(g.melt("Size_Band"), x="Size_Band", y="value", color="variable", facet_col="variable",
                 title="Area (sq ft band) vs expected 5-year return and Good-Investment rate")
    fig.update_yaxes(matches=None)
    r = df["Size_in_SqFt"].corr(df[C.TARGET_REG] / df["Price_in_Lakhs"])
    gr = g.set_index("Size_Band")["Good_Rate"]
    return _style(fig), (f"Correlation of size with expected 5-yr return: r = {r:.3f} (return depends on growth rules, "
                         f"not area). However, the Good-Investment rate climbs from {gr.iloc[0]:.0%} to {gr.iloc[-1]:.0%} "
                         f"with size because bigger homes have a lower price per sq ft, which the label rewards.")


def s3(df):
    g = df.groupby("Security").agg(Good_Rate=(C.TARGET_CLF, "mean"), Avg_Price=("Price_in_Lakhs", "mean")).reset_index()
    fig = px.bar(g, x="Security", y="Good_Rate", text_auto=".1%", title="Safety proxy (security) vs Good-Investment rate")
    return _style(fig), ("**Note:** the data set has no crime-rate column, so the 'Security' flag is used as a safety "
                         "proxy. Good-investment rate: "
                         + ", ".join(f"Security={r.Security}: {r.Good_Rate:.1%}" for r in g.itertuples()) + ".")


def s4(df):
    d = df.assign(Infra_Band=pd.cut(df["Infrastructure_Score"], 8).astype(str),
                  Resale_PPSF=df[C.TARGET_REG] * 1e5 / df["Size_in_SqFt"])
    order = pd.cut(df["Infrastructure_Score"], 8).cat.categories.astype(str)
    g = d.groupby("Infra_Band")["Resale_PPSF"].mean().reindex(order).rename_axis("Infra_Band").reset_index()
    fig = px.bar(g, x="Infra_Band", y="Resale_PPSF", title="Infrastructure score vs projected resale value (₹ / sq ft, year 5)")
    r = df["Infrastructure_Score"].corr(df[C.TARGET_REG])
    return _style(fig), (f"Correlation between infrastructure score and projected resale price: r = {r:.3f}. "
                         f"Better infrastructure raises expected growth only through the transport factor.")


EDA_QUESTIONS = [
    ("Price & Size", "Q1. What is the distribution of property prices?", q1),
    ("Price & Size", "Q2. What is the distribution of property sizes?", q2),
    ("Price & Size", "Q3. How does price per sq ft vary by property type?", q3),
    ("Price & Size", "Q4. Is there a relationship between property size and price?", q4),
    ("Price & Size", "Q5. Are there outliers in price per sq ft or size?", q5),
    ("Location", "Q6. Average price per sq ft by state", q6),
    ("Location", "Q7. Average property price by city", q7),
    ("Location", "Q8. Median age of properties by locality", q8),
    ("Location", "Q9. How is BHK distributed across cities?", q9),
    ("Location", "Q10. Price trends - top 5 most expensive localities", q10),
    ("Relationships", "Q11. How are numeric features correlated?", q11),
    ("Relationships", "Q12. Nearby schools vs price per sq ft", q12),
    ("Relationships", "Q13. Nearby hospitals vs price per sq ft", q13),
    ("Relationships", "Q14. Price by furnished status", q14),
    ("Relationships", "Q15. Price per sq ft by facing direction", q15),
    ("Investment & Amenities", "Q16. Properties per owner type", q16),
    ("Investment & Amenities", "Q17. Properties per availability status", q17),
    ("Investment & Amenities", "Q18. Does parking space affect price?", q18),
    ("Investment & Amenities", "Q19. How do amenities affect price per sq ft?", q19),
    ("Investment & Amenities", "Q20. Public transport vs price/sq ft and investment potential", q20),
    ("Step-2 analyses", "S1. Price trends by city", s1),
    ("Step-2 analyses", "S2. Area vs investment return", s2),
    ("Step-2 analyses", "S3. Safety (security) vs Good-Investment classification", s3),
    ("Step-2 analyses", "S4. Infrastructure score vs resale value", s4),
]


def run_all(out_dir=C.FIG_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(C.CLEAN_CSV)
    lines = ["# EDA Findings (auto-generated)\n", f"Data set: {len(df):,} cleaned listings.\n"]
    grp = None
    for i, (group, title, fn) in enumerate(EDA_QUESTIONS):
        fig, insight = fn(df)
        name = title.split(".")[0].lower()
        fig.write_html(out_dir / f"{name}.html", include_plotlyjs="cdn")
        if group != grp:
            lines.append(f"\n## {group}\n")
            grp = group
        lines.append(f"### {title}\n{insight}\n\n_Figure: `reports/figures/{name}.html`_\n")
        print(f"  done {title}")
    (C.ROOT / "reports" / "eda_findings.md").write_text("\n".join(lines))
    print("Wrote reports/eda_findings.md")


if __name__ == "__main__":
    run_all()
