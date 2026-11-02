# 🏙️ Real Estate Investment Advisor: Predicting Property Profitability & Future Value

Machine-learning application that (1) **classifies** whether a property is a *Good Investment* and
(2) **predicts its price after 5 years**, with MLflow experiment tracking and a Streamlit front end.
Built exactly to the project brief: Preprocessing → EDA → Model development → MLflow → Streamlit.

## Project structure
```
real_estate_advisor/
├── data/raw/india_housing_prices.csv      # provided data set (250,000 x 23)
├── data/processed/cleaned_housing.csv     # DELIVERABLE: cleaned + engineered + both targets (250,000 x 46)
├── src/                                   # BACKEND
│   ├── config.py          growth rates, thresholds, paths
│   ├── preprocessing.py   duplicates, missing values, logic fixes, IQR outlier flags
│   ├── features.py        feature engineering + Good_Investment label + Future_Price_5Y (shared w/ app)
│   ├── build_dataset.py   raw -> processed CSV + reference medians
│   ├── eda.py             all 20 EDA questions + 4 Step-2 analyses -> reports/
│   └── train.py           3 classifiers + 3 regressors, MLflow logging + Model Registry
├── app/streamlit_app.py                   # FRONTEND (6 pages)
├── models/                classifier.joblib, regressor.joblib, metrics.json, reference_stats.json
├── mlflow.db + mlruns/    MLflow experiment logs + registered models
├── reports/               eda_findings.md + 24 interactive figures (reports/figures/*.html)
├── docs/PROJECT_DOCUMENTATION.md          methodology, findings, use cases
└── tests/test_pipeline.py
```

## Quick start
```bash
pip install -r requirements.txt
./run_pipeline.sh                      # or run the three steps below individually
python -m src.build_dataset            # cleaning + features + targets
python -m src.eda                      # EDA report
python -m src.train                    # train, log to MLflow, register champions (~3 min on 1 CPU)

streamlit run app/streamlit_app.py     # the app
mlflow ui --backend-store-uri sqlite:///mlflow.db    # experiment tracking UI
pytest -q                              # tests
```
> Pre-built artifacts are included, so `streamlit run app/streamlit_app.py` works immediately.
> If you move the folder, re-run `python -m src.train` to refresh MLflow's absolute artifact paths.

## App pages
| Page | What it does |
|---|---|
| 🏠 Overview | KPIs, location heatmaps (city × property type), growth outlook by city |
| 🔮 Investment Advisor | Property form → **"Is this a Good Investment?"** + model confidence, **estimated price after 5 years**, 5-rule scorecard, price path, feature importance, comparable listings |
| 🔎 Property Explorer | Filter by state, city, type, BHK, area, price, availability; ranked table, CSV download, heatmap |
| 📊 Market Insights | The 20 brief questions + 4 Step-2 analyses, each with an auto-generated insight |
| 🧠 Model Performance | Metrics, confusion matrix, ROC, feature importance, champion/registry info |
| 📘 About | Methodology and data limits |

## Results (20 % hold-out, 50,000 listings)
| Task | Model | Key metrics |
|---|---|---|
| Classification | Logistic Regression | Acc 0.889 · F1 0.896 · AUC 0.963 |
| | Random Forest | Acc 0.994 · F1 0.994 · AUC 0.999 |
| | **XGBoost (champion)** | **Acc 0.996 · Prec 0.996 · Rec 0.997 · F1 0.996 · AUC 1.000** |
| Regression | Linear Regression | RMSE 8.59 L · MAE 6.05 L · R² 0.9985 |
| | **Random Forest (champion)** | **RMSE 0.38 L · MAE 0.24 L · R² 1.000** |
| | XGBoost | RMSE 1.25 L · MAE 1.00 L · R² 1.000 |

## ⚠️ Important honesty notes
1. **Targets are rule-derived.** The CSV has no future price or investment label, so, as the brief allows
   ("Fixed rate / By location-type / Feature-based", "Price vs median / Multi-factor score"), both are generated from
   documented rules (`src/features.py`, `src/config.py`). The near-perfect scores therefore show the models
   recover those rules; they are **not** evidence of real-world forecasting ability.
2. **The data set is essentially synthetic/uniform** (price independent of size, city and amenities; BHK, size and
   price near-uniform). EDA therefore finds very small differences between groups, and the app reports that honestly.
3. **No crime-rate, RERA or sale-date columns exist.** Security is used as the safety proxy; year built replaces
   time for "trend" charts.
4. **Data fixes:** 116,304 rows (46.5 %) had Floor_No > Total_Floors; Floor_No was capped at Total_Floors
   (flag column `Floor_No_Was_Invalid` kept). Locality names (`Locality_1..500`) repeat in every city, so Locality is
   used for EDA/filters but not as a model feature.
