# Project Documentation: Real Estate Investment Advisor

## 1. Methodology
### 1.1 Data preprocessing (`src/preprocessing.py`)
| Step | Result |
|---|---|
| Duplicates (excluding ID) | 0 found |
| Missing values | 0 found (code imputes median / mode if present) |
| Floor_No > Total_Floors | 116,304 rows (46.5 %) capped to Total_Floors, flagged |
| Age_of_Property | Re-derived as 2025 − Year_Built (already consistent) |
| Price_per_SqFt | Source column is in lakhs/sq ft (0.13); added `Price_per_SqFt_INR` (≈ ₹13,000) |
| Outliers (IQR 1.5×) | Price: 0 · Size: 0 · Price/sq ft: 19,723 (7.9 %): flagged, kept |
| Scaling / encoding | `StandardScaler` on numeric, `OneHotEncoder` on City, Property_Type, Furnished_Status, Facing, Owner_Type, Availability_Status, inside one sklearn `Pipeline` (no train/serve skew) |

### 1.2 Feature engineering (`src/features.py`)
Price per sq ft (INR), Amenities_Count + 5 amenity flags, Parking/Security flags, Transport_Score (Low .33 / Med .67 /
High 1), **School_Density_Score** (schools/10), **Infrastructure_Score** (0–10 composite of schools, hospitals,
transport, parking, security, amenities), Floor_Ratio, Size_per_BHK, Is_Metro.

### 1.3 Targets
**Good Investment**: 1 if ≥ 3 of 5 rules pass (53.3 % positive):
1. price/sq ft ≤ median of same city & property type · 2. price ≤ city median · 3. BHK ≥ 3 · 4. ready to move ·
5. infrastructure score ≥ market median.

**Future_Price_5Y** = `Price × (1 + r)^5`, `r = 8 % + type adj (Apt 0, House +0.4, Villa +0.8 pts) + transport adj
(Low 0, Med +0.4, High +1.0) + under-construction +0.8 + metro city +1.2`. Realised r spans 8.0–11.8 %.
All rates are editable in `src/config.py`.

### 1.4 Models & evaluation
Stratified 80/20 split (seed 42). Classifiers: Logistic Regression, Random Forest, XGBoost, evaluated with Accuracy,
Precision, Recall, F1, ROC-AUC and confusion matrix. Regressors: Linear Regression, Random Forest, XGBoost, evaluated
with RMSE, MAE, R². Best by F1 / RMSE becomes the champion.

### 1.5 MLflow
Two experiments (`RealEstate_Good_Investment_Classification`, `RealEstate_Future_Price_Regression`) log params,
metrics, training time, confusion-matrix PNG, feature-importance CSV and the full pipeline. Champions are registered
as `RealEstate_GoodInvestment_Classifier` and `RealEstate_FuturePrice_Regressor` with alias `champion`.
Backend: SQLite (`mlflow.db`). Models use cloudpickle serialization (self-trained models only).

## 2. Findings
See `reports/eda_findings.md` (auto-generated, one insight per question). Headlines:
* Prices (₹10–500 L) and sizes (500–5,000 sq ft) are near-uniform; no right skew.
* Size vs total price r ≈ 0. Price/sq ft falls with size (r ≈ −0.61), purely mechanically.
* State, city, type, furnishing, facing, parking, amenities and schools/hospitals all change average prices by only
  ~0–3 %. Listings are spread evenly over owner types and the two availability statuses.
* Under the rule set, transport access and metro location drive projected returns; availability, BHK and
  infrastructure score drive the Good-Investment call (top classifier features).

## 3. Use cases
* **Investors**: check a listing's long-term return and see which rules it passes/fails.
* **Buyers**: filter by city/BHK/budget and rank by investment score or growth.
* **Real-estate platforms**: batch-score listings with `classifier.joblib` / `regressor.joblib`.
* **Customer trust**: every verdict shows confidence, rule scorecard and feature importance.

## 4. Limitations & next steps
Targets are assumption-based; replace `annual_growth_rate()` with real historical price indices and the label with
real realised returns when available. Add geo-coordinates for map heatmaps, crime and RERA data, and time-stamped
sales for genuine trend analysis.
