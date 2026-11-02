"""Steps 3 & 4 - Model development + MLflow integration.

Run:  python -m src.train            (full data)
      python -m src.train --quick    (20 % sample, for smoke tests)

For both tasks three candidate models are trained, evaluated and logged to
MLflow (params, metrics, artifacts, model). The best model per task is
registered in the MLflow Model Registry with the alias ``champion`` and also
saved to ``models/*.joblib`` so the Streamlit app can run without an MLflow
server.
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from . import config as C
from .features import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES

CLF_EXP = "RealEstate_Good_Investment_Classification"
REG_EXP = "RealEstate_Future_Price_Regression"
CLF_REG_NAME = "RealEstate_GoodInvestment_Classifier"
REG_REG_NAME = "RealEstate_FuturePrice_Regressor"


def make_preprocessor() -> ColumnTransformer:
    """Scale numeric features, one-hot encode categorical ones."""
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])


def classifiers():
    rs = C.RANDOM_STATE
    return {
        "LogisticRegression": (LogisticRegression(max_iter=1000, C=1.0), {"C": 1.0, "max_iter": 1000}),
        "RandomForest": (RandomForestClassifier(n_estimators=60, max_depth=12, max_samples=0.5,
                                                n_jobs=-1, random_state=rs),
                         {"n_estimators": 60, "max_depth": 12, "max_samples": 0.5}),
        "XGBoost": (XGBClassifier(n_estimators=250, max_depth=6, learning_rate=0.1, subsample=0.9,
                                  colsample_bytree=0.9, tree_method="hist", n_jobs=-1,
                                  random_state=rs, eval_metric="logloss"),
                    {"n_estimators": 250, "max_depth": 6, "learning_rate": 0.1,
                     "subsample": 0.9, "colsample_bytree": 0.9}),
    }


def regressors():
    rs = C.RANDOM_STATE
    return {
        "LinearRegression": (LinearRegression(), {}),
        "RandomForest": (RandomForestRegressor(n_estimators=60, max_depth=14, max_samples=0.5,
                                               n_jobs=-1, random_state=rs),
                         {"n_estimators": 60, "max_depth": 14, "max_samples": 0.5}),
        "XGBoost": (XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.9,
                                 colsample_bytree=0.9, tree_method="hist", n_jobs=-1, random_state=rs),
                    {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.1,
                     "subsample": 0.9, "colsample_bytree": 0.9}),
    }


def feature_importance(pipe: Pipeline) -> pd.DataFrame:
    """Importance per one-hot column and aggregated per original feature."""
    names = pipe.named_steps["prep"].get_feature_names_out()
    model = pipe.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    else:
        coef = np.abs(np.atleast_2d(model.coef_)[0] if hasattr(model, "coef_") else model.coef_)
        imp = coef / coef.sum()
    df = pd.DataFrame({"encoded_feature": names, "importance": imp})
    df["feature"] = df["encoded_feature"].str.replace(r"^(num|cat)__", "", regex=True)
    for cat in CATEGORICAL_FEATURES:  # collapse one-hot columns back to their parent
        df.loc[df["feature"].str.startswith(cat + "_"), "feature"] = cat
    agg = df.groupby("feature", as_index=False)["importance"].sum()
    return agg.sort_values("importance", ascending=False).reset_index(drop=True)


def save_confusion_png(cm, name, path):
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center", color="black")
    ax.set_xticks([0, 1], ["Not Good", "Good"])
    ax.set_yticks([0, 1], ["Not Good", "Good"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix - {name}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def register_champion(client: MlflowClient, model_uri: str, name: str, metric_note: str):
    mv = mlflow.register_model(model_uri, name)
    client.set_registered_model_alias(name, "champion", mv.version)
    client.update_model_version(name, mv.version, description=f"Champion model. {metric_note}")
    return mv.version


def main(quick: bool = False):
    t0 = time.time()
    C.MODELS_DIR.mkdir(exist_ok=True)
    tmp = C.MODELS_DIR / "_tmp"
    tmp.mkdir(exist_ok=True)

    df = pd.read_csv(C.CLEAN_CSV)
    if quick:
        df = df.sample(frac=0.2, random_state=C.RANDOM_STATE)
    X = df[MODEL_FEATURES]
    print(f"Training on {len(df):,} rows, {len(MODEL_FEATURES)} raw features")

    mlflow.set_tracking_uri(C.MLFLOW_URI)
    client = MlflowClient()
    summary: dict = {"n_rows": int(len(df)), "classification": {}, "regression": {}}

    # ------------------------------------------------------------------ classification
    y = df[C.TARGET_CLF]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=C.RANDOM_STATE)
    mlflow.set_experiment(CLF_EXP)
    best = {"f1": -1}
    for name, (est, params) in classifiers().items():
        pipe = Pipeline([("prep", make_preprocessor()), ("model", est)])
        with mlflow.start_run(run_name=name) as run:
            ts = time.time()
            pipe.fit(Xtr, ytr)
            pred = pipe.predict(Xte)
            proba = pipe.predict_proba(Xte)[:, 1]
            m = {
                "accuracy": accuracy_score(yte, pred),
                "precision": precision_score(yte, pred),
                "recall": recall_score(yte, pred),
                "f1": f1_score(yte, pred),
                "roc_auc": roc_auc_score(yte, proba),
            }
            cm = confusion_matrix(yte, pred)
            fi = feature_importance(pipe)
            mlflow.log_params({"model": name, "n_train": len(Xtr), "n_features": len(MODEL_FEATURES),
                               "threshold_score": C.GOOD_INVESTMENT_SCORE_THRESHOLD, **params})
            mlflow.log_metrics({k: float(v) for k, v in m.items()})
            mlflow.log_metric("train_seconds", time.time() - ts)
            cm_path = tmp / f"cm_{name}.png"
            save_confusion_png(cm, name, cm_path)
            mlflow.log_artifact(str(cm_path))
            fi_path = tmp / f"fi_clf_{name}.csv"
            fi.to_csv(fi_path, index=False)
            mlflow.log_artifact(str(fi_path))
            info = mlflow.sklearn.log_model(pipe, name="model", input_example=Xtr.head(3),
                                          serialization_format="cloudpickle")
            print(f"[CLF] {name:20s} " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()))
            fpr, tpr, _ = roc_curve(yte, proba)
            idx = np.linspace(0, len(fpr) - 1, 100).astype(int)
            summary["classification"][name] = {
                **{k: float(v) for k, v in m.items()},
                "confusion_matrix": cm.tolist(),
                "roc": {"fpr": fpr[idx].tolist(), "tpr": tpr[idx].tolist()},
                "feature_importance": fi.head(25).to_dict("records"),
                "run_id": run.info.run_id,
            }
            if m["f1"] > best["f1"]:
                best = {"f1": m["f1"], "name": name, "pipe": pipe, "uri": info.model_uri, "m": m}
    ver = register_champion(client, best["uri"], CLF_REG_NAME,
                            f"{best['name']} F1={best['f1']:.4f}")
    joblib.dump(best["pipe"], C.MODELS_DIR / "classifier.joblib", compress=3)
    summary["best_classifier"] = {"name": best["name"], "registry_version": ver, **{k: float(v) for k, v in best["m"].items()}}

    # ------------------------------------------------------------------ regression
    y = df[C.TARGET_REG]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=C.RANDOM_STATE)
    mlflow.set_experiment(REG_EXP)
    best = {"rmse": np.inf}
    for name, (est, params) in regressors().items():
        pipe = Pipeline([("prep", make_preprocessor()), ("model", est)])
        with mlflow.start_run(run_name=name) as run:
            ts = time.time()
            pipe.fit(Xtr, ytr)
            pred = pipe.predict(Xte)
            m = {"rmse": float(np.sqrt(mean_squared_error(yte, pred))),
                 "mae": float(mean_absolute_error(yte, pred)),
                 "r2": float(r2_score(yte, pred))}
            fi = feature_importance(pipe)
            mlflow.log_params({"model": name, "n_train": len(Xtr), "n_features": len(MODEL_FEATURES),
                               "horizon_years": C.HORIZON_YEARS, **params})
            mlflow.log_metrics(m)
            mlflow.log_metric("train_seconds", time.time() - ts)
            fi_path = tmp / f"fi_reg_{name}.csv"
            fi.to_csv(fi_path, index=False)
            mlflow.log_artifact(str(fi_path))
            info = mlflow.sklearn.log_model(pipe, name="model", input_example=Xtr.head(3),
                                          serialization_format="cloudpickle")
            print(f"[REG] {name:20s} " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()))
            summary["regression"][name] = {**m, "feature_importance": fi.head(25).to_dict("records"),
                                           "run_id": run.info.run_id}
            if m["rmse"] < best["rmse"]:
                best = {"rmse": m["rmse"], "name": name, "pipe": pipe, "uri": info.model_uri, "m": m}
    ver = register_champion(client, best["uri"], REG_REG_NAME,
                            f"{best['name']} RMSE={best['rmse']:.3f} lakhs")
    joblib.dump(best["pipe"], C.MODELS_DIR / "regressor.joblib", compress=3)
    summary["best_regressor"] = {"name": best["name"], "registry_version": ver, **best["m"]}

    (C.MODELS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone in {time.time() - t0:.0f}s. Best classifier: {summary['best_classifier']['name']}, "
          f"best regressor: {summary['best_regressor']['name']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
