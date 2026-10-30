"""Step 2: train the Explainable Boosting Machine (EBM) and three baselines.

* Hyperparameters of the EBM are chosen on the 2020-2021 validation years.
* The decision threshold is also chosen on validation (max F1 for the
  heavy-rain class), so the 2022-2023 test set is only used once, in evaluate.py.
* Baselines (logistic regression, random forest, LightGBM) were covered in
  lectures; they are here only for comparison.

Run:  python src/train.py            (full search, ~30 min on 2 CPU cores)
      python src/train.py --quick    (skip the search, use the chosen settings)
"""
import argparse
import itertools
import json
import time

import pickle
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (CATEGORICAL, FEATURES, MODEL_DIR, PROCESSED_DIR, RANDOM_STATE,
                    RESULTS_DIR, TARGET)

NUMERIC = [f for f in FEATURES if f not in CATEGORICAL]
# Settings picked by the search below (kept so --quick reproduces the final model)


def load(name):
    return pd.read_csv(PROCESSED_DIR / f"{name}.csv", dtype={"weathercode": str},
                       parse_dates=["date"])


def make_ebm(outer_bags=8, **params):
    """EBM with fixed settings + the tuned ones.

    outer_bags: the model is fitted on several bootstrap-style splits and averaged;
    the spread between bags gives the uncertainty bands on the shape plots.
    validation_size / early_stopping: 15% of the training rows are held out inside
    each bag to stop boosting when log-loss stops improving.
    """
    return ExplainableBoostingClassifier(
        feature_names=FEATURES,
        feature_types=["nominal" if f in CATEGORICAL else "continuous" for f in FEATURES],
        max_bins=256, max_rounds=5000, early_stopping_rounds=100, validation_size=0.15, min_samples_leaf=4,
        outer_bags=outer_bags, n_jobs=-1, random_state=RANDOM_STATE, **params)


def best_threshold(y, p):
    """Threshold that maximises F1 for the heavy-rain class."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    i = int(np.nanargmax(f1[:-1]))
    return float(thr[i]), float(f1[i])


def search_ebm(train, valid):
    """Grid search: fit on 2010-2019, score PR-AUC on 2020-2021.
    Two bags per fit during the search (for speed); the final model uses eight."""
    grid = {
        "interactions": [0, 10, 20],
        "learning_rate": [0.02, 0.05],
        "max_leaves": [2, 3],
    }
    rows = []
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), values))
        t0 = time.time()
        m = make_ebm(outer_bags=2, **params).fit(train[FEATURES], train[TARGET])
        p = m.predict_proba(valid[FEATURES])[:, 1]
        rows.append({**params,
                     "valid_pr_auc": round(average_precision_score(valid[TARGET], p), 4),
                     "valid_roc_auc": round(roc_auc_score(valid[TARGET], p), 4),
                     "fit_seconds": round(time.time() - t0, 1)})
        print(rows[-1], flush=True)
    res = pd.DataFrame(rows).sort_values("valid_pr_auc", ascending=False)
    res.to_csv(RESULTS_DIR / "ebm_hyperparameter_search.csv", index=False)
    best = res.iloc[0]
    return {"interactions": int(best["interactions"]), "learning_rate": float(best["learning_rate"]),
            "max_leaves": int(best["max_leaves"])}


def baselines():
    ohe = ColumnTransformer([("num", StandardScaler(), NUMERIC),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    trees = ColumnTransformer([("num", "passthrough", NUMERIC),
                               ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    return {
        "Logistic Regression": Pipeline([("prep", ohe), ("clf", LogisticRegression(
            max_iter=2000, C=1.0))]),
        "Random Forest": Pipeline([("prep", trees), ("clf", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=20, n_jobs=-1, random_state=RANDOM_STATE))]),
        "LightGBM": Pipeline([("prep", trees), ("clf", LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, random_state=RANDOM_STATE, verbose=-1))]),
    }


def main(quick: bool):
    MODEL_DIR.mkdir(exist_ok=True)
    train, valid = load("train"), load("valid")
    if quick:
        params = json.loads((MODEL_DIR / "model_meta.json").read_text())["ebm_params"]
    else:
        params = search_ebm(train, valid)
    print("EBM settings:", params)

    models = {"EBM (proposed)": make_ebm(**params)}
    models.update(baselines())
    meta = {"ebm_params": params, "models": {}}
    for name, model in models.items():
        t0 = time.time()
        model.fit(train[FEATURES], train[TARGET])
        secs = time.time() - t0
        p = model.predict_proba(valid[FEATURES])[:, 1]
        thr, f1 = best_threshold(valid[TARGET], p)
        meta["models"][name] = {"threshold": thr, "valid_f1": round(f1, 4),
                                "valid_pr_auc": round(average_precision_score(valid[TARGET], p), 4),
                                "valid_roc_auc": round(roc_auc_score(valid[TARGET], p), 4),
                                "fit_seconds": round(secs, 1)}
        fname = name.split(" (")[0].lower().replace(" ", "_")
        with open(MODEL_DIR / f"{fname}.pkl", "wb") as fh:  # saved with pickle
            pickle.dump(model, fh)
        print(name, meta["models"][name], flush=True)

    (MODEL_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
