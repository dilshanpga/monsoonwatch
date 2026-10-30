"""Step 3: evaluate every model ONCE on the untouched 2022-2023 test set.

Outputs: reports/model_comparison.csv, reports/metrics.json and figures
(ROC, precision-recall, calibration, confusion matrix, per-city recall).

Run:  python src/evaluate.py
"""
import json

import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve, precision_recall_curve)

import plot_style as ps
from config import FEATURES, FIG_DIR, MODEL_DIR, PROCESSED_DIR, RESULTS_DIR, TARGET

FILES = {"EBM (proposed)": "ebm", "Logistic Regression": "logistic_regression",
         "Random Forest": "random_forest", "LightGBM": "lightgbm"}


def metrics(y, p, thr):
    yhat = (p >= thr).astype(int)
    return {
        "ROC-AUC": roc_auc_score(y, p), "PR-AUC": average_precision_score(y, p),
        "F1": f1_score(y, yhat), "Precision": precision_score(y, yhat),
        "Recall": recall_score(y, yhat), "Accuracy": accuracy_score(y, yhat),
        "Balanced acc.": balanced_accuracy_score(y, yhat), "Brier": brier_score_loss(y, p),
        "Threshold": thr,
    }


def main():
    ps.apply()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(PROCESSED_DIR / "test.csv", dtype={"weathercode": str}, parse_dates=["date"])
    meta = json.loads((MODEL_DIR / "model_meta.json").read_text())
    y = test[TARGET].values

    probs, rows = {}, []
    for name, f in FILES.items():
        with open(MODEL_DIR / f"{f}.pkl", "rb") as fh:
            model = pickle.load(fh)
        probs[name] = model.predict_proba(test[FEATURES])[:, 1]
        m = metrics(y, probs[name], meta["models"][name]["threshold"])
        rows.append({"Model": name, **{k: round(v, 4) for k, v in m.items()}})
    # Two naive references any useful model must beat
    persist = (test["rain_today"] >= 20).astype(int)
    rows.append({"Model": "Persistence (heavy today -> heavy tomorrow)",
                 "F1": round(f1_score(y, persist), 4), "Precision": round(precision_score(y, persist), 4),
                 "Recall": round(recall_score(y, persist), 4), "Accuracy": round(accuracy_score(y, persist), 4),
                 "Balanced acc.": round(balanced_accuracy_score(y, persist), 4)})
    rows.append({"Model": "Always 'no heavy rain'", "Accuracy": round(1 - y.mean(), 4),
                 "F1": 0.0, "Recall": 0.0, "Balanced acc.": 0.5})
    # 95% bootstrap confidence intervals (1,000 resamples of test days)
    rng = np.random.RandomState(0)
    idx = [rng.randint(0, len(y), len(y)) for _ in range(1000)]
    for r in rows[:len(FILES)]:
        p = probs[r["Model"]]
        roc = [roc_auc_score(y[i], p[i]) for i in idx]
        pr = [average_precision_score(y[i], p[i]) for i in idx]
        r["ROC-AUC 95% CI"] = f"{np.percentile(roc, 2.5):.3f}-{np.percentile(roc, 97.5):.3f}"
        r["PR-AUC 95% CI"] = f"{np.percentile(pr, 2.5):.3f}-{np.percentile(pr, 97.5):.3f}"
    ebm_p_ = probs["EBM (proposed)"]
    sig = []
    for other in list(FILES)[1:]:
        diff = [average_precision_score(y[i], ebm_p_[i]) - average_precision_score(y[i], probs[other][i]) for i in idx]
        sig.append(f"PR-AUC gain of EBM over {other}: {np.mean(diff):.3f} "
                   f"(95% CI {np.percentile(diff, 2.5):.3f} to {np.percentile(diff, 97.5):.3f})")
        print(sig[-1])
    (RESULTS_DIR / "significance.txt").write_text("\n".join(sig) + "\n")
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    print(table.to_string(index=False))

    ebm_p, ebm_thr = probs["EBM (proposed)"], meta["models"]["EBM (proposed)"]["threshold"]
    cm = confusion_matrix(y, (ebm_p >= ebm_thr).astype(int))

    # Per-city and per-season results for the EBM (used in the fairness discussion)
    test = test.assign(p=ebm_p, pred=(ebm_p >= ebm_thr).astype(int))
    def grp(d):
        return pd.Series({"days": len(d), "heavy_days": int(d[TARGET].sum()),
                          "recall": recall_score(d[TARGET], d["pred"], zero_division=0),
                          "precision": precision_score(d[TARGET], d["pred"], zero_division=0),
                          "pr_auc": average_precision_score(d[TARGET], d["p"]) if d[TARGET].any() else np.nan})
    by_city = test.groupby("city").apply(grp).sort_values("recall")
    by_city.round(3).to_csv(RESULTS_DIR / "ebm_by_city.csv")
    season = test["date"].dt.month.map(lambda m: "SW monsoon (May-Sep)" if 5 <= m <= 9 else
                                       "NE monsoon (Dec-Feb)" if m in (12, 1, 2) else "Inter-monsoon")
    by_season = test.assign(season=season).groupby("season").apply(grp)
    by_season.round(3).to_csv(RESULTS_DIR / "ebm_by_season.csv")

    (RESULTS_DIR / "metrics.json").write_text(json.dumps({
        "test_rows": int(len(test)), "test_heavy_rate": float(y.mean()),
        "ebm_confusion_matrix": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                                 "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
        "comparison": rows}, indent=2, default=float))

    # ---- Figures --------------------------------------------------------
    order = list(FILES)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for i, name in enumerate(order):
        fpr, tpr, _ = roc_curve(y, probs[name])
        axes[0].plot(fpr, tpr, color=ps.SERIES[i], label=f"{name} ({roc_auc_score(y, probs[name]):.3f})")
        pr, rc, _ = precision_recall_curve(y, probs[name])
        axes[1].plot(rc, pr, color=ps.SERIES[i], label=f"{name} ({average_precision_score(y, probs[name]):.3f})")
    axes[0].plot([0, 1], [0, 1], ls="--", color=ps.GRID, lw=1)
    axes[0].set(title="ROC curve (test 2022-23)", xlabel="False positive rate", ylabel="True positive rate")
    axes[1].axhline(y.mean(), ls="--", color=ps.GRID, lw=1)
    axes[1].set(title="Precision-recall curve (test 2022-23)", xlabel="Recall", ylabel="Precision")
    for a in axes:
        a.legend(fontsize=8, loc="lower right" if a is axes[0] else "upper right")
    fig.savefig(FIG_DIR / "roc_pr_curves.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.2))
    for i, name in enumerate(order):
        fr, mp = calibration_curve(y, probs[name], n_bins=10, strategy="quantile")
        ax.plot(mp, fr, marker="o", ms=4, color=ps.SERIES[i], label=name)
    lim = 0.45
    ax.plot([0, lim], [0, lim], ls="--", color=ps.INK_2, lw=1, label="Perfect calibration")
    ax.set(title="Calibration (test, 10 equal-size bins)", xlabel="Predicted probability",
           ylabel="Observed heavy-rain frequency", xlim=(0, lim), ylim=(0, lim))
    ax.legend(fontsize=8)
    fig.savefig(FIG_DIR / "calibration.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues"); ax.grid(False)
    for (r, c), v in np.ndenumerate(cm):
        ax.text(c, r, f"{v:,}", ha="center", va="center", fontsize=12,
                color="white" if v > cm.max() / 2 else ps.INK)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["No heavy rain", "Heavy rain"],
           yticklabels=["No heavy rain", "Heavy rain"], xlabel="Predicted", ylabel="Actual",
           title=f"EBM confusion matrix (threshold {ebm_thr:.2f})")
    fig.savefig(FIG_DIR / "confusion_matrix.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6.5))
    bc = by_city.sort_values("recall")
    ax.barh(bc.index, bc["recall"], color=ps.SERIES[0], height=0.7)
    ax.axvline(cm[1, 1] / (cm[1, 0] + cm[1, 1]), color=ps.SERIES[1], ls="--", lw=1.5, label="Overall recall")
    ax.set(title="EBM recall of heavy-rain days by city (test)", xlabel="Recall", xlim=(0, 1))
    ax.legend(loc="lower right")
    fig.savefig(FIG_DIR / "recall_by_city.png"); plt.close(fig)


if __name__ == "__main__":
    main()
