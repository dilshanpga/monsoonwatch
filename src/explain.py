"""Step 4: explain the model with four complementary methods.

1. EBM global term importance   - what matters on average (built into the model)
2. EBM shape functions          - HOW each feature changes the risk (exact, not approximated)
3. SHAP                         - model-agnostic check: permutation SHAP on the EBM and
                                  TreeSHAP on the LightGBM baseline; do they agree?
4. Partial dependence plots     - EBM vs LightGBM on the key drivers
5. Local explanations           - why the EBM flagged (or missed) specific days

Run:  python src/explain.py
"""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

import plot_style as ps
from config import FEATURES, FIG_DIR, LABELS, MODEL_DIR, PROCESSED_DIR, RANDOM_STATE, RESULTS_DIR, TARGET

MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()


def label(term):
    return " × ".join(LABELS.get(t.strip(), t.strip()) for t in term.split("&"))


def shape_data(ebm, feature):
    """Bin edges / categories and the log-odds score for one main-effect term."""
    i = ebm.term_names_.index(feature)
    d = ebm.explain_global().data(i)
    return d["names"], np.asarray(d["scores"]), d.get("lower_bounds"), d.get("upper_bounds")


def plot_importance(ebm):
    imp = pd.Series(ebm.term_importances(), index=ebm.term_names_).sort_values()
    top = imp.tail(15)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    colors = [ps.SERIES[1] if "&" in t else ps.SERIES[0] for t in top.index]
    ax.barh([label(t) for t in top.index], top.values, color=colors, height=0.7)
    ax.set(title="EBM global importance (mean |contribution|, log-odds)", xlabel="Mean absolute score")
    if any("&" in t for t in top.index):
        ax.scatter([], [], color=ps.SERIES[0], marker="s", label="Single feature")
        ax.scatter([], [], color=ps.SERIES[1], marker="s", label="Pairwise interaction")
        ax.legend(loc="lower right")
    n_int = sum("&" in t for t in imp.index)
    ax.text(0.99, 0.02, f"{len(imp) - n_int} single-feature terms + {n_int} pairwise interactions in the model;\n"
            f"the strongest interaction ranks #{list(imp.sort_values(ascending=False).index).index(next(t for t in imp.sort_values(ascending=False).index if '&' in t)) + 1}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=ps.INK_2)
    fig.savefig(FIG_DIR / "ebm_global_importance.png"); plt.close(fig)
    return imp.sort_values(ascending=False)


def plot_shapes(ebm, feats):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.8))
    for ax, f in zip(axes.flat, feats):
        names, scores, lo, hi = shape_data(ebm, f)
        if f == "month" or f == "weathercode":
            # treat as discrete: midpoint of each bin
            if f == "month":
                # score of the bin that contains each calendar month
                edges = np.asarray(names, dtype=float)
                x = np.arange(1, 13)
                scores = scores[np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(scores) - 1)]
            else:
                x = np.arange(len(names))
            ax.bar(x, scores, color=np.where(scores > 0, ps.UP, ps.DOWN), width=0.7)
            if f == "month":
                ax.set_xticks(range(1, 13)); ax.set_xticklabels([m[0] for m in MONTHS])
            else:
                ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
        else:
            edges = np.asarray(names, dtype=float)
            ax.step(edges[:-1], scores, where="post", color=ps.SERIES[0])
            if lo is not None:
                ax.fill_between(edges[:-1], lo, hi, step="post", color=ps.SERIES[0], alpha=0.15, lw=0)
            # clip the long right tail so the informative range is visible
            ax.set_xlim(edges[0], np.percentile(edges, 97) if f.startswith("rain") else edges[-2])
        ax.axhline(0, color=ps.INK_2, lw=0.8)
        ax.set_title(LABELS[f], fontsize=10)
        ax.set_ylabel("Contribution (log-odds)")
    fig.suptitle("EBM shape functions: how each feature moves tomorrow's heavy-rain risk", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ebm_shape_functions.png"); plt.close(fig)


def plot_interaction(ebm, term):
    i = ebm.term_names_.index(term)
    d = ebm.explain_global().data(i)
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    x, y = np.asarray(d["left_names"], dtype=float), np.asarray(d["right_names"], dtype=float)
    z = np.asarray(d["scores"]).T
    lim = np.abs(z).max()
    m = ax.pcolormesh(x, y, z, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="flat")
    a, b = term.split(" & ")
    ax.set(xlabel=LABELS[a], ylabel=LABELS[b], title=f"Interaction: {label(term)}")
    if a.startswith("rain"):
        ax.set_xlim(x[0], np.percentile(x, 95))
    ax.grid(False)
    fig.colorbar(m, ax=ax, label="Contribution (log-odds)")
    fig.savefig(FIG_DIR / "ebm_top_interaction.png"); plt.close(fig)


def logit(model, X):
    p = np.clip(model.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def run_shap(ebm, lgbm, train, test):
    rng = np.random.RandomState(RANDOM_STATE)
    bg = train[FEATURES].sample(100, random_state=RANDOM_STATE)
    # oversample heavy-rain days so the explained set is informative
    pos = test[test[TARGET] == 1].sample(100, random_state=RANDOM_STATE)
    neg = test[test[TARGET] == 0].sample(200, random_state=RANDOM_STATE)
    X = pd.concat([pos, neg])[FEATURES]

    # Permutation SHAP is model-agnostic, so it works directly on the EBM.
    # SHAP needs a numeric matrix, so the weather code travels as a number and is
    # turned back into the category label inside the wrapped model.
    def numeric(d):
        return d.assign(weathercode=d["weathercode"].astype(float)).astype(float)

    def f(arr):
        d = pd.DataFrame(arr, columns=FEATURES)
        d["weathercode"] = d["weathercode"].round().astype(int).astype(str)
        return logit(ebm, d)
    expl = shap.PermutationExplainer(f, shap.maskers.Independent(numeric(bg), max_samples=100))
    sv_ebm = expl(numeric(X), max_evals=2 * len(FEATURES) + 1, silent=True)
    sv_ebm.feature_names = [LABELS[c] for c in FEATURES]

    # TreeSHAP on LightGBM for comparison (needs the transformed matrix)
    prep, clf = lgbm.named_steps["prep"], lgbm.named_steps["clf"]
    Xt = prep.transform(X)
    tree_vals = shap.TreeExplainer(clf).shap_values(Xt)
    tree_vals = tree_vals[1] if isinstance(tree_vals, list) else tree_vals
    names_t = list(prep.get_feature_names_out())
    # fold the one-hot weathercode columns back into one feature
    lgb_imp = {}
    for j, n in enumerate(names_t):
        base = "weathercode" if n.startswith("cat__") else n.split("__", 1)[1]
        lgb_imp[base] = lgb_imp.get(base, 0) + np.abs(tree_vals[:, j]).mean()

    ebm_imp = pd.Series(np.abs(sv_ebm.values).mean(0), index=FEATURES)
    lgb_imp = pd.Series(lgb_imp)[FEATURES]
    rho = spearmanr(ebm_imp, lgb_imp).statistic

    plt.figure()
    shap.plots.beeswarm(sv_ebm, max_display=12, show=False, plot_size=(8, 5.5))
    plt.title("SHAP values for the EBM (300 test days)", fontweight="bold")
    plt.savefig(FIG_DIR / "shap_beeswarm_ebm.png"); plt.close("all")

    comp = pd.DataFrame({"EBM (permutation SHAP)": ebm_imp / ebm_imp.sum(),
                         "LightGBM (TreeSHAP)": lgb_imp / lgb_imp.sum()})
    comp = comp.sort_values("EBM (permutation SHAP)").tail(12)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    yy = np.arange(len(comp))
    ax.barh(yy + 0.2, comp.iloc[:, 0], height=0.38, color=ps.SERIES[0], label=comp.columns[0])
    ax.barh(yy - 0.2, comp.iloc[:, 1], height=0.38, color=ps.SERIES[1], label=comp.columns[1])
    ax.set_yticks(yy); ax.set_yticklabels([LABELS[c] for c in comp.index])
    ax.set(xlabel="Share of total mean |SHAP|", title=f"Do the two models rely on the same features? (Spearman ρ = {rho:.2f})")
    ax.legend(loc="lower right")
    fig.savefig(FIG_DIR / "shap_ebm_vs_lightgbm.png"); plt.close(fig)
    return ebm_imp.sort_values(ascending=False), lgb_imp.sort_values(ascending=False), float(rho)


def pdp(model, X, feature, grid):
    out = []
    for v in grid:
        Xc = X.copy(); Xc[feature] = v
        out.append(model.predict_proba(Xc)[:, 1].mean())
    return np.array(out)


def plot_pdps(ebm, lgbm, test):
    X = test[FEATURES].sample(2000, random_state=RANDOM_STATE)
    specs = [("rain_7d", np.linspace(0, 150, 31)), ("month", np.arange(1, 13)),
             ("wind_dir", np.arange(0, 361, 15)), ("radiation", np.linspace(8, 26, 19))]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    out = {}
    for ax, (f, grid) in zip(axes, specs):
        e, l = pdp(ebm, X, f, grid), pdp(lgbm, X, f, grid)
        out[f] = {"grid": grid.tolist(), "ebm": e.round(4).tolist(), "lightgbm": l.round(4).tolist()}
        ax.plot(grid, e, color=ps.SERIES[0], label="EBM")
        ax.plot(grid, l, color=ps.SERIES[1], label="LightGBM")
        ax.set(title=LABELS[f], ylabel="Avg. predicted probability")
        if f == "month":
            ax.set_xticks(range(1, 13)); ax.set_xticklabels([m[0] for m in MONTHS])
    axes[0].legend()
    fig.suptitle("Partial dependence (2,000 test days)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pdp_ebm_vs_lightgbm.png"); plt.close(fig)
    return out


def local_cases(ebm, test, thr):
    p = ebm.predict_proba(test[FEATURES])[:, 1]
    t = test.assign(p=p, pred=(p >= thr).astype(int))
    picks = {
        "Correct warning (true positive)": t[(t[TARGET] == 1) & (t.pred == 1)].sort_values("p").iloc[-1],
        "False alarm (false positive)": t[(t[TARGET] == 0) & (t.pred == 1)].sort_values("p").iloc[-1],
        "Missed heavy rain (false negative)": t[(t[TARGET] == 1) & (t.pred == 0)].sort_values("rain_tomorrow_mm").iloc[-1],
    }
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    cases = {}
    for ax, (title, row) in zip(axes, picks.items()):
        loc = ebm.explain_local(row[FEATURES].to_frame().T, np.array([row[TARGET]]))
        d = loc.data(0)
        s = pd.Series(d["scores"], index=d["names"])
        top = s.reindex(s.abs().sort_values().index).tail(8)
        ax.barh([label(n) for n in top.index], top.values,
                color=[ps.UP if v > 0 else ps.DOWN for v in top.values], height=0.7)
        ax.axvline(0, color=ps.INK_2, lw=0.8)
        ax.set_title(f"{title}\n{row['city']}, {pd.Timestamp(row['date']).date()} → next day "
                     f"{row['rain_tomorrow_mm']:.0f} mm, p = {row['p']:.2f}", fontsize=9.5)
        ax.set_xlabel("Contribution (log-odds)")
        cases[title] = {"city": row["city"], "date": str(pd.Timestamp(row["date"]).date()),
                        "rain_tomorrow_mm": float(row["rain_tomorrow_mm"]), "prob": float(row["p"]),
                        "top_contributions": {k: round(float(v), 3) for k, v in top[::-1].items()},
                        "intercept": float(d["extra"]["scores"][0])}
    fig.tight_layout()
    fig.savefig(FIG_DIR / "local_explanations.png"); plt.close(fig)
    return cases


def main():
    ps.apply()
    train = pd.read_csv(PROCESSED_DIR / "train.csv", dtype={"weathercode": str}, parse_dates=["date"])
    test = pd.read_csv(PROCESSED_DIR / "test.csv", dtype={"weathercode": str}, parse_dates=["date"])
    meta = json.loads((MODEL_DIR / "model_meta.json").read_text())
    ebm = joblib.load(MODEL_DIR / "ebm.joblib")
    lgbm = joblib.load(MODEL_DIR / "lightgbm.joblib")

    imp = plot_importance(ebm)
    plot_shapes(ebm, ["rain_7d", "month", "wind_dir", "radiation", "rain_today", "elevation"])
    inter = [t for t in imp.index if "&" in t]
    if inter:
        plot_interaction(ebm, inter[0])
    shap_ebm, shap_lgb, rho = run_shap(ebm, lgbm, train, test)
    pdps = plot_pdps(ebm, lgbm, test)
    cases = local_cases(ebm, test, meta["models"]["EBM (proposed)"]["threshold"])

    out = {
        "ebm_term_importance": imp.round(4).to_dict(),
        "shap_mean_abs_ebm": shap_ebm.round(4).to_dict(),
        "shap_mean_abs_lightgbm": shap_lgb.round(4).to_dict(),
        "spearman_rank_agreement": round(rho, 3),
        "pdp": pdps,
        "local_cases": cases,
    }
    (RESULTS_DIR / "explanations.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "pdp"}, indent=2)[:4000])


if __name__ == "__main__":
    main()
