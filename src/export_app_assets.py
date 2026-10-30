"""Step 5: copy everything the web app needs into app/ so that folder can be
deployed on its own (Azure App Service only receives the app/ folder).

Run:  python src/export_app_assets.py
"""
import json
import shutil

import pandas as pd

from config import CATEGORICAL, FEATURES, LABELS, MODEL_DIR, PROCESSED_DIR, ROOT, RESULTS_DIR

APP = ROOT / "app"
ASSETS = APP / "assets"


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    shutil.copy(MODEL_DIR / "ebm.pkl", ASSETS / "ebm.pkl")
    meta = json.loads((MODEL_DIR / "model_meta.json").read_text())
    test = pd.read_csv(PROCESSED_DIR / "test.csv", dtype={"weathercode": str})

    # All 30 cities are offered in the app; each is mapped to the grid cell kept in training
    summary = json.loads((PROCESSED_DIR / "data_summary.json").read_text())
    to_grid = {m: rep for rep, ms in summary["quality_checks"]["merged_duplicate_cities"].items() for m in ms}
    raw = pd.read_csv(ROOT / "data" / "raw" / "SriLanka_Weather_Dataset.csv",
                      usecols=["city", "latitude", "longitude", "elevation"])
    cities = raw.groupby("city").first().round(2).reset_index().sort_values("city")
    cities["grid_city"] = cities["city"].map(lambda c: to_grid.get(c, c))
    cities.to_csv(ASSETS / "cities.csv", index=False)
    # Real days from the unseen test period, so users can replay actual events
    test[["city", "date"] + FEATURES + ["rain_tomorrow_mm", "heavy_rain_tomorrow"]] \
        .to_csv(ASSETS / "test_days.csv.gz", index=False, compression="gzip")

    comparison = pd.read_csv(RESULTS_DIR / "model_comparison.csv")
    info = {
        "features": FEATURES, "categorical": CATEGORICAL, "labels": LABELS,
        "threshold": meta["models"]["EBM (proposed)"]["threshold"],
        "ebm_params": meta["ebm_params"],
        "comparison": comparison.fillna("").to_dict(orient="records"),
        "train_medians": pd.read_csv(PROCESSED_DIR / "train.csv")[
            [f for f in FEATURES if f not in CATEGORICAL]].median().round(2).to_dict(),
    }
    (ASSETS / "model_info.json").write_text(json.dumps(info, indent=2))
    for fig in ["roc_pr_curves.png", "calibration.png", "recall_by_city.png"]:
        shutil.copy(RESULTS_DIR / "figures" / fig, ASSETS / fig)
    print("App assets written to", ASSETS.relative_to(ROOT))


if __name__ == "__main__":
    main()
