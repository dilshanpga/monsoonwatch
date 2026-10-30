"""MonsoonWatch - next-day heavy-rainfall early warning for Sri Lankan cities.

Streamlit front-end for the Explainable Boosting Machine trained in src/train.py.
Local run:   streamlit run app.py
Azure:       see DEPLOY_AZURE.md
"""
import json
from datetime import date, timedelta
from pathlib import Path

import altair as alt
import pickle
import numpy as np
import pandas as pd
import streamlit as st

ASSETS = Path(__file__).parent / "assets"
UP, DOWN = "#e34948", "#2a78d6"
WEATHER_CODES = {
    "0": "0 - Clear sky", "1": "1 - Mainly clear", "2": "2 - Partly cloudy", "3": "3 - Overcast",
    "51": "51 - Light drizzle", "53": "53 - Moderate drizzle", "55": "55 - Dense drizzle",
    "61": "61 - Slight rain", "63": "63 - Moderate rain", "65": "65 - Heavy rain",
}

st.set_page_config(page_title="MonsoonWatch", page_icon="🌧️", layout="wide")


@st.cache_resource
def load_model():
    with open(ASSETS / "ebm.pkl", "rb") as fh:
        return pickle.load(fh)


@st.cache_data
def load_assets():
    info = json.loads((ASSETS / "model_info.json").read_text())
    cities = pd.read_csv(ASSETS / "cities.csv")
    days = pd.read_csv(ASSETS / "test_days.csv.gz", dtype={"weathercode": str}, parse_dates=["date"])
    return info, cities, days


ebm = load_model()
info, cities, days = load_assets()
FEATURES, LABELS, THR = info["features"], info["labels"], info["threshold"]


def label(term):
    return " × ".join(LABELS.get(t.strip(), t.strip()) for t in term.split("&"))


def risk_band(p):
    if p >= max(0.6, THR * 1.5):
        return "HIGH", "🔴", "Heavy rain (≥ 20 mm) is likely tomorrow. Check flood and landslide advisories."
    if p >= THR:
        return "ELEVATED", "🟠", "Above the warning threshold: heavy rain is a real possibility tomorrow."
    if p >= THR / 2:
        return "WATCH", "🟡", "Below the warning threshold but higher than a typical day."
    return "LOW", "🟢", "Heavy rain tomorrow is unlikely."


# ---------------------------------------------------------------- header
st.title("🌧️ MonsoonWatch")
st.caption("Next-day heavy-rainfall (≥ 20 mm) early warning for 30 Sri Lankan cities · "
           "Explainable Boosting Machine · student project, not an official forecast")

tab_predict, tab_learned, tab_perf, tab_about = st.tabs(
    ["Predict & explain", "What the model learned", "Model performance", "About the data"])

# ---------------------------------------------------------------- inputs
with st.sidebar:
    st.header("Today's conditions")
    mode = st.radio("Input mode", ["Replay a real day (2022-23)", "Enter values manually"],
                    help="Replay fills every field from a real day the model never saw in training.")
    city = st.selectbox("City", cities["city"].tolist(),
                        index=int(np.flatnonzero(cities["city"] == "Colombo")[0]))
    loc = cities.set_index("city").loc[city]
    grid_city = loc["grid_city"]
    defaults = {f: info["train_medians"].get(f) for f in FEATURES}
    actual = None
    if mode.startswith("Replay"):
        if grid_city != city:
            st.caption(f"{city} shares a weather-model grid cell with {grid_city}; replaying {grid_city}'s records.")
        cdays = days[days["city"] == grid_city].sort_values("date")
        heavy_only = st.checkbox("Only show days followed by heavy rain", value=False)
        if heavy_only:
            cdays = cdays[cdays["heavy_rain_tomorrow"] == 1]
        pick = st.selectbox("Day", cdays["date"].dt.date.tolist()[::-1])
        row = cdays[cdays["date"].dt.date == pick].iloc[0]
        defaults.update(row[FEATURES].to_dict())
        actual = float(row["rain_tomorrow_mm"])
        month = int(row["month"])
    else:
        pick = st.date_input("Today's date", value=date.today())
        month = pick.month
    defaults.update({"latitude": loc.latitude, "longitude": loc.longitude,
                     "elevation": loc.elevation, "month": month})

    def num(f, lo, hi, step=0.1):
        return st.number_input(LABELS[f], min_value=float(lo), max_value=float(hi),
                               value=float(np.clip(defaults[f], lo, hi)), step=step, key=f"{f}_{city}_{pick}")

    x = {}
    with st.expander("Rainfall", expanded=True):
        x["rain_today"] = num("rain_today", 0, 400)
        x["rain_hours_today"] = num("rain_hours_today", 0, 24, 1.0)
        x["rain_yesterday"] = num("rain_yesterday", 0, 400)
        x["rain_3d"] = num("rain_3d", 0, 800)
        x["rain_7d"] = num("rain_7d", 0, 1200)
        x["rain_14d"] = num("rain_14d", 0, 2000)
        x["wet_days_7d"] = num("wet_days_7d", 0, 7, 1.0)
        # Keep the rainfall history physically consistent. The model never saw
        # impossible combinations (e.g. 3-day total smaller than today + yesterday),
        # and its interaction surfaces are not trustworthy there.
        fixes = []
        for longer, need in [("rain_3d", x["rain_today"] + x["rain_yesterday"]),
                             ("rain_7d", None), ("rain_14d", None)]:
            if need is None:
                need = x["rain_3d"] if longer == "rain_7d" else x["rain_7d"]
            if x[longer] < need:
                fixes.append(f"{LABELS[longer]} raised to {need:.1f}")
                x[longer] = need
        if fixes:
            st.warning("Adjusted to stay consistent: " + "; ".join(fixes))
    with st.expander("Temperature & sunshine"):
        x["tmax"] = num("tmax", 10, 45)
        x["tmin"] = num("tmin", 5, 35)
        x["temp_range"] = round(x["tmax"] - x["tmin"], 2)
        st.caption(f"Daily range (computed): {x['temp_range']:.1f} °C")
        x["tmean_change"] = num("tmean_change", -10, 10)
        x["feels_like_gap"] = num("feels_like_gap", -5, 15)
        x["radiation"] = num("radiation", 0, 35)
        x["radiation_change"] = num("radiation_change", -30, 30)
        x["et0"] = num("et0", 0, 10)
    with st.expander("Wind & sky"):
        x["wind_max"] = num("wind_max", 0, 100)
        x["gust_max"] = num("gust_max", 0, 150)
        x["wind_dir"] = num("wind_dir", 0, 360, 1.0)
        codes = list(WEATHER_CODES)
        x["weathercode"] = st.selectbox(LABELS["weathercode"], codes,
                                        index=codes.index(str(defaults["weathercode"])) if str(defaults["weathercode"]) in codes else 4,
                                        format_func=WEATHER_CODES.get, key=f"wc_{city}_{pick}")
    for f in ["latitude", "longitude", "elevation", "month"]:
        x[f] = defaults[f]

X = pd.DataFrame([x])[FEATURES]
X["weathercode"] = X["weathercode"].astype(str)
for f in FEATURES:
    if f != "weathercode":
        X[f] = X[f].astype(float)

# ---------------------------------------------------------------- predict
with tab_predict:
    p = float(ebm.predict_proba(X)[0, 1])
    band, icon, advice = risk_band(p)
    when = (pick + timedelta(days=1)).strftime("%a %d %b %Y")
    c1, c2, c3 = st.columns([1.1, 1, 1.4])
    c1.metric(f"Heavy-rain probability · {city}, {when}", f"{p:.0%}")
    c2.metric("Risk level", f"{icon} {band}")
    c3.markdown(f"**What it means**  \n{advice}  \nWarning threshold: {THR:.0%} "
                f"(chosen on 2020-21 data to balance missed events and false alarms).")
    if actual is not None:
        hit = (actual >= 20) == (p >= THR)
        st.info(f"**What actually happened:** {actual:.1f} mm fell on {when}. "
                f"The model's warning was {'✅ correct' if hit else '❌ wrong'} for this day.")

    st.subheader("Why this prediction?")
    loc_exp = ebm.explain_local(X).data(0)
    contrib = pd.DataFrame({"term": loc_exp["names"], "score": loc_exp["scores"]})
    contrib["feature"] = contrib["term"].map(label)
    contrib["value"] = [
        " / ".join(str(X.iloc[0][t.strip()]) for t in term.split("&")) for term in contrib["term"]]
    contrib["direction"] = np.where(contrib["score"] > 0, "Raises risk", "Lowers risk")
    top = contrib.reindex(contrib["score"].abs().sort_values(ascending=False).index).head(10)
    base = float(loc_exp["extra"]["scores"][0])
    chart = alt.Chart(top).mark_bar(cornerRadius=3).encode(
        x=alt.X("score:Q", title="Contribution to log-odds of heavy rain"),
        y=alt.Y("feature:N", sort=alt.EncodingSortField("score", op="max", order="descending"), title=None,
                axis=alt.Axis(labelLimit=320, labelOverlap=False, labelFontSize=12)),
        color=alt.Color("direction:N", scale=alt.Scale(domain=["Raises risk", "Lowers risk"], range=[UP, DOWN]),
                        legend=alt.Legend(title=None, orient="bottom")),
        tooltip=["feature", "value", alt.Tooltip("score:Q", format="+.3f")],
    ).properties(height=36 * len(top))
    st.altair_chart(chart, width="stretch")
    st.caption(f"The EBM is additive: the baseline log-odds {base:+.2f} (heavy rain is rare) plus the bars above "
               f"(and smaller ones not shown) gives the final log-odds {np.log(p / (1 - p)):+.2f} → {p:.0%}. "
               "These contributions are exact, not approximations.")

# ---------------------------------------------------------------- global view
with tab_learned:
    g = ebm.explain_global()
    imp = pd.DataFrame({"term": ebm.term_names_, "importance": ebm.term_importances()})
    imp["feature"] = imp["term"].map(label)
    imp["type"] = np.where(imp["term"].str.contains("&"), "Interaction", "Single feature")
    st.subheader("Which features matter most overall")
    st.altair_chart(alt.Chart(imp.sort_values("importance", ascending=False).head(15)).mark_bar(cornerRadius=3).encode(
        x=alt.X("importance:Q", title="Mean |contribution| (log-odds)"),
        y=alt.Y("feature:N", sort="-x", title=None, axis=alt.Axis(labelLimit=340, labelOverlap=False)),
        color=alt.Color("type:N", scale=alt.Scale(domain=["Single feature", "Interaction"], range=["#2a78d6", "#eb6834"]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["feature", alt.Tooltip("importance:Q", format=".3f")]).properties(height=28 * 15),
        width="stretch")

    st.subheader("How one feature changes the risk")
    mains = [t for t in ebm.term_names_ if "&" not in t]
    f = st.selectbox("Feature", mains, format_func=label, index=mains.index("rain_7d"))
    d = g.data(ebm.term_names_.index(f))
    if f == "weathercode":
        sdf = pd.DataFrame({"x": [WEATHER_CODES.get(n, n) for n in d["names"]], "score": d["scores"]})
        ch = alt.Chart(sdf).mark_bar(cornerRadius=3).encode(x=alt.X("x:N", title=label(f)), y=alt.Y("score:Q", title="Contribution (log-odds)"))
    else:
        edges = np.asarray(d["names"], dtype=float)
        sdf = pd.DataFrame({"x": edges[:-1], "score": d["scores"],
                            "lo": d["lower_bounds"], "hi": d["upper_bounds"]})
        if f.startswith("rain"):
            sdf = sdf[sdf["x"] <= np.percentile(edges, 97)]
        band_ = alt.Chart(sdf).mark_area(opacity=0.18, interpolate="step-after", color=DOWN).encode(
            x=alt.X("x:Q", title=label(f)), y="lo:Q", y2="hi:Q")
        line = alt.Chart(sdf).mark_line(interpolate="step-after", color=DOWN, strokeWidth=2).encode(
            x="x:Q", y=alt.Y("score:Q", title="Contribution (log-odds)"), tooltip=["x", alt.Tooltip("score:Q", format="+.3f")])
        ch = band_ + line
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#888").encode(y="y:Q")
    st.altair_chart((ch + zero).properties(height=320), width="stretch")
    st.caption("Above zero = pushes tomorrow's heavy-rain risk up; below zero = pushes it down. "
               "The shaded band shows the spread across the model's bagged fits.")

# ---------------------------------------------------------------- performance
with tab_perf:
    st.subheader("Test results (Jan 2022 - Jun 2023, never used for training or tuning)")
    comp = pd.DataFrame(info["comparison"]).replace("", np.nan)
    st.dataframe(comp.set_index("Model").drop(columns=["Threshold"], errors="ignore"), width="stretch")
    col1, col2 = st.columns(2)
    col1.image(str(ASSETS / "roc_pr_curves.png"), width="stretch")
    col2.image(str(ASSETS / "calibration.png"), width="stretch")
    st.image(str(ASSETS / "recall_by_city.png"), width=620)

# ---------------------------------------------------------------- about
with tab_about:
    st.markdown("""
**Data.** Daily weather for 30 Sri Lankan cities, 2010-01-01 to 2023-06-17 (147,480 rows), from the public
*Sri Lanka Weather Dataset* on Kaggle (CC0), which was compiled from the Open-Meteo historical weather API.
These are **weather-model values for a grid cell**, not rain-gauge readings, so local storms can be smoothed out.
Neighbouring cities that share a grid cell have identical rainfall, so the model was trained on one city per
cell: **15 locations, 73,530 days**.

**Target.** Tomorrow is a *very heavy precipitation day* if it gets ≥ 20 mm of rain (the WMO ETCCDI index R20mm).
About 5.5 % of days meet this.

**Model.** An Explainable Boosting Machine (InterpretML): a generalised additive model whose per-feature
functions are learned by cyclic gradient boosting, plus a few automatically detected pairwise interactions.
Every prediction is an exact sum of readable feature contributions.

**Limits.** This is a coursework prototype. It is not a substitute for the Department of Meteorology's forecasts
or the Disaster Management Centre's warnings. Please follow official advisories.
""")
