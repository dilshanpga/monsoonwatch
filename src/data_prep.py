"""Step 1: clean the raw Open-Meteo export and build the modelling table.

Each row = one city on one day. Features use only information available by the
end of that day; the target is whether the NEXT day brings >= 20 mm of rain.

Run:  python src/data_prep.py
"""
import json

import numpy as np
import pandas as pd

from config import (CATEGORICAL, FEATURES, HEAVY_RAIN_MM, PROCESSED_DIR, RAW_CSV,
                    TARGET, TRAIN_END, VALID_END)


def load_raw(path=RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"])
    # Columns that carry no information for this tropical country or duplicate others
    df = df.drop(columns=["snowfall_sum", "rain_sum", "country", "sunrise", "sunset"])
    df = df.drop_duplicates(subset=["city", "time"]).sort_values(["city", "time"])
    return df.reset_index(drop=True)


# Several neighbouring cities fall in the same coarse rainfall grid cell of the
# source weather model and therefore have IDENTICAL rainfall, wind and radiation
# series (only temperature differs slightly with elevation). Keeping all of them
# would count the same weather event up to 7 times and over-weight the Western
# Province, so one representative city is kept per grid cell.
PREFERRED = ["Colombo", "Gampaha", "Galle", "Kurunegala"]


def deduplicate_grid_cells(df: pd.DataFrame):
    series = df.pivot(index="time", columns="city", values="precipitation_sum")
    groups = {}
    for city in series.columns:
        groups.setdefault(tuple(series[city].round(2)), []).append(city)
    keep, merged = [], {}
    for members in groups.values():
        rep = next((c for c in PREFERRED if c in members), sorted(members)[0])
        keep.append(rep)
        if len(members) > 1:
            merged[rep] = sorted(m for m in members if m != rep)
    return df[df["city"].isin(keep)].reset_index(drop=True), merged


def check_quality(df: pd.DataFrame) -> dict:
    """Simple sanity checks; results are written into the data report."""
    report = {
        "rows": int(len(df)),
        "cities": int(df["city"].nunique()),
        "date_min": str(df["time"].min().date()),
        "date_max": str(df["time"].max().date()),
        "missing_values": int(df.isna().sum().sum()),
        "negative_rain": int((df["precipitation_sum"] < 0).sum()),
        "tmin_above_tmax": int((df["temperature_2m_min"] > df["temperature_2m_max"]).sum()),
    }
    # Gaps in the daily sequence per city
    gaps = df.groupby("city")["time"].apply(lambda s: int((s.diff().dt.days > 1).sum()))
    report["date_gaps"] = int(gaps.sum())
    return report


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("city", group_keys=False)
    out = pd.DataFrame({
        "city": df["city"],
        "date": df["time"],
        "rain_today": df["precipitation_sum"],
        "rain_hours_today": df["precipitation_hours"],
        "rain_yesterday": g["precipitation_sum"].shift(1),
        "rain_3d": g["precipitation_sum"].transform(lambda s: s.rolling(3).sum()),
        "rain_7d": g["precipitation_sum"].transform(lambda s: s.rolling(7).sum()),
        "rain_14d": g["precipitation_sum"].transform(lambda s: s.rolling(14).sum()),
        "wet_days_7d": g["precipitation_sum"].transform(lambda s: (s >= 1).rolling(7).sum()),
        "tmax": df["temperature_2m_max"],
        "tmin": df["temperature_2m_min"],
        "temp_range": df["temperature_2m_max"] - df["temperature_2m_min"],
        "tmean_change": g["temperature_2m_mean"].diff(),
        # Feels-like above actual temperature signals humid air
        "feels_like_gap": df["apparent_temperature_mean"] - df["temperature_2m_mean"],
        "radiation": df["shortwave_radiation_sum"],
        "radiation_change": g["shortwave_radiation_sum"].diff(),
        "et0": df["et0_fao_evapotranspiration"],
        "wind_max": df["windspeed_10m_max"],
        "gust_max": df["windgusts_10m_max"],
        "wind_dir": df["winddirection_10m_dominant"],
        "weathercode": df["weathercode"].astype(int).astype(str),
        "latitude": df["latitude"].round(2),
        "longitude": df["longitude"].round(2),
        "elevation": df["elevation"],
        "month": df["time"].dt.month,
    })
    # Target: tomorrow's rainfall in the same city
    out["rain_tomorrow_mm"] = g["precipitation_sum"].shift(-1)
    out[TARGET] = (out["rain_tomorrow_mm"] >= HEAVY_RAIN_MM).astype("Int64")
    # Drop the warm-up rows (no 14-day history) and each city's last day (no tomorrow)
    out = out.dropna(subset=FEATURES + ["rain_tomorrow_mm"]).reset_index(drop=True)
    out[TARGET] = out[TARGET].astype(int)
    return out


def split(df: pd.DataFrame):
    train = df[df["date"] <= TRAIN_END]
    valid = df[(df["date"] > TRAIN_END) & (df["date"] <= VALID_END)]
    test = df[df["date"] > VALID_END]
    return train, valid, test


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    quality = check_quality(raw)
    deduped, merged = deduplicate_grid_cells(raw)
    quality["cities_after_dedup"] = int(deduped["city"].nunique())
    quality["rows_after_dedup"] = int(len(deduped))
    quality["merged_duplicate_cities"] = merged
    data = build_features(deduped)
    train, valid, test = split(data)
    for name, part in [("train", train), ("valid", valid), ("test", test)]:
        part.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)

    summary = {
        "quality_checks": quality,
        "modelling_rows": int(len(data)),
        "n_features": len(FEATURES),
        "categorical": CATEGORICAL,
        "split": {
            name: {
                "rows": int(len(p)),
                "from": str(p["date"].min().date()),
                "to": str(p["date"].max().date()),
                "heavy_rain_rate": round(float(p[TARGET].mean()), 4),
            }
            for name, p in [("train", train), ("valid", valid), ("test", test)]
        },
    }
    (PROCESSED_DIR / "data_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
