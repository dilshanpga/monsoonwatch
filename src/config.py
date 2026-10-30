"""Shared settings for the Sri Lanka heavy-rainfall early-warning project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "SriLanka_Weather_Dataset.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
RESULTS_DIR = ROOT / "reports"

RANDOM_STATE = 42

# Target: will TOMORROW be a "very heavy precipitation day"?
# Threshold follows the WMO/ETCCDI climate index R20mm (daily precipitation >= 20 mm).
HEAVY_RAIN_MM = 20.0

# Chronological split (no shuffling, so the model never sees the future).
TRAIN_END = "2019-12-31"   # 2010-2019  -> training
VALID_END = "2021-12-31"   # 2020-2021  -> validation (tuning + threshold)
                           # 2022-2023  -> test (touched once, at the end)

FEATURES = [
    # today's rain and recent accumulation
    "rain_today", "rain_hours_today", "rain_yesterday",
    "rain_3d", "rain_7d", "rain_14d", "wet_days_7d",
    # temperature and moisture proxies
    "tmax", "tmin", "temp_range", "tmean_change", "feels_like_gap",
    "radiation", "radiation_change", "et0",
    # wind
    "wind_max", "gust_max", "wind_dir",
    # sky state reported by the model
    "weathercode",
    # location and season
    "latitude", "longitude", "elevation", "month",
]
CATEGORICAL = ["weathercode"]
TARGET = "heavy_rain_tomorrow"

# Human-readable names used in plots, the app and the report.
LABELS = {
    "rain_today": "Rain today (mm)",
    "rain_hours_today": "Hours of rain today",
    "rain_yesterday": "Rain yesterday (mm)",
    "rain_3d": "Rain, last 3 days (mm)",
    "rain_7d": "Rain, last 7 days (mm)",
    "rain_14d": "Rain, last 14 days (mm)",
    "wet_days_7d": "Wet days in last 7 (>=1 mm)",
    "tmax": "Max temperature (°C)",
    "tmin": "Min temperature (°C)",
    "temp_range": "Daily temperature range (°C)",
    "tmean_change": "Change in mean temp vs yesterday (°C)",
    "feels_like_gap": "Feels-like minus actual temp (°C)",
    "radiation": "Solar radiation (MJ/m²)",
    "radiation_change": "Change in radiation vs yesterday (MJ/m²)",
    "et0": "Evapotranspiration ET0 (mm)",
    "wind_max": "Max wind speed (km/h)",
    "gust_max": "Max wind gust (km/h)",
    "wind_dir": "Dominant wind direction (°)",
    "weathercode": "Weather code today",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "elevation": "Elevation (m)",
    "month": "Month",
}
