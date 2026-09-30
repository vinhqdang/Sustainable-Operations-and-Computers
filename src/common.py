"""Shared configuration and data loading."""
import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RES = ROOT / "results"
FIG = ROOT / "figures"

# Candidate data-centre sites (NESO region ids) and assumed facility parameters
SITES = {2: "South Scotland", 3: "North West England", 8: "West Midlands", 13: "London"}
SITE_IDS = list(SITES)
PUE = np.array([1.15, 1.20, 1.25, 1.30])  # cooler northern sites assumed more efficient
H = 24  # planning / forecast horizon (hours)

TRAIN_END = "2023-07-01"
CAL_END = "2023-10-01"   # conformal calibration window [TRAIN_END, CAL_END)
VAL_END = "2024-01-01"   # policy tuning window [CAL_END, VAL_END)
TEST_END = "2025-01-01"


def load_hourly():
    """Hourly carbon intensity (gCO2/kWh) and generation-mix shares, wide by region."""
    cache = DATA / "gb_hourly.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    d = pd.read_csv(DATA / "raw" / "gb_regional_ci_2022_2024.csv.gz", parse_dates=["time"])
    d["time"] = d["time"].dt.tz_localize(None)
    d = d[d.time >= "2022-01-01"]
    d["hour"] = d["time"].dt.floor("h")
    g = d.groupby(["hour", "region"])[["ci", "wind", "solar", "gas"]].mean()
    wide = g.unstack("region")
    idx = pd.date_range("2022-01-01", "2024-12-31 23:00", freq="h")
    wide = wide.reindex(idx)
    missing = wide["ci"].isna().any(axis=1)
    # Causal gap filling: carry the last observation forward (persistence), so that no
    # filled value depends on data published after it. Filled hours are flagged and
    # excluded from forecast training, calibration and forecast evaluation.
    wide = wide.ffill()
    wide.attrs["n_missing_hours"] = int(missing.sum())
    wide.attrs["filled"] = missing.to_numpy()
    wide.to_pickle(cache)
    return wide


def filled_mask():
    """Boolean array over the hourly index: True where the value was gap-filled."""
    return np.asarray(load_hourly().attrs["filled"], bool)
