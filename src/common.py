"""Shared configuration and data loading."""
import os
import pathlib
import numpy as np
import pandas as pd
from fleets import FLEETS, CAP as FLEET_CAP, ORIGIN_P as FLEET_ORIGIN_P

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FLEET = os.environ.get("FLEET", "GB").upper()        # selects the grid/fleet; GB is the main study
_F = FLEETS[FLEET]
RES = ROOT / "results" if FLEET == "GB" else ROOT / "results" / "fleets" / FLEET.lower()
RES.mkdir(parents=True, exist_ok=True)
FIG = ROOT / "figures"

# Candidate data-centre sites (NESO region ids) and assumed facility parameters
SITES = dict(zip(_F["zones"], _F["names"]))
SITE_IDS = list(SITES)
PUE = np.array(_F["pue"])  # Great Britain: cooler northern sites assumed more efficient
TZ = np.array(_F["tz"], float)                      # UTC offset of local standard time (h)
BASE_CAP = np.array(FLEET_CAP)
BASE_ORIGIN_P = np.array(FLEET_ORIGIN_P)
REF = 18 if FLEET == "GB" else "REF"               # reference (national/fleet-mean) intensity column
HOURLY_CACHE = DATA / ("gb_hourly.pkl" if FLEET == "GB" else f"hourly_{FLEET.lower()}.pkl")
FORECAST_PATH = DATA / ("forecasts.pkl" if FLEET == "GB" else f"forecasts_{FLEET.lower()}.pkl")
H = 24  # planning / forecast horizon (hours)

TRAIN_END = "2023-07-01"
CAL_END = "2023-10-01"   # conformal calibration window [TRAIN_END, CAL_END)
VAL_END = "2024-01-01"   # policy tuning window [CAL_END, VAL_END)
TEST_END = "2025-01-01"


def load_hourly():
    """Hourly carbon intensity (gCO2/kWh) and generation-mix shares, wide by zone."""
    if HOURLY_CACHE.exists():
        return pd.read_pickle(HOURLY_CACHE)
    idx = pd.date_range("2022-01-01", "2024-12-31 23:00", freq="h")
    if FLEET == "GB":
        d = pd.read_csv(DATA / "raw" / "gb_regional_ci_2022_2024.csv.gz", parse_dates=["time"])
        d["time"] = d["time"].dt.tz_localize(None)
        d = d[d.time >= "2022-01-01"]
        d["hour"] = d["time"].dt.floor("h")
        g = d.groupby(["hour", "region"])[["ci", "wind", "solar", "gas"]].mean()
        wide = g.unstack("region")
    else:
        parts = {}
        for z in _F["zones"]:
            zd = pd.read_csv(DATA / "raw" / "grids" / f"{z}.csv.gz", parse_dates=["hour"]).set_index("hour")
            parts[z] = zd[["ci", "wind", "solar", "gas"]]
        wide = pd.concat(parts, axis=1).swaplevel(axis=1).sort_index(axis=1)
    wide = wide.reindex(idx)
    missing = wide["ci"].isna().any(axis=1)
    # Causal gap filling: carry the last observation forward (persistence), so that no
    # filled value depends on data published after it. Filled hours are flagged and
    # excluded from forecast training, calibration and forecast evaluation.
    wide = wide.ffill().bfill()            # bfill only for gaps before the first observation
    if FLEET != "GB":                      # fleet-mean reference intensity
        for v in ("ci",):
            wide[(v, "REF")] = wide[v][_F["zones"]].mean(axis=1)
    wide.attrs["n_missing_hours"] = int(missing.sum())
    wide.attrs["filled"] = missing.to_numpy()
    wide.to_pickle(HOURLY_CACHE)
    return wide


def filled_mask():
    """Boolean array over the hourly index: True where the value was gap-filled."""
    return np.asarray(load_hourly().attrs["filled"], bool)
