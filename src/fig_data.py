"""Figures on data and forecasts (Figs. 2-3)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": 300})
COL = ["#0072B2", "#009E73", "#CC79A7", "#D55E00"]          # color-blind-safe (Okabe-Ito)
LSTY = ["-", "--", "-.", ":"]
FIG.mkdir(exist_ok=True)
w = load_hourly()["ci"][SITE_IDS]

fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.6, 1]})
m = w.resample("MS").mean()
for s, c in zip(SITE_IDS, COL):
    ax[0].plot(m.index, m[s], color=c, lw=1.4, ls=LSTY[SITE_IDS.index(s)], label=SITES[s])
ax[0].set_ylabel("Monthly mean CI (gCO$_2$/kWh)")
ax[0].axvspan(pd.Timestamp(VAL_END), pd.Timestamp("2024-12-31"), color="0.92", zorder=0)
ax[0].text(pd.Timestamp("2024-07-01"), 238, "test year", ha="center", fontsize=8)
ax[0].set_ylim(0, 250)
import matplotlib.dates as mdates
ax[0].xaxis.set_major_locator(mdates.YearLocator()); ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
h_, l_ = ax[0].get_legend_handles_labels()
fig.legend(h_, l_, frameon=False, fontsize=7.5, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.15))
ax[0].set_title("(a) Regional carbon intensity, 2022-2024", fontsize=9, loc="left")
dh = w[w.index >= VAL_END].groupby(w[w.index >= VAL_END].index.hour).mean()
for s, c in zip(SITE_IDS, COL):
    ax[1].plot(dh.index, dh[s], color=c, lw=1.4, ls=LSTY[SITE_IDS.index(s)])
ax[1].set_xlabel("Hour of day (UTC)"); ax[1].set_xticks([0, 6, 12, 18, 23])
ax[1].set_title("(b) Mean diurnal profile, 2024", fontsize=9, loc="left")
fig.savefig(FIG / "fig2_data.pdf"); plt.close(fig)

acc = pd.read_csv(RES / "forecast_accuracy_by_h.csv")
cal = pd.read_csv(RES / "forecast_calibration.csv")
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
for (name, g), c, ls in zip(acc.groupby("model", sort=False), ["0.5", "0.3", "#D55E00"], [":", "--", "-"]):
    gh = g.groupby("h").mae.mean()
    ax[0].plot(gh.index, gh.values, ls, color=c, lw=1.6, label=name)
ax[0].set_xlabel("Forecast horizon h (hours)"); ax[0].set_ylabel("MAE (gCO$_2$/kWh)")
ax[0].legend(frameon=False, fontsize=7); ax[0].set_title("(a) Accuracy by horizon, 2024", fontsize=9, loc="left")
for (lab, g), c, mk in zip(cal.groupby("method", sort=False), ["0.4", "#D55E00"], ["s", "o"]):
    ax[1].errorbar(g.level + (0.006 if mk == "o" else -0.006), g.coverage,
                   yerr=[g.coverage - g.worst_month, g.best_month - g.coverage],
                   fmt=mk, color=c, ms=4, capsize=2, lw=1, label=lab)
ax[1].plot([0.45, 0.95], [0.45, 0.95], color="0.6", lw=0.8)
ax[1].set_xlabel("Target coverage"); ax[1].set_ylabel("Empirical coverage")
ax[1].legend(frameon=False, fontsize=7); ax[1].set_title("(b) Upper-bound calibration (monthly range)", fontsize=9, loc="left")
fig.savefig(FIG / "fig3_forecast.pdf"); plt.close(fig)
print("ok")
