"""Hourly carbon intensity for grid zones outside Great Britain, computed from the
public generation mix (2022-01-01 to 2024-12-31, UTC).

Sources (all open, no API key):
  eu : Energy-Charts public power (Fraunhofer ISE), countries DE, FR, PL, ES
  us : EIA-930 hourly net generation by fuel, bulk file EBA.zip (balancing authorities)
  br : ONS Brazil hourly generation by plant (subsystems N, NE, S, SE/CO)
  nz : NZ Electricity Authority, generation by plant, half-hourly (national)

Intensity of a zone-hour = sum_f g_f EF_f / sum_f g_f (production-based, attributional),
with the IPCC (2014, AR5) lifecycle median emission factors below. Storage discharge,
cross-border flows and consumption are excluded from the mix, so imports are not
accounted for. These values differ in definition from the direct-emission factors that
NESO applies to Great Britain; only ratios within a fleet matter for scheduling.

Output: data/raw/grids/<zone>.csv.gz with columns hour, ci, wind, solar, gas (shares
of generation, percent). Missing hours are left absent; common.load_hourly fills them
causally and flags them."""
import sys, io, json, gzip, time, zipfile, subprocess, urllib.request, pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "grids"
TMP = pathlib.Path("/tmp/gridfetch")
IDX = pd.date_range("2022-01-01", "2024-12-31 23:00", freq="h")

# gCO2eq/kWh, IPCC AR5 lifecycle medians (other/unknown: 500)
EF = dict(coal=820, gas=490, oil=650, biomass=230, nuclear=12, hydro=24, wind=11, solar=45,
          geothermal=38, waste=400, other=500)


def finish(zone, gen):
    """gen: DataFrame indexed by UTC hour, columns = EF categories (MW or MWh)."""
    gen = gen.reindex(columns=list(EF), fill_value=0.0).clip(lower=0.0)
    tot = gen.sum(axis=1)
    ok = tot > 0
    ci = (gen * pd.Series(EF)).sum(axis=1)[ok] / tot[ok]
    out = pd.DataFrame(dict(ci=ci.round(1), wind=100 * gen.wind[ok] / tot[ok], solar=100 * gen.solar[ok] / tot[ok],
                            gas=100 * gen.gas[ok] / tot[ok])).round(2)
    out = out[(out.index >= IDX[0]) & (out.index <= IDX[-1])]
    out.index.name = "hour"
    OUT.mkdir(parents=True, exist_ok=True)
    out.reset_index().to_csv(OUT / f"{zone}.csv.gz", index=False)
    n_miss = len(IDX) - len(out.index.intersection(IDX))
    print(f"{zone}: {len(out)} hours, missing {n_miss}, mean ci {out.ci.mean():.0f}", flush=True)


# ------------------------------------------------------------------ Europe
EC_MAP = {"Hydro Run-of-River": "hydro", "Hydro water reservoir": "hydro", "Biomass": "biomass",
          "Fossil brown coal / lignite": "coal", "Fossil hard coal": "coal", "Fossil oil": "oil",
          "Fossil coal-derived gas": "coal", "Fossil gas": "gas", "Geothermal": "geothermal",
          "Others": "other", "Waste": "waste", "Wind offshore": "wind", "Wind onshore": "wind",
          "Solar": "solar", "Nuclear": "nuclear"}
EC_SKIP = {"Hydro pumped storage consumption", "Cross border electricity trading", "Load", "Residual load",
           "Renewable share of load", "Renewable share of generation", "Hydro pumped storage",
           "Battery Storage", "Battery Storage Consumption", "Battery Storage Charge", "Battery Storage Discharge"}


def get_json(url, tries=6):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.load(r)
        except Exception as e:
            time.sleep(2 ** k)
    raise RuntimeError(url)


def fetch_eu(countries=("de", "fr", "pl", "es")):
    for c in countries:
        parts = []
        for m in pd.period_range("2022-01", "2024-12", freq="M"):
            a, b = m.start_time.date(), (m.end_time + pd.Timedelta(days=1)).date()
            d = get_json(f"https://api.energy-charts.info/public_power?country={c}&start={a}&end={b}")
            t = pd.to_datetime(d["unix_seconds"], unit="s")
            df = pd.DataFrame({p["name"]: pd.Series(p["data"], dtype="float64") for p in d["production_types"]})
            df.index = t
            parts.append(df)
        df = pd.concat(parts)
        df = df[~df.index.duplicated()]
        unknown = [c_ for c_ in df.columns if c_ not in EC_MAP and c_ not in EC_SKIP]
        if unknown:
            print(c, "unmapped types (ignored):", unknown)
        gen = pd.DataFrame({k: sum((df[n].fillna(0) for n, v in EC_MAP.items() if v == k and n in df),
                                   start=pd.Series(0.0, index=df.index)) for k in EF})
        gen = gen.groupby(gen.index.floor("h")).mean()      # 15-min or hourly MW -> hourly mean
        finish(c.upper(), gen)


# ------------------------------------------------------------------ United States
US_FUEL = {"COL": "coal", "NG": "gas", "NUC": "nuclear", "OIL": "oil", "OTH": "other", "SUN": "solar",
           "UNK": "other", "WAT": "hydro", "WND": "wind"}
US_BA = ["CISO", "ERCO", "PJM", "BPAT", "MISO", "NYIS", "SOCO", "TVA", "ISNE", "SWPP"]


def fetch_us():
    TMP.mkdir(exist_ok=True)
    zp = TMP / "EBA.zip"
    if not zp.exists():
        subprocess.run(["curl", "-s", "-m", "1800", "-o", str(zp), "https://www.eia.gov/opendata/bulk/EBA.zip"], check=True)
    want = {f"EBA.{b}-ALL.NG.{f}.H": (b, f) for b in US_BA for f in US_FUEL}
    got = {}
    with zipfile.ZipFile(zp) as z, z.open("EBA.txt") as fh:
        for raw in io.TextIOWrapper(fh, encoding="utf-8"):
            if not raw.startswith('{"series_id":"EBA.'):
                continue
            sid = raw[14:raw.index('"', 14)]
            if sid in want:
                d = json.loads(raw)
                s = pd.Series({pd.Timestamp(t[:8] + " " + t[9:11] + ":00"): float(v) for t, v in d["data"] if v not in (None, "")})
                got[want[sid]] = s
    for b in US_BA:
        cols = {}
        for f, k in US_FUEL.items():
            if (b, f) in got:
                cols[k] = cols.get(k, 0) + got[(b, f)].clip(lower=0).reindex(IDX)
        if not cols:
            print(b, "no data"); continue
        gen = pd.DataFrame(cols)
        gen = gen[gen.notna().any(axis=1)].fillna(0.0)
        finish(b, gen)


# ------------------------------------------------------------------ Brazil
BR_FUEL = {"Hidráulica": "hydro", "Eólica": "wind", "Fotovoltaica": "solar", "Nuclear": "nuclear", "Gás": "gas",
           "Carvão Mineral": "coal", "Carvão": "coal", "Biomassa": "biomass", "Óleo": "oil", "Diesel": "oil",
           "Bagaço": "biomass", "Resíduos": "waste"}
BR_SUB = {"N": "BR_N", "NE": "BR_NE", "S": "BR_S", "SE": "BR_SE"}


def fetch_br():
    TMP.mkdir(exist_ok=True)
    acc = {z: [] for z in BR_SUB.values()}
    unknown = set()
    for m in pd.period_range("2022-01", "2024-12", freq="M"):
        f = TMP / f"br_{m}.csv"
        url = f"https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/geracao_usina_2_ho/GERACAO_USINA-2_{m.year}_{m.month:02d}.csv"
        subprocess.run(["curl", "-s", "-m", "600", "-o", str(f), url], check=True)
        d = pd.read_csv(f, sep=";", usecols=["din_instante", "id_subsistema", "nom_tipocombustivel", "val_geracao"])
        f.unlink()
        d["k"] = d.nom_tipocombustivel.map(BR_FUEL)
        unknown |= set(d.loc[d.k.isna(), "nom_tipocombustivel"].unique())
        d["k"] = d.k.fillna("other")
        d["val_geracao"] = pd.to_numeric(d.val_geracao, errors="coerce").fillna(0.0)
        g = d.groupby(["id_subsistema", "din_instante", "k"]).val_geracao.sum().unstack("k")
        for s, z in BR_SUB.items():
            if s in g.index.get_level_values(0):
                acc[z].append(g.loc[s])
        print(m, flush=True)
    if unknown:
        print("Brazil fuel types mapped to other:", unknown)
    for z, parts in acc.items():
        gen = pd.concat(parts)
        gen.index = pd.to_datetime(gen.index) + pd.Timedelta(hours=3)      # local (UTC-3) to UTC
        finish(z, gen)


# ------------------------------------------------------------------ New Zealand
NZ_FUEL = {"Hydro": "hydro", "Wind": "wind", "Solar": "solar", "Gas": "gas", "Coal": "coal", "Geo": "geothermal",
           "Wood": "biomass", "Diesel": "oil"}


def fetch_nz():
    TMP.mkdir(exist_ok=True)
    rows = []
    for m in pd.period_range("2022-01", "2024-12", freq="M"):
        f = TMP / f"nz_{m}.csv"
        url = f"https://emidatasets.blob.core.windows.net/publicdata/Datasets/Wholesale/Generation/Generation_MD/{m.year}{m.month:02d}_Generation_MD.csv"
        subprocess.run(["curl", "-sL", "-m", "300", "-o", str(f), url], check=True)
        d = pd.read_csv(f)
        f.unlink()
        tp = [c for c in d.columns if c.startswith("TP")]
        long = d.melt(id_vars=["Fuel_Code", "Trading_Date"], value_vars=tp, var_name="tp", value_name="v").dropna(subset=["v"])
        long["n"] = long.tp.str[2:].astype(int)
        long["k"] = long.Fuel_Code.map(NZ_FUEL).fillna("other")
        mid = pd.to_datetime(long.Trading_Date).dt.tz_localize("Pacific/Auckland", ambiguous="NaT", nonexistent="shift_forward")
        long["t"] = (mid.dt.tz_convert("UTC") + pd.to_timedelta((long.n - 1) * 30, unit="m")).dt.tz_localize(None)
        long = long.dropna(subset=["t"])
        per_tp = long.groupby(["t", "k"]).v.sum().unstack("k")            # all plants, per half-hour
        rows.append(per_tp.groupby(per_tp.index.floor("h")).mean())       # mean power of the hour
        print(m, flush=True)
    gen = pd.concat(rows).groupby(level=0).sum()
    finish("NZ", gen)


if __name__ == "__main__":
    for w in sys.argv[1:]:
        {"eu": fetch_eu, "us": fetch_us, "br": fetch_br, "nz": fetch_nz}[w]()
