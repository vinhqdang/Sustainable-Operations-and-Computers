"""Download half-hourly regional carbon intensity for Great Britain
from the National Energy System Operator (NESO) Carbon Intensity API
and store it as a tidy CSV (one row per half hour and region)."""
import json, time, datetime as dt, urllib.request, pathlib
import pandas as pd

API = "https://api.carbonintensity.org.uk/regional/intensity/{a}/{b}"
OUT = pathlib.Path(__file__).resolve().parents[1] / "data"


def fetch(start, end, step_days=13):
    rows, cur = [], start
    while cur < end:
        nxt = min(cur + dt.timedelta(days=step_days), end)
        url = API.format(a=cur.strftime("%Y-%m-%dT%H:%MZ"), b=nxt.strftime("%Y-%m-%dT%H:%MZ"))
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    payload = json.load(r)
                break
            except Exception as e:  # network hiccup
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError(url)
        for blk in payload["data"]:
            for reg in blk["regions"]:
                mix = {m["fuel"]: m["perc"] for m in reg["generationmix"]}
                rows.append(dict(time=blk["from"], region=reg["regionid"],
                                 name=reg["shortname"], ci=reg["intensity"]["forecast"],
                                 wind=mix.get("wind"), solar=mix.get("solar"), gas=mix.get("gas")))
        print(cur.date(), len(rows), flush=True)
        cur = nxt
    return pd.DataFrame(rows)


def repair(df):
    """Some multi-day requests return incomplete blocks. Re-request every day with
    missing half-hours in one-day chunks and add the rows that are returned."""
    t = pd.to_datetime(df["time"]).dt.tz_convert(None)
    full = pd.date_range(t.min(), t.max(), freq="30min")
    have = set(t[df.region == 13])
    days = sorted({x.floor("D") for x in full if x not in have})
    extra = [fetch(d.to_pydatetime(), (d + pd.Timedelta(days=1)).to_pydatetime(), step_days=1) for d in days]
    return pd.concat([df] + extra).drop_duplicates(["time", "region"]) if extra else df


if __name__ == "__main__":
    df = fetch(dt.datetime(2022, 1, 1), dt.datetime(2025, 1, 2))
    df = repair(df.drop_duplicates(["time", "region"]))
    df = df.sort_values(["time", "region"])
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "raw" / "gb_regional_ci_2022_2024.csv.gz", index=False)
    print(df.shape)
