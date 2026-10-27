"""Data loading, cleaning and feature engineering for the river flood early-warning model.

Used by both the notebook (training) and the Streamlit app (live scoring), so the
features the app computes are exactly the features the model was trained on.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

HORIZON_H = 6            # predict a new flood within the next 6 hours
FFILL_LIMIT_H = 3        # forward-fill gaps of at most 3 hours
RISING_FAST = 0.10       # a 3-h rise above 10% of the normal-to-flood margin counts as "rising quickly"

FEATURES = [
    "pos_now", "rise_1h", "rise_3h", "rise_6h", "rise_12h", "rise_24h", "accel_3h",
    "pos_max_24h", "pos_std_6h",
    "up_pos_max", "up_rise_3h_max", "has_upstream",
    "basin_pos_max_other", "basin_rise_3h_mean",
    "n_alert_national", "n_rising_national",
    "margin_m",
]

FEATURE_DESCRIPTIONS = {
    "pos_now": "Current level on a 0-1 scale: 0 = station's normal (median) level, 1 = minor-flood level",
    "rise_1h": "Change in level over the last 1 h (as a fraction of the normal-to-flood margin)",
    "rise_3h": "Change in level over the last 3 h (fraction of margin)",
    "rise_6h": "Change in level over the last 6 h (fraction of margin)",
    "rise_12h": "Change in level over the last 12 h (fraction of margin)",
    "rise_24h": "Change in level over the last 24 h (fraction of margin)",
    "accel_3h": "Acceleration: last-3h rise minus the previous-3h rise",
    "pos_max_24h": "Highest level (0-1 scale) in the last 24 h",
    "pos_std_6h": "Variability of the level over the last 6 h",
    "up_pos_max": "Highest current level (0-1 scale) at the gauges directly upstream",
    "up_rise_3h_max": "Fastest 3-h rise at the gauges directly upstream",
    "has_upstream": "1 if the station has a gauged station upstream",
    "basin_pos_max_other": "Highest level (0-1 scale) at any other gauge in the same river basin",
    "basin_rise_3h_mean": "Average 3-h rise across the river basin",
    "n_alert_national": "Number of gauges in Sri Lanka currently at or above their alert level",
    "n_rising_national": "Number of gauges in Sri Lanka rising quickly (3-h rise > 10% of their normal-to-flood margin)",
    "margin_m": "Metres between the station's normal level and its minor-flood level",
}

HAMPEL_HALF_WINDOW = 2   # 5-reading centred window
HAMPEL_N_SIGMA = 4       # flag readings > 4 robust std-devs from the local median ...
HAMPEL_MIN_JUMP_M = 0.5  # ... and at least 0.5 m away from it (ignores tiny wiggles)


# ----------------------------------------------------------------------------- loading
def load_stations(stations_path, rivers_path) -> pd.DataFrame:
    """Station table indexed by name: river, basin, coordinates, official thresholds and the
    gauged stations directly upstream (derived from each river's source -> mouth location list)."""
    st = pd.DataFrame(json.loads(Path(stations_path).read_text(encoding="utf-8")))
    rivers = json.loads(Path(rivers_path).read_text(encoding="utf-8"))
    st = st[st.minor_flood_level_m > 0].rename(columns={
        "alert_level_m": "alert", "minor_flood_level_m": "minor", "major_flood_level_m": "major"})
    st["basin"] = st.river_name.map({r["name"]: r["basin_name"] for r in rivers})

    gauged = set(st.name)
    downstream: dict[str, set] = {}
    for r in rivers:
        locs = r["location_names"]
        for a, b in zip(locs, locs[1:]):
            downstream.setdefault(a, set()).add(b)
    upstream: dict[str, set] = {n: set() for n in gauged}
    for s in gauged:  # walk downstream through ungauged places until the next gauge
        stack, seen = list(downstream.get(s, ())), set()
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            if n in gauged:
                upstream[n].add(s)
            else:
                stack.extend(downstream.get(n, ()))
    st["upstream"] = [sorted(upstream[n]) for n in st.name]
    st["lat"], st["lng"] = st.lat_lng.str[0], st.lat_lng.str[1]
    cols = ["river_name", "basin", "district_id", "lat", "lng", "alert", "minor", "major", "upstream"]
    return st.set_index("name")[cols].sort_index()


def readings_from_records(records) -> pd.DataFrame:
    """[{station_name, time_ut, water_level_m}, ...] -> DataFrame(station, t, level) in Sri Lanka time."""
    df = pd.DataFrame(records)
    df["t"] = pd.to_datetime(df.time_ut, unit="s") + pd.Timedelta(hours=5, minutes=30)
    df = df.rename(columns={"station_name": "station", "water_level_m": "level"})
    return df[["station", "t", "level"]].sort_values(["station", "t"]).reset_index(drop=True)


def load_readings(path) -> pd.DataFrame:
    """Load the archived readings (all.json from nuuuwan/lk_irrigation, gzipped)."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return readings_from_records(json.load(f))


# ----------------------------------------------------------------------------- cleaning
def hampel_flags(s: pd.Series, half_window: int = HAMPEL_HALF_WINDOW, n_sigma: float = HAMPEL_N_SIGMA,
                 min_jump: float = HAMPEL_MIN_JUMP_M) -> pd.Series:
    """True for isolated spikes / drops (Hampel filter on one station's readings in time order)."""
    s = pd.Series(s).reset_index(drop=True)
    w = 2 * half_window + 1
    med = s.rolling(w, center=True, min_periods=1).median()
    mad = (s - med).abs().rolling(w, center=True, min_periods=1).median() * 1.4826
    return (s - med).abs() > np.maximum(n_sigma * mad, min_jump)


def clean_readings(raw: pd.DataFrame, stations: pd.DataFrame):
    """Drop unknown stations, duplicates and sensor spikes. Returns (clean, report)."""
    report = {"raw_rows": len(raw)}
    df = raw[raw.station.isin(stations.index)]
    report["unknown_station_rows"] = len(raw) - len(df)
    n = len(df)
    df = df.drop_duplicates(["station", "t"]).sort_values(["station", "t"]).reset_index(drop=True)
    report["duplicates"] = n - len(df)
    spikes = np.concatenate([hampel_flags(g.level).values for _, g in df.groupby("station", sort=True)])
    report["spikes_removed"] = int(spikes.sum())
    df = df[~spikes].reset_index(drop=True)
    report["clean_rows"] = len(df)
    return df, report


def to_hourly(clean: pd.DataFrame, ffill_limit: int = FFILL_LIMIT_H, end=None) -> pd.DataFrame:
    """Regular hourly grid per station: last reading in each hour, gaps of <= 3 h forward-filled.
    Columns: station, t, level, observed (1 if a real reading fell in that hour).
    `end` (used by the app for live data) extends every station's grid to a common last hour."""
    out = []
    for name, g in clean.assign(t=clean.t.dt.floor("h")).groupby("station", sort=True):
        s = g.groupby("t")["level"].last()
        last = s.index.max() if end is None else pd.Timestamp(end).floor("h")
        s = s.reindex(pd.date_range(s.index.min(), last, freq="h"))
        out.append(pd.DataFrame({"station": name, "t": s.index, "level": s.ffill(limit=ffill_limit).values,
                                 "observed": s.notna().astype(int).values}))
    return pd.concat(out, ignore_index=True)


def station_base_levels(hourly: pd.DataFrame) -> pd.Series:
    """Each station's normal level = median hourly level (computed on training months only)."""
    return hourly.groupby("station")["level"].median()


# ----------------------------------------------------------------------------- features
def _wide(hourly: pd.DataFrame, col: str) -> pd.DataFrame:
    return hourly.pivot(index="t", columns="station", values=col).asfreq("h")


def build_features(hourly: pd.DataFrame, stations: pd.DataFrame, base: pd.Series,
                   with_target: bool = True) -> pd.DataFrame:
    """Station-hour feature table. Every feature uses only information available at hour t.
    Target y = 1 if the level reaches the minor-flood level in any of the next 6 hours."""
    L = _wide(hourly, "level")
    obs = _wide(hourly, "observed")
    names = list(L.columns)
    st = stations.loc[names]
    b = base.reindex(names)
    margin = st.minor - b

    pos = (L - b) / margin
    rise = {k: (L - L.shift(k)) / margin for k in (1, 3, 6, 12, 24)}
    accel = ((L - L.shift(3)) - (L.shift(3) - L.shift(6))) / margin
    pos_max_24h = pos.rolling(24, min_periods=1).max()
    pos_std_6h = pos.rolling(6, min_periods=3).std()

    is_alert = (L >= st.alert).astype(float).where(L.notna())
    is_flood = (L >= st.minor).astype(float).where(L.notna())
    is_major = (L >= st.major).astype(float).where(L.notna())
    n_alert = (L >= st.alert).sum(axis=1).astype(float)
    n_rising = (rise[3] > RISING_FAST).sum(axis=1)

    basin_pos_other, basin_rise_mean, up_pos, up_rise = {}, {}, {}, {}
    for s in names:
        same = [o for o in names if st.basin[o] == st.basin[s]]
        others = [o for o in same if o != s]
        # a neighbour with no reading this hour falls back to the station's own value
        basin_pos_other[s] = (pos[others].max(axis=1) if others else pos[s]).fillna(pos[s])
        basin_rise_mean[s] = rise[3][same].mean(axis=1)
        ups = [u for u in st.upstream[s] if u in names]
        up_pos[s] = (pos[ups].max(axis=1) if ups else pos[s]).fillna(pos[s])   # no upstream: own value
        up_rise[s] = (rise[3][ups].max(axis=1) if ups else rise[3][s]).fillna(rise[3][s])

    if with_target:
        # max level over t+1 ... t+6 (needs at least 3 of those hours to be known)
        future_max = L[::-1].rolling(HORIZON_H, min_periods=3).max()[::-1].shift(-1)
        y = (future_max >= st.minor).astype(float).where(future_max.notna())

    def long(df):
        return df.stack(future_stack=True)

    cols = {"level": L, "observed": obs, "pos_now": pos,
            **{f"rise_{k}h": rise[k] for k in rise}, "accel_3h": accel,
            "pos_max_24h": pos_max_24h, "pos_std_6h": pos_std_6h}
    out = pd.DataFrame({k: long(v) for k, v in cols.items()})
    out.index.names = ["t", "station"]
    out = out.reset_index()
    out = out[out.level.notna() | out.observed.notna()]
    out = out[out.observed.notna()]                       # rows inside each station's own grid
    out["observed"] = out.observed.astype(int)
    out = out[["t", "level", "observed", "station"] + [c for c in cols if c not in ("level", "observed")]]
    out["margin_m"] = out.station.map(margin)
    out["basin"] = out.station.map(st.basin)

    def put(name, wide):
        out[name] = wide.stack(future_stack=True).reindex(pd.MultiIndex.from_arrays([out.t, out.station])).values

    put("is_alert", is_alert); put("is_flood", is_flood); put("is_major", is_major)
    if with_target:
        put("y", y)
    out["n_alert_national"] = out.t.map(n_alert).values
    out["n_rising_national"] = out.t.map(n_rising).astype(int).values
    put("basin_pos_max_other", pd.DataFrame(basin_pos_other))
    put("basin_rise_3h_mean", pd.DataFrame(basin_rise_mean))
    put("up_pos_max", pd.DataFrame(up_pos))
    put("up_rise_3h_max", pd.DataFrame(up_rise))
    out["has_upstream"] = out.station.map(lambda s: int(len(st.upstream[s]) > 0)).astype(int)
    return out.sort_values(["station", "t"]).reset_index(drop=True)
