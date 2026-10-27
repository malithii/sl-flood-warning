"""Live data access and scoring for the Streamlit app.

The same functions as the notebook (src.features) turn raw gauge readings into model
features, so the app scores exactly what the model was trained on.
"""
from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from src.features import FEATURES, build_features, clean_readings, to_hourly

# Rolling 28-day window of readings, refreshed about hourly by the archive's GitHub Action.
LIVE_URL = "https://raw.githubusercontent.com/nuuuwan/lk_irrigation/main/data/alert_data.last_28_days.json"
ARCHIVE_REPO = "https://github.com/nuuuwan/lk_irrigation"
DASHBOARD_URL = "https://www.arcgis.com/apps/dashboards/2cffe83c9ff5497d97375498bdf3ff38"
SL_TZ = timezone(timedelta(hours=5, minutes=30))
HISTORY_H = 72          # hours of history kept for charts (features need 24 h)

STATUS_ORDER = ["Major flood", "Flooding", "Flood likely within 6 h", "Alert level", "Normal", "No recent data"]


def now_sl() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(SL_TZ).replace(tzinfo=None))


def parse_live_payload(payload: dict) -> pd.DataFrame:
    """alert_data.last_28_days.json -> DataFrame(station, t, level); times are Sri Lanka local time."""
    rows = []
    for station, days in payload["event_data"].items():
        for day, times in days.items():
            for hhmmss, level in times.items():
                if level is None:
                    continue
                rows.append((station, f"{day}{int(hhmmss):06d}", float(level)))
    df = pd.DataFrame(rows, columns=["station", "stamp", "level"])
    df["t"] = pd.to_datetime(df.stamp, format="%Y%m%d%H%M%S")
    return df[["station", "t", "level"]].sort_values(["station", "t"]).reset_index(drop=True)


def fetch_live_readings(timeout: int = 20) -> pd.DataFrame:
    req = urllib.request.Request(LIVE_URL, headers={"User-Agent": "flood-early-warning-app"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return parse_live_payload(json.load(r))


def snapshot_hour(readings: pd.DataFrame, at=None, min_share: float = 0.75) -> pd.Timestamp:
    """The hour to report on: the latest hour (<= `at`) in which most gauges have reported,
    so a half-finished current hour does not count as 'no change'."""
    r = readings if at is None else readings[readings.t < pd.Timestamp(at).floor("h") + pd.Timedelta(hours=1)]
    per_hour = r.assign(h=r.t.dt.floor("h")).groupby("h").station.nunique()
    n = readings.station.nunique()
    good = per_hour[per_hour >= min_share * n]
    return good.index.max() if len(good) else per_hour.index.max()


def score_window(readings: pd.DataFrame, bundle: dict, at=None, history_h: int = HISTORY_H):
    """Score every station-hour in the `history_h` hours up to the snapshot hour.

    Returns (table, snapshot_t, cleaning_report): table has one row per station-hour with
    level, thresholds, the 17 features, the model probability and a plain-language status.
    """
    stations, base = bundle["stations"], bundle["base_levels"]
    t_end = snapshot_hour(readings, at)
    t_start = t_end - pd.Timedelta(hours=history_h + 30)      # +30 h so 24-h features exist at the start
    r = readings[(readings.t >= t_start) & (readings.t < t_end + pd.Timedelta(hours=1))]
    clean, report = clean_readings(r, stations)
    hourly = to_hourly(clean, end=t_end)
    feats = build_features(hourly, stations, base, with_target=False)
    feats = feats[feats.t > t_end - pd.Timedelta(hours=history_h)].reset_index(drop=True)

    feats = feats.join(stations[["river_name", "alert", "minor", "major", "lat", "lng"]], on="station")
    ok = feats[FEATURES].notna().all(axis=1) & (feats.is_flood == 0)
    feats["risk"] = np.nan
    if ok.any():
        feats.loc[ok, "risk"] = bundle["model"].predict_proba(feats.loc[ok, FEATURES])[:, 1]
    thr = bundle["threshold"]
    feats["status"] = np.select(
        [feats.level.isna(), feats.is_major == 1, feats.is_flood == 1, feats.risk >= thr, feats.is_alert == 1],
        ["No recent data", "Major flood", "Flooding", "Flood likely within 6 h", "Alert level"], "Normal")
    feats["warn"] = feats.status.isin(["Flood likely within 6 h"]).astype(int)
    return feats, t_end, report


def latest(table: pd.DataFrame, t_end) -> pd.DataFrame:
    """One row per station at the snapshot hour, most serious first."""
    snap = table[table.t == t_end].copy()
    snap["order"] = snap.status.map({s: i for i, s in enumerate(STATUS_ORDER)})
    snap["to_flood_m"] = snap.minor - snap.level
    return snap.sort_values(["order", "risk"], ascending=[True, False]).reset_index(drop=True)


def features_from_manual(inp: dict, station: str, bundle: dict) -> pd.DataFrame:
    """Build the 17 features for one station from a handful of typed-in readings (what-if tool).

    Hours between the readings the user gives are filled by straight lines, and basin
    context is approximated from the upstream gauge, so this is a simplified version of the
    full pipeline used on real data.
    """
    st = bundle["stations"].loc[station]
    b = bundle["base_levels"][station]
    m = st.minor - b
    pts = {0: inp["level_now"], 1: inp["level_1h"], 3: inp["level_3h"], 6: inp["level_6h"],
           12: inp["level_12h"], 24: inp["level_24h"]}
    hours = np.arange(0, 25)
    series = np.interp(hours, sorted(pts), [pts[h] for h in sorted(pts)])       # series[h] = level h hours ago
    pos = (series - b) / m
    f = {"pos_now": pos[0]}
    for k in (1, 3, 6, 12, 24):
        f[f"rise_{k}h"] = (pts[0] - pts[k]) / m
    f["accel_3h"] = ((pts[0] - pts[3]) - (pts[3] - pts[6])) / m
    f["pos_max_24h"] = (max(inp["max_24h"], series.max()) - b) / m
    f["pos_std_6h"] = float(pd.Series(pos[:6]).std())
    ups = list(st.upstream)
    if ups and inp.get("up_now") is not None:
        u = bundle["stations"].loc[ups[0]]
        ub = bundle["base_levels"][ups[0]]
        um = u.minor - ub
        f["up_pos_max"] = (inp["up_now"] - ub) / um
        f["up_rise_3h_max"] = (inp["up_now"] - inp["up_3h"]) / um
    else:
        f["up_pos_max"], f["up_rise_3h_max"] = f["pos_now"], f["rise_3h"]
    f["has_upstream"] = int(len(ups) > 0)
    f["basin_pos_max_other"] = f["up_pos_max"]
    f["basin_rise_3h_mean"] = (f["rise_3h"] + f["up_rise_3h_max"]) / 2 if ups else f["rise_3h"]
    f["n_alert_national"] = float(inp["n_alert"])
    f["n_rising_national"] = int(inp["n_rise"])
    f["margin_m"] = m
    return pd.DataFrame([f])[FEATURES]
