"""Shared loaders, colours and small UI components for the Streamlit app."""
from __future__ import annotations

import html
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from src.features import FEATURE_DESCRIPTIONS, load_readings
from src.live import fetch_live_readings, latest, now_sl, score_window

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "flood_model.pkl"
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"
ARCHIVE_PATH = ROOT / "data" / "raw" / "readings.json.gz"

# ------------------------------------------------------------------ colours
INK, RIVER, SHALLOW, LINE, MUTED = "#13232F", "#0B5C7A", "#EEF4F7", "#D3DFE6", "#5B6B77"
STATUS_COLORS = {
    "Major flood": "#7A0E0E",
    "Flooding": "#C62828",
    "Flood likely within 6 h": "#E8590C",
    "Alert level": "#C99A06",
    "Normal": "#2F7D55",
    "No recent data": "#8A9BA8",
}
STATUS_HELP = {
    "Major flood": "At or above the official major-flood level",
    "Flooding": "At or above the official minor-flood level",
    "Flood likely within 6 h": "Below flood level now, but the model expects it to reach flood level within 6 hours",
    "Alert level": "Above the official alert level, model does not expect a flood within 6 hours",
    "Normal": "Below the alert level",
    "No recent data": "No reading in the last 3 hours",
}
THRESH_COLORS = {"alert": "#C99A06", "minor": "#E8590C", "major": "#B42318"}

FRIENDLY = {
    "pos_now": "How close the river is to flood level now",
    "rise_1h": "Rise in the last hour",
    "rise_3h": "Rise in the last 3 hours",
    "rise_6h": "Rise in the last 6 hours",
    "rise_12h": "Rise in the last 12 hours",
    "rise_24h": "Rise in the last 24 hours",
    "accel_3h": "Is the rise speeding up?",
    "pos_max_24h": "Highest level in the last 24 hours",
    "pos_std_6h": "How unsteady the level has been (6 h)",
    "up_pos_max": "How high the upstream gauge is",
    "up_rise_3h_max": "How fast the upstream gauge is rising",
    "has_upstream": "Has a gauge upstream",
    "basin_pos_max_other": "Highest other gauge in the same basin",
    "basin_rise_3h_mean": "Average rise across the basin",
    "n_alert_national": "Gauges at alert level nationally",
    "n_rising_national": "Gauges rising fast nationally",
    "margin_m": "Gap between normal and flood level (m)",
}

# Plotly look shared by every chart
pio.templates["river"] = go.layout.Template(layout=dict(
    font=dict(family="IBM Plex Sans, sans-serif", size=13, color=INK),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    colorway=["#0B5C7A", "#E8590C", "#2F7D55", "#C99A06", "#7A4EA3", "#8A9BA8", "#B42318", "#3B9AB2"],
    xaxis=dict(gridcolor="#E6EDF1", linecolor=LINE, zeroline=False, ticks="outside", tickcolor=LINE),
    yaxis=dict(gridcolor="#E6EDF1", linecolor=LINE, zeroline=False),
    margin=dict(l=10, r=10, t=40, b=10), hoverlabel=dict(font_family="IBM Plex Sans"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""),
    title=dict(font=dict(size=15), x=0, xanchor="left"),
))
pio.templates.default = "river"


# ------------------------------------------------------------------ loading
@st.cache_resource(show_spinner="Loading the trained model…")
def load_bundle():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_resource(show_spinner=False)
def shap_explainer():
    import shap
    return shap.TreeExplainer(load_bundle()["model"])


@st.cache_data(ttl=900, show_spinner="Downloading the latest river levels…")
def live_readings():
    """Cached for 15 minutes; the archive itself updates about once an hour."""
    df = fetch_live_readings()
    return df, now_sl()


@st.cache_data(show_spinner="Loading the archived readings…")
def archive_readings():
    return load_readings(ARCHIVE_PATH)


@st.cache_data(show_spinner="Scoring every gauge…")
def scored(source_key: str, at, fetched_at=None):
    """fetched_at is part of the cache key, so new live data is re-scored."""
    bundle = load_bundle()
    if source_key == "live":
        readings, _ = live_readings()
    else:
        readings = archive_readings()
    table, t_end, report = score_window(readings, bundle, at=at)
    return table, t_end, report


@st.cache_data
def read_csv(name, **kw):
    p = REPORTS / name
    return pd.read_csv(p, **kw) if p.exists() else None


def figure(name):
    p = FIGS / f"{name}.png"
    return str(p) if p.exists() else None


# ------------------------------------------------------------------ context shared by the monitor pages
def get_context():
    """Data source chosen in the sidebar -> (table, snapshot time, latest-per-station, source label)."""
    src = st.session_state.get("source", "live")
    at = st.session_state.get("replay_at") if src == "archive" else None
    error = None
    fetched = None
    if src == "live":
        try:
            _, fetched = live_readings()
            table, t_end, _ = scored("live", None, fetched)
        except Exception as e:  # network or format problem: fall back to the archive
            error = str(e)
            src = "archive"
            at = st.session_state.get("replay_at")
    if src == "archive":
        table, t_end, _ = scored("archive", at)
    return {"table": table, "t": t_end, "snap": latest(table, t_end), "source": src, "error": error,
            "fetched": fetched}


# ------------------------------------------------------------------ components
def page_css():
    st.html("""
<style>
  .block-container {padding-top: 5rem; max-width: 1280px;}
  h1, h2, h3 {letter-spacing: -0.01em;}
  .headline {font-size: 2.05rem; line-height: 1.2; font-weight: 600; color: #13232F; margin: 0 0 .35rem 0; max-width: 34ch;}
  .subline {color: #5B6B77; font-size: 0.98rem; margin-bottom: 1.1rem;}
  .ribbon {display: flex; width: 100%; height: 46px; border-radius: 6px; overflow: hidden; margin: .4rem 0 .5rem 0;}
  .ribbon div {display: flex; align-items: center; justify-content: center; color: #fff; font-weight: 600;
               font-size: .95rem; min-width: 34px; white-space: nowrap;}
  .legend {display: flex; flex-wrap: wrap; gap: .4rem 1.2rem; font-size: .88rem; color: #3B4B57; margin-bottom: .6rem;}
  .legend span.sw {display: inline-block; width: 11px; height: 11px; border-radius: 2px; margin-right: .4rem; vertical-align: -1px;}
  .pill {display: inline-block; padding: .18rem .62rem; border-radius: 999px; color: #fff; font-weight: 600; font-size: .85rem;}
  .note {background: #EEF4F7; border-left: 3px solid #0B5C7A; padding: .7rem .9rem; border-radius: 0 6px 6px 0;
         color: #22333F; font-size: .93rem; margin: .4rem 0 1rem 0;}
  .staff-wrap {display: flex; gap: 14px; align-items: stretch;}
  .staff {position: relative; width: 74px; height: 320px; background: repeating-linear-gradient(to top, #fff 0 9px, #F1F5F7 9px 18px);
          border: 2px solid #13232F; border-radius: 4px; overflow: hidden;}
  .staff .water {position: absolute; left: 0; right: 0; bottom: 0; background: linear-gradient(#3B9AB2, #0B5C7A);}
  .staff .mark {position: absolute; left: 0; right: 0; height: 0; border-top: 2px dashed;}
  .staff-labels {position: relative; height: 324px; width: 150px; font-size: .82rem;}
  .staff-labels div {position: absolute; left: 0; transform: translateY(50%); white-space: nowrap;}
  .bignum {font-size: 2.4rem; font-weight: 700; line-height: 1; font-variant-numeric: tabular-nums;}
  .smalllabel {color: #5B6B77; font-size: .88rem;}
  [data-testid="stSidebar"] .smalllabel {color: #A9BCC7;}
  @media (prefers-reduced-motion: reduce) { * {transition: none !important; animation: none !important;} }
</style>""")


def pill(status: str) -> str:
    return f'<span class="pill" style="background:{STATUS_COLORS[status]}">{html.escape(status)}</span>'


def status_ribbon(snap: pd.DataFrame):
    counts = snap.status.value_counts()
    parts, legend = [], []
    total = len(snap)
    for s, c in STATUS_COLORS.items():
        n = int(counts.get(s, 0))
        if n:
            parts.append(f'<div style="background:{c};flex:{n}" title="{s}: {n}">{n}</div>')
        legend.append(f'<span title="{html.escape(STATUS_HELP[s])}"><span class="sw" style="background:{c}"></span>{s} · {n}</span>')
    st.html(f'<div class="ribbon" role="img" aria-label="Status of {total} gauges">{"".join(parts)}</div>'
            f'<div class="legend">{"".join(legend)}</div>')


def situation_headline(snap: pd.DataFrame) -> str:
    c = snap.status.value_counts()
    flooding = int(c.get("Flooding", 0) + c.get("Major flood", 0))
    likely = int(c.get("Flood likely within 6 h", 0))
    if flooding == 0 and likely == 0:
        return "No river is flooding, and none is expected to flood in the next 6 hours."
    a = f"{flooding} gauge{'s are' if flooding != 1 else ' is'} at flood level" if flooding else "No gauge is at flood level"
    b = (f"{likely} more {'are' if likely != 1 else 'is'} likely to flood within 6 hours" if likely
         else "no other gauge is expected to flood within 6 hours")
    return f"{a}, and {b}."


def staff_gauge(level, base, alert, minor, major, risk=None):
    """A river staff gauge drawn in HTML: water height against the official levels."""
    lo = min(base - 0.25 * (minor - base), level if level == level else base) - 0.1
    hi = max(major + 0.12 * (major - base), level if level == level else major) + 0.1
    frac = lambda v: (v - lo) / (hi - lo) * 100
    marks = [("Major flood", major, THRESH_COLORS["major"]), ("Minor flood", minor, THRESH_COLORS["minor"]),
             ("Alert", alert, THRESH_COLORS["alert"]), ("Normal", base, "#2F7D55")]
    lines = "".join(f'<div class="mark" style="bottom:{frac(v):.1f}%;border-color:{c}"></div>' for _, v, c in marks)
    labels = "".join(f'<div style="bottom:{frac(v):.1f}%;color:{c}"><b>{n}</b> {v:.2f} m</div>' for n, v, c in marks)
    water = f'<div class="water" style="height:{max(0, frac(level)):.1f}%"></div>' if level == level else ""
    return (f'<div class="staff-wrap" role="img" aria-label="Water level {level:.2f} m">'
            f'<div class="staff">{water}{lines}</div><div class="staff-labels">{labels}</div></div>')


def level_chart(g: pd.DataFrame, station_row, threshold: float, height=380):
    """Water level with the official thresholds, and the model's 6-hour flood risk underneath."""
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.06)
    fig.add_trace(go.Scatter(x=g.t, y=g.level, name="Water level", mode="lines",
                             line=dict(color=RIVER, width=2.4), hovertemplate="%{x|%d %b %H:%M}<br>%{y:.2f} m<extra></extra>"),
                  row=1, col=1)
    for key, label in [("alert", "Alert"), ("minor", "Minor flood"), ("major", "Major flood")]:
        v = float(station_row[key])
        fig.add_hline(y=v, line=dict(color=THRESH_COLORS[key], dash="dash", width=1.4), row=1, col=1,
                      annotation_text=f"{label} {v:.2f} m", annotation_position="top left",
                      annotation_font=dict(color=THRESH_COLORS[key], size=11))
    fig.add_trace(go.Scatter(x=g.t, y=g.risk, name="Flood risk (next 6 h)", mode="lines", fill="tozeroy",
                             line=dict(color="#E8590C", width=1.6), fillcolor="rgba(232,89,12,.18)",
                             hovertemplate="%{x|%d %b %H:%M}<br>risk %{y:.2f}<extra></extra>"), row=2, col=1)
    fig.add_hline(y=threshold, line=dict(color="#7A0E0E", dash="dot", width=1.2), row=2, col=1,
                  annotation_text=f"warning threshold {threshold:.2f}", annotation_position="top left",
                  annotation_font=dict(size=11, color="#7A0E0E"))
    fl = g[g.is_flood == 1]
    if len(fl):
        fig.add_trace(go.Scatter(x=fl.t, y=[0.02] * len(fl), mode="markers", name="At flood level",
                                 marker=dict(color="#C62828", size=6, symbol="square"),
                                 hovertemplate="%{x|%d %b %H:%M} at flood level<extra></extra>"), row=2, col=1)
    top = max(float(station_row["major"]), float(np.nanmax(g.level))) if g.level.notna().any() else float(station_row["major"])
    bottom = float(np.nanmin(g.level)) if g.level.notna().any() else 0
    fig.update_yaxes(title_text="Level (m)", range=[bottom - 0.05 * (top - bottom) - 0.1, top * 1.03 + 0.1], row=1, col=1)
    fig.update_yaxes(title_text="Risk", range=[0, 1], row=2, col=1)
    fig.update_layout(height=height, showlegend=True, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def shap_bar(values: pd.Series, feature_values: pd.Series, base_value: float, height=420):
    """Horizontal bar chart of SHAP contributions, largest first, in plain language."""
    s = values.reindex(values.abs().sort_values(ascending=True).index)
    s = s.iloc[-10:]
    labels = [f"{FRIENDLY.get(k, k)}  ({feature_values[k]:.2f})" for k in s.index]
    colors = ["#C62828" if v > 0 else "#2F7D55" for v in s.values]
    fig = go.Figure(go.Bar(x=s.values, y=labels, orientation="h", marker_color=colors,
                           text=[f"{v:+.3f}" for v in s.values], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}<br>pushes risk by %{x:+.3f}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=INK, width=1))
    span = float(np.abs(s.values).max()) or 0.01
    fig.update_layout(height=height, xaxis=dict(title=f"Effect on risk (starting from the average risk {base_value:.3f})",
                                                range=[-span * 1.35 if (s < 0).any() else -span * 0.05, span * 1.35]),
                      yaxis=dict(automargin=True), margin=dict(l=10, r=20, t=10, b=10))
    return fig


def fmt_t(t) -> str:
    return pd.Timestamp(t).strftime("%a %d %b %Y, %H:%M")


def require_model():
    b = load_bundle()
    if b is None:
        st.error("The trained model file `models/flood_model.pkl` was not found. "
                 "Run the notebook `notebooks/flood_warning.ipynb` from top to bottom once; "
                 "its last section saves the model, then reload this page.")
        st.stop()
    return b


def describe(feature):
    return FEATURE_DESCRIPTIONS.get(feature, "")
