import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.common import (STATUS_COLORS, STATUS_HELP, fmt_t, get_context, pill, require_model,
                       situation_headline, status_ribbon)

bundle = require_model()
ctx = get_context()
snap, t = ctx["snap"], ctx["t"]
thr = bundle["threshold"]

if ctx["error"]:
    st.warning("The live readings could not be downloaded, so a past moment from the archive is shown instead. "
               f"Check the internet connection and press **Check for new readings**. Details: {ctx['error'][:160]}")

# ---------------------------------------------------------------- headline
st.html(f'<h1 class="headline">{html.escape(situation_headline(snap))}</h1>')
if ctx["source"] == "live":
    age_h = (pd.Timestamp.now(tz="Asia/Colombo").tz_localize(None) - t).total_seconds() / 3600
    fresh = "" if age_h < 3 else f" The newest complete hour is {age_h:.0f} hours old, so the gauges may be offline."
    sub = f"Live readings up to {fmt_t(t)} (Sri Lanka time), {snap.station.nunique()} gauges in 18 river basins.{fresh}"
else:
    sub = f"Replaying archived readings at {fmt_t(t)} (Sri Lanka time). Nothing after this moment is used."
st.html(f'<div class="subline">{html.escape(sub)}</div>')
status_ribbon(snap)

# ---------------------------------------------------------------- map + attention list
left, right = st.columns([7, 5], gap="large")

with left:
    order = list(STATUS_COLORS)[::-1]                        # draw the serious ones on top
    fig = go.Figure()
    for s in order:
        d = snap[snap.status == s]
        if d.empty:
            continue
        size = {"Major flood": 22, "Flooding": 19, "Flood likely within 6 h": 19, "Alert level": 14}.get(s, 11)
        risk_txt = d.risk.map(lambda r: "—" if pd.isna(r) else f"{r:.2f}")
        fig.add_trace(go.Scattermap(
            lat=d.lat, lon=d.lng, mode="markers", name=s,
            marker=dict(size=size, color=STATUS_COLORS[s], opacity=0.92),
            customdata=list(zip(d.station, d.river_name, d.level, d.minor, risk_txt)),
            hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]}<br>Level %{customdata[2]:.2f} m "
                          "(flood level %{customdata[3]:.2f} m)<br>Risk %{customdata[4]}<extra>" + s + "</extra>"))
    fig.update_layout(map=dict(style="carto-positron", center=dict(lat=7.55, lon=80.75), zoom=6.35),
                      height=600, margin=dict(l=0, r=0, t=0, b=0),
                      legend=dict(orientation="v", x=0.01, y=0.99, bgcolor="rgba(255,255,255,.85)"))
    st.plotly_chart(fig, config={"displayModeBar": False, "scrollZoom": True})

with right:
    st.subheader("Needs attention")
    att = snap[snap.status.isin(["Major flood", "Flooding", "Flood likely within 6 h", "Alert level"])]
    if att.empty:
        st.html('<div class="note">Every gauge is below its alert level. Nothing needs attention right now.</div>')
    else:
        rows = []
        for _, r in att.head(12).iterrows():
            gap = r.minor - r.level
            gap_txt = f"{-gap:.2f} m above flood level" if gap <= 0 else f"{gap:.2f} m below flood level"
            risk_txt = "" if pd.isna(r.risk) else f" · risk {r.risk:.2f}"
            rows.append(f'<div style="padding:.55rem 0;border-bottom:1px solid #E3EBF0">'
                        f'<div style="display:flex;justify-content:space-between;gap:.5rem;align-items:center">'
                        f'<b>{html.escape(r.station)}</b>{pill(r.status)}</div>'
                        f'<div class="smalllabel">{html.escape(r.river_name)} · {r.level:.2f} m, {gap_txt}{risk_txt}</div></div>')
        st.html("".join(rows))
        if len(att) > 12:
            st.caption(f"…and {len(att) - 12} more in the table below.")
    st.caption(f"A gauge is marked *Flood likely within 6 h* when the model's risk is at least {thr:.3f}. "
               "That threshold was tuned to catch most floods (recall counts twice as much as precision).")

# ---------------------------------------------------------------- full table
st.subheader("All gauges")
st.caption("Select a row to open that station's detail page.")
tab = snap[["station", "status", "river_name", "level", "alert", "minor", "to_flood_m", "risk"]].copy()
tab.loc[snap.status.isin(["Flooding", "Major flood"]).values, "risk"] = 1.0
event = st.dataframe(
    tab, hide_index=True, height=35 * (len(tab) + 1) + 4, on_select="rerun", selection_mode="single-row", key="all_gauges",
    column_config={
        "station": st.column_config.TextColumn("Station", width="medium"),
        "status": st.column_config.TextColumn("Status", help="; ".join(f"{k}: {v}" for k, v in STATUS_HELP.items())),
        "river_name": "River",
        "level": st.column_config.NumberColumn("Level (m)", format="%.2f"),
        "alert": st.column_config.NumberColumn("Alert (m)", format="%.2f"),
        "minor": st.column_config.NumberColumn("Flood level (m)", format="%.2f"),
        "to_flood_m": st.column_config.NumberColumn("Below flood level (m)", format="%.2f",
                                                    help="Negative means the river is above its flood level"),
        "risk": st.column_config.ProgressColumn("Flood risk, next 6 h", min_value=0.0, max_value=1.0, format="%.2f",
                                                help="Model probability. Shown as 1.00 for gauges already flooding."),
    })
if event.selection.rows:
    st.session_state["station"] = tab.iloc[event.selection.rows[0]].station
    st.switch_page("views/station.py")
