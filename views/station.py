import html

import numpy as np
import pandas as pd
import streamlit as st

from src.features import FEATURES
from ui.common import (FRIENDLY, STATUS_COLORS, fmt_t, get_context, level_chart, pill, require_model,
                       shap_bar, shap_explainer, staff_gauge)

bundle = require_model()
ctx = get_context()
table, snap, t = ctx["table"], ctx["snap"], ctx["t"]
stations, base, thr = bundle["stations"], bundle["base_levels"], bundle["threshold"]

names = sorted(snap.station)
default = st.session_state.get("station", snap.station.iloc[0])
station = st.selectbox("Station", names, index=names.index(default) if default in names else 0,
                       help="Stations are listed A–Z. The River situation page lists the most serious first.")
st.session_state["station"] = station

r = snap.set_index("station").loc[station]
s = stations.loc[station]
g = table[table.station == station].sort_values("t")

st.html(f'<h1 class="headline" style="margin-bottom:.2rem">{html.escape(station)} {pill(r.status)}</h1>'
        f'<div class="subline">{html.escape(s.river_name)} · {html.escape(s.basin)} basin · '
        f'reading for {fmt_t(t)}</div>')

c1, c2, c3 = st.columns([3, 5, 4], gap="large")
with c1:
    st.html(staff_gauge(r.level, base[station], s.alert, s.minor, s.major))
with c2:
    if r.status in ("Flooding", "Major flood"):
        big, label = f"{r.level - s.minor:+.2f} m", "above the minor-flood level"
    elif pd.isna(r.risk):
        big, label = "—", "not enough recent readings to score"
    else:
        big, label = f"{r.risk:.2f}", f"chance-like score of flooding within 6 h (warning at {thr:.3f})"
    st.html(f'<div class="bignum" style="color:{STATUS_COLORS[r.status]}">{big}</div>'
            f'<div class="smalllabel" style="margin-bottom:1rem">{label}</div>')
    rise3 = r.rise_3h * r.margin_m if pd.notna(r.rise_3h) else np.nan
    st.markdown(
        f"- Level now **{r.level:.2f} m**; flood level **{s.minor:.2f} m** "
        f"({'above it' if r.level >= s.minor else f'{s.minor - r.level:.2f} m to go'})\n"
        f"- Change in the last 3 hours: **{rise3:+.2f} m**\n"
        f"- Normal level here is about **{base[station]:.2f} m**\n"
        f"- Highest in the last 24 hours: **{base[station] + r.pos_max_24h * r.margin_m:.2f} m**")
    if r.status in ("Flooding", "Major flood"):
        st.html('<div class="note">This gauge is already at flood level, so it is reported by the official '
                'threshold rule. The model only scores gauges that are not yet flooding.</div>')
with c3:
    st.markdown("**Upstream gauges**")
    if not s.upstream:
        st.caption("No gauged station upstream. The model leans on this gauge's own trend and on "
                   "the national picture.")
    for u in s.upstream:
        ur = snap.set_index("station").loc[u] if u in set(snap.station) else None
        if ur is None:
            st.caption(f"{u}: no recent data")
            continue
        st.html(f'<div style="padding:.4rem 0;border-bottom:1px solid #E3EBF0"><b>{html.escape(u)}</b> '
                f'{pill(ur.status)}<div class="smalllabel">{ur.level:.2f} m · 3-h change '
                f'{ur.rise_3h * ur.margin_m:+.2f} m</div></div>')
    st.markdown("**National picture**")
    st.markdown(f"- {int(r.n_alert_national)} of 39 gauges at or above alert level\n"
                f"- {int(r.n_rising_national)} gauges rising fast")

st.subheader("Last 3 days")
st.plotly_chart(level_chart(g, s, thr), config={"displayModeBar": False})

st.subheader("Why the model gave this score")
if pd.isna(r.risk):
    st.info("There is no model score for this hour, so there is nothing to explain. "
            "Pick a gauge that is not flooding, or replay an earlier moment.")
else:
    with st.spinner("Calculating SHAP values…"):
        ex = shap_explainer()
        x = pd.DataFrame([r[FEATURES].astype(float).values], columns=FEATURES)
        sv = ex(x)
        vals = sv.values[0, :, 1] if sv.values.ndim == 3 else sv.values[0]
        base_val = sv.base_values[0][1] if np.ndim(sv.base_values) == 2 else float(np.ravel(sv.base_values)[0])
    contrib = pd.Series(vals, index=FEATURES)
    left, right = st.columns([8, 4], gap="large")
    with left:
        st.plotly_chart(shap_bar(contrib, x.iloc[0], base_val), config={"displayModeBar": False})
    with right:
        up = contrib[contrib > 0].sort_values(ascending=False).head(3)
        down = contrib[contrib < 0].sort_values().head(3)
        st.markdown("**Raising the risk**")
        st.markdown("\n".join(f"- {FRIENDLY[k]}" for k in up.index) or "- nothing notable")
        st.markdown("**Lowering the risk**")
        st.markdown("\n".join(f"- {FRIENDLY[k]}" for k in down.index) or "- nothing notable")
        st.caption("SHAP splits the score into one contribution per input. Red bars push the risk up, "
                   "green bars push it down; they add up to the model's score. Values in brackets are the "
                   "inputs on the model's 0–1 scale, where 0 is the normal level and 1 is the flood level.")
