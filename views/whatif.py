import html

import numpy as np
import pandas as pd
import streamlit as st

from src.live import features_from_manual
from ui.common import STATUS_COLORS, get_context, pill, require_model, shap_bar, shap_explainer, staff_gauge

bundle = require_model()
ctx = get_context()
stations, base, thr = bundle["stations"], bundle["base_levels"], bundle["threshold"]
table = ctx["table"]

st.html('<h1 class="headline">What if the river keeps rising?</h1>'
        '<div class="subline">Type in recent readings for one gauge and see the model\'s 6-hour flood risk. '
        'Start from a real warning or the latest readings, then change them to test a scenario.</div>')

names = sorted(stations.index)
ex_station = bundle["example"]["station"]
c0, c1 = st.columns([5, 4], vertical_alignment="bottom")
station = c0.selectbox("Station", names, index=names.index(ex_station), key="whatif_station",
                       help="Starts on the gauge from the saved real warning; pick any gauge to test it.")
start = c1.segmented_control("Start from", ["A real warning from the test period", "Latest readings"],
                             default="A real warning from the test period", key="whatif_start", required=True)
s = stations.loc[station]
ups = list(s.upstream)


def defaults():
    ex = bundle["example"]
    if start == "A real warning from the test period" and ex["station"] == station:
        return ex
    g = table[table.station == station].set_index("t").level
    t = ctx["t"]
    get = lambda h, series=g: float(series.get(t - pd.Timedelta(hours=h), np.nan))
    d = {k: get(h) for k, h in [("level_now", 0), ("level_1h", 1), ("level_3h", 3), ("level_6h", 6),
                                 ("level_12h", 12), ("level_24h", 24)]}
    d["max_24h"] = float(g[g.index > t - pd.Timedelta(hours=24)].max())
    if ups:
        gu = table[table.station == ups[0]].set_index("t").level
        d["up_now"], d["up_3h"] = get(0, gu), get(3, gu)
    snap = ctx["snap"].iloc[0]
    d["n_alert"], d["n_rise"] = int(snap.n_alert_national), int(snap.n_rising_national)
    fallback = float(base[station])
    return {k: (fallback if (v is None or (isinstance(v, float) and np.isnan(v))) else v) for k, v in d.items()}


if start == "A real warning from the test period" and bundle["example"]["station"] != station:
    st.info(f"The saved real warning is for **{bundle['example']['station']}**. "
            "Choose that station to load it, or start from the latest readings.")
d = defaults()
key = f"{station}-{start}-{ctx['t']}"

with st.form("whatif"):
    st.markdown(f"**Water level at {station} (m)** · flood level {s.minor:.2f} m, alert {s.alert:.2f} m")
    cols = st.columns(6)
    labels = [("level_now", "Now"), ("level_1h", "1 h ago"), ("level_3h", "3 h ago"), ("level_6h", "6 h ago"),
              ("level_12h", "12 h ago"), ("level_24h", "24 h ago")]
    vals = {k: cols[i].number_input(lab, value=float(d[k]), step=0.05, format="%.2f", key=f"{k}-{key}")
            for i, (k, lab) in enumerate(labels)}
    vals["max_24h"] = st.number_input("Highest level in the last 24 h (m)", value=float(d["max_24h"]), step=0.05,
                                      format="%.2f", key=f"max-{key}")
    if ups:
        st.markdown(f"**Upstream gauge: {ups[0]}** · flood level {stations.loc[ups[0]].minor:.2f} m")
        u1, u2 = st.columns(2)
        vals["up_now"] = u1.number_input("Now", value=float(d.get("up_now", base[ups[0]])), step=0.05, format="%.2f",
                                         key=f"upn-{key}")
        vals["up_3h"] = u2.number_input("3 h ago", value=float(d.get("up_3h", base[ups[0]])), step=0.05,
                                        format="%.2f", key=f"up3-{key}")
    st.markdown("**Across Sri Lanka**")
    n1, n2 = st.columns(2)
    vals["n_alert"] = n1.slider("Gauges at alert level or higher", 0, 39, int(d["n_alert"]), key=f"na-{key}")
    vals["n_rise"] = n2.slider("Gauges rising fast", 0, 39, int(d["n_rise"]), key=f"nr-{key}")
    submitted = st.form_submit_button("Check flood risk", type="primary", icon=":material/water_drop:")

lvl = vals["level_now"]
if lvl >= s.minor:
    status = "Major flood" if lvl >= s.major else "Flooding"
    st.html(f'<div style="margin:.6rem 0">{pill(status)}</div>'
            '<div class="note">At this level the gauge is already flooding, so the official threshold rule '
            'applies and the model is not used. Lower the current level to test an early warning.</div>')
    st.stop()

X = features_from_manual(vals, station, bundle)
risk = float(bundle["model"].predict_proba(X)[:, 1][0])
status = "Flood likely within 6 h" if risk >= thr else ("Alert level" if lvl >= s.alert else "Normal")

a, b_ = st.columns([3, 9], gap="large")
with a:
    st.html(staff_gauge(lvl, base[station], s.alert, s.minor, s.major))
with b_:
    st.html(f'<div class="bignum" style="color:{STATUS_COLORS[status]}">{risk:.2f}</div>'
            f'<div class="smalllabel">risk of reaching flood level within 6 hours</div>'
            f'<div style="margin:.8rem 0">{pill(status)}</div>')
    verdict = ("The model would raise a warning." if risk >= thr
               else f"No warning: the score is below the threshold of {thr:.3f}.")
    st.markdown(verdict)
    st.caption("Scores come from a class-balanced model, so treat them as a ranking of risk rather than an "
               "exact probability.")
st.markdown("**Why this score**")
sv = shap_explainer()(X)
vals_ = sv.values[0, :, 1] if sv.values.ndim == 3 else sv.values[0]
bv = sv.base_values[0][1] if np.ndim(sv.base_values) == 2 else float(np.ravel(sv.base_values)[0])
st.plotly_chart(shap_bar(pd.Series(vals_, index=X.columns), X.iloc[0], bv, height=380),
                config={"displayModeBar": False})
st.caption("Red bars push the risk up, green bars push it down. Values in brackets are the inputs on the "
           "model's 0–1 scale, where 0 is the normal level and 1 is the flood level.")

with st.expander("How the inputs become model features"):
    st.markdown("Hours between the readings you type are filled with straight lines. Basin context is "
                "approximated from the upstream gauge. On the Station detail page every feature comes from "
                "the full hourly pipeline instead, so small differences between the two pages are expected.")
    st.dataframe(X.T.rename(columns={0: "value"}).style.format("{:.3f}"))
