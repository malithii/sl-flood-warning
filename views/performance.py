import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.common import RIVER, figure, read_csv, require_model

bundle = require_model()
best = bundle["model_name"]
res = read_csv("test_results.csv", index_col=0)
tun = read_csv("tuning_results.csv", index_col=0)
ev = read_csv("test_flood_events.csv", parse_dates=["onset"])
if res is None:
    st.error("`reports/test_results.csv` is missing. Run the notebook once to create the reports.")
    st.stop()

m, a = res.loc[best], res.loc["Baseline: alert rule"]
n_on = len(ev) if ev is not None else None
fewer = 1 - m["false-alarm hours"] / a["false-alarm hours"]

warned = round(m["onsets warned ≥1 h"] * n_on) if n_on else None
share = (f"every one of the {n_on} new floods" if warned == n_on else f"{warned} of {n_on} new floods") if n_on \
    else f'{m["onsets warned ≥1 h"]:.0%} of new floods'
st.html(f'<h1 class="headline">On unseen data, {best} warned of {share} at least an hour ahead, '
        f'with {fewer:.0%} fewer false alarms than the alert rule.</h1>'
        f'<div class="subline">Test period 1 Aug – 23 Oct 2026, never used for training or tuning'
        f'{f": {n_on} flood onsets at {ev.station.nunique()} gauges" if ev is not None else ""}. '
        'A flood onset is the first flood hour after at least 12 hours below flood level.</div>')

# ---------------------------------------------------------------- model vs today's rule
st.subheader(f"{best} compared with today's alert-level rule")
rows = [("Flood onsets warned at least 1 h ahead", "onsets warned ≥1 h", "{:.0%}", True),
        ("Flood onsets warned at least 3 h ahead", "onsets warned ≥3 h", "{:.0%}", True),
        ("Median warning time (hours)", "median lead (h)", "{:.1f}", True),
        ("False-alarm station-hours", "false-alarm hours", "{:,.0f}", False),
        ("Precision (share of warnings that were real)", "precision", "{:.2f}", True),
        ("Recall (share of new-flood hours warned)", "recall", "{:.2f}", True),
        ("PR-AUC (ranking quality, higher is better)", "PR-AUC", "{:.3f}", True)]
cmp = pd.DataFrame([{"Measure": lab, best: fmt.format(m[c]), "Alert-level rule": fmt.format(a[c]),
                     "Better": best if ((m[c] > a[c]) == hi and m[c] != a[c]) else ("same" if m[c] == a[c] else "Alert rule")}
                    for lab, c, fmt, hi in rows])
st.dataframe(cmp, hide_index=True, column_config={"Better": st.column_config.TextColumn("Better", width="small")})
st.caption("The alert rule warns whenever a river is above its official alert level. It gives early warnings "
           "but fires on many hours that never become floods, which teaches people to ignore warnings.")

# ---------------------------------------------------------------- all models
st.subheader("Eight models and three baselines")
c1, c2 = st.columns(2, gap="large")
ml = res[~res.index.str.startswith("Baseline")]
with c1:
    d = pd.DataFrame({"Cross-validation (development)": tun["CV PR-AUC"] if tun is not None else np.nan,
                      "Test": ml["PR-AUC"]}).sort_values("Cross-validation (development)")
    fig = go.Figure()
    fig.add_bar(y=d.index, x=d.iloc[:, 0], orientation="h", name="Cross-validation (development)", marker_color="#9DB9C9")
    fig.add_bar(y=d.index, x=d["Test"], orientation="h", name="Test", marker_color=RIVER)
    fig.add_vline(x=a["PR-AUC"], line=dict(color="#E8590C", dash="dash"),
                  annotation_text="alert rule (test)", annotation_position="bottom right",
                  annotation_font_color="#E8590C")
    fig.update_layout(barmode="group", height=460, title="PR-AUC: the model was chosen on cross-validation",
                      xaxis_title="PR-AUC", legend=dict(orientation="h", y=-0.2, yanchor="top"),
                      margin=dict(t=50, b=10))
    st.plotly_chart(fig, config={"displayModeBar": False})
with c2:
    fig = go.Figure()
    for name, r in res[res["false-alarm hours"] > 0].iterrows():
        base_ = name.startswith("Baseline")
        label = name == best or base_
        fig.add_scatter(x=[r["false-alarm hours"]], y=[r["onsets warned ≥3 h"]],
                        mode="markers+text" if label else "markers",
                        text=[name.replace("Baseline: ", "") if label else ""],
                        textposition="bottom center" if name == best else "top center", showlegend=False,
                        marker=dict(size=15 if name == best else 11, symbol="triangle-up" if base_ else "circle",
                                    color="#E8590C" if base_ else (RIVER if name == best else "#8A9BA8"),
                                    line=dict(width=1.5, color="white")),
                        hovertemplate=f"<b>{name}</b><br>%{{x:,}} false-alarm hours<br>"
                                      f"%{{y:.0%}} of onsets warned ≥3 h<extra></extra>")
    fig.update_layout(height=460, title="Early warning against false alarms (top-left is best)",
                      xaxis=dict(title="False-alarm station-hours (log scale)", type="log",
                                 tickvals=[300, 500, 1000, 2000, 3000], ticktext=["300", "500", "1,000", "2,000", "3,000"]),
                      yaxis=dict(title="Onsets warned ≥ 3 h ahead", tickformat=".0%", range=[-0.05, 1.05]))
    st.plotly_chart(fig, config={"displayModeBar": False})

show = res.copy()
show.insert(0, "type", np.where(show.index.str.startswith("Baseline"), "Rule", "Model"))
st.dataframe(show.reset_index().rename(columns={"index": "model"}), hide_index=True, height=35 * (len(show) + 1) + 4,
             column_config={
    "model": st.column_config.TextColumn("Model", width="medium"),
    "PR-AUC": st.column_config.NumberColumn(format="%.3f"), "ROC-AUC": st.column_config.NumberColumn(format="%.3f"),
    "precision": st.column_config.NumberColumn("Precision", format="%.3f"),
    "recall": st.column_config.NumberColumn("Recall", format="%.3f"),
    "F2": st.column_config.NumberColumn(format="%.3f"),
    "false-alarm hours": st.column_config.NumberColumn("False alarms (h)", format="%d"),
    "onsets warned ≥1 h": st.column_config.ProgressColumn("Warned ≥1 h", format="percent", min_value=0, max_value=1),
    "onsets warned ≥3 h": st.column_config.ProgressColumn("Warned ≥3 h", format="percent", min_value=0, max_value=1),
    "median lead (h)": st.column_config.NumberColumn("Median lead (h)", format="%.1f")})
st.caption("Thresholds for every model were chosen to maximise F2 on out-of-fold development predictions. "
           "Accuracy is not shown: predicting “no flood” every hour is over 99.8% accurate and useless.")
if tun is not None:
    with st.expander("Tuning results (cross-validation by calendar month)"):
        st.dataframe(tun, column_config={"CV PR-AUC": st.column_config.NumberColumn(format="%.3f"),
                                         "CV PR-AUC std": st.column_config.NumberColumn(format="%.3f"),
                                         "threshold (F2, out-of-fold)": st.column_config.NumberColumn(format="%.3f")})

# ---------------------------------------------------------------- lead time per flood
if ev is not None:
    st.subheader("Warning time for every flood in the test period")
    e = ev.sort_values("onset").reset_index(drop=True)
    e["label"] = e.station + "  " + e.onset.dt.strftime("%d %b %H:%M")
    fig = go.Figure()
    for i, r in e.iterrows():
        fig.add_shape(type="line", x0=r.alert_rule_lead_h, x1=r.model_lead_h, y0=r.label, y1=r.label,
                      line=dict(color="#C9D6DE", width=3))
    fig.add_scatter(x=e.alert_rule_lead_h, y=e.label, mode="markers", name="Alert-level rule",
                    marker=dict(symbol="triangle-up", size=11, color="#E8590C"))
    fig.add_scatter(x=e.model_lead_h, y=e.label, mode="markers", name=best, marker=dict(size=11, color=RIVER))
    fig.update_layout(height=max(380, 24 * len(e)), xaxis=dict(title="Hours of warning before the flood (0–6)",
                                                               range=[-0.4, 6.4], dtick=1),
                      yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, config={"displayModeBar": False})
    st.caption("0 h means no warning in the 6 hours before the flood. The longest possible warning is 6 h, "
               "the model's horizon.")

f1, f2 = figure("confusion"), figure("model_comparison")
if f1 or f2:
    with st.expander("Charts from the notebook"):
        if f1:
            st.image(f1, caption="Confusion matrices on the test period")
        if f2:
            st.image(f2, caption="PR-AUC and precision–recall curves")
