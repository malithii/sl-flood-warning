import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ui.common import RIVER, read_csv, require_model

require_model()
z = read_csv("fairness_by_zone.csv", index_col=0)
b = read_csv("fairness_by_basin.csv", index_col=0)
if z is None:
    st.error("`reports/fairness_by_zone.csv` is missing. Run the notebook once to create the reports.")
    st.stop()

wet, dry = z.loc["Wet zone (south-west)"], z.loc["Dry & intermediate zone"]
st.html(f'<h1 class="headline">Dry-zone communities get less notice: {dry["onsets warned ≥3 h"]:.0%} of their '
        f'floods were warned 3 hours ahead, against {wet["onsets warned ≥3 h"]:.0%} in the wet zone.</h1>'
        '<div class="subline">Most training floods came from the south-west wet zone (Kelani, Kalu, Gin, Nilwala, '
        'Attanagalu and Maha Oya basins), so the model was checked separately for each region.</div>')

fig = make_subplots(rows=1, cols=3, subplot_titles=["Floods warned ≥ 3 h ahead", "Recall (flood hours warned)",
                                                    "False alarms per gauge per day"])
colors = {"Wet zone (south-west)": RIVER, "Dry & intermediate zone": "#C99A06"}
for i, col in enumerate(["onsets warned ≥3 h", "recall (hours)", "false alarms per station-day"], start=1):
    for zone, r in z.iterrows():
        fig.add_bar(x=[zone.split(" (")[0]], y=[r[col]], marker_color=colors.get(zone, "#8A9BA8"), name=zone,
                    showlegend=(i == 1), text=[f"{r[col]:.0%}" if i < 3 else f"{r[col]:.2f}"], textposition="outside",
                    cliponaxis=False, row=1, col=i)
fig.update_yaxes(tickformat=".0%", range=[0, 1.1], row=1, col=1)
fig.update_yaxes(tickformat=".0%", range=[0, 1.1], row=1, col=2)
fig.update_layout(height=380, margin=dict(t=60), showlegend=False)
st.plotly_chart(fig, config={"displayModeBar": False})

c1, c2, c3 = st.columns(3, gap="large")
c1.markdown(f"**Unequal early warning.** Every test flood in both zones got at least 1 hour of warning, but only "
            f"{dry['onsets warned ≥3 h']:.0%} of dry-zone floods got 3 hours (wet zone {wet['onsets warned ≥3 h']:.0%}).")
c2.markdown(f"**Unequal false alarms.** Wet-zone gauges raise about {wet['false alarms per station-day']:.2f} "
            f"false-alarm hours per gauge per day, dry-zone gauges about {dry['false alarms per station-day']:.2f}. "
            "Warning fatigue is a bigger risk in the wet zone.")
n_unverified = int(b["test flood onsets"].eq(0).sum()) if b is not None else 11
c3.markdown(f"**Unverified regions.** {n_unverified} of {len(b) if b is not None else 18} basins had no floods in the "
            "test period, so performance there cannot be measured yet.")

if b is not None:
    st.subheader("By river basin")
    st.dataframe(b.sort_values("test new-flood hours", ascending=False), height=35 * (len(b) + 1) + 4, column_config={
        "basin": "Basin", "stations": "Gauges",
        "development new-flood hours": st.column_config.NumberColumn("Training flood hours"),
        "test new-flood hours": st.column_config.NumberColumn("Test flood hours"),
        "test flood onsets": st.column_config.NumberColumn("Test floods"),
        "onsets warned ≥1 h": st.column_config.NumberColumn("Warned ≥1 h", format="percent"),
        "onsets warned ≥3 h": st.column_config.NumberColumn("Warned ≥3 h", format="percent"),
        "recall (hours)": st.column_config.NumberColumn("Recall", format="%.2f"),
        "false alarms per station-day": st.column_config.NumberColumn("False alarms / gauge / day", format="%.3f")})
    st.caption("“None” means the basin had no flood in the test period, so it cannot be scored.")
