import streamlit as st

from ui.common import FRIENDLY, figure, read_csv, require_model

bundle = require_model()

st.html('<h1 class="headline">The model asks three questions: is the river already high, is water coming '
        'from upstream, and is a storm active?</h1>'
        '<div class="subline">SHAP and LIME explanations of the selected model, computed in the notebook on the '
        'test period. For a live explanation of any gauge, open Station detail.</div>')

g, loc, lime, cmpt = st.tabs(["What matters overall", "One real warning", "LIME", "SHAP and LIME compared"])

with g:
    c1, c2 = st.columns(2, gap="large")
    if figure("shap_bar"):
        c1.image(figure("shap_bar"), caption="Average impact of each input (mean |SHAP value|)")
    if figure("shap_beeswarm"):
        c2.image(figure("shap_beeswarm"), caption="Every dot is one station-hour. Red = high input value; "
                                                   "dots to the right push towards a flood warning.")
    st.markdown("""
**How to read it**

- **How close the river is to its flood level** (`pos_now`, `pos_max_24h`) matters most. This is how flood
  forecasters work: the closer the stage is to flood level, the less extra water is needed.
- **The upstream gauge** (`up_pos_max`) matters because a flood wave travels downstream. The notebook measured
  this delay directly, for example about 3 hours from Glencourse to Hanwella.
- **Many rivers rising at once** (`n_rising_national`, `n_alert_national`) is the signature of a monsoon surge or
  cyclone, which also saturates catchments.
- **A wide gap between normal and flood level** (`margin_m`) lowers the risk: big, deep channels need much more
  rain to overtop.
""")
    if figure("shap_dependence"):
        st.image(figure("shap_dependence"), caption="How the three most important inputs change the prediction")
    with st.expander("What each input means"):
        st.dataframe([{"Input": k, "Plain meaning": v} for k, v in FRIENDLY.items()], hide_index=True)

with loc:
    ex = bundle.get("example", {})
    st.markdown(f"**{ex.get('label', 'Baddegama, 21 Sep 2026')}**")
    c1, c2 = st.columns([5, 6], gap="large")
    if figure("shap_waterfall"):
        c1.image(figure("shap_waterfall"), caption="SHAP waterfall for the warning hour")
    with c2:
        st.markdown(f"""
At 09:00 the Gin Ganga at Baddegama stood at **{ex.get('level_now', 3.91):.2f} m**, just below its
**4.0 m** flood level.

- Being this close to flood level pushed the risk **up** strongly.
- No rivers rising nationally, and a **falling** upstream gauge at Thawalama, pushed it **down**.
- The net score was still above the warning threshold of **{bundle['threshold']:.3f}**, so the model warned
  6 hours before the river flooded at 15:00.

One behaviour a hydrologist would question: a falling upstream gauge lowers the risk even when this station is
close to flooding. That is sensible on average, but wrong when heavy rain falls between the two gauges.
""")
    if figure("event_timeline"):
        st.image(figure("event_timeline"), caption="Model risk (shaded) and water level around the flood")

with lime:
    for name, cap in [("lime_true_warning", "A correct warning"), ("lime_false_alarm", "The highest-scoring false alarm"),
                      ("lime_missed_flood", "The clearest missed flood")]:
        if figure(name):
            st.image(figure(name), caption=cap)
    st.markdown("""
- **False alarm:** Deraniyagala, 20 Sep 2026. The river jumped 1.5 m in three hours during a national storm, then
  receded before reaching its 5.8 m flood level. A near miss worth knowing about.
- **Missed flood:** Holombuwa, 1 Aug 2026. The river was at its normal level, then rose 3.2 m in four hours. A
  flash flood like this leaves no sign in river levels beforehand; only rainfall data could warn of it.
""")

with cmpt:
    sl = read_csv("shap_vs_lime.csv", index_col=0)
    if sl is not None:
        st.dataframe(sl.sort_values("SHAP rank"), column_config={
            "value": st.column_config.NumberColumn("Input value", format="%.3f"),
            "SHAP value": st.column_config.NumberColumn(format="%.3f"),
            "LIME weight": st.column_config.NumberColumn(format="%.4f")})
    st.markdown("""
LIME agrees that the level inputs matter most, but its overall ranking differs and its weights are tiny.
LIME explains by sampling typical inputs from the training data. Those are mostly calm rivers where the risk is
almost zero, so its simple local model is nearly flat around a rare, sharp flood boundary. SHAP's TreeExplainer
is exact for tree ensembles, so it is the more trustworthy explanation here. Using both, and seeing where they
differ, is itself a useful check.
""")
