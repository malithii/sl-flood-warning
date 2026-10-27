import streamlit as st

from src.features import FEATURE_DESCRIPTIONS, FEATURES
from src.live import ARCHIVE_REPO, DASHBOARD_URL
from ui.common import figure, require_model

bundle = require_model()

st.html('<h1 class="headline">Can the river levels Sri Lanka already publishes warn of a flood before it starts?</h1>'
        '<div class="subline">IT5504 Applied Machine Learning, final project. River flood early warning from '
        'Irrigation Department gauge data.</div>')

st.markdown(f"""
### The question
When a river gauge is **not yet flooding**, will it reach its official minor-flood level within the
**next 6 hours**, and can a model say so with fewer false alarms than today's threshold rules?

### The data
- **Source:** Sri Lanka Irrigation Department, Hydrology and Disaster Management Division. Water levels are
  published on a [live dashboard]({DASHBOARD_URL}) and archived about hourly at [{ARCHIVE_REPO.split('.com/')[1]}]({ARCHIVE_REPO}).
- **Size:** 296,111 near-hourly readings from 39 gauges in 18 river basins, 23 Nov 2025 – 23 Oct 2026.
- **Labels:** the department's own alert, minor-flood and major-flood levels for each gauge.
- **This app** downloads the same archive's last 28 days for the live view, and uses the saved archive for replays.
""")

c1, c2 = st.columns(2, gap="large")
if figure("eda_levels"):
    c1.image(figure("eda_levels"), caption="Water levels and official thresholds at four gauges")
if figure("eda_rise_before_flood"):
    c2.image(figure("eda_rise_before_flood"), caption="Rivers rise sharply in the hours before they flood")

st.markdown("""
### How the model is built
1. **Clean** the readings: a Hampel filter removes 650 isolated sensor spikes and drops to 0 m.
2. **Regular hourly grid:** the last reading in each hour; gaps of up to 3 hours are forward-filled, longer
   outages stay missing.
3. **Put every gauge on one scale:** 0 = the gauge's normal (median) level, 1 = its minor-flood level. A raw
   metre value means nothing across gauges (Glencourse floods at 16.5 m, Nagalagam Street at 1.5 m).
4. **Two-stage warning.** A gauge already at flood level is reported by the official rule. The model only
   predicts **new** floods, for gauges that are not flooding yet.
5. **Split by time:** develop on Nov 2025 – Jul 2026, test once on Aug – Oct 2026. Cross-validation uses whole
   calendar months, so neighbouring hours never sit on both sides of a fold.
6. **Compare eight models** (logistic regression, KNN, SVM, decision tree, random forest, Extra Trees, XGBoost,
   LightGBM) against three rule baselines, using PR-AUC because floods are rare (0.15% of hours).
7. **Choose the threshold** that maximises F2, which counts a missed flood as worse than a false alarm.
""")

st.markdown(f"### The {len(FEATURES)} inputs")
st.dataframe([{"Input": f, "Meaning": FEATURE_DESCRIPTIONS[f]} for f in FEATURES], hide_index=True,
             column_config={"Input": st.column_config.TextColumn(width="medium")})
st.caption("Calendar inputs (month, hour) were left out on purpose: with 11 months of data, a model could "
           "memorise storm dates instead of learning river behaviour.")

st.markdown("""
### Limitations
- Only 11 months of data and a handful of storms; 187 of the 266 development flood hours come from one cyclone.
- River levels only, no rainfall: flash floods with no earlier rise are missed.
- Scores come from a class-balanced model and are not calibrated probabilities.
- A 6-hour horizon and minor floods only.
- The history exists because a third party archives the public dashboard; thresholds are used as published.

### Responsible use
- **Benefit:** a median 4 hours of extra notice for new floods, from data the government already collects.
- **Harm from errors:** a missed flood can give false reassurance; frequent false alarms teach people to ignore
  warnings. This app supports, and does not replace, official warnings from the Irrigation Department and the
  Disaster Management Centre.
- **Data ethics:** public sensor data about rivers, with no personal information.
""")

st.markdown("### Run it yourself")
st.code("pip install -r requirements.txt\njupyter notebook notebooks/flood_warning.ipynb   # trains and saves the model\n"
        "streamlit run app.py", language="bash")
