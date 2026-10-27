"""River Flood Early Warning for Sri Lanka — Streamlit front end.

Run from the project folder:   streamlit run app.py
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="River Flood Early Warning · Sri Lanka", page_icon="🌊",
                   layout="wide", initial_sidebar_state="expanded")

from ui.common import fmt_t, page_css  # noqa: E402
from src.live import ARCHIVE_REPO  # noqa: E402

page_css()

# Moments from the archive that are worth replaying in a demo (all are real readings).
REPLAYS = {
    "Gin Ganga, 6 h before Baddegama flooded (21 Sep 2026, 09:00)": pd.Timestamp("2026-09-21 09:00"),
    "Nilwala Ganga, before Panadugama flooded (13 Oct 2026, 00:00)": pd.Timestamp("2026-10-13 00:00"),
    "Kelani & Mahaweli floods (3 Aug 2026, 03:00)": pd.Timestamp("2026-08-03 03:00"),
    "Cyclone Ditwah (27 Nov 2025, 18:00)": pd.Timestamp("2025-11-27 18:00"),
    "Choose a date and time…": None,
}

pages = {
    "Monitor": [
        st.Page("views/overview.py", title="River situation", icon=":material/water:", default=True),
        st.Page("views/station.py", title="Station detail", icon=":material/location_on:"),
        st.Page("views/whatif.py", title="What-if check", icon=":material/tune:"),
    ],
}
nav = st.navigation(pages)

# Keep the sidebar choices when moving between pages.
for _k in ("mode", "replay_choice", "replay_date", "replay_time"):
    if _k in st.session_state:
        st.session_state[_k] = st.session_state[_k]

st.logo(str(ROOT / "assets" / "logo.svg"), size="large", icon_image=str(ROOT / "assets" / "icon.svg"))

with st.sidebar:
    mode = st.radio("River levels to use", ["Live", "Replay a past moment"], key="mode",
                    help="Live downloads the latest readings (updated about hourly). Replay re-runs the "
                         "model on archived readings, so you can see how it behaved before real floods.")
    st.session_state["source"] = "live" if mode == "Live" else "archive"
    if mode != "Live":
        choice = st.selectbox("Moment", list(REPLAYS), key="replay_choice")
        at = REPLAYS[choice]
        if at is None:
            st.session_state.setdefault("replay_date", pd.Timestamp("2026-09-21").date())
            st.session_state.setdefault("replay_time", pd.Timestamp("2026-09-21 09:00").time())
            c1, c2 = st.columns(2)
            d = c1.date_input("Date", min_value=pd.Timestamp("2025-11-25").date(),
                              max_value=pd.Timestamp("2026-10-23").date(), key="replay_date")
            h = c2.time_input("Time", step=3600, key="replay_time")
            at = pd.Timestamp.combine(d, h)
        st.session_state["replay_at"] = pd.Timestamp(at).floor("h")
    else:
        st.html('<div class="smalllabel">Source: Irrigation Department gauges, via the public archive '
                f'<a href="{ARCHIVE_REPO}" style="color:#7CC4DD">nuuuwan/lk_irrigation</a>. '
                'Downloaded at most every 15 minutes.</div>')
        if st.button("Check for new readings", icon=":material/refresh:", width="stretch"):
            st.cache_data.clear()
            st.rerun()

nav.run()
