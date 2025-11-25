import streamlit as st
import pandas as pd
import uuid
from pathlib import Path

# ---- CONFIG ----
DATA_PATH = Path("/Users/jeremyfeagan/Library/Mobile Documents/com~apple~CloudDocs/GitHub/NTSB_Project/data/processed/narratives_for_coding.parquet")
SAVE_PATH = Path("data/coding_responses.csv")

# ---- LOAD DATA ----
@st.cache_data
def load_data():
    df = pd.read_parquet(DATA_PATH)
    # Make sure we only send her the fields she needs
    cols = ["ev_id", "ev_date", "narrative_full"]
    return df[cols].reset_index(drop=True)

df = load_data()

st.title("Aviation Narrative Coding")
st.write("Please read each narrative and answer the questions below.")

# ---- SESSION STATE ----
if "index" not in st.session_state:
    st.session_state.index = 0

# ---- CURRENT RECORD ----
i = st.session_state.index
if i >= len(df):
    st.success("🎉 You have completed all assigned narratives!")
else:
    record = df.iloc[i]

    st.subheader(f"Event {i+1} of {len(df)} — EV_ID: {record['ev_id']}")
    st.write(f"**Date:** {record['ev_date']}")

    st.markdown("### Narrative")
    st.text_area("Narrative text", record["narrative_full"], height=350, disabled=True)

    # ---- CODING QUESTIONS ----
    st.markdown("### Coding Questions")

    fcm = st.radio(
        "Does this narrative indicate a monitoring or cue-usage failure (FCM)?",
        ["Yes", "No", "Cannot Determine"],
        index=None,
    )

    loc = st.radio(
        "Does this narrative indicate a loss of control (LOC)?",
        ["Yes", "No", "Cannot Determine"],
        index=None,
    )

    notes = st.text_area("Optional Notes")

    # ---- SAVE BUTTON ----
    if st.button("Save and Next"):
        if fcm is None or loc is None:
            st.error("Please answer both coding questions before continuing.")
        else:
            # Save row
            response = {
                "response_id": str(uuid.uuid4()),
                "ev_id": record["ev_id"],
                "ev_date": record["ev_date"],
                "FCM": fcm,
                "LOC": loc,
                "Notes": notes,
            }

            # Append to CSV
            SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([response]).to_csv(
                SAVE_PATH, mode="a", index=False, header=not SAVE_PATH.exists()
            )

            # Increment index
            st.session_state.index += 1
            st.experimental_rerun()
            