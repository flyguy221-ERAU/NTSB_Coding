import streamlit as st
import pandas as pd
import uuid
import numpy as np
import gspread


from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List
from google.oauth2 import service_account
from google.oauth2.service_account import Credentials


# Scopes: Sheets + Drive (Drive is needed if you open by title)
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

def get_gsheets_client():
    """Create an authenticated gspread client using Streamlit secrets."""
    try:
        sa_info = dict(st.secrets["gcp_service_account"])
    except Exception as e:
        st.warning("Google service account config not found in secrets.")
        raise

    creds = service_account.Credentials.from_service_account_info(
        sa_info,
        scopes=SCOPES,
    )
    return gspread.authorize(creds)


def get_worksheet():
    """Open the target sheet + worksheet from secrets."""
    gc = get_gsheets_client()

    sheet_name = st.secrets["sheets"]["sheet_name"]
    worksheet_name = st.secrets["sheets"]["worksheet_name"]

    # Option A: open by sheet title (requires Drive scope)
    sh = gc.open(sheet_name)

    # If the worksheet doesn't exist, create it
    try:
        ws = sh.worksheet(worksheet_name)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=worksheet_name, rows=1000, cols=20)

    return ws

def normalize_for_sheets_value(value: Any) -> Any:
    """Convert values to types that can be JSON-encoded for Google Sheets."""
    # Pandas / datetime objects → ISO string
    if isinstance(value, (pd.Timestamp, datetime.datetime, datetime.date)):
        return value.isoformat()

    # NumPy scalars → native Python types
    if isinstance(value, (np.generic,)):
        return value.item()

    # Anything else: just pass through (str, int, float, None, etc.)
    return value


def normalize_row_for_sheets(row_dict: Dict[str, Any], header: List[str]) -> List[Any]:
    """Return a row list in header order with serializable values."""
    return [normalize_for_sheets_value(row_dict.get(col, "")) for col in header]


def save_response_to_sheets(row_dict: dict):
    ws = get_worksheet()

    # Fetch header row (row 1)
    header = ws.row_values(1)

    # If header missing or incomplete, write a correct one
    if not header or header != SHEET_COLUMNS:
        ws.update("A1", [SHEET_COLUMNS])
        header = SHEET_COLUMNS

    # Build safe row
    row = []
    for col in header:
        val = row_dict.get(col, "")
        if isinstance(val, (pd.Timestamp, datetime)):
            val = val.isoformat()
        if isinstance(val, np.generic):
            val = val.item()
        row.append(val)

    ws.append_row(row)

# ---- CONFIG ----
DATA_PATH = Path("/Users/jeremyfeagan/Library/Mobile Documents/com~apple~CloudDocs/GitHub/NTSB_Project/data/processed/narratives_for_coding.parquet")
SAVE_PATH = Path("data/coding_responses.csv")

SHEET_COLUMNS = [
    "response_id",
    "event_id",
    "ev_date",
    "fcm_code",
    "loc_code",
    "notes",
    "saved_at_utc",
]

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
    "event_id": str(record["ev_id"]),
    "ev_date": str(record["ev_date"]),   # convert Timestamp → string
    "fcm_code": fcm,
    "loc_code": loc,
    "notes": notes,
    "saved_at_utc": datetime.now(timezone.utc).isoformat(),
}

            save_response_to_sheets(response)

            # Append to CSV
            SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([response]).to_csv(
                SAVE_PATH, mode="a", index=False, header=not SAVE_PATH.exists()
            )

        st.session_state.index += 1

        # Support both older and newer Streamlit versions
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()
            