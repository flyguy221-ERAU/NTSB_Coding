import streamlit as st
import pandas as pd
import uuid
import numpy as np
import datetime as dt
import gspread

from pathlib import Path
from typing import Any, Dict, List, Set
from google.oauth2 import service_account

# ------------- CONFIG -------------

# Who is this app for? (used in sheet + optional filtering)
RATER_ID = "Ashley"

# Scopes: Sheets + Drive (Drive is needed if you open by title)
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# For local dev you might use an absolute path.
# For Streamlit Cloud, make sure this is a *relative* path
# and that the file is in your repo.
DATA_PATH = Path("data/processed/Ashley_Coding.parquet")

# Also keep a local CSV log if you want
SAVE_PATH = Path("data/coding_responses.csv")

# ------------- GOOGLE SHEETS HELPERS -------------

def get_gsheets_client():
    """Create an authenticated gspread client using Streamlit secrets."""
    try:
        sa_info = dict(st.secrets["gcp_service_account"])
    except Exception:
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

    # Open by Google Sheet title
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
    if isinstance(value, (pd.Timestamp, dt.datetime, dt.date)):
        return value.isoformat()

    # NumPy scalars → native Python types
    if isinstance(value, (np.generic,)):
        return value.item()

    # Everything else (str, int, float, None) is fine
    return value


def normalize_row_for_sheets(row_dict: Dict[str, Any], header: List[str]) -> List[Any]:
    """Return a row list in header order with serializable values."""
    return [normalize_for_sheets_value(row_dict.get(col, "")) for col in header]


def save_response_to_sheets(row_dict: dict):
    """Append one coded row to Google Sheets."""
    ws = get_worksheet()

    # Current header row from the sheet (if any)
    header = ws.row_values(1)

    if not header:
        # First write: create header from keys in row_dict
        header = list(row_dict.keys())
        ws.append_row(header)

    # Build row in header order, with all values JSON-safe
    row_values = normalize_row_for_sheets(row_dict, header)
    ws.append_row(row_values)


def get_completed_ids_from_sheets() -> Set[str]:
    """Return set of ev_id values already coded in this worksheet."""
    try:
        ws = get_worksheet()
        records = ws.get_all_records()
        return {str(r.get("ev_id")) for r in records if r.get("ev_id")}
    except Exception:
        # If Sheets is unavailable, just treat as 0 completed
        return set()

# ------------- DATA LOADING -------------

@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)

    # Keep only what the rater needs
    cols = ["ev_id", "ev_date", "narrative_full"]
    df = df[cols].reset_index(drop=True)

    # OPTIONAL: if your parquet already has an assignment column, filter
    # so Ashley only sees her portion.
    # Example expected column: "assigned_rater"
    if "assigned_rater" in df.columns:
        df = df[df["assigned_rater"] == RATER_ID].reset_index(drop=True)

    # Force ev_id to string for consistency with Sheets
    df["ev_id"] = df["ev_id"].astype(str)

    return df


def compute_start_index(df: pd.DataFrame, completed_ids: Set[str]) -> int:
    """Find first index in df whose ev_id has NOT yet been coded."""
    ev_ids = df["ev_id"].tolist()
    for idx, ev in enumerate(ev_ids):
        if ev not in completed_ids:
            return idx
    # All done
    return len(df)


# ------------- MAIN APP -------------

df = load_data()
total_n = len(df)

st.title("Aviation Narrative Coding")
st.write(
    "Please read each narrative and answer the questions below. "
    "You can close this page and come back later; your completed work is saved."
)

# Get which IDs are already coded in this worksheet
completed_ids = get_completed_ids_from_sheets()
completed_count = sum(ev in completed_ids for ev in df["ev_id"].tolist())

# Progress bar
if total_n > 0:
    progress_value = completed_count / total_n
else:
    progress_value = 0.0

st.progress(progress_value, text=f"{completed_count} of {total_n} narratives coded")

# Session state index (resume from last completed item)
if "index" not in st.session_state:
    st.session_state.index = compute_start_index(df, completed_ids)

i = st.session_state.index

if total_n == 0:
    st.info("No narratives are assigned to you.")
elif i >= total_n:
    st.success("🎉 You have completed all assigned narratives!")
else:
    record = df.iloc[i]

    st.subheader(f"Event {i+1} of {total_n} — EV_ID: {record['ev_id']}")
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
            # Build response row
            response = {
                "response_id": str(uuid.uuid4()),
                "rater_id": RATER_ID,
                "event_id": record["ev_id"],
                "ev_date": record["ev_date"],
                "fcm_code": fcm,
                "loc_code": loc,
                "notes": notes,
                "saved_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            }

            # 1) Save to Google Sheets
            try:
                save_response_to_sheets(response)
            except Exception as e:
                st.error(f"Could not sync to Google Sheets: {e}")

            # 2) Also append to local CSV (optional backup)
            SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([response]).to_csv(
                SAVE_PATH, mode="a", index=False, header=not SAVE_PATH.exists()
            )

            # 3) Move to next record and rerun
            st.session_state.index += 1
            st.rerun()