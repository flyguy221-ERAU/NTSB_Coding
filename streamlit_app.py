import streamlit as st
import pandas as pd
import uuid
import numpy as np
import datetime as dt
import gspread
import streamlit.components.v1 as components  

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
        return {str(r.get("event_id")) for r in records if r.get("event_id")}
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

# Handle input reset request before widgets render
if st.session_state.get("reset_inputs", False):
    st.session_state.fcm_choice = None
    st.session_state.loc_choice = None
    st.session_state.notes_text = ""
    st.session_state.reset_inputs = False

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

# ---- GLOBAL STYLE TWEAKS ----
st.markdown(
    """
    <style>
    .ntsb-card {
        padding: 1.2rem 1.4rem;
        border-radius: 0.8rem;
        border: 1px solid rgba(200,200,200,0.6);
        background-color: rgb(74, 112, 169);
        margin-bottom: 1.0rem;
    }
    .ntsb-header {
        font-size: 1.3rem;
        font-weight: 600;
        font-color: rgba(143, 171, 212);
        margin-bottom: 0.3rem;
    }
    .ntsb-subheader {
        font-size: 0.95rem;
        font-color: rgba(74, 112, 169);
        margin-bottom: 0.6rem;
    }
    .ntsb-narrative {
        font-family: "SF Mono", Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        white-space: pre-wrap;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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

# Initialize index from completed_count on first load (resume)
if "index" not in st.session_state or st.session_state.index < completed_count:
    st.session_state.index = completed_count

# ---- SESSION STATE INIT ----
if "index" not in st.session_state:
    st.session_state.index = 0

if "reset_inputs" not in st.session_state:
    st.session_state.reset_inputs = False

# If we requested an input reset on the previous save, clear widget state now
if st.session_state.reset_inputs:
    st.session_state.fcm_choice = None
    st.session_state.loc_choice = None
    st.session_state.notes_text = ""
    st.session_state.reset_inputs = False


# ---- CURRENT RECORD ----
i = st.session_state.index
if i >= len(df):
    st.success("🎉 You have completed all assigned narratives!")
else:
    record = df.iloc[i]

    st.markdown(
            f"""
            <div class="ntsb-card">
            <div class="ntsb-header">Event {i+1} of {total_n} — EV_ID: {record["ev_id"]}</div>
            <div class="ntsb-subheader">Date: {record["ev_date"]}</div>
            <div class="ntsb-narrative">
            """,
            unsafe_allow_html=True,
        )
    st.markdown(record["narrative_full"])
    st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown("### Narrative")
    st.text_area("Narrative text", record["narrative_full"], height=350, disabled=True)

    # ---- CODING QUESTIONS ----
    st.markdown("### Coding Questions")

    fcm = st.radio(
        "Does this narrative indicate a monitoring or cue-usage failure (FCM)?",
        ["Yes", "No", "Cannot Determine"],
        index=None,
        key="fcm_choice",
    )

    loc = st.radio(
        "Does this narrative indicate a loss of control (LOC)?",
        ["Yes", "No", "Cannot Determine"],
        index=None,
        key="loc_choice",
    )

    notes = st.text_area("Optional Notes", key="notes_text")

    st.caption("Keyboard shortcuts: 1 = Yes, 2 = No, 3 = Cannot Determine")

    # ---- Keyboard shortcut wiring (1/2/3 → radio choices) ----
    KEYBOARD_JS = """
    <script>
    document.addEventListener('keydown', function(e) {
    const key = e.key;
    if (!['1','2','3'].includes(key)) return;

    // Find all radio labels
    const labels = Array.from(
        window.parent.document.querySelectorAll('label[data-baseweb="radio"]')
    );

    function clickFirstLabelWithText(text) {
        const label = labels.find(l => l.innerText.trim().startsWith(text));
        if (label) { label.click(); }
    }

    if (key === '1') clickFirstLabelWithText('Yes');
    if (key === '2') clickFirstLabelWithText('No');
    if (key === '3') clickFirstLabelWithText('Cannot Determine');
    });
    </script>
    """

    components.html(KEYBOARD_JS, height=0, width=0)

    #notes = st.text_area("Optional Notes", key="notes_text")

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

        # Show spinner while saving
        with st.spinner("Saving response..."):
            # Google Sheets save
            save_response_to_sheets(response)

            # Local CSV backup
            #SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
            #pd.DataFrame([response]).to_csv(
            #    SAVE_PATH,
            #    mode="a",
            #    index=False,
            #    header=not SAVE_PATH.exists(),
            #)

        # Toast confirmation (bottom-right)
        st.toast("Saved! Moving to the next narrative.", icon="✅")

        # Remember last saved ID (if you use it in a status message)
        st.session_state.last_saved_ev_id = record["ev_id"]

        # Request reset of widget values on next rerun
        st.session_state.reset_inputs = True

        # Advance to next record
        st.session_state.index += 1

        # Trigger rerun
        st.rerun()