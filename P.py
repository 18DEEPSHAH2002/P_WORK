"""
Task Management Dashboard - Full Streamlit App (Extended / Robust)
Author: Generated for Deep Shah
Purpose: Full-featured dashboard to read tasks from a Google Sheet (CSV export),
         normalize and clean data, and provide summary dashboards and
         detailed, filterable views.
         
Version 2: Incorporates 'Deadline' for Overdue/Performance tracking
         and adds a new 'Dashboard Summary' page as the default.

Notes:
- The script uses the gviz CSV endpoint:
  https://docs.google.com/spreadsheets/d/{KEY}/gviz/tq?tqx=out:csv&gid={GID}
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import StringIO
import requests
import re
import datetime
import base64
from typing import Tuple, List, Dict
import openpyxl

# ------------------------------
# Page config & CSS
# ------------------------------
st.set_page_config(
    page_title="Task Management Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
/* App background */
.stApp {
    background-color: #dbeafe !important;  /* light blue */
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    color: #000000 !important;
}

/* Force all text to black */
body, p, span, div, label, .stMarkdown, .stText, .stMetric {
    color: #000000 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #bfdbfe !important; /* softer blue */
    border-right: 1px solid #93c5fd;
    padding: 1rem;
}

/* Metric cards */
.metric-card, .stMetric {
    background-color: #eff6ff !important;
    color: #000000 !important;
    border-radius: 8px;
    padding: 1rem;
    border: 1px solid #bfdbfe;
}

/* Highlight 'Overdue' metric in red */
.stMetric[aria-label^="Total Overdue"] > div:nth-child(2) {
    color: #dc2626 !important; /* red-600 */
}

/* --- Dropdown / Select Box STYLING --- */

/* 1. The Label ABOVE the select box */
label[data-baseweb="select"] {
    color: #0c4a6e !important;
    font-weight: 600 !important;
}

/* 2. The main, visible select box */
div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #1e293b !important;
    border: 2px solid #3b82f6 !important;
    border-radius: 8px !important;
}

/* 3. The dropdown menu that appears on click */
ul[role="listbox"] {
    background-color: #f0f8ff !important;
    border: 1px solid #93c5fd !important;
    border-radius: 8px !important;
}

/* 4. Individual options in the dropdown list */
ul[role="listbox"] li {
    color: #1e293b !important;
    padding: 8px 12px !important;
}

/* 5. How an option looks when you hover or select it */
ul[role="listbox"] li:hover,
ul[role="listbox"] li[aria-selected="true"] {
    background-color: #dbeafe !important;
    color: #0c4a6e !important;
}

/* Table header */
.stDataFrame th {
    background-color: #3b82f6;
    color: white;
}

/* Highlight urgent rows */
.urgent-highlight {
    background-color: #fee2e2; /* light red */
    font-weight: bold;
}
/* Highlight overdue rows */
.overdue-highlight {
    background-color: #fecaca; /* stronger red */
    font-weight: bold;
    color: #b91c1c;
}

</style>

""",
    unsafe_allow_html=True,
)

# ------------------------------
# Constants and defaults
# ------------------------------
DEFAULT_SHEET_GVIZ_CSV = (
    "https://docs.google.com/spreadsheets/d/14-idXJHzHKCUQxxaqGZi-6S0G20gvPUhK4G16ci2FwI"
    "/gviz/tq?tqx=out:csv&gid=213021534"
)

# A richer priority canonicalization dictionary covering many variants
PRIORITY_CANONICAL = {
    "most urgent": "Most Urgent",
    "mosturgent": "Most Urgent",
    "most_urgent": "Most Urgent",
    "urgent": "Most Urgent",
    "highest": "Most Urgent",
    "high": "High",
    "medium": "Medium",
    "med": "Medium",
    "low": "Low",  # optionally present in some sheets
    "not urgent": "Low",
}

# ------------------------------
# Utility functions
# ------------------------------
def safe_request_csv(url: str, timeout: int = 12) -> pd.DataFrame:
    """
    Request CSV text from a URL and convert to DataFrame.
    Returns an empty DataFrame on failure.
    """
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        text = resp.text
        return pd.read_csv(StringIO(text))
    except Exception as e:
        st.warning(f"Failed to fetch CSV from URL: {e}")
        return pd.DataFrame()

def normalize_string(val: object) -> str:
    """
    Convert a value to a normalized, trimmed, lower-case string with collapsing spaces.
    """
    if pd.isna(val):
        return ""
    s = str(val)
    s = s.replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()

def canonical_priority(val: str) -> str:
    """
    Map a normalized priority string to a canonical display value.
    If unknown, return 'Medium' as a neutral default (can be adjusted).
    """
    if val is None:
        return "Medium"
    val_norm = normalize_string(val)
    if val_norm == "":
        return "Medium"
    # Direct dictionary match
    if val_norm in PRIORITY_CANONICAL:
        return PRIORITY_CANONICAL[val_norm]
    # Try fuzzy-like checks
    if "urgent" in val_norm:
        return "Most Urgent"
    if "high" in val_norm:
        return "High"
    if "medium" in val_norm or "med" in val_norm:
        return "Medium"
    if "low" in val_norm:
        return "Low"
    # fallback
    return "Medium"

def valid_sr_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows where 'Sr' is present and not equal to header strings.
    Handles numeric and textual Sr values.
    """
    if "Sr" not in df.columns:
        return df
    mask = df["Sr"].notna() & (df["Sr"].astype(str).str.strip() != "") & (df["Sr"].astype(str).str.strip().str.lower() != "sr")
    return df[mask].copy()

def create_clickable_file_link(file_value: str, sr_number: object) -> str:
    """
    Generate a clickable link for the File column.
    """
    if pd.isna(file_value) or str(file_value).strip() == "" or str(file_value).strip().lower() == "file":
        return "No file"
    
    file_str = str(file_value).strip()

    # Case 1: Google Drive link (extract FILE_ID)
    if "drive.google.com" in file_str:
        match = re.search(r"[-\w]{25,}", file_str)  # Drive file IDs are 25+ chars
        if match:
            file_id = match.group(0)
            return f'<a class="file-link" href="https://drive.google.com/file/d/{file_id}/view" target="_blank">📎 Open File</a>'
        else:
            return f'<a class="file-link" href="{file_str}" target="_blank">📎 Open File</a>'

    # Case 2: Any other http/https URL
    if file_str.startswith("http://") or file_str.startswith("https://"):
        return f'<a class="file-link" href="{file_str}" target="_blank">📎 Open Link</a>'

    # Case 3: Plain filename → no link
    return "No file"

def df_to_csv_download_link(df: pd.DataFrame, filename: str = "export.csv") -> str:
    """
    Convert DataFrame to CSV and create a Streamlit downloadable link (HTML).
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download CSV</a>'
    return href

def summarize_priority_counts(df: pd.DataFrame) -> pd.Series:
    """
    Returns counts series for priority column in canonical order.
    """
    counts = df["Priority"].value_counts()
    # Ensure consistent index order: Most Urgent, High, Medium, Low
    ordered = ["Most Urgent", "High", "Medium", "Low"]
    return pd.Series({k: int(counts.get(k, 0)) for k in ordered})

def parse_date_flexible(x):
    """
    Attempt to parse dates from a variety of formats. Returns ISO date string
    or original input if parsing fails.
    """
    if pd.isna(x):
        return ""
    s = str(x).strip()
    # Common formats: DD/MM/YYYY, D/M/YYYY, YYYY-MM-DD, DD-MM-YYYY
    patterns = [
        ("%d/%m/%Y", r"^\d{1,2}/\d{1,2}/\d{4}$"),
        ("%d-%m-%Y", r"^\d{1,2}-\d{1,2}/\d{4}$"),
        ("%Y-%m-%d", r"^\d{4}-\d{1,2}-\d{1,2}$"),
        ("%d %b %Y", r"^\d{1,2} [A-Za-z]{3} \d{4}$"),
    ]
    for fmt, pat in patterns:
        if re.match(pat, s):
            try:
                dt = datetime.datetime.strptime(s, fmt)
                return dt.date().isoformat()
            except Exception:
                pass
    # Try pandas parser fallback
    try:
        dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
        if not pd.isna(dt):
            return dt.date().isoformat()
    except Exception:
        pass
    return s  # return original if unable to parse

# ------------------------------
# Sample data generator (large)
# ------------------------------
def create_sample_data_large() -> pd.DataFrame:
    """
    Create a larger sample dataset for demonstration when CSV fetch fails.
    """
    n = 120
    officers = ["CMFO", "DRO", "ADC (RD)", "ADC G", "Legal Cell", "AC G", "DyESA", "Election Tehsildar", "ADC (W)", "EO"]
    priorities = ["Most Urgent", "High", "Medium", "Low"]
    statuses = ["In progress", "Completed", "In progress", "In progress"]
    rows = []
    today = datetime.date.today()
    for i in range(1, n + 1):
        entry_date = (today - datetime.timedelta(days=np.random.randint(1, 60)))
        deadline_date = entry_date + datetime.timedelta(days=np.random.choice([7, 14, 30]))
        # Make some overdue
        if i % 5 == 0:
            deadline_date = entry_date - datetime.timedelta(days=np.random.randint(1, 10))
            
        status = statuses[i % len(statuses)]
        response_date = np.nan
        if status == "Completed":
            response_date = (entry_date + datetime.timedelta(days=np.random.randint(1, 20))).strftime("%d/%m/%Y")
            
        rows.append({
            "Sr": i,
            "Marked to Officer": officers[i % len(officers)],
            "Priority": priorities[i % len(priorities)],
            "Status": status,
            "Subject": f"Task {i} - Administrative item regarding process {i%7}",
            "File": f"document_{i:03d}.pdf" if i % 3 != 0 else f"https://example.com/doc_{i}.pdf",
            "Entry Date": entry_date.strftime("%d/%m/%Y"),
            "Deadline": deadline_date.strftime("%d/%m/%Y"),
            "Response Recieved on": response_date,
            "Remarks": "Auto-generated sample data" if i % 5 else "Requires signature"
        })
    return pd.DataFrame(rows)

# ------------------------------
# Data loading & processing
# ------------------------------
@st.cache_data(ttl=300)
def load_and_process(sheet_url: str) -> pd.DataFrame:
    """
    Load CSV from Google Sheets URL and perform robust cleaning:
    - Ensure required columns exist
    - Normalize 'Priority'
    - Clean officer names and status
    - Filter invalid 'Sr' if present
    - Parse dates and calculate task status (Overdue, etc.)
    """
    raw = safe_request_csv(sheet_url)
    # If fetch failed, provide a richer sample fallback for app demo
    if raw.empty:
        raw = create_sample_data_large()

    # Trim column names
    raw.columns = [str(c).strip() for c in raw.columns]

    # Try to detect common header rows
    if "Sr" in raw.columns:
        first_row_vals = raw.iloc[0].astype(str).str.strip().str.lower().tolist()
        if "sr" in first_row_vals:
            # drop the first row (likely header repeated)
            raw = raw.iloc[1:].reset_index(drop=True)

    # Keep only rows with valid Sr if Sr exists
    raw = valid_sr_filter(raw)

    # Ensure required columns exist
    expected_cols = ["Marked to Officer", "Priority", "Status", "File", "Subject", "Entry Date", "Remarks", "Sr", "Deadline", "Response Recieved on"]
    for col in expected_cols:
        if col not in raw.columns:
            raw[col] = np.nan
            st.sidebar.warning(f"Missing expected column: '{col}'. Column added with NAs.")

    # Clean officer names
    raw["Marked to Officer"] = raw["Marked to Officer"].fillna("Unknown").astype(str).str.strip()

    # Normalize priority robustly
    raw["Priority"] = raw["Priority"].apply(lambda v: canonical_priority(v))

    # Clean status strings
    raw["Status"] = raw["Status"].fillna("In progress").astype(str).str.strip()

    # Parse key dates
    raw["Entry Date (Parsed)"] = pd.to_datetime(raw["Entry Date"].apply(parse_date_flexible), errors="coerce")
    raw["Deadline (Parsed)"] = pd.to_datetime(raw["Deadline"].apply(parse_date_flexible), errors="coerce")
    raw["Response Date (Parsed)"] = pd.to_datetime(raw["Response Recieved on"].apply(parse_date_flexible), errors="coerce")

    # Create File Link column (HTML)
    raw["File Link"] = raw.apply(lambda r: create_clickable_file_link(r["File"], r.get("Sr", "")), axis=1)

    # Ensure Sr is numeric where possible, but keep original text
    raw["Sr_original"] = raw["Sr"].astype(str)
    
    # --- NEW: Calculate Task Status based on Deadline ---
    today = pd.Timestamp.today().normalize()
    is_pending = raw["Status"].str.lower() == "in progress"
    is_completed = raw["Status"].str.lower() == "completed"
    is_overdue = (raw["Deadline (Parsed)"] < today) & is_pending & raw["Deadline (Parsed)"].notna()
    is_due_soon = (raw["Deadline (Parsed)"] >= today) & (raw["Deadline (Parsed)"] <= today + pd.Timedelta(days=3)) & is_pending
    
    conditions = [
        is_completed,
        is_overdue,
        is_due_soon,
        is_pending
    ]
    choices = [
        "Completed",
        "Overdue",
        "Due Soon",
        "Pending"
    ]
    raw["Task_Status"] = np.select(conditions, choices, default="Pending")
    # ----------------------------------------------------

    # Reorder columns
    cols_order = ["Sr_original", "Marked to Officer", "Priority", "Status", "Task_Status", "Subject", "Entry Date", "Deadline", "Response Recieved on", "File Link", "Remarks"]
    # add any other columns that were present
    for c in raw.columns:
        # We MUST keep the parsed date columns for calculations, so remove them from the exclusion list.
        if c not in cols_order and c not in ["File", "Sr"]: # <-- REMOVED PARSED DATES FROM EXCLUSION LIST
            cols_order.append(c)
    raw = raw[[c for c in cols_order if c in raw.columns]]

    return raw

# ------------------------------
# UI helper components
# ------------------------------
def sidebar_controls():
    """
    Build controls in the sidebar and return settings.
    """
    st.sidebar.title("Controls & Settings")
    sheet_url = st.sidebar.text_input("Google Sheet CSV URL (gviz CSV recommended)", value=DEFAULT_SHEET_GVIZ_CSV)
    show_debug = st.sidebar.checkbox("Show debug info (raw head)", value=False)
    highlight_urgent = st.sidebar.checkbox("Highlight Urgent/Overdue tasks", value=True)
    
    refresh_now = st.sidebar.button(" Refresh Data Now")
    if refresh_now:
        st.cache_data.clear()
        st.experimental_rerun()
        
    return {
        "sheet_url": sheet_url.strip(),
        "show_debug": show_debug,
        "highlight_urgent": highlight_urgent,
    }

def render_global_metrics(df: pd.DataFrame):
    st.markdown('<h1 class="main-header"> Task Management Dashboard</h1>', unsafe_allow_html=True)
    # Quick top-row metrics
    total_tasks = len(df)
    pending_df = df[df["Task_Status"].isin(["Pending", "Due Soon", "Overdue"])]
    total_pending = len(pending_df)
    total_overdue = len(df[df["Task_Status"] == "Overdue"])
    unique_officers = df["Marked to Officer"].nunique()
    most_urgent_total = len(df[(df["Priority"] == "Most Urgent") & (df["Task_Status"] != "Completed")])

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Tasks (All)", total_tasks)
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Pending Tasks", total_pending)
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        # This metric is specifically styled with CSS to be red
        st.metric("Total Overdue", total_overdue)
        st.markdown("</div>", unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Pending 'Most Urgent'", most_urgent_total)
        st.markdown("</div>", unsafe_allow_html=True)
    with c5:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Officers", unique_officers)
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")

def dashboard_summary_page(df: pd.DataFrame, settings: dict):
    """
    NEW: Main dashboard page showing performance overview,
    based on the user's sketch.
    """
    st.header("Dashboard Summary")

    # Data prep
    pending_df = df[df["Task_Status"].isin(["Pending", "Due Soon", "Overdue"])].copy()
    completed_df = df[df["Task_Status"] == "Completed"].copy()
    
    # --- Performance Stats Calculation ---
    
    # 1. Pending/Overdue counts
    officer_stats = pending_df.groupby("Marked to Officer")["Task_Status"].value_counts().unstack(fill_value=0)
    if "Overdue" not in officer_stats.columns: officer_stats["Overdue"] = 0
    if "Due Soon" not in officer_stats.columns: officer_stats["Due Soon"] = 0
    if "Pending" not in officer_stats.columns: officer_stats["Pending"] = 0
    officer_stats = officer_stats.reindex(columns=["Overdue", "Due Soon", "Pending"], fill_value=0)
    officer_stats["Total Pending"] = officer_stats.sum(axis=1)
    # --> RESET INDEX HERE
    officer_stats = officer_stats.reset_index() # Now 'Marked to Officer' is a column

    # 2. Completed in last 7 days
    today = pd.Timestamp.today().normalize()
    last_week = today - pd.Timedelta(days=7)
    recent_completed = completed_df[
        (completed_df["Response Date (Parsed)"].notna()) &
        (completed_df["Response Date (Parsed)"] >= last_week) &
        (completed_df["Response Date (Parsed)"] <= today)
    ]
    completed_counts_7d = recent_completed.groupby("Marked to Officer").size().reset_index(name="Completed (Last 7 Days)")
    
    # 3. Merge stats
    officer_summary = officer_stats.merge(
        completed_counts_7d, 
        on="Marked to Officer", 
        how="outer" # Outer merge on the 'Marked to Officer' column
    )
    
    # Fill NaNs created by the outer merge
    fill_cols = ["Overdue", "Due Soon", "Pending", "Total Pending", "Completed (Last 7 Days)"]
    for col in fill_cols:
        if col not in officer_summary.columns:
            officer_summary[col] = 0 # Add column if it doesn't exist at all (e.g., no pending tasks ever)
        officer_summary[col] = officer_summary[col].fillna(0).astype(int)

    # 4. Filter for the bar chart.
    # We only want to chart officers with pending tasks.
    officer_pending_counts = officer_summary[officer_summary["Total Pending"] > 0].copy()
    officer_pending_counts = officer_pending_counts.sort_values("Total Pending", ascending=True)
    
    # --- Layout (as per sketch) ---
    col1, col2 = st.columns([2, 1])

    with col1:
        # --- BAR GRAPH OFFICER LIST WITH THE NO. OF TASK PENDING ---
        st.subheader("Officer-wise Pending Tasks")
        if not officer_pending_counts.empty:
            fig = px.bar(
                officer_pending_counts,
                x="Total Pending",
                y="Marked to Officer",
                orientation="h",
                title="Total Pending Tasks (Overdue + Due Soon + Pending)",
                labels={"Total Pending": "Number of Tasks", "Marked to Officer": "Officer"},
                color="Total Pending",
                color_continuous_scale="Blues",
                height=450,
                text_auto=True
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No pending tasks to show.")

        st.markdown("---")
        
        # --- TABLE WITH THE OFFICER COMPLETED IN LAST 7 DAYS ---
        st.subheader("Officer Performance (Last 7 Days)")
        st.dataframe(
            officer_summary.sort_values("Completed (Last 7 Days)", ascending=False),
            use_container_width=True,
            hide_index=True
        )

    with col2:
        # --- TOP 5 BEST PERFORMANCE ---
        st.subheader("Top 5 Best Performance")
        st.markdown("<small>(Based on: Fewest Overdue, then Fewest Total Pending)</small>", unsafe_allow_html=True)
        best_5 = officer_summary.sort_values(
            by=["Overdue", "Total Pending"], 
            ascending=[True, True]
        ).head(5)
        st.dataframe(best_5[["Marked to Officer", "Overdue", "Total Pending", "Completed (Last 7 Days)"]], use_container_width=True, hide_index=True)
        
        # --- TOP 5 WORST PERFORMANCE ---
        st.subheader("Top 5 Worst Performance")
        st.markdown("<small>(Based on: Most Overdue, then Most Total Pending)</small>", unsafe_allow_html=True)
        worst_5 = officer_summary.sort_values(
            by=["Overdue", "Total Pending"], 
            ascending=[False, False]
        ).head(5)
        st.dataframe(worst_5[["Marked to Officer", "Overdue", "Total Pending", "Completed (Last 7 Days)"]], use_container_width=True, hide_index=True)

        st.markdown("---")
        
        # --- TABLE WITH THE OFFICER COMPLETED IN LAST 7 DAYS ---
        st.subheader("Officer Performance (Last 7 Days)")
        st.dataframe(
            officer_summary.sort_values("Completed (Last 7 Days)", ascending=False),
            use_container_width=True
        )

    with col2:
        # --- TOP 5 BEST PERFORMANCE ---
        st.subheader("Top 5 Best Performance")
        st.markdown("<small>(Based on: Fewest Overdue, then Fewest Total Pending)</small>", unsafe_allow_html=True)
        best_5 = officer_summary.sort_values(
            by=["Overdue", "Total Pending"], 
            ascending=[True, True]
        ).head(5)
        st.dataframe(best_5[["Overdue", "Total Pending", "Completed (Last 7 Days)"]], use_container_width=True)
        
        # --- TOP 5 WORST PERFORMANCE ---
        st.subheader("Top 5 Worst Performance")
        st.markdown("<small>(Based on: Most Overdue, then Most Total Pending)</small>", unsafe_allow_html=True)
        worst_5 = officer_summary.sort_values(
            by=["Overdue", "Total Pending"], 
            ascending=[False, False]
        ).head(5)
        st.dataframe(worst_5[["Overdue", "Total Pending", "Completed (Last 7 Days)"]], use_container_width=True)

        st.markdown("---")
        
        # --- TOTALS ---
        st.subheader("Overall Status")
        total_pending = officer_summary["Total Pending"].sum()
        total_overdue = officer_summary["Overdue"].sum()
        total_tasks = len(df)
        percent_pending = (total_pending / total_tasks * 100) if total_tasks > 0 else 0
        
        st.metric("Total Tasks Pending", f"{total_pending} / {total_tasks}")
        st.metric("% of Tasks Pending", f"{percent_pending:.1f}%")
        st.metric("Total Overdue", total_overdue)

def all_tasks_page(df: pd.DataFrame, settings: dict):
    st.header(" All Tasks (Filtered View)")
    st.markdown("Use the filters below to inspect all rows. You can sort columns by clicking headers.")

    # Provide filters: officer, priority, status, date range
    col1, col2, col3, col4 = st.columns(4)
    officers = sorted(df["Marked to Officer"].fillna("Unknown").unique().tolist())
    with col1:
        officer_filter = st.multiselect("Filter by Officer", options=["All Officers"] + officers, default="All Officers")
    with col2:
        priority_options = sorted(df["Priority"].unique().tolist())
        priority_filter = st.multiselect("Filter by Priority", options=["All"] + priority_options, default="All")
    with col3:
        # Use new Task_Status
        status_options = sorted(df["Task_Status"].unique().tolist())
        status_filter = st.multiselect("Filter by Task Status", options=["All"] + status_options, default="All")
    with col4:
        q = st.text_input("Search subject / remarks:", value="")

    filtered = df.copy()
    if "All Officers" not in officer_filter:
        filtered = filtered[filtered["Marked to Officer"].isin(officer_filter)]
    if "All" not in priority_filter:
        filtered = filtered[filtered["Priority"].isin(priority_filter)]
    if "All" not in status_filter:
        filtered = filtered[filtered["Task_Status"].isin(status_filter)]
    if q.strip():
        qlow = q.strip().lower()
        mask_subject = filtered["Subject"].astype(str).str.lower().str.contains(qlow, na=False)
        mask_remarks = filtered["Remarks"].astype(str).str.lower().str.contains(qlow, na=False)
        filtered = filtered[mask_subject | mask_remarks]

    st.markdown(f"**Showing {len(filtered)} rows**")
    
    # Define columns for display
    display_cols = ["Sr_original", "Marked to Officer", "Task_Status", "Priority", "Subject", "Entry Date", "Deadline", "File Link", "Remarks"]
    available_cols = [c for c in display_cols if c in filtered.columns]
    
    # Use st.dataframe for sortable columns
    st.dataframe(filtered[available_cols], use_container_width=True, hide_index=True)

    st.markdown(df_to_csv_download_link(filtered, filename="all_tasks_filtered.csv"), unsafe_allow_html=True)


def officers_overview_page_deepdive(df: pd.DataFrame, settings: dict):
    st.header(" Officer-wise Deep Dive")
    st.markdown("Select an officer to see their full task list. Includes pending and completed.")
    
    officers_list = ["All Officers"] + sorted(df["Marked to Officer"].unique().tolist())
    selected = st.selectbox("Select Officer", options=officers_list, index=0)

    # Search box for free text
    q = st.text_input("Search subject / remarks (simple substring search):", value="")

    filtered = df.copy()
    if selected != "All Officers":
        filtered = filtered[filtered["Marked to Officer"] == selected]
    if q.strip():
        qlow = q.strip().lower()
        mask_subject = filtered["Subject"].astype(str).str.lower().str.contains(qlow, na=False)
        mask_remarks = filtered["Remarks"].astype(str).str.lower().str.contains(qlow, na=False)
        filtered = filtered[mask_subject | mask_remarks]

    # Sort by status (Overdue first)
    filtered = filtered.sort_values(by=["Task_Status", "Priority"], ascending=[False, False])

    st.markdown(f"**Showing {len(filtered)} tasks**")
    
    display_cols = ["Sr_original", "Task_Status", "Priority", "Subject", "Entry Date", "Deadline", "File Link", "Remarks"]
    available_cols = [c for c in display_cols if c in filtered.columns]
    
    # Simple HTML table to show highlights
    if settings["highlight_urgent"]:
        def style_row_html(row):
            cls = ""
            if row["Task_Status"] == "Overdue":
                cls = "overdue-highlight"
            elif row["Priority"] == "Most Urgent":
                cls = "urgent-highlight"
            
            cells = f"<tr class='{cls}'>"
            for col in available_cols:
                cells += f"<td>{row[col]}</td>"
            cells += "</tr>"
            return cells

        # Build HTML table manually
        header = "".join(f"<th>{col}</th>" for col in available_cols)
        rows_html = "".join(filtered.apply(style_row_html, axis=1))
        
        table_html = f"""
        <style>
            .styled-table {{
                border-collapse: collapse;
                width: 100%;
                font-size: 0.9rem;
            }}
            .styled-table th, .styled-table td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            .styled-table th {{
                background-color: #3b82f6;
                color: white;
            }}
            .styled-table tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
        </style>
        <table class="styled-table">
            <thead><tr>{header}</tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        """
        st.markdown(table_html, unsafe_allow_html=True)
    else:
        st.dataframe(filtered[available_cols], use_container_width=True, hide_index=True)

    # Option to download filtered CSV
    st.markdown(df_to_csv_download_link(filtered, filename=f"{selected}_tasks.csv"), unsafe_allow_html=True)


def priority_analysis_page(df: pd.DataFrame, settings: dict):
    st.header(" Priority-wise Task Analysis")
    pending_df = df[df["Task_Status"].isin(["Pending", "Due Soon", "Overdue"])].copy()

    # Top metrics
    total_pending = len(pending_df)
    counts = summarize_priority_counts(pending_df)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Pending Tasks", total_pending)
    with c2:
        st.metric("Most Urgent", counts.get("Most Urgent", 0), delta=f"{(counts.get('Most Urgent', 0)/total_pending*100):.1f}%" if total_pending else "0%")
    with c3:
        st.metric("High", counts.get("High", 0), delta=f"{(counts.get('High', 0)/total_pending*100):.1f}%" if total_pending else "0%")
    with c4:
        st.metric("Medium", counts.get("Medium", 0), delta=f"{(counts.get('Medium', 0)/total_pending*100):.1f}%" if total_pending else "0%")

    st.markdown("---")
    # For each priority, show officer-wise distribution
    priority_order = ["Most Urgent", "High", "Medium", "Low"]
    priority_colors = {"Most Urgent": "#ff4b4b", "High": "#ff8c00", "Medium": "#ffd700", "Low": "#94d2bd"}

    for p in priority_order:
        st.subheader(f"{p} Priority Tasks - Officer-wise Distribution")
        p_df = pending_df[pending_df["Priority"] == p]
        if p_df.empty:
            st.info(f"No {p} priority tasks found.")
            st.markdown("---")
            continue
        counts_by_officer = p_df.groupby("Marked to Officer").size().reset_index(name="Task Count").sort_values("Task Count", ascending=True)
        fig = px.bar(
            counts_by_officer,
            x="Task Count",
            y="Marked to Officer",
            orientation="h",
            title=f"{p} Priority Tasks by Officer ({len(p_df)} total)",
            labels={"Task Count": "Number of Tasks", "Marked to Officer": "Officer"},
            color_discrete_sequence=[priority_colors.get(p, "#636EFA")],
            height=360,
            text_auto=True
        )
        st.plotly_chart(fig, use_container_width=True)
        with st.expander(f"View {p} Priority Task Details ({len(p_df)} rows)"):
            display_cols = ["Sr_original", "Marked to Officer", "Task_Status", "Subject", "Entry Date", "Deadline", "File Link", "Remarks"]
            av = [c for c in display_cols if c in p_df.columns]
            st.dataframe(p_df[av], use_container_width=True, hide_index=True)
        st.markdown("---")

# ------------------------------
# Main
# ------------------------------
def main():
    settings = sidebar_controls()
    
    # Load and process data
    sheet_url = settings["sheet_url"] or DEFAULT_SHEET_GVIZ_CSV
    df = load_and_process(sheet_url)

    # Debug info
    if settings["show_debug"]:
        st.sidebar.markdown("### Debug Info")
        st.sidebar.write("Dataframe shape:", df.shape)
        st.sidebar.write("Columns:", df.columns.tolist())
        st.sidebar.write("Task_Status values:", df["Task_Status"].unique().tolist())
        st.sidebar.write("Sample rows:")
        st.sidebar.dataframe(df.head(10))

    # Top header and metrics (now rendered on dashboard page)
    render_global_metrics(df)

    # Navigation
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Select Page", 
        [
            "Dashboard Summary", 
            "All Tasks (Filtered View)", 
            "Officer-wise Deep Dive", 
            "Priority-wise Analysis", 
            "About / Help"
        ]
    )
    
    if page == "Dashboard Summary":
        dashboard_summary_page(df, settings)
    elif page == "All Tasks (Filtered View)":
        all_tasks_page(df, settings)
    elif page == "Officer-wise Deep Dive":
        officers_overview_page_deepdive(df, settings)
    elif page == "Priority-wise Analysis":
        priority_analysis_page(df, settings)
    else:
        st.header(" About ")
        st.markdown(
            """
            THIS IS WORKING DASHBOARD FOR LUDHIANA ADMINISTRATION ONLY 

             THIS DASHBOARD IS BUILD BY DEEP SHAH  
             THE OWNERSHIP IS UNDER DC LUDHIANA OFFICE 
            
            """
        )
        st.markdown("### Contact / Notes")
        st.markdown("If any changes happen in the excel and get any bug or loophole, contact: +918905309441; gmail:18deep.shah2002@gmail.com")


if __name__ == "__main__":
    main()


