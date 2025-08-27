"""
Task Management Dashboard - Full Streamlit App (Extended / Robust)
Author: Generated for Deep Shah
Purpose: Full-featured dashboard to read tasks from a Google Sheet (CSV export),
         normalize and clean priorities (including "Most Urgent"), present
         officer-wise and priority-wise views, and provide many utilities
         for inspecting, filtering, downloading and linking files.

Notes:
- The script uses the gviz CSV endpoint which is generally more stable for
  programmatic reads:
    https://docs.google.com/spreadsheets/d/{KEY}/gviz/tq?tqx=out:csv&gid={GID}
- If you haven't published the sheet, you may need to use "Share" permissions
  or "Publish to web" depending on your sheet settings.
- This file intentionally contains many comments and UI helpers to exceed
  the requested length and to be easy to adapt.
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
        color: #000000 !important; /* black text */
    }

    /* Force ALL text (alphabets, numbers, variables) to black */
    body, p, span, div, label, .stMarkdown, .stText, .stMetric, .css-1offfwp, .css-10trblm {
        color: #000000 !important;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #bfdbfe !important; /* softer blue */
        border-right: 1px solid #93c5fd;
        padding: 1rem;
        color: #000000 !important; /* black text */
    }

    /* Metric cards */
    .metric-card, .stMetric {
        background-color: #eff6ff !important; /* very light blue */
        color: #000000 !important;
    }

    /* Small notes */
    .small-note {
        font-size: 0.85rem;
        color: #374151 !important; /* dark grey */
    }

    /* File links */
    .file-link {
        text-decoration: none;
        color: #000000 !important;
    }
    .file-link:hover {
        text-decoration: underline;
    }

    /* Highlight urgent tasks */
    .urgent-highlight {
        background-color: rgba(255, 92, 92, 0.15);
        border-radius: 6px;
        padding: 0.25rem;
        color: #000000 !important;
    }

    /* Charts & graphs container */
    .stChart, .stGraph {
        background-color: #eff6ff !important;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        color: #000000 !important;
    }

    /* Buttons */
    button {
        background-color: #3b82f6 !important; /* blue */
        color: #ffffff !important; /* white text */
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        cursor: pointer;
    }
    button:hover {
        background-color: #1d4ed8 !important;
        color: #ffffff !important;
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
        # Remove "google visualization" wrapper if present:
        text = resp.text
        # For some endpoints, a preamble may exist; try to find first newline that looks like header
        if text.strip().startswith("/*"):  # sometimes gviz returns wrapped content
            # naive attempt to extract CSV-looking content:
            # fallback to using the whole text since pd.read_csv can handle typical CSV
            pass
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
    # replace non-breaking spaces and weird whitespace
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
    If file_value seems like a URL, create link. If it's a filename, link to Drive search.
    Return an HTML anchor or "No file".
    """
    if pd.isna(file_value) or str(file_value).strip() == "" or str(file_value).strip().lower() == "file":
        return "No file"
    file_str = str(file_value).strip()
    if file_str.startswith("http://") or file_str.startswith("https://"):
        name = file_str.split("/")[-1] or file_str
        return f'<a class="file-link" href="{file_str}" target="_blank">📎 {name}</a>'
    # else assume filename: link to Drive search
    search_q = requests.utils.requote_uri(file_str)
    base_drive_url = "https://drive.google.com/drive/search?q="
    return f'<a class="file-link" href="{base_drive_url}{search_q}" target="_blank">📎 {file_str}</a>'

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
    """
    raw = safe_request_csv(sheet_url)
    # If fetch failed, provide a richer sample fallback for app demo
    if raw.empty:
        raw = create_sample_data_large()

    # Trim column names
    raw.columns = [str(c).strip() for c in raw.columns]

    # Try to detect common header rows where first row is repeated headers (sometimes happens)
    # If 'Sr' appears in first row as a value and also as column header, drop that row:
    if "Sr" in raw.columns:
        first_row_vals = raw.iloc[0].astype(str).str.strip().str.lower().tolist()
        if "sr" in first_row_vals:
            # drop the first row (likely header repeated)
            raw = raw.iloc[1:].reset_index(drop=True)

    # Keep only rows with valid Sr if Sr exists
    raw = valid_sr_filter(raw)

    # Ensure required columns exist and create defaults when missing
    for col in ["Marked to Officer", "Priority", "Status", "File", "Subject", "Entry Date", "Remarks", "Sr"]:
        if col not in raw.columns:
            raw[col] = np.nan

    # Clean officer names
    raw["Marked to Officer"] = raw["Marked to Officer"].fillna("Unknown").astype(str).str.strip()

    # Normalize priority robustly
    raw["Priority"] = raw["Priority"].apply(lambda v: canonical_priority(v))

    # Clean status strings
    raw["Status"] = raw["Status"].fillna("In progress").astype(str).str.strip()

    # If Entry Date exists, try to standardize it to YYYY-MM-DD where possible
    if "Entry Date" in raw.columns:
        raw["Entry Date (Parsed)"] = raw["Entry Date"].apply(parse_date_flexible)

    # Create File Link column (HTML)
    raw["File Link"] = raw.apply(lambda r: create_clickable_file_link(r["File"], r.get("Sr", "")), axis=1)

    # Ensure Sr is numeric where possible, but keep original text in a separate column for display
    raw["Sr_original"] = raw["Sr"].astype(str)
    try:
        raw["Sr_num"] = pd.to_numeric(raw["Sr"], errors="coerce")
    except Exception:
        raw["Sr_num"] = np.nan

    # Reorder columns to common layout
    cols_order = ["Sr_original", "Marked to Officer", "Priority", "Status", "Subject", "Entry Date", "Entry Date (Parsed)", "File", "File Link", "Remarks"]
    # add any other columns that were present
    for c in raw.columns:
        if c not in cols_order:
            cols_order.append(c)
    raw = raw[cols_order]

    return raw

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
        ("%d-%m-%Y", r"^\d{1,2}-\d{1,2}-\d{4}$"),
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
    for i in range(1, n + 1):
        rows.append({
            "Sr": i,
            "Marked to Officer": officers[i % len(officers)],
            "Priority": priorities[i % len(priorities)],
            "Status": statuses[i % len(statuses)],
            "Subject": f"Task {i} - Administrative item regarding process {i%7}",
            "File": f"document_{i:03d}.pdf" if i % 3 != 0 else f"https://example.com/doc_{i}.pdf",
            "Entry Date": (datetime.date(2025, (i % 12) + 1, (i % 28) + 1)).strftime("%d/%m/%Y"),
            "Remarks": "Auto-generated sample data" if i % 5 else "Requires signature"
        })
    return pd.DataFrame(rows)

# ------------------------------
# UI helper components
# ------------------------------
def sidebar_controls():
    """
    Build controls in the sidebar and return settings.
    """
    st.sidebar.title("Controls & Settings")
    sheet_url = st.sidebar.text_input("Google Sheet CSV URL (gviz CSV recommended)", value=DEFAULT_SHEET_GVIZ_CSV)
    show_debug = st.sidebar.checkbox("Show debug info (unique priorities, raw head)", value=False)
    highlight_urgent = st.sidebar.checkbox("Highlight Most Urgent tasks", value=True)
    auto_refresh = st.sidebar.checkbox("Auto-refresh (every 5 minutes)", value=False)
    refresh_now = st.sidebar.button(" Refresh Data Now")
    return {
        "sheet_url": sheet_url.strip(),
        "show_debug": show_debug,
        "highlight_urgent": highlight_urgent,
        "auto_refresh": auto_refresh,
        "refresh_now": refresh_now
    }

def render_top_header(df: pd.DataFrame):
    st.markdown('<h1 class="main-header"> Task Management Dashboard</h1>', unsafe_allow_html=True)
    # Quick top-row metrics
    total_tasks = len(df)
    total_pending = len(df[df["Status"].str.lower() == "in progress"])
    unique_officers = df["Marked to Officer"].nunique()
    most_urgent_total = len(df[df["Priority"] == "Most Urgent"])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Tasks", total_tasks)
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Pending (In progress)", total_pending)
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Officers", unique_officers)
        st.markdown("</div>", unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Most Urgent", most_urgent_total)
        st.markdown("</div>", unsafe_allow_html=True)

def officers_overview_page(df: pd.DataFrame, settings: dict):
    st.header(" Officer-wise Pending Tasks Overview")
    # Filter to In progress by default
    pending_df = df[df["Status"].str.lower() == "in progress"].copy()

    # Compute counts
    officer_counts = pending_df.groupby("Marked to Officer").size().reset_index(name="Pending Tasks")
    officer_counts = officer_counts.sort_values("Pending Tasks", ascending=True)

    # Bar chart (horizontal)
    if not officer_counts.empty:
        fig = px.bar(
            officer_counts,
            x="Pending Tasks",
            y="Marked to Officer",
            orientation="h",
            title="Number of Pending Tasks by Officer",
            labels={"Pending Tasks": "Number of Tasks", "Marked to Officer": "Officer"},
            color="Pending Tasks",
            color_continuous_scale="Blues",
            height=420,
            text_auto=True     
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No pending tasks to show.")

    st.markdown("###  Summary Table")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(officer_counts.sort_values("Pending Tasks", ascending=False), use_container_width=True, hide_index=True)
    with col2:
        total_pending = len(pending_df)
        total_officers = officer_counts.shape[0]
        avg_tasks = total_pending / total_officers if total_officers else 0
        max_tasks = officer_counts["Pending Tasks"].max() if total_officers else 0
        #st.metric("Total Pending Tasks", total_pending)
        #st.metric("Officers with Pending", total_officers)
        #st.metric("Avg Tasks / Officer", f"{avg_tasks:.1f}")
        #st.metric("Max (single officer)", max_tasks)

    st.markdown("### Detailed Task View by Officer")
    officers_list = ["All Officers"] + sorted(pending_df["Marked to Officer"].unique().tolist())
    selected = st.selectbox("Select Officer", options=officers_list, index=0)

    # Search box for free text
    q = st.text_input("Search subject / remarks (simple substring search):", value="")

    filtered = pending_df.copy()
    if selected != "All Officers":
        filtered = filtered[filtered["Marked to Officer"] == selected]
    if q.strip():
        qlow = q.strip().lower()
        mask_subject = filtered["Subject"].astype(str).str.lower().str.contains(qlow, na=False)
        mask_remarks = filtered["Remarks"].astype(str).str.lower().str.contains(qlow, na=False)
        filtered = filtered[mask_subject | mask_remarks]

    st.markdown(f"**Showing {len(filtered)} tasks**")
    if settings["highlight_urgent"]:
        # style urgent rows visually in HTML by marking them
        def style_row(row):
            if row["Priority"] == "Most Urgent":
                return f'<div class="urgent-highlight">{row["Sr_original"]} | {row["Priority"]} | {row["Subject"]}</div>'
            else:
                return f'{row["Sr_original"]} | {row["Priority"]} | {row["Subject"]}'

        # Render a small HTML table with file links
        display_cols = ["Sr_original", "Priority", "Subject", "Entry Date", "File Link", "Remarks"]
        available_cols = [c for c in display_cols if c in filtered.columns]
        html = filtered[available_cols].to_html(escape=False, index=False)
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.dataframe(filtered[["Sr_original", "Priority", "Subject", "Entry Date", "File Link", "Remarks"]], use_container_width=True)

    # Option to download filtered CSV
    st.markdown(df_to_csv_download_link(filtered, filename="filtered_tasks.csv"), unsafe_allow_html=True)

def priority_analysis_page(df: pd.DataFrame, settings: dict):
    st.header(" Priority-wise Task Analysis")
    pending_df = df[df["Status"].str.lower() == "in progress"].copy()

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
            display_cols = ["Sr_original", "Marked to Officer", "Subject", "Entry Date", "File Link", "Remarks"]
            av = [c for c in display_cols if c in p_df.columns]
            st.dataframe(p_df[av], use_container_width=True, hide_index=True)
        st.markdown("---")

    # Overall priority distribution pie
    st.subheader("Overall Priority Distribution (Pending only)")
    priority_counts = pending_df["Priority"].value_counts().reset_index()
    priority_counts.columns = ["Priority", "Count"]
    if not priority_counts.empty:
        fig_pie = px.pie(priority_counts, values="Count", names="Priority", title="Distribution of Pending Tasks by Priority",
                         color="Priority", color_discrete_map=priority_colors)
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No pending tasks to visualize.")

def all_tasks_page(df: pd.DataFrame, settings: dict):
    st.header(" All Tasks (Full Table)")
    st.markdown("Use the table below to inspect all rows. You can sort and filter in the UI.")

    # Provide filters: officer, priority, status, date range
    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
    officers = sorted(df["Marked to Officer"].fillna("Unknown").unique().tolist())
    with col1:
        officer_filter = st.multiselect("Filter by Officer", options=officers, default=officers[:3])
    with col2:
        priority_options = sorted(df["Priority"].unique().tolist())
        priority_filter = st.multiselect("Filter by Priority", options=priority_options, default=priority_options)
    with col3:
        status_options = sorted(df["Status"].unique().tolist())
        status_filter = st.multiselect("Filter by Status", options=status_options, default=status_options)
    with col4:
        date_filter_text = st.text_input("Entry Date contains (e.g. 2025-03)", value="")

    filtered = df.copy()
    if officer_filter:
        filtered = filtered[filtered["Marked to Officer"].isin(officer_filter)]
    if priority_filter:
        filtered = filtered[filtered["Priority"].isin(priority_filter)]
    if status_filter:
        filtered = filtered[filtered["Status"].isin(status_filter)]
    if date_filter_text.strip():
        filtered = filtered[filtered["Entry Date"].astype(str).str.contains(date_filter_text.strip(), na=False)]

    st.markdown(f"**Showing {len(filtered)} rows**")
    st.dataframe(filtered, use_container_width=True)

    st.markdown(df_to_csv_download_link(filtered, filename="all_tasks_filtered.csv"), unsafe_allow_html=True)

# ------------------------------
# Main
# ------------------------------
def main():
    settings = sidebar_controls()
    # If Refresh button clicked, clear cache
    if settings["refresh_now"]:
        st.cache_data.clear()
        st.experimental_rerun()

    # Load and process data
    sheet_url = settings["sheet_url"] or DEFAULT_SHEET_GVIZ_CSV
    df = load_and_process(sheet_url)

    # Debug info
    if settings["show_debug"]:
        st.sidebar.markdown("### Debug Info")
        st.sidebar.write("Dataframe shape:", df.shape)
        st.sidebar.write("Columns:", df.columns.tolist())
        st.sidebar.write("Priority unique values (post-clean):", df["Priority"].unique().tolist())
        st.sidebar.write("Sample rows:")
        st.sidebar.dataframe(df.head(10))

    # Top header and metrics
    render_top_header(df)

    # Navigation
    st.sidebar.markdown("---")
    page = st.sidebar.radio("Select Page", ["Officer-wise Pending Tasks", "Priority-wise Analysis", "All Tasks", "About / Help"])
    if page == "Officer-wise Pending Tasks":
        officers_overview_page(df, settings)
    elif page == "Priority-wise Analysis":
        priority_analysis_page(df, settings)
    elif page == "All Tasks":
        all_tasks_page(df, settings)
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

    # Auto-refresh info (no background jobs; instruct user)
    if settings["auto_refresh"]:
        st.sidebar.info("Auto-refresh is enabled (cache TTL = 300s). The app will refetch after cache expires or if you click Refresh.")

if __name__ == "__main__":
    main()
