# app.py
"""
Task Management Dashboard - Full, Self-contained Streamlit App
(Designed to be long / comprehensive — 600+ lines)
Author: Generated for Deep Shah
Purpose:
 - Read tasks from Google Sheets (gviz CSV recommended)
 - Normalize priorities (detect "Most Urgent" and variants)
 - Render clickable file links (e.g., 📎 52.pdf) without showing raw HTML escape
 - Show bar charts with numeric labels and annotations
 - Provide debugging controls, download/export functionality
 - Robust fallbacks and sample data when network or sheet permissions block access

Usage:
 1. Save as app.py
 2. streamlit run app.py
"""

# --------------------------
# Imports
# --------------------------
import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
import re
import datetime
import base64
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Any, Dict, List, Tuple

# --------------------------
# App-wide Configuration
# --------------------------
st.set_page_config(
    page_title="Task Management Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------
# Constants & Defaults
# --------------------------
# Use the gviz CSV form for stable CSV responses
DEFAULT_GVIZ_CSV = (
    "https://docs.google.com/spreadsheets/d/14-idXJHzHKCUQxxaqGZi-6S0G20gvPUhK4G16ci2FwI"
    "/gviz/tq?tqx=out:csv&gid=213021534"
)

# Priority canonicalization mapping
PRIORITY_CANONICAL_MAP = {
    "most urgent": "Most Urgent",
    "mosturgent": "Most Urgent",
    "most_urgent": "Most Urgent",
    "urgent": "Most Urgent",
    "highest": "Most Urgent",
    "high": "High",
    "medium": "Medium",
    "med": "Medium",
    "low": "Low",
    "not urgent": "Low"
}

# Default column names we expect (human readable)
EXPECTED_COLUMNS = [
    "Sr",
    "Marked to Officer",
    "Priority",
    "Status",
    "Subject",
    "File",
    "Entry Date",
    "Remarks"
]

# --------------------------
# Utility helper functions
# --------------------------

def _log(msg: str) -> None:
    """Simple console log (useful during development)"""
    # Using print so logs show up if you run streamlit with logs
    print(f"[TaskDashboard] {msg}")

def fetch_gviz_csv(url: str, timeout: int = 12) -> pd.DataFrame:
    """
    Fetch CSV text from a gviz CSV endpoint or any CSV URL and return a DataFrame.
    Returns empty DataFrame on failure (we handle fallback later).
    """
    try:
        _log(f"Fetching CSV from URL: {url}")
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        text = r.text
        # Some endpoints produce JSON-like wrappers — pandas can still parse many CSVs.
        df = pd.read_csv(StringIO(text))
        _log(f"Fetched {len(df)} rows")
        return df
    except Exception as e:
        _log(f"Failed to fetch CSV: {e}")
        return pd.DataFrame()

def normalize_string(x: Any) -> str:
    """
    Turn value into normalized string: trim, collapse whitespace, lower-case.
    Non-string inputs are coerced to string safely.
    """
    if pd.isna(x):
        return ""
    s = str(x)
    # replace non-breaking spaces
    s = s.replace("\u00A0", " ")
    # collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()

def canonical_priority(val: Any) -> str:
    """
    Map a raw priority value (various capitalizations / typos) to canonical set.
    Default fallback: 'Medium'
    """
    if pd.isna(val):
        return "Medium"
    norm = normalize_string(val)
    if norm == "":
        return "Medium"
    # direct map
    if norm in PRIORITY_CANONICAL_MAP:
        return PRIORITY_CANONICAL_MAP[norm]
    # fuzzy rules
    if "urgent" in norm:
        return "Most Urgent"
    if "high" in norm:
        return "High"
    if "medium" in norm or "med" in norm:
        return "Medium"
    if "low" in norm:
        return "Low"
    # fallback
    return "Medium"

def safe_make_clickable(href: str, display_name: Optional[str] = None) -> str:
    """
    Build a very simple HTML anchor for links. Keep attributes minimal to avoid
    string-escaping issues inside DataFrame/HTML rendering.
    Example: '<a href="https://..." target="_blank">📎 52.pdf</a>'
    """
    if not href or pd.isna(href):
        return ""
    href_str = str(href).strip()
    if display_name is None:
        display_name = href_str.split("/")[-1] or href_str
    # We purposely don't use class attributes to avoid double-quote confusion.
    return f'<a href="{href_str}" target="_blank">📎 {display_name}</a>'

def df_to_download_link(df: pd.DataFrame, filename: str = "tasks_export.csv") -> str:
    """
    Convert DataFrame to CSV and return an HTML anchor link (base64).
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download CSV</a>'

def parse_date_either(val: Any) -> Optional[datetime.date]:
    """
    Try common date formats and pandas fallback. Returns a date or None.
    """
    if pd.isna(val):
        return None
    s = str(val).strip()
    # Try common explicit formats
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d %b %Y", "%d %B %Y"):
        try:
            return datetime.datetime.strptime(s, fmt).date()
        except Exception:
            pass
    # pandas fallback (dayfirst True)
    try:
        dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
        if not pd.isna(dt):
            return dt.date()
    except Exception:
        pass
    # failed parse
    return None

# --------------------------
# Data preparation pipeline
# --------------------------

def prepare_raw_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Take raw DataFrame from CSV and normalize it:
    - Ensure expected columns exist
    - Normalize values: Priority, Status, Officer names
    - Create 'File Link HTML' column for clickable links
    - Parse entry dates
    - Ensure Sr filtering (drop empty header rows)
    """

    if raw is None:
        return pd.DataFrame()

    df = raw.copy()

    # normalize column names: strip only (we keep original names for match)
    df.columns = [str(c).strip() for c in df.columns]

    # If first row appears to be repeated header, drop it.
    if "Sr" in df.columns:
        try:
            first_row_vals = df.iloc[0].astype(str).str.strip().str.lower().tolist()
            if "sr" in first_row_vals:
                _log("Dropping first row because it appears to be duplicate header")
                df = df.iloc[1:].reset_index(drop=True)
        except Exception:
            pass

    # Ensure expected columns exist
    for c in EXPECTED_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan

    # Filter out rows with empty Sr
    try:
        mask = df["Sr"].notna() & (df["Sr"].astype(str).str.strip() != "") & (df["Sr"].astype(str).str.strip().str.lower() != "sr")
        df = df[mask].copy()
    except Exception:
        # If column weirdness, keep as-is
        pass

    # Normalize officer name
    df["Marked to Officer"] = df["Marked to Officer"].fillna("Unknown").astype(str).str.strip()

    # Clean priority
    df["Priority_Raw"] = df["Priority"].astype(str)
    df["Priority"] = df["Priority_Raw"].apply(canonical_priority)

    # Normalize status
    df["Status"] = df["Status"].fillna("In progress").astype(str).str.strip()

    # Entry Date parse
    df["Entry Date Parsed"] = df["Entry Date"].apply(lambda v: parse_date_either(v) if pd.notna(v) else None)
    # Convert parsed date to ISO string for display
    df["Entry Date Parsed"] = df["Entry Date Parsed"].astype(str).replace("None", "")

    # Create File Link HTML
    def make_file_link(file_val):
        if pd.isna(file_val) or str(file_val).strip() == "" or str(file_val).strip().lower() == "file":
            return ""
        s = str(file_val).strip()
        # if a full URL:
        if s.startswith("http://") or s.startswith("https://"):
            display = s.split("/")[-1] or s
            return safe_make_clickable(s, display)
        # else treat as filename: link to Google Drive search
        query = requests.utils.requote_uri(s)
        drive_url = f"https://drive.google.com/drive/search?q={query}"
        return safe_make_clickable(drive_url, s)

    df["File Link HTML"] = df["File"].apply(make_file_link)

    # Keep a Sr_display and Sr_num
    df["Sr_display"] = df["Sr"].astype(str)
    try:
        df["Sr_num"] = pd.to_numeric(df["Sr"], errors="coerce")
    except Exception:
        df["Sr_num"] = np.nan

    # Return in a friendly column order
    display_cols = [
        "Sr_display",
        "Marked to Officer",
        "Priority",
        "Status",
        "Subject",
        "Entry Date",
        "Entry Date Parsed",
        "File",
        "File Link HTML",
        "Remarks"
    ]
    # add any other existing columns after display ones
    extra_cols = [c for c in df.columns if c not in display_cols]
    ordered = display_cols + extra_cols
    df = df[ordered]
    return df

# --------------------------
# Sample data (fallback)
# --------------------------
def sample_data_factory(n: int = 150) -> pd.DataFrame:
    """
    Create a fairly large sample DataFrame for testing or offline use.
    The sample includes many variants of 'Most Urgent' to test canonicalization.
    """
    officers = [
        "CMFO", "DRO", "ADC (RD)", "ADC G", "Legal Cell", "AC G",
        "DyESA", "Election Tehsildar", "ADC (W)", "EO", "Inspector"
    ]
    priority_variants = ["Most Urgent", "most urgent", "MOST URGENT", "Urgent", "High", "Medium", "Low", "mosturgent", "most_urgent", "med"]
    statuses = ["In progress", "Completed", "Pending", "In progress", "In progress"]
    rows = []
    for i in range(1, n + 1):
        pr = priority_variants[i % len(priority_variants)]
        stt = statuses[i % len(statuses)]
        officer = officers[i % len(officers)]
        subj = f"Task {i} - action item regarding process {i%12}"
        # create some file names and occasionally full URLs
        if i % 7 == 0:
            file_val = f"https://drive.google.com/file/d/fakefileid_{i}/view?usp=sharing"
        elif i % 5 == 0:
            file_val = f"{i}.pdf"
        else:
            file_val = ""
        entry_date = (datetime.date(2025, ((i % 12) + 1), ((i % 28) + 1))).strftime("%d/%m/%Y")
        remark = "Requires signature" if i % 4 == 0 else "Follow up"
        rows.append({
            "Sr": i,
            "Marked to Officer": officer,
            "Priority": pr,
            "Status": stt,
            "Subject": subj,
            "File": file_val,
            "Entry Date": entry_date,
            "Remarks": remark
        })
    return pd.DataFrame(rows)

# --------------------------
# Caching & loader wrapper
# --------------------------
@st.cache_data(ttl=300)
def load_and_prepare_sheet(url: str) -> pd.DataFrame:
    """
    Fetch, prepare and return DataFrame. Falls back to sample if network fails.
    """
    # Fetch CSV
    df_raw = fetch_gviz_csv(url)
    if df_raw is None or df_raw.empty:
        _log("Using sample data fallback because raw fetch failed or returned empty.")
        df_raw = sample_data_factory(200)
    df_prepared = prepare_raw_dataframe(df_raw)
    return df_prepared

# --------------------------
# Sidebar controls
# --------------------------
st.sidebar.title("Task Dashboard Controls")
sheet_url = st.sidebar.text_input("Google Sheet (gviz CSV) URL", value=DEFAULT_GVIZ_CSV)
use_sample = st.sidebar.checkbox("Force use sample data (ignore sheet)", value=False)
show_debug = st.sidebar.checkbox("Show debug info", value=False)
highlight_urgent_rows = st.sidebar.checkbox("Highlight Most Urgent rows in listings", value=True)
auto_refresh = st.sidebar.checkbox("Auto-refresh (cache TTL 5 min)", value=False)
refresh_now = st.sidebar.button("🔄 Refresh Now (clear & reload)")

if refresh_now:
    st.cache_data.clear()
    st.experimental_rerun()

# --------------------------
# Load data
# --------------------------
if use_sample:
    raw_df = sample_data_factory(220)
    df = prepare_raw_dataframe(raw_df)
else:
    df = load_and_prepare_sheet(sheet_url)

# handle empty data graceful
if df is None or df.empty:
    st.title("📊 Task Management Dashboard")
    st.warning("No data available. Either the sheet URL is invalid, permissions prevent access, or the sheet returned empty. Toggle 'Force use sample data' to test the app.")
    st.stop()

# --------------------------
# Top-level header & KPIs
# --------------------------
st.markdown('<h1 style="text-align:center;color:#1f77b4">📊 Task Management Dashboard</h1>', unsafe_allow_html=True)

total_tasks = len(df)
pending_mask = df["Status"].str.lower().str.contains("in progress") | df["Status"].str.lower().str.contains("pending")
total_pending = int(pending_mask.sum())
unique_officers = df["Marked to Officer"].nunique()
most_urgent_count = int((df["Priority"] == "Most Urgent").sum())

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Tasks", total_tasks)
with c2:
    st.metric("Pending (In progress)", total_pending)
with c3:
    st.metric("Officers", unique_officers)
with c4:
    st.metric("Most Urgent", most_urgent_count)

st.markdown("---")

# Debugging panel
if show_debug:
    st.sidebar.markdown("### Debug Info")
    st.sidebar.write("Data shape:", df.shape)
    st.sidebar.write("Columns:", df.columns.tolist())
    st.sidebar.write("Unique Priorities:", df["Priority"].unique().tolist())
    st.sidebar.dataframe(df.head(10))

# --------------------------
# Filters (sidebar)
# --------------------------
st.sidebar.markdown("---")
st.sidebar.header("Filters")
officers_opt = sorted(df["Marked to Officer"].unique().tolist())
selected_officers = st.sidebar.multiselect("Officers", options=officers_opt, default=officers_opt)

priority_order = ["Most Urgent", "High", "Medium", "Low"]
priorities_present = sorted(list(set(df["Priority"].tolist())), key=lambda x: priority_order.index(x) if x in priority_order else 99)
selected_priorities = st.sidebar.multiselect("Priorities", options=priorities_present, default=priorities_present)

statuses_present = sorted(df["Status"].unique().tolist())
selected_statuses = st.sidebar.multiselect("Statuses", options=statuses_present, default=statuses_present)

search_text = st.sidebar.text_input("Search Subject or Remarks (substring)", value="")

# Apply filters
filtered_df = df.copy()
if selected_officers:
    filtered_df = filtered_df[filtered_df["Marked to Officer"].isin(selected_officers)]
if selected_priorities:
    filtered_df = filtered_df[filtered_df["Priority"].isin(selected_priorities)]
if selected_statuses:
    filtered_df = filtered_df[filtered_df["Status"].isin(selected_statuses)]
if search_text.strip():
    q = search_text.strip().lower()
    mask_subject = filtered_df["Subject"].astype(str).str.lower().str.contains(q, na=False)
    mask_remarks = filtered_df["Remarks"].astype(str).str.lower().str.contains(q, na=False)
    filtered_df = filtered_df[mask_subject | mask_remarks]

# --------------------------
# Navigation pages
# --------------------------
page = st.radio("Select Page", options=["Officer-wise Pending Tasks", "Priority-wise Analysis", "All Tasks & Export", "About/Help"], index=0, horizontal=True)

# --------------------------
# Page: Officer-wise Pending Tasks
# --------------------------
if page == "Officer-wise Pending Tasks":
    st.header("📋 Officer-wise Pending Tasks Overview")

    pending_df = filtered_df[filtered_df["Status"].str.lower().str.contains("in progress") | filtered_df["Status"].str.lower().str.contains("pending")]

    if pending_df.empty:
        st.info("No pending tasks for selected filters.")
    else:
        # compute counts per officer
        officer_counts = pending_df.groupby("Marked to Officer").size().reset_index(name="Pending Tasks")
        officer_counts = officer_counts.sort_values("Pending Tasks", ascending=True)

        # Plot horizontal bar chart with values
        fig = px.bar(
            officer_counts,
            x="Pending Tasks",
            y="Marked to Officer",
            orientation="h",
            text="Pending Tasks",
            labels={"Pending Tasks":"Number of Tasks", "Marked to Officer":"Officer"},
            title="Number of Pending Tasks by Officer"
        )
        fig.update_traces(textposition="outside")
        # Add annotations (robust visibility)
        for idx, row in officer_counts.iterrows():
            fig.add_annotation(
                x=row["Pending Tasks"] + 0.1,
                y=row["Marked to Officer"],
                text=str(int(row["Pending Tasks"])),
                showarrow=False,
                xanchor="left",
                font=dict(size=11)
            )
        fig.update_layout(height=520, margin=dict(l=140, r=40, t=60, b=40))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Summary Table")
        left_col, right_col = st.columns([2, 1])
        with left_col:
            st.dataframe(officer_counts.sort_values("Pending Tasks", ascending=False), use_container_width=True, hide_index=True)
        with right_col:
            st.markdown("### Quick Stats")
            total_pending_local = len(pending_df)
            total_officers_local = len(officer_counts)
            avg_tasks = total_pending_local / total_officers_local if total_officers_local else 0
            max_tasks = int(officer_counts["Pending Tasks"].max()) if not officer_counts.empty else 0
            st.metric("Total Pending (filtered)", total_pending_local)
            st.metric("Officers with Pending", total_officers_local)
            st.metric("Avg Tasks / Officer", f"{avg_tasks:.1f}")
            st.metric("Max Tasks (Single Officer)", max_tasks)

        st.markdown("---")
        st.subheader("🔎 Detailed Task View by Officer")
        officer_options = ["All Officers"] + sorted(pending_df["Marked to Officer"].unique().tolist())
        selected_officer = st.selectbox("Select an Officer:", options=officer_options, index=0)

        display_df = pending_df.copy()
        if selected_officer != "All Officers":
            display_df = display_df[display_df["Marked to Officer"] == selected_officer]

        # Columns to show
        show_columns = ["Sr_display", "Priority", "Subject", "Entry Date", "File Link HTML", "Remarks"]
        available_cols = [c for c in show_columns if c in display_df.columns]

        if display_df.empty:
            st.info("No tasks for this filter.")
        else:
            # Render HTML anchors using to_html(escape=False)
            df_show = display_df[available_cols].rename(columns={"Sr_display":"Sr", "File Link HTML":"File"})
            st.markdown(df_show.to_html(escape=False, index=False), unsafe_allow_html=True)

            st.markdown("### Open tasks (expander view)")
            for _, row in display_df.iterrows():
                header = f"Sr {row['Sr_display']} — {row.get('Subject','No Subject')} — {row.get('Priority','')}"
                with st.expander(header):
                    st.write(f"**Officer:** {row.get('Marked to Officer','')}")
                    st.write(f"**Priority:** {row.get('Priority','')}")
                    st.write(f"**Status:** {row.get('Status','')}")
                    if row.get("Entry Date",""):
                        st.write(f"**Entry Date:** {row.get('Entry Date','')}")
                    if row.get("File Link HTML",""):
                        st.markdown(row.get("File Link HTML",""), unsafe_allow_html=True)
                    else:
                        st.write("No file attached.")
                    st.write(f"**Remarks:** {row.get('Remarks','')}")

# --------------------------
# Page: Priority-wise Analysis
# --------------------------
elif page == "Priority-wise Analysis":
    st.header("⚡ Priority-wise Task Analysis")

    # Consider pending set as per the earlier definition
    pending_df = filtered_df[filtered_df["Status"].str.lower().str.contains("in progress") | filtered_df["Status"].str.lower().str.contains("pending")]
    total_pending_local = len(pending_df)

    counts = pending_df["Priority"].value_counts().reindex(["Most Urgent", "High", "Medium", "Low"]).fillna(0).astype(int)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Pending Tasks", total_pending_local)
    with m2:
        st.metric("Most Urgent", counts.get("Most Urgent", 0), delta=f"{(counts.get('Most Urgent',0)/total_pending_local*100):.1f}%" if total_pending_local else "0%")
    with m3:
        st.metric("High", counts.get("High", 0), delta=f"{(counts.get('High',0)/total_pending_local*100):.1f}%" if total_pending_local else "0%")
    with m4:
        st.metric("Medium", counts.get("Medium", 0), delta=f"{(counts.get('Medium',0)/total_pending_local*100):.1f}%" if total_pending_local else "0%")

    st.markdown("---")
    colors = {"Most Urgent":"#ff4b4b","High":"#ff8c00","Medium":"#ffd700","Low":"#94d2bd"}

    for p in ["Most Urgent","High","Medium","Low"]:
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
            labels={"Task Count":"Number of Tasks","Marked to Officer":"Officer"},
            color_discrete_sequence=[colors.get(p, "#636EFA")]
        )
        # show values on bars
        fig.update_traces(text=counts_by_officer["Task Count"].astype(int), textposition="outside")
        # add annotations for robust visibility
        for idx, row in counts_by_officer.iterrows():
            fig.add_annotation(
                x=row["Task Count"] + 0.05,
                y=row["Marked to Officer"],
                text=str(int(row["Task Count"])),
                showarrow=False,
                xanchor="left",
                font=dict(size=11)
            )
        fig.update_layout(height=360, margin=dict(l=140))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander(f"View {p} Priority Task Details ({len(p_df)} rows)"):
            display_cols = ["Sr_display","Marked to Officer","Subject","Entry Date","File Link HTML","Remarks"]
            av = [c for c in display_cols if c in p_df.columns]
            if av:
                df_show = p_df[av].rename(columns={"Sr_display":"Sr","File Link HTML":"File"})
                st.markdown(df_show.to_html(escape=False, index=False), unsafe_allow_html=True)
            else:
                st.write("No columns available to show.")

    st.markdown("---")
    st.subheader("📊 Overall Priority Distribution")
    overall_counts = pending_df["Priority"].value_counts().reset_index()
    overall_counts.columns = ["Priority","Count"]
    if not overall_counts.empty:
        fig_pie = px.pie(overall_counts, values="Count", names="Priority", title="Distribution of Pending Tasks by Priority", color_discrete_map=colors)
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No pending tasks to visualize.")

# --------------------------
# Page: All Tasks & Export
# --------------------------
elif page == "All Tasks & Export":
    st.header("📚 All Tasks - Inspect & Export")

    display_cols = ["Sr_display","Marked to Officer","Priority","Status","Subject","Entry Date","File Link HTML","Remarks"]
    available_cols = [c for c in display_cols if c in df.columns]
    df_show = df[available_cols].rename(columns={"Sr_display":"Sr","File Link HTML":"File"})
    st.markdown("### Full Table")
    st.markdown(df_show.to_html(escape=False, index=False), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(df_to_download_link(df, filename="tasks_full_export.csv"), unsafe_allow_html=True)
    st.markdown("Tip: If links do not open, check file permissions or the link format. Drive search links might require authentication.")

# --------------------------
# Page: About/Help
# --------------------------
else:
    st.header("ℹ️ About & Help")
    st.markdown("""
    **Task Management Dashboard** — full-featured dashboard to analyze tasks exported from Google Sheets.

    **How to use**
    1. Use the default gviz CSV URL or paste your own gviz CSV link.
       - Recommended format:
         `https://docs.google.com/spreadsheets/d/{KEY}/gviz/tq?tqx=out:csv&gid={GID}`
    2. If your sheet is private, either publish it temporarily, or I can provide a version using the Google Sheets API (service account).
    3. Toggle 'Force use sample data' to test locally without sheet access.

    **Why 'Most Urgent' isn't showing?**
    - This app normalizes common variants like `MOST URGENT`, `mosturgent`, `most urgent`, `Urgent`. If your sheet contains hidden characters, trailing spaces, or non-standard text, enable "Show debug info" to inspect the raw values.
    - If the 'Priority' column is misspelled or placed in a different column, ensure the column header exactly matches "Priority".

    **File links**
    - If your sheet column contains full URLs (https://...), the app renders them directly as clickable anchors.
    - If your sheet contains filenames (like `52.pdf`), the app links them to Google Drive search (`https://drive.google.com/drive/search?q=52.pdf`). If Drive search requires permissions, users need appropriate access.

    **Need private-sheet access?**
    - I can adapt the app to use Google Sheets API (service account) — this requires you to create a service account and share the sheet with it. Ask me and I'll provide the secure version.

    **Want customizations?**
    - I can add alerts, email notifications, Slack integration, or automatic sorting and deep filters.
    """)

# --------------------------
# Footer
# --------------------------
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit. Ask me to adapt this for Google Sheets API access, or to add email/SMS alerts for 'Most Urgent' tasks.")
