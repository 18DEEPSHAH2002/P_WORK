# app.py
"""
Task Management Dashboard - Full Ready-to-Run Streamlit App (~600+ lines)
Author: Generated for Deep Shah
Purpose:
 - Read tasks from a Google Sheet via gviz CSV (recommended)
 - Robust normalization of Priority (detects Many forms of "Most Urgent")
 - Sidebar navigation (Overview | Priority-wise Analysis & Export)
 - Filters only appear and apply on the Priority-wise Analysis & Export page
 - Merge of "All Tasks & Export" into "Priority-wise Analysis & Export" page
 - Clickable file links render correctly (📎 filename) rather than raw HTML text
 - Bar charts show numeric labels on/above bars and add annotations
 - Download/export CSV/Excel functionality included
 - Lots of helpful utilities, debug toggles, and safe fallbacks (sample data)
 - Comments and structured code to make it easy to modify
Notes:
 - Save this file as app.py and run: `streamlit run app.py`
 - Install required packages if missing: `pip install streamlit pandas plotly requests openpyxl`
"""

# ------------------------------
# Imports
# ------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO, BytesIO
import re
import datetime
import base64
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Any, List, Dict

# ------------------------------
# Page configuration
# ------------------------------
st.set_page_config(
    page_title="Task Management Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------
# Constants & Defaults
# ------------------------------
# Recommended: gviz CSV endpoint for robust CSV output
DEFAULT_GVIZ_CSV = (
    "https://docs.google.com/spreadsheets/d/14-idXJHzHKCUQxxaqGZi-6S0G20gvPUhK4G16ci2FwI"
    "/gviz/tq?tqx=out:csv&gid=213021534"
)

# Canonicalization map for common priority variants
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

# The columns we expect / want to support; if missing we'll create defaults
EXPECTED_COLUMNS = [
    "Sr",
    "Marked to Officer",
    "Priority",
    "Status",
    "Subject",
    "File",
    "Entry Date",
    "Remarks",
]

# ------------------------------
# Utility functions
# ------------------------------
def _log(msg: str) -> None:
    """Internal simple logger (prints to stdout)"""
    print(f"[TaskDash] {msg}")

def fetch_csv_from_url(url: str, timeout: int = 12) -> pd.DataFrame:
    """
    Fetch CSV text from URL and parse to DataFrame.
    Returns empty DataFrame on failure.
    """
    try:
        _log(f"Fetching CSV from: {url}")
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        text = resp.text
        # read via pandas
        df = pd.read_csv(StringIO(text))
        _log(f"Fetched {len(df)} rows")
        return df
    except Exception as e:
        _log(f"Fetch failed: {e}")
        return pd.DataFrame()

def normalize_string(val: Any) -> str:
    """Normalize text: strip, collapse whitespace, replace non-breaking spaces, lower-case"""
    if pd.isna(val):
        return ""
    s = str(val)
    s = s.replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()

def canonical_priority(val: Any) -> str:
    """
    Map variants of priority strings to canonical categories:
    'Most Urgent', 'High', 'Medium', 'Low'
    Defaults to 'Medium' when uncertain.
    """
    if pd.isna(val):
        return "Medium"
    norm = normalize_string(val)
    if norm == "":
        return "Medium"
    if norm in PRIORITY_CANONICAL_MAP:
        return PRIORITY_CANONICAL_MAP[norm]
    # fuzzy rules
    if "urgent" in norm:
        return "Most Urgent"
    if "high" in norm:
        return "High"
    if "med" in norm or "medium" in norm:
        return "Medium"
    if "low" in norm:
        return "Low"
    return "Medium"

def safe_html_anchor(url: str, display: Optional[str] = None) -> str:
    """
    Produce a simple HTML anchor string for a link, for rendering with unsafe_allow_html.
    Avoids extra classes/attributes to reduce escaping issues.
    Example returned string: '<a href="https://..." target="_blank">📎 name.pdf</a>'
    """
    if not url or pd.isna(url):
        return ""
    u = str(url).strip()
    if display is None:
        display = u.split("/")[-1] or u
    return f'<a href="{u}" target="_blank">📎 {display}</a>'

def drive_search_link_for_filename(filename: str) -> str:
    """If cell contains only a filename, link to Google Drive search for that filename."""
    if not filename or pd.isna(filename) or str(filename).strip() == "":
        return ""
    q = requests.utils.requote_uri(str(filename).strip())
    return f"https://drive.google.com/drive/search?q={q}"

def make_file_link_html(cell_value: Any) -> str:
    """
    If cell_value is a full URL, use it as-is. If it's a filename, create a Drive search link.
    Returns HTML anchor string or empty string.
    """
    if pd.isna(cell_value):
        return ""
    s = str(cell_value).strip()
    if s == "" or s.lower() == "file":
        return ""
    if s.startswith("http://") or s.startswith("https://"):
        # display last segment nicely
        display = s.split("/")[-1] or s
        return safe_html_anchor(s, display)
    # else treat as filename: link to Drive search
    ds = drive_search_link_for_filename(s)
    return safe_html_anchor(ds, s)

def parse_date_flexibly(val: Any) -> Optional[datetime.date]:
    """
    Try to parse common date formats. Returns date or None.
    """
    if pd.isna(val):
        return None
    s = str(val).strip()
    # try explicit formats
    formats = ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d %b %Y", "%d %B %Y")
    for fmt in formats:
        try:
            dt = datetime.datetime.strptime(s, fmt)
            return dt.date()
        except Exception:
            pass
    # pandas fallback
    try:
        dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
        if not pd.isna(dt):
            return dt.date()
    except Exception:
        pass
    return None

def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    """
    Convert DataFrame to Excel file bytes for download.
    Requires openpyxl installed.
    """
    with BytesIO() as b:
        with pd.ExcelWriter(b, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Tasks")
        return b.getvalue()

def df_to_csv_base64_link(df: pd.DataFrame, filename: str = "tasks_export.csv") -> str:
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download CSV</a>'

def df_to_excel_download_button(df: pd.DataFrame, filename: str = "tasks_export.xlsx") -> None:
    """Use Streamlit's download_button to provide Excel download"""
    try:
        excel_bytes = df_to_excel_bytes(df)
        st.download_button(label="📥 Download Excel (.xlsx)", data=excel_bytes, file_name=filename, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception as e:
        st.error(f"Failed to create Excel file: {e}")

# ------------------------------
# Data preparation & normalization
# ------------------------------
def prepare_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a raw CSV DataFrame from the sheet:
     - Ensure expected columns exist
     - Normalize priority values
     - Normalize status values
     - Create File Link HTML column for rendering anchors
     - Parse entry date into a stable string column
     - Filter rows with bad 'Sr' if necessary
    """
    if raw is None or raw.empty:
        return pd.DataFrame()

    df = raw.copy()

    # Trim headers
    df.columns = [str(c).strip() for c in df.columns]

    # If first row is repeated header, drop it (common issue for some CSV exports)
    try:
        if "Sr" in df.columns:
            first_row_vals = df.iloc[0].astype(str).str.strip().str.lower().tolist()
            if "sr" in first_row_vals:
                df = df.iloc[1:].reset_index(drop=True)
    except Exception:
        pass

    # Ensure expected columns exist
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    # Filter out rows where Sr is empty/header-like (but if Sr column is messy, keep all)
    try:
        mask = df["Sr"].notna() & (df["Sr"].astype(str).str.strip() != "") & (df["Sr"].astype(str).str.strip().str.lower() != "sr")
        df = df[mask].copy()
    except Exception:
        pass

    # Normalize Marked to Officer
    df["Marked to Officer"] = df["Marked to Officer"].fillna("Unknown").astype(str).str.strip()

    # Normalize Priority
    df["Priority_Raw"] = df["Priority"].astype(str)
    df["Priority"] = df["Priority_Raw"].apply(canonical_priority)

    # Normalize Status
    df["Status"] = df["Status"].fillna("In progress").astype(str).str.strip()

    # Parse Entry Date (into ISO-like strings for easy filtering & display)
    df["Entry Date Parsed"] = df["Entry Date"].apply(lambda x: parse_date_flexibly(x) if pd.notna(x) else None)
    df["Entry Date Parsed"] = df["Entry Date Parsed"].astype(str).replace("None", "")

    # Create File Link HTML column
    df["File Link HTML"] = df["File"].apply(make_file_link_html)

    # Maintain displayable Sr and attempt numeric conversion for sorting
    df["Sr_display"] = df["Sr"].astype(str)
    try:
        df["Sr_num"] = pd.to_numeric(df["Sr"], errors="coerce")
    except Exception:
        df["Sr_num"] = np.nan

    # Reorder important columns first
    front_cols = ["Sr_display", "Marked to Officer", "Priority", "Status", "Subject", "Entry Date", "Entry Date Parsed", "File", "File Link HTML", "Remarks"]
    for c in front_cols:
        if c not in df.columns:
            df[c] = ""
    # Append remaining columns
    rest = [c for c in df.columns if c not in front_cols]
    ordered_cols = front_cols + rest
    df = df[ordered_cols]
    return df

# ------------------------------
# Sample data fallback generator (useful for offline testing)
# ------------------------------
def sample_data(n: int = 220) -> pd.DataFrame:
    officers = ["CMFO", "DRO", "ADC (RD)", "ADC G", "Legal Cell", "AC G", "DyESA", "Election Tehsildar", "ADC (W)", "EO"]
    priority_vars = ["Most Urgent", "most urgent", "MOST URGENT", "Urgent", "High", "Medium", "Low", "mosturgent", "most_urgent", "med"]
    statuses = ["In progress", "Completed", "In progress", "Pending"]
    rows = []
    for i in range(1, n + 1):
        pr = priority_vars[i % len(priority_vars)]
        stt = statuses[i % len(statuses)]
        officer = officers[i % len(officers)]
        subj = f"Task {i} - admin process {i%12}"
        # file sometimes a link, sometimes just filename, sometimes blank
        if i % 9 == 0:
            file_val = f"https://drive.google.com/file/d/fakeid_{i}/view?usp=sharing"
        elif i % 4 == 0:
            file_val = f"{i}.pdf"
        else:
            file_val = ""
        entry_date = (datetime.date(2025, (i % 12) + 1, (i % 28) + 1)).strftime("%d/%m/%Y")
        remark = "Requires signature" if i % 5 == 0 else "Follow up"
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

# ------------------------------
# Caching loader
# ------------------------------
@st.cache_data(ttl=300)
def load_and_prepare(url: str, force_sample: bool = False) -> pd.DataFrame:
    """
    Load a sheet via URL or return sample data if fetch fails or force_sample True.
    Returns prepared DataFrame (normalized).
    """
    if force_sample:
        raw = sample_data(220)
        prepared = prepare_dataframe(raw)
        return prepared

    df_raw = fetch_csv_from_url(url)
    if df_raw is None or df_raw.empty:
        # fallback to sample
        _log("Using sample data fallback (fetch failed or returned empty).")
        raw = sample_data(220)
        prepared = prepare_dataframe(raw)
        return prepared
    prepared = prepare_dataframe(df_raw)
    return prepared

# ------------------------------
# Sidebar: top-level controls & navigation
# ------------------------------
st.sidebar.title("Controls & Navigation")

sheet_url = st.sidebar.text_input("Google Sheet (gviz CSV) URL", value=DEFAULT_GVIZ_CSV)
force_sample = st.sidebar.checkbox("Force sample data (ignore sheet)", value=False)
show_debug = st.sidebar.checkbox("Show Debug Info (inspect loaded values)", value=False)

# Page selection in sidebar (user requested)
page = st.sidebar.selectbox("Select Page", options=["Dashboard Overview", "Priority-wise Analysis & Export", "About / Help"], index=0)

# Refresh button to clear cache and reload (useful while editing sheet)
if st.sidebar.button("🔄 Refresh Data (clear cache)"):
    st.cache_data.clear()
    st.experimental_rerun()

# ------------------------------
# Load & prepare data
# ------------------------------
df = load_and_prepare(sheet_url, force_sample=force_sample)

# Basic guard
if df is None or df.empty:
    st.title("Task Management Dashboard")
    st.error("No data loaded. Either the sheet URL is invalid or the sheet is private. Toggle 'Force sample data' to test locally.")
    st.stop()

# Debug panel
if show_debug:
    st.sidebar.markdown("### Debug Info")
    st.sidebar.write("Data shape:", df.shape)
    st.sidebar.write("Columns:", df.columns.tolist())
    st.sidebar.write("Unique Priority values:", df["Priority"].unique().tolist())
    st.sidebar.dataframe(df.head(10))

# ------------------------------
# Top header and high-level metrics
# ------------------------------
st.markdown('<h1 style="text-align:center;color:#1f77b4">📊 Task Management Dashboard</h1>', unsafe_allow_html=True)

total_tasks = len(df)
pending_mask = df["Status"].str.lower().str.contains("in progress") | df["Status"].str.lower().str.contains("pending")
total_pending = int(pending_mask.sum())
unique_officers = df["Marked to Officer"].nunique()
most_urgent_total = int((df["Priority"] == "Most Urgent").sum())

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Tasks", total_tasks)
with c2:
    st.metric("Pending (In progress)", total_pending)
with c3:
    st.metric("Officers", unique_officers)
with c4:
    st.metric("Most Urgent", most_urgent_total)

st.markdown("---")

# ------------------------------
# Page: Dashboard Overview
# ------------------------------
if page == "Dashboard Overview":
    st.header("📈 Dashboard Overview")

    # Status distribution (pie or bar)
    st.subheader("Status Distribution")
    if "Status" in df.columns:
        status_counts = df["Status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        fig_status = px.pie(status_counts, values="Count", names="Status", title="Tasks by Status", hole=0.3)
        st.plotly_chart(fig_status, use_container_width=True)
    else:
        st.info("No Status column available to show distribution.")

    # Priority distribution
    st.subheader("Priority Distribution (all tasks)")
    if "Priority" in df.columns:
        pri_counts = df["Priority"].value_counts().reindex(["Most Urgent", "High", "Medium", "Low"]).fillna(0).astype(int).reset_index()
        pri_counts.columns = ["Priority", "Count"]
        fig_pri = px.bar(pri_counts, x="Priority", y="Count", text="Count", title="All Tasks by Priority")
        fig_pri.update_traces(textposition="outside")
        st.plotly_chart(fig_pri, use_container_width=True)
    else:
        st.info("No Priority column present.")

    # Show some upcoming deadlines (if Entry Date parsed)
    st.subheader("Upcoming Entries (by Entry Date Parsed)")
    if "Entry Date Parsed" in df.columns and df["Entry Date Parsed"].str.strip().ne("").any():
        df_deadline = df[df["Entry Date Parsed"].astype(str) != ""].copy()
        # convert to datetime where possible (Entry Date Parsed currently stored as string representation of date)
        try:
            df_deadline["ED_dt"] = pd.to_datetime(df_deadline["Entry Date Parsed"], errors="coerce")
            upcoming = df_deadline.sort_values("ED_dt").head(7)
            show_cols = ["Sr_display", "Marked to Officer", "Subject", "Priority", "Status", "Entry Date", "File Link HTML"]
            av = [c for c in show_cols if c in upcoming.columns]
            display_upcoming = upcoming[av].rename(columns={"Sr_display": "Sr", "File Link HTML": "File"})
            st.markdown(display_upcoming.to_html(escape=False, index=False), unsafe_allow_html=True)
        except Exception:
            st.info("Could not parse Entry Date Parsed column to show upcoming entries.")
    else:
        st.info("No parsed 'Entry Date' values available to show upcoming entries.")

    # Quick table preview
    st.subheader("Sample Tasks")
    st.dataframe(df.head(10), use_container_width=True)

# ------------------------------
# Page: Priority-wise Analysis & Export (merged with All Tasks)
# ------------------------------
elif page == "Priority-wise Analysis & Export":
    st.header("🚨 Priority-wise Analysis & Export")

    # Sidebar filters (only appear on this page - implemented in main UI section here)
    st.sidebar.markdown("---")
    st.sidebar.header("Filters (Priority-wise Analysis & Export)")
    priority_options = sorted(df["Priority"].unique().tolist(), key=lambda x: ["Most Urgent","High","Medium","Low"].index(x) if x in ["Most Urgent","High","Medium","Low"] else 99)
    status_options = sorted(df["Status"].unique().tolist())
    officer_options = sorted(df["Marked to Officer"].unique().tolist())

    sel_priorities = st.sidebar.multiselect("Select Priorities", options=priority_options, default=priority_options)
    sel_status = st.sidebar.multiselect("Select Status", options=status_options, default=status_options)
    sel_officers = st.sidebar.multiselect("Select Officers", options=officer_options, default=officer_options)
    txt_search = st.sidebar.text_input("Search Subject or Remarks (substring)", value="")

    # Apply filters
    filtered = df.copy()
    if sel_priorities:
        filtered = filtered[filtered["Priority"].isin(sel_priorities)]
    if sel_status:
        filtered = filtered[filtered["Status"].isin(sel_status)]
    if sel_officers:
        filtered = filtered[filtered["Marked to Officer"].isin(sel_officers)]
    if txt_search.strip():
        q = txt_search.strip().lower()
        mask_sub = filtered["Subject"].astype(str).str.lower().str.contains(q, na=False)
        mask_rem = filtered["Remarks"].astype(str).str.lower().str.contains(q, na=False)
        filtered = filtered[mask_sub | mask_rem]

    # Top metrics for filtered set
    st.subheader("Filtered Metrics")
    total_filtered = len(filtered)
    most_urgent_filtered = int((filtered["Priority"] == "Most Urgent").sum())
    high_filtered = int((filtered["Priority"] == "High").sum())
    medium_filtered = int((filtered["Priority"] == "Medium").sum())

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total (filtered)", total_filtered)
    with m2:
        st.metric("Most Urgent", most_urgent_filtered, delta=f"{(most_urgent_filtered/total_filtered*100):.1f}%" if total_filtered else "0%")
    with m3:
        st.metric("High", high_filtered, delta=f"{(high_filtered/total_filtered*100):.1f}%" if total_filtered else "0%")
    with m4:
        st.metric("Medium", medium_filtered, delta=f"{(medium_filtered/total_filtered*100):.1f}%" if total_filtered else "0%")

    st.markdown("---")

    # For each priority, show officer-wise distribution with values on bars
    priority_order = ["Most Urgent", "High", "Medium", "Low"]
    color_map = {"Most Urgent":"#ff4b4b","High":"#ff8c00","Medium":"#ffd700","Low":"#94d2bd"}

    for p in priority_order:
        st.subheader(f"{p} Priority Tasks - Officer-wise Distribution")
        p_df = filtered[filtered["Priority"] == p]
        if p_df.empty:
            st.info(f"No {p} priority tasks available for selected filters.")
            continue
        counts_by_officer = p_df.groupby("Marked to Officer").size().reset_index(name="Task Count").sort_values("Task Count", ascending=True)
        fig = px.bar(
            counts_by_officer,
            x="Task Count",
            y="Marked to Officer",
            orientation="h",
            text="Task Count",
            labels={"Task Count": "Number of Tasks", "Marked to Officer": "Officer"},
            color_discrete_sequence=[color_map.get(p, "#636EFA")]
        )
        fig.update_traces(textposition="outside")
        # add numeric annotation to ensure visibility
        for _, r in counts_by_officer.iterrows():
            fig.add_annotation(
                x=r["Task Count"] + 0.05,
                y=r["Marked to Officer"],
                text=str(int(r["Task Count"])),
                showarrow=False,
                xanchor="left",
                font=dict(size=11)
            )
        fig.update_layout(height=360, margin=dict(l=140))
        st.plotly_chart(fig, use_container_width=True)

        # Details table for this priority (render anchors)
        with st.expander(f"View {p} Priority Task Details ({len(p_df)} rows)"):
            cols_show = ["Sr_display","Marked to Officer","Subject","Entry Date","File Link HTML","Remarks","Status"]
            av = [c for c in cols_show if c in p_df.columns]
            df_show = p_df[av].rename(columns={"Sr_display":"Sr","File Link HTML":"File"})
            st.markdown(df_show.to_html(escape=False, index=False), unsafe_allow_html=True)

        st.markdown("---")

    # Overall priority distribution pie
    st.subheader("📊 Overall Priority Distribution (Filtered)")
    overall_counts = filtered["Priority"].value_counts().reset_index()
    overall_counts.columns = ["Priority","Count"]
    if not overall_counts.empty:
        fig_pie = px.pie(overall_counts, names="Priority", values="Count", title="Filtered Distribution by Priority", color_discrete_map=color_map)
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No tasks to visualize for selected filters.")

    st.markdown("---")
    # Merged "All Tasks" table & Export functionality (integrated here)
    st.subheader("📋 All Tasks (Filtered View & Export)")
    show_cols = ["Sr_display","Marked to Officer","Priority","Status","Subject","Entry Date","File Link HTML","Remarks"]
    avail_cols = [c for c in show_cols if c in filtered.columns]
    if filtered.empty:
        st.info("No tasks to display for current filters.")
    else:
        df_table = filtered[avail_cols].rename(columns={"Sr_display":"Sr","File Link HTML":"File"})
        st.markdown(df_table.to_html(escape=False, index=False), unsafe_allow_html=True)

        # Export options
        st.markdown("**Export Options**")
        # CSV download link (base64 anchor)
        st.markdown(df_to_csv_base64_link(filtered, filename="tasks_filtered_export.csv"), unsafe_allow_html=True)
        # Excel download via download_button (binary)
        df_to_excel_download_button(filtered, filename="tasks_filtered_export.xlsx")

    # Optionally show full raw underlying table (toggle)
    if st.checkbox("Show raw underlying filtered DataFrame (debug)", value=False):
        st.dataframe(filtered, use_container_width=True)

# ------------------------------
# Page: About / Help
# ------------------------------
else:
    st.header("ℹ️ About & Help")
    st.markdown("""
    **Task Management Dashboard**
    - Sidebar contains the page navigation.
    - The \"Priority-wise Analysis & Export\" page includes both the analysis charts and the full filtered table with export options.
    - Filters are shown only on the \"Priority-wise Analysis & Export\" page (per your request).
    - Clickable file links appear as `📎 filename` which open in a new tab. If the sheet contains only filenames (e.g., `52.pdf`), the app links them to Google Drive search for that filename.
    - If your Google Sheet is private, the gviz CSV endpoint will not work. Use \"Force sample data\" to test locally or ask for a version using the Google Sheets API (service account).
    """)

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and Plotly. If you want alerts (email/Slack) for new Most Urgent tasks, or Google Sheets API support for private sheets, I can add that.")
