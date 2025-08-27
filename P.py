# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Optional, Dict

# ========= BASIC CONFIG =========
st.set_page_config(page_title="Pending Works Dashboard", layout="wide")

# ======== SHEET CONFIG (EDIT IF NEEDED) ========
SHEET_ID = "14-idXJHzHKCUQxxaqGZi-6S0G20gvPUhK4G16ci2FwI"
# Option A: by gid (preferred if you know it)
GID = "213021534"
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"
# Option B (fallback): by sheet name — uncomment and comment CSV_URL above if you prefer
# SHEET_NAME = "Sheet1"
# CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={SHEET_NAME}"

# ========= UTILITIES =========
def _norm(s: str) -> str:
    """normalize a column name for matching"""
    return " ".join(str(s).strip().lower().replace("\n", " ").split())

def pick_column(df: pd.DataFrame, candidates) -> Optional[str]:
    """
    Returns the *actual* column name in df that best matches any candidate names.
    Matching is case/space insensitive.
    """
    norm_map: Dict[str, str] = {_norm(c): c for c in df.columns}
    for cand in candidates:
        key = _norm(cand)
        if key in norm_map:
            return norm_map[key]
    # try contains matching for flexible cases
    for key, real in norm_map.items():
        for cand in candidates:
            if _norm(cand) in key:
                return real
    return None

@st.cache_data
def load_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    df.columns = [c.strip().replace("\n", " ") for c in df.columns]
    return df

def filter_blank_unmarked_status(df: pd.DataFrame, status_col: Optional[str]) -> pd.DataFrame:
    """Keep only rows where Status is Blank/Unmarked (the 'Star Sheet')."""
    if status_col is None:
        return df.copy()  # if no status col, just return as-is
    s = df[status_col].astype(str).str.strip().str.lower()
    # treat "", "blank", "unmarked", "na" as blank-like
    mask = (s == "") | (s == "blank") | (s == "unmarked")
    return df.loc[mask].copy()

def officer_counts_bar(data: pd.DataFrame, officer_col: str, title: str, y_label: str = "Pending Tasks"):
    counts = (
        data[officer_col]
        .dropna()
        .astype(str)
        .str.strip()
        .value_counts()
        .reset_index()
    )
    counts.columns = ["Officer", y_label]
    if counts.empty:
        st.info("No data to plot.")
        return counts
    fig = px.bar(
        counts,
        x="Officer",
        y=y_label,
        title=title,
        text=y_label,
    )
    fig.update_layout(xaxis={"categoryorder": "total descending"})
    st.plotly_chart(fig, use_container_width=True)
    return counts

def linkify_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Use Streamlit's LinkColumn if available; otherwise return df unchanged."""
    # We will use column_config in st.dataframe call; here just ensure strings are URLs or None
    out = df.copy()
    if col in out.columns:
        out[col] = out[col].apply(lambda x: x if (isinstance(x, str) and x.startswith("http")) else None)
    return out

# ========= LOAD DATA =========
df_raw = load_data(CSV_URL)

# ========= DETECT COLUMNS (robust to name variations) =========
officer_col  = pick_column(df_raw, ["Marked to Officer", "Marked to Nodal Officer", "Nodal Officer", "Officer", "Assigned Officer"])
status_col   = pick_column(df_raw, ["Status", "Current Status"])
priority_col = pick_column(df_raw, ["Priority", "Priority Level", "Urgency"])  # <-- handles your KeyError
subject_col  = pick_column(df_raw, ["Subject", "Task", "Task Description", "Description"])
file_col     = pick_column(df_raw, ["File", "File Link", "Drive Link", "Link", "Document Link"])
file_date_col= pick_column(df_raw, ["File Entry Date", "Entry Date", "Received Date", "Date of Receipt of letter", "Date"])
from_col     = pick_column(df_raw, ["Received From", "Sender", "From"])

# Keep only blank/unmarked status rows (Star Sheet)
df_star = filter_blank_unmarked_status(df_raw, status_col)

# ========= SIDEBAR NAV =========
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Pending Tasks Overview", "Priority Insights"])

# ========= PAGE 1: Pending Tasks Overview =========
if page == "Pending Tasks Overview":
    st.title("📊 Pending Tasks by Officer (Blank/Unmarked Status)")

    # Guardrails for missing columns
    if officer_col is None:
        st.error("❌ Could not find an 'Officer' column. Please ensure your sheet has a column like 'Marked to Officer' or 'Officer'.")
        st.caption(f"Available columns: {list(df_raw.columns)}")
        st.stop()

    # Bar chart of pending per officer
    officer_pending = officer_counts_bar(
        data=df_star,
        officer_col=officer_col,
        title="Pending Tasks per Officer",
        y_label="Pending Tasks"
    )

    # Table of counts (left: officer, right: count)
    st.subheader("Pending Tasks Count by Officer")
    st.dataframe(officer_pending, use_container_width=True)

    # Officer detail table with clickable File links
    st.subheader("🔍 Task Details by Officer")
    if not officer_pending.empty:
        officer_choice = st.selectbox("Select Officer:", officer_pending["Officer"].tolist())
        details = df_star.loc[df_star[officer_col].astype(str).str.strip() == officer_choice].copy()

        # Choose columns to display (only those present)
        cols_to_show = [c for c in [
            officer_col, priority_col, subject_col, file_date_col, from_col, status_col, file_col
        ] if c is not None and c in details.columns]

        if file_col in details.columns:
            details = linkify_column(details, file_col)

        # Render table; use LinkColumn for 'File'
        if file_col in details.columns:
            st.dataframe(
                details[cols_to_show],
                use_container_width=True,
                column_config={
                    file_col: st.column_config.LinkColumn(file_col, display_text="Open File")
                }
            )
        else:
            st.dataframe(details[cols_to_show], use_container_width=True)
    else:
        st.info("No pending tasks found for any officer.")

# ========= PAGE 2: Priority Insights =========
elif page == "Priority Insights":
    st.title("📌 Priority-wise Pending Tasks (Blank/Unmarked Status)")

    total_pending = len(df_star)

    # Build safe subsets for priorities
    def subset_for(keys):
        if priority_col is None or priority_col not in df_star.columns:
            return df_star.iloc[0:0].copy()
        patt = "|".join([pd.regex.escape(k) for k in keys])
        return df_star[df_star[priority_col].astype(str).str.contains(patt, case=False, na=False)]

    most_urgent_df = subset_for(["Most Urgent", "Urgent"])
    medium_df      = subset_for(["Medium"])
    high_df        = subset_for(["High"])

    colA, colB, colC, colD = st.columns(4)
    colA.metric("Total Pending", total_pending)
    colB.metric("Most Urgent Pending", len(most_urgent_df))
    colC.metric("Medium Pending", len(medium_df))
    colD.metric("High Pending", len(high_df))

    # Charts by officer for each priority group
    if officer_col is None:
        st.error("❌ Could not find an 'Officer' column to build charts.")
        st.caption(f"Available columns: {list(df_raw.columns)}")
        st.stop()

    st.subheader("Most Urgent Pending by Officer")
    if most_urgent_df.empty:
        st.info("No Most Urgent/Urgent tasks pending.")
    else:
        officer_counts_bar(most_urgent_df, officer_col, "Most Urgent Pending by Officer", y_label="Most Urgent")

    st.subheader("Medium Pending by Officer")
    if medium_df.empty:
        st.info("No Medium tasks pending.")
    else:
        officer_counts_bar(medium_df, officer_col, "Medium Pending by Officer", y_label="Medium Pending")

    st.subheader("High Pending by Officer")
    if high_df.empty:
        st.info("No High tasks pending.")
    else:
        officer_counts_bar(high_df, officer_col, "High Pending by Officer", y_label="High Pending")
