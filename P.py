import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------------
# Load Google Sheet Data
# -----------------------------
@st.cache_data
def load_data():
    sheet_id = "14-idXJHzHKCUQxxaqGZi-6S0G20gvPUhK4G16ci2FwI"
    sheet_name = "Sheet1"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    df = pd.read_csv(url)
    return df

df = load_data()

# -----------------------------
# Streamlit Dashboard Layout
# -----------------------------
st.set_page_config(page_title="Pending Works Dashboard", layout="wide")

st.title("📊 Pending Works Dashboard")

# Sidebar Navigation
page = st.sidebar.radio("Navigate", ["📌 Pending Works Overview", "🚨 Urgent Pending Works"])

# -----------------------------
# Page 1: Pending Works Overview
# -----------------------------
if page == "📌 Pending Works Overview":
    st.header("Pending Works Overview")

    # Summary Stats
    total_tasks = len(df)
    pending_tasks = df["Status"].str.contains("Pending", case=False, na=False).sum()
    completed_tasks = df["Status"].str.contains("Completed", case=False, na=False).sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Tasks", total_tasks)
    col2.metric("Pending Tasks", pending_tasks)
    col3.metric("Completed Tasks", completed_tasks)

    # Chart: Pending Works by Officer
    if "Marked to Officer" in df.columns:
        officer_chart = px.bar(
            df["Marked to Officer"].value_counts().reset_index(),
            x="index", y="Marked to Officer",
            title="Pending Works by Officer",
            labels={"index": "Officer", "Marked to Officer": "Number of Tasks"}
        )
        st.plotly_chart(officer_chart, use_container_width=True)

    # Chart: Pending Works by Department
    if "Department" in df.columns:
        dept_chart = px.pie(
            df,
            names="Department",
            title="Pending Works by Department"
        )
        st.plotly_chart(dept_chart, use_container_width=True)

# -----------------------------
# Page 2: Urgent Pending Works
# -----------------------------
elif page == "🚨 Urgent Pending Works":
    st.header("🚨 Urgent Pending Works by Officer")

    # Filter urgent tasks
    urgent_df = df[df["Priority"].str.contains("Urgent", case=False, na=False)]

    if urgent_df.empty:
        st.warning("✅ No urgent pending works found.")
    else:
        urgent_chart = px.bar(
            urgent_df["Marked to Officer"].value_counts().reset_index(),
            x="index", y="Marked to Officer",
            title="Most Urgent Pending Works by Officer",
            labels={"index": "Officer", "Marked to Officer": "Urgent Tasks"}
        )
        st.plotly_chart(urgent_chart, use_container_width=True)

        st.subheader("Urgent Task Details")
        st.dataframe(urgent_df, use_container_width=True)
