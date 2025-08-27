import streamlit as st
import pandas as pd
import plotly.express as px

# Google Sheet URL
sheet_url = "https://docs.google.com/spreadsheets/d/14-idXJHzHKCUQxxaqGZi-6S0G20gvPUhK4G16ci2FwI/export?format=csv&gid=213021534"

@st.cache_data
def load_data():
    return pd.read_csv(sheet_url)

# Load data
df = load_data()

st.set_page_config(page_title="Task Dashboard", layout="wide")
st.title("📊 Task Monitoring Dashboard")

# -------------------------------
# Summary Metrics
# -------------------------------
total_tasks = len(df)
pending_tasks = len(df[df["Status"].str.lower() == "pending"])
completed_tasks = len(df[df["Status"].str.lower() == "completed"])
urgent_tasks = len(df[df["Urgency"].str.lower() == "urgent"])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Tasks", total_tasks)
col2.metric("Pending", pending_tasks)
col3.metric("Completed", completed_tasks)
col4.metric("Urgent", urgent_tasks)

# -------------------------------
# Task Status Overview
# -------------------------------
status_chart = px.pie(
    df,
    names="Status",
    title="Task Status Distribution"
)
st.plotly_chart(status_chart, use_container_width=True)

# -------------------------------
# Urgent Tasks Trend
# -------------------------------
urgent_df = df[df["Urgency"].str.lower() == "urgent"]

if not urgent_df.empty:
    urgent_chart = px.bar(
        urgent_df["Marked to Officer"].value_counts().reset_index(),
        x="index", y="Marked to Officer",
        labels={"index": "Officer", "Marked to Officer": "No. of Urgent Tasks"},
        title="Urgent Tasks by Officer"
    )
    st.plotly_chart(urgent_chart, use_container_width=True)
else:
    st.info("✅ No urgent tasks found!")
