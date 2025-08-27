import streamlit as st
import pandas as pd
import plotly.express as px

# ---------------------------
# Load Data from Google Sheets
# ---------------------------
sheet_id = "14-idXJHzHKCUQxxaqGZi-6S0G20gvPUhK4G16ci2FwI"
sheet_name = "Sheet1"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

@st.cache_data
def load_data():
    return pd.read_csv(url)

df = load_data()

# ---------------------------
# Sidebar Navigation
# ---------------------------
st.sidebar.title("📊 Dashboard Navigation")
page = st.sidebar.selectbox("Go to:", ["Overview", "Urgent Tasks"])

# ---------------------------
# Page 1: Overview
# ---------------------------
if page == "Overview":
    st.title("📌 Dashboard Overview")

    st.write("This dashboard shows pending and urgent tasks assigned to officers.")

    # Summary KPIs
    total_tasks = len(df)
    urgent_tasks = df[df["Urgency"] == "High"].shape[0]
    officers = df["Marked to Officer"].nunique()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Tasks", total_tasks)
    col2.metric("Urgent Tasks", urgent_tasks)
    col3.metric("Officers Assigned", officers)

    # Tasks by Officer
    st.subheader("📈 Tasks by Officer")
    officer_chart = px.bar(
        df["Marked to Officer"].value_counts().reset_index(),
        x="index", y="Marked to Officer",
        title="Tasks Assigned per Officer",
        labels={"index": "Officer", "Marked to Officer": "No. of Tasks"},
        color="Marked to Officer",
        text="Marked to Officer"
    )
    st.plotly_chart(officer_chart, use_container_width=True)

# ---------------------------
# Page 2: Urgent Tasks
# ---------------------------
elif page == "Urgent Tasks":
    st.title("⏳ Most Urgent Pending Tasks by Officer")

    urgent_df = df[df["Urgency"] == "High"]

    if urgent_df.empty:
        st.warning("✅ No urgent tasks found.")
    else:
        urgent_chart = px.bar(
            urgent_df["Marked to Officer"].value_counts().reset_index(),
            x="index", y="Marked to Officer",
            title="Most Urgent Pending by Officer",
            labels={"index": "Officer", "Marked to Officer": "Urgent Tasks"},
            color="Marked to Officer",
            text="Marked to Officer"
        )
        st.plotly_chart(urgent_chart, use_container_width=True)

        st.subheader("📋 Urgent Task Details")
        st.dataframe(urgent_df)
