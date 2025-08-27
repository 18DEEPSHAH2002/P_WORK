import streamlit as st
import pandas as pd
import plotly.express as px

# Google Sheets CSV export link
sheet_url = "https://docs.google.com/spreadsheets/d/14-idXJHzHKCUQxxaqGZi-6S0G20gvPUhK4G16ci2FwI/export?format=csv&gid=213021534"

# Load data
df = pd.read_csv(sheet_url)

# Clean the data
df["Status"] = df["Status"].fillna("").astype(str).str.strip()
df = df[df["Status"].str.lower() != ""]   # keep only non-blank
pending_df = df[df["Status"].str.contains("progress", case=False)]

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Pending Tasks Overview", "Priority Insights"])

# ------------------- PAGE 1 -------------------
if page == "Pending Tasks Overview":
    st.title("📊 Pending Tasks by Officer")

    officer_pending = pending_df["Marked to Officer"].value_counts().reset_index()
    officer_pending.columns = ["Officer", "Pending Tasks"]

    # Bar chart
    fig = px.bar(officer_pending, 
                 x="Officer", y="Pending Tasks", 
                 title="Pending Tasks per Officer",
                 text="Pending Tasks")
    st.plotly_chart(fig)

    # Table
    st.subheader("Pending Tasks Count by Officer")
    st.dataframe(officer_pending)

    # Officer task detail view
    st.subheader("🔍 Task Details by Officer")
    officer_choice = st.selectbox("Select Officer:", officer_pending["Officer"].unique())

    officer_tasks = pending_df[pending_df["Marked to Officer"] == officer_choice][[
        "Marked to Officer", "Priority", "Subject", "File Entry Date", "Received From", "Status"
    ]]

    # Make file link clickable (assuming "File" column exists with Google Drive links)
    if "File" in pending_df.columns:
        officer_tasks["File"] = officer_tasks["File"].apply(
            lambda x: f"[Open File]({x})" if str(x).startswith("http") else x
        )

    st.write(officer_tasks.to_markdown(index=False), unsafe_allow_html=True)


# ------------------- PAGE 2 -------------------
elif page == "Priority Insights":
    st.title("📌 Priority Wise Pending Tasks")

    total_pending = len(pending_df)
    urgent_pending = len(pending_df[pending_df["Priority"].str.contains("Urgent", case=False, na=False)])
    medium_pending = len(pending_df[pending_df["Priority"].str.contains("Medium", case=False, na=False)])
    high_pending = len(pending_df[pending_df["Priority"].str.contains("High", case=False, na=False)])

    st.metric("Total Pending Tasks", total_pending)
    col1, col2, col3 = st.columns(3)
    col1.metric("Most Urgent Pending", urgent_pending)
    col2.metric("Medium Pending", medium_pending)
    col3.metric("High Pending", high_pending)

    # Officer-wise priority breakdowns
    st.subheader("Most Urgent Tasks by Officer")
    urgent_df = pending_df[pending_df["Priority"].str.contains("Urgent", case=False, na=False)]
    if not urgent_df.empty:
        urgent_chart = px.bar(urgent_df["Marked to Officer"].value_counts().reset_index(),
                              x="index", y="Marked to Officer",
                              title="Most Urgent Pending by Officer")
        st.plotly_chart(urgent_chart)

    st.subheader("Medium Tasks by Officer")
    medium_df = pending_df[pending_df["Priority"].str.contains("Medium", case=False, na=False)]
    if not medium_df.empty:
        medium_chart = px.bar(medium_df["Marked to Officer"].value_counts().reset_index(),
                              x="index", y="Marked to Officer",
                              title="Medium Pending by Officer")
        st.plotly_chart(medium_chart)

    st.subheader("High Tasks by Officer")
    high_df = pending_df[pending_df["Priority"].str.contains("High", case=False, na=False)]
    if not high_df.empty:
        high_chart = px.bar(high_df["Marked to Officer"].value_counts().reset_index(),
                            x="index", y="Marked to Officer",
                            title="High Pending by Officer")
        st.plotly_chart(high_chart)
