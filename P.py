import streamlit as st
import pandas as pd
import plotly.express as px

# ================== LOAD DATA ==================
sheet_id = "14-idXJHzHKCUQxxaqGZi-6S0G20gvPUhK4G16ci2FwI"
sheet_name = "Sheet1"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

@st.cache_data
def load_data():
    df = pd.read_csv(url)
    # Remove rows where Status is blank or unmarked
    df = df[df["Status"].notna() & (df["Status"].str.strip() != "")]
    return df

df = load_data()

# ================== SIDEBAR NAVIGATION ==================
st.sidebar.title("📊 Dashboard Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Task List"])

# ================== PAGE 1 : DASHBOARD ==================
if page == "Dashboard":
    st.title("📌 Task Dashboard")

    # KPI Metrics
    total_tasks = df.shape[0]
    completed_tasks = df[df["Status"] == "Completed"].shape[0]
    pending_tasks = df[df["Status"] == "Pending"].shape[0]
    urgent_tasks = df[df["Urgency"] == "High"].shape[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Tasks", total_tasks)
    col2.metric("✅ Completed", completed_tasks)
    col3.metric("⏳ Pending", pending_tasks)
    col4.metric("⚡ Urgent", urgent_tasks)

    st.markdown("---")

    # Chart: Task Status Distribution
    status_counts = df["Status"].value_counts().reset_index()
    status_counts.columns = ["Status", "Count"]

    fig1 = px.pie(status_counts, names="Status", values="Count", title="Task Status Distribution")
    st.plotly_chart(fig1, use_container_width=True)

    # Chart: Urgency vs Status
    if "Urgency" in df.columns:
        urgency_counts = df.groupby(["Urgency", "Status"]).size().reset_index(name="Count")
        fig2 = px.bar(
            urgency_counts,
            x="Urgency",
            y="Count",
            color="Status",
            barmode="group",
            title="Urgency vs Status"
        )
        st.plotly_chart(fig2, use_container_width=True)

# ================== PAGE 2 : TASK LIST ==================
elif page == "Task List":
    st.title("📋 Task List")

    # Filters
    status_filter = st.selectbox("Filter by Status", options=["All"] + df["Status"].unique().tolist())
    urgency_filter = st.selectbox("Filter by Urgency", options=["All"] + df["Urgency"].dropna().unique().tolist())

    filtered_df = df.copy()
    if status_filter != "All":
        filtered_df = filtered_df[filtered_df["Status"] == status_filter]
    if urgency_filter != "All":
        filtered_df = filtered_df[filtered_df["Urgency"] == urgency_filter]

    st.dataframe(filtered_df, use_container_width=True)

    # Download Option
    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Filtered Data",
        data=csv,
        file_name="task_list.csv",
        mime="text/csv"
    )
