import streamlit as st
import pandas as pd
import plotly.express as px

# Google Sheets CSV export link
SHEET_URL = "https://docs.google.com/spreadsheets/d/14-idXJHzHKCUQxxaqGZi-6S0G20gvPUhK4G16ci2FwI/export?format=csv&gid=213021534"

# Load data
@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip().str.replace("\n", " ", regex=True)

    # Keep only rows where Status is blank/unmarked (pending work)
    df["Status"] = df["Status"].fillna("").astype(str).str.strip()
    df = df[df["Status"] == ""]

    return df

df = load_data(SHEET_URL)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Pending Tasks Overview", "Priority Insights"])

# Helper: Officer bar chart
def officer_bar_chart(data, title):
    officer_counts = data["Marked to Officer"].value_counts().reset_index()
    officer_counts.columns = ["Officer", "Pending Tasks"]

    fig = px.bar(
        officer_counts, x="Officer", y="Pending Tasks",
        title=title, text="Pending Tasks",
        color="Pending Tasks", color_continuous_scale="Blues"
    )
    fig.update_layout(xaxis={'categoryorder':'total descending'})
    st.plotly_chart(fig, use_container_width=True)
    return officer_counts

# Helper: make file links clickable
def make_clickable(val):
    if isinstance(val, str) and val.startswith("http"):
        return f'<a href="{val}" target="_blank">Open File</a>'
    return val

# ------------------- PAGE 1 -------------------
if page == "Pending Tasks Overview":
    st.title("📊 Pending Tasks by Officer")

    if "Marked to Officer" not in df.columns:
        st.error("❌ 'Marked to Officer' column missing in sheet")
    else:
        officer_pending = officer_bar_chart(df, "Pending Tasks per Officer")

        # Table
        st.subheader("Pending Tasks Count by Officer")
        st.dataframe(officer_pending, use_container_width=True)

        # Officer task detail view
        st.subheader("🔍 Task Details by Officer")
        officer_choice = st.selectbox("Select Officer:", officer_pending["Officer"].unique())

        officer_tasks = df[df["Marked to Officer"] == officer_choice]

        columns_to_show = [c for c in [
            "Marked to Officer", "Priority", "Subject",
            "File Entry Date", "Received From", "Status", "File"
        ] if c in officer_tasks.columns]

        if "File" in officer_tasks.columns:
            officer_tasks = officer_tasks.copy()
            officer_tasks["File"] = officer_tasks["File"].apply(make_clickable)

        st.write(officer_tasks[columns_to_show].to_html(escape=False, index=False), unsafe_allow_html=True)

        # Download button
        st.download_button(
            label="⬇️ Download Officer Tasks (CSV)",
            data=officer_tasks.to_csv(index=False).encode("utf-8"),
            file_name=f"{officer_choice}_tasks.csv",
            mime="text/csv"
        )

# ------------------- PAGE 2 -------------------
elif page == "Priority Insights":
    st.title("📌 Priority Wise Pending Tasks")

    total_pending = len(df)
    st.metric("Total Pending Tasks", total_pending)

    # Show counts for all priorities dynamically
    st.subheader("📋 Tasks by Priority")
    priority_counts = df["Priority"].value_counts().reset_index()
    priority_counts.columns = ["Priority", "Count"]

    st.dataframe(priority_counts, use_container_width=True)

    # Metrics row
    cols = st.columns(len(priority_counts))
    for i, row in priority_counts.iterrows():
        cols[i].metric(row["Priority"], row["Count"])

    # Officer-wise charts
    for priority in priority_counts["Priority"]:
        st.subheader(f"{priority} Tasks by Officer")
        subset = df[df["Priority"].str.contains(priority, case=False, na=False)]
        if not subset.empty:
            officer_bar_chart(subset, f"{priority} Pending by Officer")
        else:
            st.info(f"No {priority} tasks pending.")
