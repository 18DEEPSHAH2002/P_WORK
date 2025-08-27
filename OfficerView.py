import streamlit as st
import pandas as pd

# Google Sheet URL
sheet_url = "https://docs.google.com/spreadsheets/d/14-idXJHzHKCUQxxaqGZi-6S0G20gvPUhK4G16ci2FwI/export?format=csv&gid=213021534"

@st.cache_data
def load_data():
    return pd.read_csv(sheet_url)

df = load_data()

st.set_page_config(page_title="Urgent Tasks by Officer", layout="wide")
st.title("🚨 Urgent Tasks by Officer")

# Filter only urgent tasks
urgent_df = df[df["Urgency"].str.lower() == "urgent"]

if urgent_df.empty:
    st.info("✅ No urgent tasks available.")
else:
    officer_list = urgent_df["Marked to Officer"].unique()
    officer = st.selectbox("Select Officer", officer_list)

    officer_tasks = urgent_df[urgent_df["Marked to Officer"] == officer]

    st.subheader(f"Tasks assigned to **{officer}**")
    st.dataframe(officer_tasks[["Task ID", "Task Description", "Status", "Due Date"]])

    st.metric("Total Urgent Tasks", len(officer_tasks))
