import streamlit as st
import pandas as pd

# --- Page config ---
st.set_page_config(page_title="Pending Tasks Dashboard", layout="wide")

st.title("Pending Tasks Dashboard")

# --- Load Google Sheet ---
sheet_id = "14-idXJHzHKCUQxxaqGZi-6S0G20gvPUhK4G16ci2FwI"
sheet_name = "Sheet1"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
df = pd.read_csv(url)

# --- Clean column names ---
df.columns = df.columns.str.strip()

# --- Filter pending tasks ---
pending_df = df[df['Status'] != 'Completed']

# --- Select Nodal Officer ---
officer_list = pending_df['Marked to Officer'].unique().tolist()
selected_officer = st.selectbox("Select Nodal Officer:", officer_list)

# --- Filter tasks for selected officer ---
officer_tasks = pending_df[pending_df['Marked to Officer'] == selected_officer]

# --- Display tasks ---
st.subheader(f"Pending Tasks for {selected_officer}")

# Check if 'File Link' exists to show it
columns_to_show = ['Task Name', 'Status']  # Replace 'Task Name' with your actual column name
if 'File Link' in officer_tasks.columns:
    columns_to_show.append('File Link')

# Make links clickable if 'File Link' exists
if 'File Link' in columns_to_show:
    officer_tasks_display = officer_tasks[columns_to_show].copy()
    officer_tasks_display['File Link'] = officer_tasks_display['File Link'].apply(
        lambda x: f"[Link]({x})" if pd.notna(x) else ""
    )
    st.write(officer_tasks_display.to_html(escape=False), unsafe_allow_html=True)
else:
    st.dataframe(officer_tasks[columns_to_show])
