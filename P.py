import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# --- Page config ---
st.set_page_config(page_title="Pending Tasks Dashboard", layout="wide")

# --- Load Google Sheet ---
sheet_id = "14-idXJHzHKCUQxxaqGZi-6S0G20gvPUhK4G16ci2FwI"
sheet_name = "Sheet1"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
df = pd.read_csv(url)

# --- Clean column names to remove spaces ---
df.columns = df.columns.str.strip()

# --- Filter pending tasks ---
pending_df = df[df['Status'] != 'Completed']

# --- Count pending tasks per officer ---
task_counts = pending_df['Marked to Officer'].value_counts()
pending_table = task_counts.reset_index()
pending_table.columns = ['Nodal Officer', 'Pending Tasks']

# --- Streamlit layout ---
st.title("Pending Tasks Dashboard")

# Display table
st.subheader("Pending Tasks Table")
st.dataframe(pending_table)

# Display bar chart
st.subheader("Pending Tasks Bar Graph")
fig, ax = plt.subplots(figsize=(10,6))
bars = ax.bar(task_counts.index, task_counts.values, color='skyblue')

# Highlight officer with max pending tasks
max_index = task_counts.idxmax()
bars[list(task_counts.index).index(max_index)].set_color('red')

# Add values on top of bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height, str(height),
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel("Nodal Officer")
ax.set_ylabel("Pending Tasks")
ax.set_title("Pending Tasks per Officer")
plt.xticks(rotation=45)

st.pyplot(fig)

# --- New: Selection option for officer ---
st.subheader("View Pending Tasks for a Specific Officer")
officer_list = pending_table['Nodal Officer'].tolist()
selected_officer = st.selectbox("Choose Officer:", officer_list)

# Filter pending tasks for selected officer
officer_tasks = pending_df[pending_df['Marked to Officer'] == selected_officer]

# Display tasks table with optional file link if present
columns_to_show = ['Task Name', 'Status']  # Replace 'Task Name' with your actual column
if 'File Link' in officer_tasks.columns:
    columns_to_show.append('File Link')

st.dataframe(officer_tasks[columns_to_show])
