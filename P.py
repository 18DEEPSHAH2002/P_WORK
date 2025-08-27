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

# --- Filter pending tasks ---
pending_df = df[df['Status'] != 'Completed']

# --- Count pending tasks per officer ---
task_counts = pending_df['Marked to Officer'].value_counts()
pending_table = task_counts.reset_index()
pending_table.columns = ['Nodal Officer', 'Pending Tasks']

# --- Streamlit layout ---
st.title("Pending Tasks Dashboard")

# --- Select officer ---
st.subheader("Select Nodal Officer")
officer_list = pending_table['Nodal Officer'].tolist()
selected_officer = st.selectbox("Choose Officer:", officer_list)

# --- Display selected officer info ---
st.subheader(f"Pending Tasks for {selected_officer}")
officer_tasks = pending_df[pending_df['Marked to Officer'] == selected_officer]

# Display table of pending tasks with file links
if 'File Link' in officer_tasks.columns:
    st.dataframe(officer_tasks[['Task Name', 'Status', 'File Link']])
else:
    st.dataframe(officer_tasks[['Task Name', 'Status']])

# --- Display bar chart of all officers ---
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
