import pandas as pd
from IPython.display import display, HTML

# --- Load Google Sheet ---
sheet_id = "14-idXJHzHKCUQxxaqGZi-6S0G20gvPUhK4G16ci2FwI"
sheet_name = "Sheet1"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

df = pd.read_csv(url)

# --- Clean column names ---
df.columns = df.columns.str.strip()

# --- Filter pending tasks ---
pending_df = df[df['Status'] != 'Completed']

# --- Specify the officer ---
a = "Officer Name"  # Replace with the officer's exact name

# --- Filter tasks for that officer ---
officer_tasks = pending_df[pending_df['Marked to Officer'] == a]

# --- Columns to display ---
columns_to_show = ['Sr', 'Priority', 'Dealing Branch', 'Subject', 'Received From', 'File', 'Entry Date', 'Status', 'Response Recieved']

officer_tasks_display = officer_tasks[columns_to_show].copy()

# --- Make File links clickable ---
def make_clickable(link):
    if pd.isna(link) or link.strip() == "":
        return ""
    # If the link is a Google Drive "sharing link", convert to direct download
    if "drive.google.com" in link and "id=" in link:
        file_id = link.split("id=")[-1]
        return f"<a href='https://drive.google.com/uc?id={file_id}' target='_blank'>Open File</a>"
    # Otherwise, assume it's a normal URL
    if link.startswith("http"):
        return f"<a href='{link}' target='_blank'>Open File</a>"
    # Otherwise, just display text
    return str(link)

officer_tasks_display['File'] = officer_tasks_display['File'].apply(make_clickable)

# --- Display the table ---
display(HTML(officer_tasks_display.to_html(escape=False, index=False)))
