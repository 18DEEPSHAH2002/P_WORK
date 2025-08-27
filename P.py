import streamlit as st
import pandas as pd
import plotly.express as px


SHEET_URL = "https://docs.google.com/spreadsheets/d/14-idXJHzHKCUQxxaqGZi-6S0G20gvPUhK4G16ci2FwI/export?format=csv&gid=213021534"
# Load Data
@st.cache_data
def load_data():
    df = pd.read_excel(SHEET_URL, sheet_name="Star Marked Letters")
    # Filter only non-blank/Unmarked Sr rows, keep meaningful tasks
    df = df[df["Sr"].notna()]
    df = df[df["Sr"].astype(str).str.strip() != ""]
    return df

def make_file_link(row):
    if isinstance(row["File"], str) and ".pdf" in row["File"]:
        # Make downloadable link for the PDF files
        return f'<a href="https://drive.google.com/file/d/{row["File"].replace(".pdf","")}/view" target="_blank">{row["File"]}</a>'
    else:
        return ""

df = load_data()

# Only "In progress" tasks
pending_df = df[df["Status"].str.strip().str.lower() == "in progress"]

### ---------- Page 1: Officer Wise Pending Analysis ----------

def page1():
    st.header("Officer Pending Task Analysis")

    # Officer-wise pending count
    pending_counts = pending_df["Marked to Officer"].value_counts().sort_values(ascending=False)
    officers = pending_counts.index.tolist()
    counts = pending_counts.values.tolist()

    # Bar Chart
    fig = px.bar(x=officers, y=counts, labels={'x':'Officer Name','y':'Pending Tasks'},
                 title="Pending Tasks per Officer", color=counts, color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)

    # Table below bar chart
    st.subheader("Pending Task Table (Officer-wise)")
    table_df = pd.DataFrame({'Nodal Officer': officers, 'Number of Pending Tasks': counts})
    st.dataframe(table_df, use_container_width=True, hide_index=True)

    st.subheader("Select Officer to View Pending Task Details:")
    selected_officer = st.selectbox("Choose Officer", officers)
    officer_tasks = pending_df[pending_df["Marked to Officer"] == selected_officer]

    # Display details with file links
    st.markdown(f"### Pending Tasks for Officer: {selected_officer}")
    display_df = officer_tasks.copy()
    display_df["File Link"] = officer_tasks.apply(make_file_link, axis=1)
    display_df_show = display_df[["Sr","Priority","Dealing Branch ","Subject","Received From","File Link","Entry Date","Status","Remarks"]]
    st.markdown(display_df_show.to_html(escape=False, index=False), unsafe_allow_html=True)

### ---------- Page 2: Priority and Officerwise Analytics ----------

def page2():
    st.header("Priority-wise Pending Analytics")

    # Top metrics
    total_pending = pending_df.shape[0]
    most_urgent = pending_df[pending_df["Priority"].str.contains("Most Urgent",case=False)].shape[0]
    medium = pending_df[pending_df["Priority"].str.contains("Medium",case=False)].shape[0]
    high = pending_df[pending_df["Priority"].str.contains("High",case=False)].shape[0]
    st.columns(1)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Pending",total_pending)
    col2.metric("Most Urgent Pending",most_urgent)
    col3.metric("Medium Pending",medium)
    col4.metric("High Pending",high)
    st.markdown("---")

    def priority_chart_data(priority):
        return pending_df[pending_df["Priority"].str.contains(priority, case=False)]["Marked to Officer"].value_counts()

    # Charts
    for priority, color in zip(["Most Urgent","Medium","High"],["crimson","orange","green"]):
        count_data = priority_chart_data(priority)
        fig = px.bar(x=count_data.index, y=count_data.values, labels={'x':'Officer','y':f'{priority} Tasks'},
                     title=f"{priority} Priority Tasks Officer-wise", color=count_data.values, color_continuous_scale=[color])
        st.plotly_chart(fig, use_container_width=True)

# ------ Sidebar Navigation -------
st.sidebar.title("Navigation")
selected_page = st.sidebar.radio("Select Page", ["Officer Analytics","Priority Analysis"])

if selected_page == "Officer Analytics":
    page1()
elif selected_page == "Priority Analysis":
    page2()
