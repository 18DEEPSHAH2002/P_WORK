import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from datetime import datetime

# --- Page Configuration ---
# Set the layout and appearance of the Streamlit page.
st.set_page_config(
    page_title="Task Management Dashboard",
    page_icon="✅",
    layout="wide",
)

# --- Caching Function for Data Loading ---
# Using st.cache_data ensures that the data is loaded only once,
# speeding up the app's performance on subsequent runs.
@st.cache_data
def load_data(url):
    """
    Loads and preprocesses data from the specified Google Sheet URL.
    It renames columns, cleans data, standardizes priorities, and calculates pending days.
    """
    try:
        # The URL is the direct CSV export link.
        df = pd.read_csv(url)

        # --- Data Cleaning and Preparation ---
        # Strip any leading/trailing whitespace from all column names.
        df.columns = df.columns.str.strip()

        # --- UPDATE: Changed to 'Entry Date' to match the new column name from your sheet ---
        # Rename columns to be more script-friendly to match the sheet's actual headers
        df.rename(columns={
            'Entry Date': 'Start Date',
            'Marked to Officer': 'Assign To',
            'Status': 'Task Status' 
        }, inplace=True)

        # Convert date columns to datetime objects, handling potential errors
        df['Start Date'] = pd.to_datetime(df['Start Date'], errors='coerce')

        # --- UPDATE: Clean text columns and 'File' column separately ---
        # Clean columns that need to be lowercased for consistent filtering
        text_columns_to_lower = ['Priority', 'Task Status', 'Dealing Branch', 'Assign To']
        for col in text_columns_to_lower:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower()
                df[col].replace(['', 'nan'], 'unknown', inplace=True)

        # Clean the 'File' column separately to preserve the original URL case and content
        if 'File' in df.columns:
            df['File'] = df['File'].astype(str).str.strip()
            # Replace placeholder values so they don't appear as broken links
            df['File'].replace(['nan', 'unknown'], '', inplace=True)
        
        # --- FIX: Standardize Priority Values to handle variations ---
        # This function maps various text inputs (e.g., "high priority") to a standard set.
        def standardize_priority(priority_text):
            if 'most urgent' in priority_text:
                return 'most urgent'
            elif 'high' in priority_text:
                return 'high'
            elif 'medium' in priority_text:
                return 'medium'
            else:
                return 'unknown'

        # Apply the standardization function to the 'Priority' column.
        if 'Priority' in df.columns:
            df['Priority'] = df['Priority'].apply(standardize_priority)

        # --- Feature: Calculate Days Pending ---
        # Calculate the number of days a task has been pending from the 'Start Date'.
        df['Days Pending'] = (datetime.now() - df['Start Date']).dt.days
        df['Days Pending'] = df['Days Pending'].fillna(0).astype(int)

        return df
    except KeyError as e:
        st.error(f"Column Mismatch Error: Could not find the column {e} in your Google Sheet.")
        st.info("Please check your Google Sheet for the exact spelling and capitalization of the column headers.")
        try:
            temp_df = pd.read_csv(url)
            st.write("Here are the column names I found in your sheet:", temp_df.columns.tolist())
        except Exception as read_e:
            st.error(f"Could not read the columns from the sheet. Error: {read_e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        st.warning("Please ensure the Google Sheet is shared publicly ('Anyone with the link can view').")
        return pd.DataFrame()

# --- Data Loading ---
# Construct the direct CSV export URL from the sheet ID and GID.
sheet_id = "14howESk1k414yH06e_hG8mCE0HYUcR5VFTnbro4IdiU"
sheet_gid = "345729707"
GOOGLE_SHEET_URL = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={sheet_gid}"
df = load_data(GOOGLE_SHEET_URL)

# --- Sidebar Navigation ---
st.sidebar.title("📋 Navigation")
page = st.sidebar.radio("Go to", ["Officer Pending Tasks", "Task Priority Dashboard"])
st.sidebar.markdown("---")
st.sidebar.info("This dashboard provides an overview of pending tasks from the Google Sheet.")

# --- Main Application Logic ---
# Only proceed if the DataFrame was loaded successfully.
if not df.empty:
    # --- A task is pending if its status is NOT 'completed' ---
    pending_tasks_df = df[df['Task Status'] != 'completed'].copy()

    # --- UPDATE: Filter out 'unknown' or 'nan' assignments before displaying ---
    # This removes rows where the officer or department is unassigned from all calculations.
    pending_tasks_df = pending_tasks_df[
        ~pending_tasks_df['Assign To'].isin(['unknown', 'nan'])
    ].copy()


    # --- Page 1: Officer Pending Tasks ---
    if page == "Officer Pending Tasks":
        st.title("👨‍💼 Officer Pending Tasks Overview")
        st.markdown("This page shows the number and average age of pending tasks for each officer.")
        st.markdown("---")

        if not pending_tasks_df.empty:
            # Count pending tasks for each officer.
            officer_pending_counts = pending_tasks_df['Assign To'].value_counts().reset_index()
            officer_pending_counts.columns = ['Officer', 'Number of Pending Tasks']

            # Calculate Average Pending Days per officer
            avg_pending_days = pending_tasks_df.groupby('Assign To')['Days Pending'].mean().round(0).astype(int).reset_index()
            avg_pending_days.rename(columns={'Days Pending': 'Avg. Days Pending', 'Assign To': 'Officer'}, inplace=True)

            # Merge the counts and the average days into one summary table.
            officer_summary = pd.merge(officer_pending_counts, avg_pending_days, on='Officer')
            officer_summary['Officer'] = officer_summary['Officer'].str.title() # Capitalize for display

            # --- LAYOUT: Show graph first, then the table ---
            st.subheader("Visual Distribution")
            
            # --- UPDATE: Replaced Plotly with Matplotlib chart as requested ---
            task_counts = pending_tasks_df['Assign To'].value_counts()
            fig, ax = plt.subplots(figsize=(10,6))
            bars = ax.bar(task_counts.index, task_counts.values, color='skyblue')

            # Highlight officer with max pending tasks
            if not task_counts.empty:
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


            st.subheader("Pending Task Summary")
            st.dataframe(officer_summary, use_container_width=True, hide_index=True)
            
            # --- NEW FEATURE: Filterable Task List ---
            st.markdown("---")
            st.header("🔍 Filter and View Pending Task Details")

            col1, col2 = st.columns(2)

            with col1:
                # Create a sorted list of unique departments for the dropdown
                departments = ['All'] + sorted(pending_tasks_df['Dealing Branch'].unique().tolist())
                selected_department = st.selectbox("Select a Department", departments)

            # --- UPDATE: Dynamically update officer list based on department selection ---
            with col2:
                if selected_department == 'All':
                    # If all departments are selected, show all officers
                    officers = ['All'] + sorted(pending_tasks_df['Assign To'].unique().tolist())
                else:
                    # If a specific department is selected, show only officers from that department
                    department_specific_df = pending_tasks_df[pending_tasks_df['Dealing Branch'] == selected_department]
                    officers = ['All'] + sorted(department_specific_df['Assign To'].unique().tolist())
                
                selected_officer = st.selectbox("Select an Officer", officers)

            # Filter the dataframe based on the selections
            filtered_df = pending_tasks_df.copy()
            if selected_department != 'All':
                filtered_df = filtered_df[filtered_df['Dealing Branch'] == selected_department]
            
            if selected_officer != 'All':
                filtered_df = filtered_df[filtered_df['Assign To'] == selected_officer]

            # Display the 'File' column as clickable links
            st.dataframe(
                filtered_df,
                column_config={
                    "File": st.column_config.LinkColumn("Open File")
                }
            )

        else:
            st.warning("No pending tasks found to display.")

    # --- Page 2: Task Priority Dashboard ---
    elif page == "Task Priority Dashboard":
        st.title("📊 Task Priority Dashboard")
        st.markdown("An in-depth look at task distribution by priority level, department, and officer.")
        st.markdown("---")

        if not pending_tasks_df.empty:
            # --- Key Metrics ---
            st.header("High-Level Summary")
            total_pending = pending_tasks_df.shape[0]
            most_urgent_count = pending_tasks_df[pending_tasks_df['Priority'] == 'most urgent'].shape[0]
            high_count = pending_tasks_df[pending_tasks_df['Priority'] == 'high'].shape[0]
            medium_count = pending_tasks_df[pending_tasks_df['Priority'] == 'medium'].shape[0]
            oldest_task_days = pending_tasks_df['Days Pending'].max() if not pending_tasks_df.empty else 0

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Total Pending", total_pending)
            col2.metric("Most Urgent", most_urgent_count)
            col3.metric("High Priority", high_count)
            col4.metric("Medium Priority", medium_count)
            col5.metric("Oldest Task (Days)", oldest_task_days)
            st.markdown("---")

            def create_bar_chart(data, x_axis, title, color_by):
                # Capitalize the x-axis labels for better display in charts
                data[x_axis] = data[x_axis].str.title()
                fig = px.bar(data, x=x_axis, y='count', title=title, text='count', color=color_by, color_discrete_sequence=px.colors.qualitative.Set2)
                fig.update_traces(textposition='outside')
                fig.update_layout(yaxis_title="Task Count", xaxis_title=x_axis.replace('_', ' ').title(), showlegend=True)
                return fig

            # --- Priority Sections ---
            priority_levels = {
                "Most Urgent": "🚨",
                "High": "⚠️",
                "Medium": "🟡"
            }

            for priority, icon in priority_levels.items():
                st.header(f"{icon} {priority} Priority Tasks")
                priority_df = pending_tasks_df[pending_tasks_df['Priority'] == priority.lower()]
                
                if not priority_df.empty:
                    col1, col2 = st.columns(2)
                    with col1:
                        officer_data = priority_df['Assign To'].value_counts().reset_index()
                        st.plotly_chart(create_bar_chart(officer_data, 'Assign To', f'Officer-wise {priority} Tasks', 'Assign To'), use_container_width=True)
                    with col2:
                        # Filter out unknown/nan branches before charting
                        branch_data = priority_df[~priority_df['Dealing Branch'].isin(['unknown', 'nan'])]
                        dept_data = branch_data['Dealing Branch'].value_counts().reset_index()
                        st.plotly_chart(create_bar_chart(dept_data, 'Dealing Branch', f'Department-wise {priority} Tasks', 'Dealing Branch'), use_container_width=True)
                else:
                    st.info(f"No '{priority}' priority tasks are currently pending.")
                st.markdown("---")
            
            # --- Added an expander to show the exact data being used for the charts ---
            with st.expander("View Filtered Data for Charts"):
                st.info("This table shows the exact data being used to generate the charts above. A task is considered 'pending' if its status is not 'completed'.")
                st.dataframe(pending_tasks_df)

        else:
            st.warning("No pending tasks found to display.")
else:
    st.error("Failed to load data. Please check the Google Sheet URL and permissions.")
