import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from io import StringIO
import re

# Page configuration
st.set_page_config(
    page_title="Task Management Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .dataframe {
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data():
    """Load data from Google Sheets"""
    try:
        # Google Sheets CSV export URL
        sheet_url = "https://docs.google.com/spreadsheets/d/14-idXJHzHKCUQxxaqGZi-6S0G20gvPUhK4G16ci2FwI/export?format=csv&gid=213021534"
        
        # Try to fetch data
        response = requests.get(sheet_url, timeout=10)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
        else:
            # Fallback: Create sample data based on the structure we saw
            st.warning("Unable to fetch live data. Using sample data for demonstration.")
            df = create_sample_data()
        
        return process_data(df)
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Using sample data for demonstration.")
        return process_data(create_sample_data())

def create_sample_data():
    """Create sample data based on the structure we observed"""
    data = {
        'Sr': list(range(1, 101)),
        'Marked to Officer': ['CMFO', 'DRO', 'ADC (RD)', 'ADC G', 'Legal Cell', 'AC G', 'DyESA', 'Election Tehsildar'] * 12 + ['CMFO', 'DRO', 'ADC (RD)', 'ADC G'],
        'Priority': ['Most Urgent', 'Medium', 'High'] * 33 + ['Most Urgent'],
        'Status': ['In progress', 'Completed'] * 50,
        'Subject': [f'Task {i} - Various administrative work' for i in range(1, 101)],
        'File': [f'{i:02d}.pdf' for i in range(1, 101)],
        'Entry Date': ['25/03/2025'] * 100,
        'Remarks': ['Remarks'] * 100
    }
    return pd.DataFrame(data)

def process_data(df):
    """Process and clean the data"""
    # Handle missing Sr column values (keep only non-blank/non-null Sr entries)
    if 'Sr' in df.columns:
        df = df[df['Sr'].notna() & (df['Sr'] != '') & (df['Sr'] != 'Sr')]
    
    # Ensure required columns exist
    required_columns = ['Marked to Officer', 'Priority', 'Status', 'File']
    for col in required_columns:
        if col not in df.columns:
            df[col] = 'Unknown'
    
    # Clean officer names
    df['Marked to Officer'] = df['Marked to Officer'].fillna('Unknown')
    
    # Clean priority values
    priority_mapping = {
        'Most Urgent': 'Most Urgent',
        'Medium': 'Medium', 
        'High': 'High'
    }
    df['Priority'] = df['Priority'].fillna('Medium').map(priority_mapping).fillna('Medium')
    
    # Clean status values
    df['Status'] = df['Status'].fillna('In progress')
    
    return df

def create_clickable_file_link(file_value, sr_number):
    """Create clickable file links"""
    if pd.isna(file_value) or file_value == '' or file_value == 'File':
        return "No file"
    
    # If it's already a URL, use it directly
    if file_value.startswith('http'):
        return f'<a href="{file_value}" target="_blank">📎 {file_value.split("/")[-1]}</a>'
    
    # If it's a filename, create a Google Drive search link or placeholder
    base_drive_url = "https://drive.google.com/drive/search?q="
    search_url = f"{base_drive_url}{file_value}"
    return f'<a href="{search_url}" target="_blank">📎 {file_value}</a>'

def main():
    st.markdown('<h1 class="main-header">📊 Task Management Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    # Filter for only pending tasks (Status = "In progress")
    pending_df = df[df['Status'] == 'In progress'].copy()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select Page", ["Officer-wise Pending Tasks", "Priority-wise Analysis"])
    
    if page == "Officer-wise Pending Tasks":
        show_page1(pending_df)
    else:
        show_page2(pending_df)

def show_page1(pending_df):
    """Page 1: Officer-wise pending tasks"""
    st.header("📋 Officer-wise Pending Tasks Overview")
    
    # Calculate pending tasks by officer
    officer_counts = pending_df.groupby('Marked to Officer').size().reset_index(name='Pending Tasks')
    officer_counts = officer_counts.sort_values('Pending Tasks', ascending=True)
    
    # Bar chart
    fig_bar = px.bar(
        officer_counts, 
        x='Pending Tasks', 
        y='Marked to Officer',
        orientation='h',
        title="Number of Pending Tasks by Officer",
        labels={'Pending Tasks': 'Number of Tasks', 'Marked to Officer': 'Officer'},
        color='Pending Tasks',
        color_continuous_scale='Blues'
    )
    fig_bar.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Summary table
    st.subheader("📊 Summary Table")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Officer Task Summary")
        summary_table = officer_counts.sort_values('Pending Tasks', ascending=False)
        st.dataframe(
            summary_table, 
            use_container_width=True,
            hide_index=True,
            column_config={
                "Marked to Officer": "Officer Name",
                "Pending Tasks": st.column_config.NumberColumn(
                    "Pending Tasks",
                    format="%d",
                ),
            }
        )
    
    with col2:
        st.markdown("### Quick Stats")
        total_pending = len(pending_df)
        total_officers = len(officer_counts)
        avg_tasks_per_officer = total_pending / total_officers if total_officers > 0 else 0
        max_tasks = officer_counts['Pending Tasks'].max() if len(officer_counts) > 0 else 0
        
        st.metric("Total Pending Tasks", total_pending)
        st.metric("Officers with Pending Tasks", total_officers)
        st.metric("Average Tasks per Officer", f"{avg_tasks_per_officer:.1f}")
        st.metric("Maximum Tasks (Single Officer)", max_tasks)
    
    # Officer selection and detailed view
    st.subheader("🔍 Detailed Task View by Officer")
    
    selected_officer = st.selectbox(
        "Select an Officer to view their pending tasks:",
        options=['All Officers'] + list(officer_counts['Marked to Officer'].unique()),
        index=0
    )
    
    if selected_officer != 'All Officers':
        officer_tasks = pending_df[pending_df['Marked to Officer'] == selected_officer].copy()
        
        st.markdown(f"### Tasks for: **{selected_officer}** ({len(officer_tasks)} pending)")
        
        # Create clickable file links
        officer_tasks['File Link'] = officer_tasks.apply(
            lambda row: create_clickable_file_link(row['File'], row.get('Sr', '')), 
            axis=1
        )
        
        # Display columns to show
        display_columns = ['Sr', 'Priority', 'Subject', 'Entry Date', 'File Link', 'Remarks']
        available_columns = [col for col in display_columns if col in officer_tasks.columns]
        
        st.markdown(officer_tasks[available_columns].to_html(escape=False, index=False), unsafe_allow_html=True)
        
        # Priority breakdown for selected officer
        if len(officer_tasks) > 0:
            priority_breakdown = officer_tasks['Priority'].value_counts()
            fig_pie = px.pie(
                values=priority_breakdown.values,
                names=priority_breakdown.index,
                title=f"Priority Breakdown for {selected_officer}",
                color_discrete_map={'Most Urgent': '#ff4b4b', 'High': '#ff8c00', 'Medium': '#ffd700'}
            )
            st.plotly_chart(fig_pie, use_container_width=True)

def show_page2(pending_df):
    """Page 2: Priority-wise analysis"""
    st.header("⚡ Priority-wise Task Analysis")
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    total_pending = len(pending_df)
    most_urgent_count = len(pending_df[pending_df['Priority'] == 'Most Urgent'])
    high_count = len(pending_df[pending_df['Priority'] == 'High'])
    medium_count = len(pending_df[pending_df['Priority'] == 'Medium'])
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Pending Tasks", total_pending, delta=None)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Most Urgent Tasks", most_urgent_count, delta=f"{most_urgent_count/total_pending*100:.1f}%" if total_pending > 0 else "0%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("High Priority Tasks", high_count, delta=f"{high_count/total_pending*100:.1f}%" if total_pending > 0 else "0%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Medium Priority Tasks", medium_count, delta=f"{medium_count/total_pending*100:.1f}%" if total_pending > 0 else "0%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Priority-wise officer analysis
    priorities = ['Most Urgent', 'High', 'Medium']
    colors = ['#ff4b4b', '#ff8c00', '#ffd700']
    
    for i, (priority, color) in enumerate(zip(priorities, colors)):
        st.subheader(f"{priority} Priority Tasks - Officer-wise Distribution")
        
        priority_data = pending_df[pending_df['Priority'] == priority]
        
        if len(priority_data) > 0:
            officer_priority_counts = priority_data.groupby('Marked to Officer').size().reset_index(name='Task Count')
            officer_priority_counts = officer_priority_counts.sort_values('Task Count', ascending=True)
            
            fig = px.bar(
                officer_priority_counts,
                x='Task Count',
                y='Marked to Officer',
                orientation='h',
                title=f"{priority} Priority Tasks by Officer ({len(priority_data)} total)",
                labels={'Task Count': 'Number of Tasks', 'Marked to Officer': 'Officer'},
                color_discrete_sequence=[color]
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show detailed breakdown in expander
            with st.expander(f"View {priority} Priority Task Details"):
                display_columns = ['Sr', 'Marked to Officer', 'Subject', 'Entry Date', 'Remarks']
                available_columns = [col for col in display_columns if col in priority_data.columns]
                st.dataframe(
                    priority_data[available_columns], 
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.info(f"No {priority} priority tasks found.")
        
        if i < len(priorities) - 1:
            st.markdown("---")
    
    # Overall priority distribution
    st.subheader("📊 Overall Priority Distribution")
    priority_counts = pending_df['Priority'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_pie = px.pie(
            values=priority_counts.values,
            names=priority_counts.index,
            title="Distribution of Pending Tasks by Priority",
            color_discrete_map={'Most Urgent': '#ff4b4b', 'High': '#ff8c00', 'Medium': '#ffd700'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.markdown("### Priority Summary")
        for priority, color in zip(priorities, colors):
            count = priority_counts.get(priority, 0)
            percentage = (count / total_pending * 100) if total_pending > 0 else 0
            st.markdown(f"**{priority}**: {count} tasks ({percentage:.1f}%)")

# Refresh button
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# Auto-refresh option
if st.sidebar.checkbox("Auto-refresh (every 5 minutes)"):
    st.sidebar.info("Data will automatically refresh every 5 minutes")

if __name__ == "__main__":
    main()
