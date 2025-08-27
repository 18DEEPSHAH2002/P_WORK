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
    """Load data from Google Sheets (published CSV)"""
    try:
        # ✅ Use Publish-to-Web CSV link instead of export?format=csv
        sheet_url = "https://docs.google.com/spreadsheets/d/14-idXJHzHKCUQxxaqGZi-6S0G20gvPUhK4G16ci2FwI/gviz/tq?tqx=out:csv&gid=213021534"
        
        response = requests.get(sheet_url, timeout=10)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
        else:
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
        'Sr': list(range(1, 11)),
        'Marked to Officer': ['CMFO', 'DRO', 'ADC (RD)', 'ADC G', 'Legal Cell',
                              'AC G', 'DyESA', 'Election Tehsildar', 'CMFO', 'DRO'],
        'Priority': ['Most Urgent', 'Medium', 'High', 'Most Urgent', 'Medium',
                     'High', 'Most Urgent', 'Medium', 'High', 'Most Urgent'],
        'Status': ['In progress', 'Completed'] * 5,
        'Subject': [f'Task {i}' for i in range(1, 11)],
        'File': [f'{i:02d}.pdf' for i in range(1, 11)],
        'Entry Date': ['25/03/2025'] * 10,
        'Remarks': ['Remarks'] * 10
    }
    return pd.DataFrame(data)

def process_data(df):
    """Process and clean the data"""
    if 'Sr' in df.columns:
        df = df[df['Sr'].notna() & (df['Sr'] != '') & (df['Sr'] != 'Sr')]
    
    required_columns = ['Marked to Officer', 'Priority', 'Status', 'File']
    for col in required_columns:
        if col not in df.columns:
            df[col] = 'Unknown'
    
    df['Marked to Officer'] = df['Marked to Officer'].fillna('Unknown').str.strip()
    
    # ✅ Normalize priority values
    df['Priority'] = (
        df['Priority']
        .fillna('Medium')
        .astype(str)
        .str.strip()
        .str.replace(r'\s+', ' ', regex=True)  # collapse multiple spaces
        .str.lower()
    )

    priority_mapping = {
        'most urgent': 'Most Urgent',
        'urgent': 'Most Urgent',
        'high': 'High',
        'medium': 'Medium'
    }
    df['Priority'] = df['Priority'].map(priority_mapping).fillna('Medium')
    
    df['Status'] = df['Status'].fillna('In progress').str.strip()
    
    return df

def create_clickable_file_link(file_value, sr_number):
    """Create clickable file links"""
    if pd.isna(file_value) or file_value == '' or file_value == 'File':
        return "No file"
    
    if file_value.startswith('http'):
        return f'<a href="{file_value}" target="_blank">📎 {file_value.split("/")[-1]}</a>'
    
    base_drive_url = "https://drive.google.com/drive/search?q="
    search_url = f"{base_drive_url}{file_value}"
    return f'<a href="{search_url}" target="_blank">📎 {file_value}</a>'

def main():
    st.markdown('<h1 class="main-header">📊 Task Management Dashboard</h1>', unsafe_allow_html=True)
    
    df = load_data()

    # ✅ Debug: show raw priorities
    st.write("Unique raw priorities after cleaning:", df['Priority'].unique())

    pending_df = df[df['Status'] == 'In progress'].copy()
    
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select Page", ["Officer-wise Pending Tasks", "Priority-wise Analysis"])
    
    if page == "Officer-wise Pending Tasks":
        show_page1(pending_df)
    else:
        show_page2(pending_df)

# ... (keep rest of your show_page1 and show_page2 unchanged) ...

# Refresh button
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

if st.sidebar.checkbox("Auto-refresh (every 5 minutes)"):
    st.sidebar.info("Data will automatically refresh every 5 minutes")

if __name__ == "__main__":
    main()
