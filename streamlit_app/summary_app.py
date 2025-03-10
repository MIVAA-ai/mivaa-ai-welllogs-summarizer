import streamlit as st
import pandas as pd
import os
from utils.logger import Logger
from config.celeryconfig import las_csv_path, dlis_csv_path

# Initialize logger
summary_app_logger = Logger("streamlit_summary_app.log").get_logger()

# Set paths
csv_paths = [las_csv_path, dlis_csv_path]  # Both CSV files
summary_app_logger.info("Streamlit app started. Monitoring CSV files.")

# Function to get last modified timestamps of CSVs
def get_file_timestamps(paths):
    return {path: os.path.getmtime(path) if os.path.exists(path) else 0 for path in paths}


# Store timestamps in session state (persistent across reruns)
if "file_timestamps" not in st.session_state:
    st.session_state.file_timestamps = get_file_timestamps(csv_paths)


# Load CSV data
def load_csv(paths):
    dataframes = []
    for path in paths:
        if os.path.exists(path):
            summary_app_logger.info(f"Loading CSV: {path}")
            try:
                dataframes.append(pd.read_csv(path))
            except Exception as e:
                summary_app_logger.error(f"Error reading CSV {path}: {str(e)}")
    return pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()

# Load summary from file
def load_summary(summary_file_path):
    try:
        with open(summary_file_path, 'r') as file:
            summary_app_logger.info(f"Reading summary file: {summary_file_path}")
            return file.read()
    except FileNotFoundError:
        summary_app_logger.warning(f"Summary file not found: {summary_file_path}")
        return "Summary file not found."


# Streamlit app configuration
st.set_page_config(layout="wide")
st.title('MIVAA - AI Well Logs Summarizer')

# Check if files changed; if yes, refresh
new_timestamps = get_file_timestamps(csv_paths)
if new_timestamps != st.session_state.file_timestamps:
    st.session_state.file_timestamps = new_timestamps
    summary_app_logger.info("CSV files updated, reloading Streamlit app.")
    st.rerun()

# Create two panes: Left pane for filenames, right pane for summary
left_col, right_col = st.columns([1, 3])

# Load CSVs
csv_data = load_csv(csv_paths)

# Left Pane: List filenames
with left_col:
    st.header("Well Logs Files")

    # Manual Refresh Button
    if st.button("Refresh File List 🔄"):
        summary_app_logger.info("Manual refresh button clicked.")
        st.rerun()

    if not csv_data.empty:
        selected_file = st.selectbox(
            "Select a file:",
            options=csv_data['file_name'].tolist()
        )
        summary_app_logger.info(f"File selected: {selected_file}")
    else:
        st.write("No files available.")
        summary_app_logger.warning("No files found in CSV data.")
        selected_file = None

# Right Pane: Show summary for selected file
with right_col:
    st.header("File Summary")
    if selected_file:
        # Fetch summary file path from CSV record
        summary_path = csv_data[csv_data['file_name'] == selected_file]['output_ai_summary_text_file'].values[0]
        summary_text = load_summary(summary_path)
        st.write(summary_text)
    else:
        st.write("Select a file to view its summary.")