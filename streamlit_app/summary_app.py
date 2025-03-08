import streamlit as st
import pandas as pd
import os

# Set paths
csv_paths = ['uploads/las_scanned_files.csv', 'uploads/dlis_scanned_files.csv']  # Both CSV files

# Load CSV data
def load_csv(paths):
    dataframes = []
    for path in paths:
        if os.path.exists(path):
            dataframes.append(pd.read_csv(path))
    return pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()

# Load summary from file
def load_summary(summary_file_path):
    try:
        with open(summary_file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        return "Summary file not found."

# Streamlit app configuration
st.set_page_config(layout="wide")
st.title('File Summary Viewer')

# Create two panes: Left pane for filenames, right pane for summary
left_col, right_col = st.columns([1, 3])

# Load CSVs
csv_data = load_csv(csv_paths)

# Left Pane: List filenames
with left_col:
    st.header("Files")
    if not csv_data.empty:
        # List of files dynamically updated
        selected_file = st.radio(
            "Select a file:",
            options=csv_data['filename'].tolist()
        )
    else:
        st.write("No files available.")
        selected_file = None

# Right Pane: Show summary for selected file
with right_col:
    st.header("File Summary")

    if selected_file:
        # Fetch summary file path from CSV record
        summary_path = csv_data[csv_data['filename'] == selected_file]['output_ai_summary_text_file'].values[0]
        summary_text = load_summary(summary_path)
        st.write(summary_text)
    else:
        st.write("Select a file to view its summary.")
