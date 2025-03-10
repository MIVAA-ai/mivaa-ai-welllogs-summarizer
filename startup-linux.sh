#!/bin/bash

# Check if one argument is passed
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <base_directory>"
  exit 1
fi

BASE_DIR="$1"

# Ensure base directory exists
if [ ! -d "$BASE_DIR" ]; then
  echo "Base directory does not exist. Creating it: $BASE_DIR"
  mkdir -p "$BASE_DIR"
fi

# Define folder paths based on the .env file structure
FOLDERS=(
  "$BASE_DIR/processed"
  "$BASE_DIR/uploads"
  "$BASE_DIR/logs"
  "$BASE_DIR/jobs/in"
  "$BASE_DIR/jobs/results"
  "$BASE_DIR/jobs/summary"
)

# Create the folders
for FOLDER in "${FOLDERS[@]}"; do
  if [ ! -d "$FOLDER" ]; then
    echo "Creating folder: $FOLDER"
    mkdir -p "$FOLDER"
  else
    echo "Folder already exists: $FOLDER"
  fi
done

# Update .env file with the new folder paths
if [ ! -f ".env" ]; then
  echo ".env file not found in the current directory."
  exit 1
fi

while IFS= read -r line || [ -n "$line" ]; do
  case "$line" in
    PROCESSED_VOLUME=*) echo "PROCESSED_VOLUME=$BASE_DIR/processed" >> updated_env.tmp ;;
    UPLOADS_VOLUME=*) echo "UPLOADS_VOLUME=$BASE_DIR/uploads" >> updated_env.tmp ;;
    LOGS_VOLUME=*) echo "LOGS_VOLUME=$BASE_DIR/logs" >> updated_env.tmp ;;
    DATA_IN_VOLUME=*) echo "DATA_IN_VOLUME=$BASE_DIR/worker/data/in" >> updated_env.tmp ;;
    DATA_RESULTS_VOLUME=*) echo "DATA_RESULTS_VOLUME=$BASE_DIR/worker/data/results" >> updated_env.tmp ;;
    SUMMARY_VOLUME=*) echo "SUMMARY_VOLUME=$BASE_DIR/worker/data/summary" >> updated_env.tmp ;;
    *) echo "$line" >> updated_env.tmp ;;
  esac
done < .env

mv updated_env.tmp .env

# Run docker-compose command
echo "Starting Docker container using docker-compose..."
docker-compose --env-file .env up --build
