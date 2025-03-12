from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent


# Watcher-specific configuration
CRAWLER_CONFIG = {
    "UPLOAD_FOLDER": BASE_DIR / "uploads",
    "PROCESSED_FOLDER": BASE_DIR / "processed"
}

# Ensure directories exist
for folder in CRAWLER_CONFIG.values():
    folder.mkdir(parents=True, exist_ok=True)