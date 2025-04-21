# utils/constants.py
"""Module for storing constants used across the ETL pipeline."""

# Extraction Constants
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
)
REQUEST_TIMEOUT = 15  # seconds
REQUEST_DELAY = 0.5  # seconds between page requests

# Transformation Constants
USD_TO_IDR_RATE = 16000

# Logging Format
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"

# CSV Filepath
CSV_FILEPATH = "fashion_products.csv"

# Google Sheets
WORKSHEET_NAME = "fashion_products"

# PostgreSQL
POSTGRES_TABLE_NAME = "fashion_products"
