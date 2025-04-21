# utils/constants.py
"""Module for storing constants used across the ETL pipeline."""

# Extraction Constants
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
)
REQUEST_TIMEOUT = 15  # seconds
REQUEST_DELAY = 0.3  # seconds between page requests

# Transformation Constants
USD_TO_IDR_RATE = 16000
# Define the final schema and data types expected after transformation
FINAL_SCHEMA_TYPE_MAPPING = {
    "title": str,
    "price": float,  # IDR Price
    "rating": float,
    "colors": int,
    "size": str,
    "gender": str,
    "image_url": str,
    # 'timestamp' is handled separately to ensure datetime type
}
# Define columns that must not be null after transformation
REQUIRED_COLUMNS = list(FINAL_SCHEMA_TYPE_MAPPING.keys()) + ["timestamp"]

# Logging Format
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"

# CSV Filepath (Example, should be configurable or passed as argument)
CSV_FILEPATH = "products.csv"

# Google Sheets (Worksheet name constant)
WORKSHEET_NAME = "products"

# PostgreSQL (Table name constant)
POSTGRES_TABLE_NAME = "products"
