# main.py
"""
Main script to run the ETL pipeline for scraping fashion product data.

Orchestrates the extraction, transformation, and loading steps.
Configuration is loaded from environment variables managed via a .env file.
"""

import logging
import os
import sys

from dotenv import load_dotenv

# Ensure utils package can be found if running main.py directly
# This might be needed depending on how the project is structured and run
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.constants import (
    POSTGRES_TABLE_NAME,
    WORKSHEET_NAME,
    LOG_FORMAT,
    CSV_FILEPATH,
)
from utils.extract import scrape_all_pages
from utils.load import load_to_csv, load_to_gsheets, load_to_postgres
from utils.transform import transform_data

# Configure logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

# Load environment variables from .env file if it exists
load_dotenv()

# --- Configuration ---
SOURCE_URL = os.getenv("SOURCE_URL")
if not SOURCE_URL:
    logging.error("SOURCE_URL environment variable not set. Cannot proceed.")
    sys.exit(1)  # Exit if the source URL is missing

# Maximum number of pages to scrape
MAX_PAGES_TO_SCRAPE = 50

# API configurations
GOOGLE_SHEETS_CREDENTIALS_PATH = os.getenv("GOOGLE_SHEETS_CREDENTIALS_PATH")
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID")

# PostgreSQL configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT", 5432),  # Default PostgreSQL port
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}
POSTGRES_TABLE_NAME = os.getenv("POSTGRES_TABLE_NAME", POSTGRES_TABLE_NAME)

# Basic check for essential PostgreSQL config
POSTGRES_CONFIG_COMPLETE = all(
    DB_CONFIG.get(k) for k in ["host", "dbname", "user", "password"]
)


def run_pipeline():
    """Executes the full ETL pipeline: Extract, Transform, Load."""
    logging.info("========================================")
    logging.info("Starting ETL pipeline...")
    logging.info("========================================")

    # --- Extract ---
    logging.info(
        "Step 1: Extracting data from %s (max %d pages)...",
        SOURCE_URL,
        MAX_PAGES_TO_SCRAPE,
    )
    all_extracted_data = scrape_all_pages(SOURCE_URL, MAX_PAGES_TO_SCRAPE)

    if not all_extracted_data:
        logging.warning("Extraction yielded no data. Pipeline will terminate.")
        return  # Stop pipeline if extraction failed or returned nothing

    logging.info(
        "Extraction complete. Total raw items extracted: %d", len(all_extracted_data)
    )

    # --- Transform ---
    logging.info("Step 2: Transforming extracted data...")
    transformed_df = transform_data(all_extracted_data)

    if transformed_df is None:
        # transform_data should ideally return empty DataFrame, not None
        logging.error("Transformation step failed unexpectedly. Aborting pipeline.")
        return
    if transformed_df.empty:
        logging.warning(
            "Transformation resulted in an empty DataFrame. "
            "No data will be loaded. Check source data quality or transformation rules."
        )
        # Decide if pipeline should stop or continue (e.g., to truncate tables)
        # For now, let's stop loading if there's nothing to load.
        return

    logging.info(
        "Transformation complete. Processed %d valid products.", len(transformed_df)
    )
    logging.debug("Sample of transformed data:\n%s", transformed_df.head().to_string())

    # --- Load ---
    logging.info("Step 3: Loading transformed data...")

    # Load to CSV
    logging.info("Attempting to load data to CSV file: %s", CSV_FILEPATH)
    if load_to_csv(transformed_df, CSV_FILEPATH):
        logging.info("Successfully loaded data to %s", CSV_FILEPATH)
    else:
        logging.error(
            "Failed to load data to %s", CSV_FILEPATH
        )  # Log error but continue

    # Load to Google Sheets (Conditional)
    if GOOGLE_SHEET_ID and GOOGLE_SHEETS_CREDENTIALS_PATH:
        if os.path.exists(GOOGLE_SHEETS_CREDENTIALS_PATH):
            logging.info(
                "Attempting to load data to Google Sheet ID: %s, Worksheet: %s",
                GOOGLE_SHEET_ID,
                WORKSHEET_NAME,
            )
            if load_to_gsheets(
                transformed_df,
                GOOGLE_SHEETS_CREDENTIALS_PATH,
                GOOGLE_SHEET_ID,
                WORKSHEET_NAME,
            ):
                logging.info("Successfully loaded data to Google Sheets.")
            else:
                logging.error("Failed to load data to Google Sheets.")
        else:
            logging.warning(
                "Skipping Google Sheets load: Credentials file not found at %s",
                GOOGLE_SHEETS_CREDENTIALS_PATH,
            )
    else:
        logging.info(
            "Skipping Google Sheets load: GOOGLE_SHEET_ID or GOOGLE_SHEETS_CREDENTIALS_PATH "
            "not configured in environment variables."
        )

    # Load to PostgreSQL (Conditional)
    if POSTGRES_CONFIG_COMPLETE:
        logging.info(
            "Attempting to load data to PostgreSQL table: %s.%s",
            DB_CONFIG.get("dbname"),
            POSTGRES_TABLE_NAME,
        )
        if load_to_postgres(transformed_df, DB_CONFIG, POSTGRES_TABLE_NAME):
            logging.info("Successfully loaded data to PostgreSQL.")
        else:
            logging.error("Failed to load data to PostgreSQL.")
    else:
        logging.info(
            "Skipping PostgreSQL load: Database configuration is incomplete "
            "in environment variables (DB_HOST, DB_NAME, DB_USER, DB_PASSWORD required)."
        )

    logging.info("========================================")
    logging.info("ETL pipeline finished.")
    logging.info("========================================")


if __name__ == "__main__":
    run_pipeline()
